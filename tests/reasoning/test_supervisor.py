"""The Environmental Supervisor, against the real gate (FR-40, FR-41, FR-44).

The model is scripted; everything else is real. The supervisor's snapshot,
its tools, and the control service's arbitration and validator share one
in-process bus, so a verdict the model is handed is one the actual gate
produced -- which is the only way "the gate refuses the unsafe ones" means
anything.
"""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AdaptationState,
    CallSite,
    Comfort,
    DetectorId,
    FaultClass,
    FaultEvent,
    Goal,
    GoalSource,
    Intent,
    Mode,
    ModeState,
    PreferenceHint,
    ReasonCode,
    ReasoningOutcome,
    ReasoningRecord,
    SensorReading,
    TariffBand,
    TariffState,
    ThermalEstimate,
    Unit,
    Verdict,
)
from src.control.service import build_service as build_control
from src.reasoning.audit import ReasoningAudit
from src.reasoning.endpoint import ChatEndpoint
from src.reasoning.supervisor_agent import (
    EnvironmentalSupervisor,
    SupervisorSchedule,
    supervisor_prompt,
)
from src.reasoning.supervisor_tools import RoomSnapshot, SupervisorTools
from tests.reasoning.fakes import ScriptedClient, calls, text

READS = (
    ("get_thermal_state", {}),
    ("get_occupancy", {}),
    ("get_tariff_state", {}),
    ("get_active_faults", {}),
)


def _propose(setpoint_c, mode="NORMAL", rationale="occupied, normal tariff"):
    return calls(
        (
            "propose_setpoint",
            {"setpoint_c": setpoint_c, "mode": mode, "rationale": rationale},
        )
    )


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


class Room:
    """A supervisor and, optionally, the real gate, on one bus."""

    def __init__(self, config, replies, with_gate=True) -> None:
        self.config = config
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        self.board = Blackboard(config.mqtt, self.transport)
        self.client = ScriptedClient(list(replies))
        self.snapshot = RoomSnapshot(config, self.clock, self.board)
        self.snapshot.subscribe()
        self.tools = SupervisorTools(config, self.clock, self.board, self.snapshot)
        self.supervisor = EnvironmentalSupervisor(
            config,
            self.clock,
            ChatEndpoint(config.reasoning, self.clock, self.client),
            self.tools,
            ReasoningAudit(self.clock, self.board, config.reasoning.max_audit_chars),
        )
        self.transport.attach(self.board)
        self.control = None
        if with_gate:
            control_board = Blackboard(config.mqtt, self.transport)
            self.control = build_control(config, self.clock, control_board)
            self.control.subscribe()
            self.transport.attach(control_board)
        self.operator = Blackboard(config.mqtt, self.transport)
        self.transport.attach(self.operator)

    def publish(self, spec, message, **parameters) -> None:
        self.operator.publish(spec, message, **parameters)

    def set_mode(self, mode: Mode, fault_ids=()) -> None:
        self.publish(
            topics.SYSTEM_MODE,
            ModeState(
                ts=self.clock.now(),
                mode=mode,
                since_ts=self.clock.now(),
                active_fault_ids=tuple(fault_ids),
            ),
        )

    def set_occupied(self, occupied: bool) -> None:
        self.publish(
            topics.SENSOR_STATE,
            SensorReading(
                ts=self.clock.now(),
                sensor_id="pir_01",
                value=1.0 if occupied else 0.0,
                unit=Unit.BOOLEAN,
            ),
            sensor_id="pir_01",
        )

    def set_tariff(self, band: TariffBand) -> None:
        now = self.clock.now()
        self.publish(
            topics.CONTEXT_TARIFF,
            TariffState(
                ts=now,
                band=band,
                since_ts=now - 60.0,
                next_transition_ts=now + 3600.0,
                offset_c=1.0,
            ),
        )

    def set_temperature(self, t_in: float = 26.4, t_pred: float = 26.3) -> None:
        self.publish(
            topics.ESTIMATE_THERMAL,
            ThermalEstimate(
                ts=self.clock.now(),
                t_in=t_in,
                t_pred=t_pred,
                residual=t_in - t_pred,
                residual_sigma=0.15,
                model_confidence=0.8,
                adaptation=AdaptationState.ACTIVE,
            ),
        )

    def raise_fault(self, fault_id="f_temp01_stuck_1") -> FaultEvent:
        event = FaultEvent(
            fault_id=fault_id,
            detector=DetectorId.D2_STUCK_AT,
            subject="temp_01",
            fault_class=FaultClass.SENSOR,
            confidence=0.9,
            detected_ts=self.clock.now(),
            evidence={"variance": 0.0},
            mode_impact=Mode.DEGRADED_SENSOR,
        )
        self.publish(topics.FAULT, event, fault_id=fault_id)
        return event

    def _decoded(self, topic, schema):
        return [
            schema.model_validate_json(payload)
            for name, payload, _, _ in self.transport.published
            if name == topic and payload
        ]

    def proposals(self) -> list[Goal]:
        return self._decoded("space/goal/proposed", Goal)

    def records(self) -> list[ReasoningRecord]:
        return self._decoded("space/audit/reasoning", ReasoningRecord)


@pytest.fixture(name="room")
def _room(config):
    room = Room(config, [])
    room.set_mode(Mode.NORMAL)
    room.set_occupied(True)
    room.set_tariff(TariffBand.NORMAL)
    room.set_temperature()
    return room


class TestReadTools:
    def test_the_thermal_state_is_what_the_estimator_published(self, room):
        state = room.tools.execute("get_thermal_state", "{}").content
        assert state["t_in_c"] == 26.4 and state["t_pred_c"] == 26.3

    def test_the_setpoint_reported_is_the_one_in_force(self, room, config):
        state = room.tools.execute("get_thermal_state", "{}").content
        assert state["t_setpoint_c"] == config.controller.default_setpoint_c

    def test_occupancy_is_read_from_the_derived_sensor(self, room):
        assert room.tools.execute("get_occupancy", "{}").content["occupied"] is True

    def test_an_empty_room_says_for_how_long(self, room):
        room.set_occupied(False)
        room.clock.advance(600.0)
        occupancy = room.tools.execute("get_occupancy", "{}").content
        assert occupancy["vacancy_duration_s"] == 600.0

    def test_the_tariff_names_its_band_and_offset(self, room):
        room.set_tariff(TariffBand.PEAK)
        tariff = room.tools.execute("get_tariff_state", "{}").content
        assert tariff["band"] == "peak" and tariff["offset_c"] == 1.0

    def test_times_are_given_as_local_clock_times(self, room):
        """A model reasons about 22:00, not about 1756044000."""
        tariff = room.tools.execute("get_tariff_state", "{}").content
        assert "T" in tariff["next_transition"]

    def test_an_unknown_tariff_is_said_to_be_unknown(self, config):
        room = Room(config, [])
        assert room.tools.execute("get_tariff_state", "{}").content["band"] == "unknown"

    def test_active_faults_come_with_the_mode(self, room):
        room.raise_fault("f_1")
        room.set_mode(Mode.DEGRADED_SENSOR, ["f_1"])
        answer = room.tools.execute("get_active_faults", "{}").content
        assert answer["mode"] == "DEGRADED_SENSOR"
        assert answer["faults"][0]["sensor"] == "temp_01"

    def test_a_retired_fault_is_no_longer_active(self, room):
        room.raise_fault("f_1")
        room.set_mode(Mode.DEGRADED_SENSOR, ["f_1"])
        room.set_mode(Mode.NORMAL, [])
        assert room.tools.execute("get_active_faults", "{}").content["faults"] == []

    def test_a_read_tool_given_arguments_says_so(self, room):
        answer = room.tools.execute("get_occupancy", '{"room": "kitchen"}')
        assert "error" in answer.content and not answer.discarded

    def test_an_unknown_tool_is_an_error_not_a_crash(self, room):
        answer = room.tools.execute("open_window", "{}")
        assert "error" in answer.content and not answer.discarded

    def test_arguments_that_are_not_json_are_an_error(self, room):
        assert "error" in room.tools.execute("get_occupancy", "{oops").content

    def test_the_model_is_shown_exactly_five_tools(self, room):
        names = [schema["function"]["name"] for schema in room.tools.schemas()]
        assert names == [
            "get_thermal_state",
            "get_occupancy",
            "get_tariff_state",
            "get_active_faults",
            "propose_setpoint",
        ]


class TestTriggers:
    """FR-41: occupancy, tariff and fault transitions wake the supervisor."""

    def test_the_first_occupancy_reading_is_not_a_transition(self, config):
        room = Room(config, [])
        room.set_occupied(True)
        assert room.snapshot.drain_events() == []

    def test_an_occupancy_transition_is_an_event(self, room):
        room.snapshot.drain_events()
        room.set_occupied(False)
        assert room.snapshot.drain_events() == ["occupancy: the room became empty"]

    def test_a_repeated_reading_is_not_a_transition(self, room):
        room.snapshot.drain_events()
        room.set_occupied(True)
        assert room.snapshot.drain_events() == []

    def test_a_tariff_transition_is_an_event(self, room):
        room.snapshot.drain_events()
        room.set_tariff(TariffBand.PEAK)
        assert room.snapshot.drain_events() == ["tariff: normal -> peak"]

    def test_a_confirmed_fault_is_an_event_and_a_diagnosis_input(self, room):
        room.snapshot.drain_events()
        room.raise_fault("f_9")
        assert room.snapshot.drain_events()[0].startswith("fault confirmed")
        assert [event.fault_id for event in room.snapshot.drain_new_faults()] == ["f_9"]

    def test_a_fault_seen_twice_is_one_event(self, room):
        room.snapshot.drain_events()
        room.raise_fault("f_9")
        room.raise_fault("f_9")
        assert len(room.snapshot.drain_events()) == 1


class TestProposeSetpoint:
    def test_a_proposal_reaches_the_gate_and_its_verdict_comes_back(self, room):
        answer = room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 25.0, "mode": "NORMAL", "rationale": "occupied"}',
        )
        assert answer.verdict is not None
        assert answer.content["verdict"] == "ACCEPTED"
        assert room.control.setpoint_c == 25.0

    def test_the_proposal_is_the_supervisors(self, room):
        room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 25.0, "mode": "NORMAL", "rationale": "occupied"}',
        )
        assert room.proposals()[-1].source is GoalSource.SUPERVISOR

    def test_an_unsafe_request_is_refused_by_the_gate_visibly(self, room):
        """5 C passes the post-decode check -- it is a temperature -- and the
        validator refuses it where everyone can see (section 5.4)."""
        answer = room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 5.0, "mode": "NORMAL", "rationale": "make it cold"}',
        )
        assert answer.content["verdict"] == "CLAMPED"
        assert answer.content["applied_setpoint_c"] > 5.0
        assert room.control.setpoint_c > 5.0

    def test_a_number_that_is_not_a_room_temperature_never_reaches_the_gate(
        self, room
    ):
        answer = room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 500.0, "mode": "NORMAL", "rationale": "x"}',
        )
        assert answer.discarded
        assert room.proposals() == []

    def test_a_mode_the_system_is_not_in_is_discarded(self, room):
        """The supervisor reports the mode; choosing one is not its job."""
        answer = room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 24.0, "mode": "SAFE_HOLD", "rationale": "x"}',
        )
        assert "not the system's mode" in answer.discarded

    def test_a_proposal_with_no_reason_is_discarded(self, room):
        answer = room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 24.0, "mode": "NORMAL", "rationale": "  "}',
        )
        assert answer.discarded

    def test_a_proposal_missing_an_argument_is_discarded(self, room):
        answer = room.tools.execute("propose_setpoint", '{"setpoint_c": 24.0}')
        assert answer.discarded and room.proposals() == []

    def test_an_occupant_outranks_the_supervisor_and_the_model_is_told(self, room):
        room.publish(
            topics.CONTEXT_PREFERENCE,
            PreferenceHint(
                ts=room.clock.now(),
                intent=Intent.ENVIRONMENT,
                comfort=Comfort.COOLER,
                target_c=23.0,
                rationale="too warm",
            ),
        )
        answer = room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 26.0, "mode": "NORMAL", "rationale": "peak"}',
        )
        assert answer.content["reason"] == ReasonCode.OUTRANKED.value
        assert room.control.setpoint_c == 23.0

    def test_with_no_gate_running_the_proposal_is_submitted_not_lost(self, config):
        room = Room(config, [], with_gate=False)
        room.set_mode(Mode.NORMAL)
        answer = room.tools.execute(
            "propose_setpoint",
            '{"setpoint_c": 25.0, "mode": "NORMAL", "rationale": "occupied"}',
        )
        assert answer.content["status"] == "submitted"
        assert room.proposals()[-1].setpoint_c == 25.0


class TestTheAgentLoop:
    def _run(self, room, replies, trigger="cadence"):
        room.client.replies.extend(replies)
        return room.supervisor.run(trigger)

    def test_it_reads_then_proposes_and_the_record_says_so(self, room):
        run = self._run(room, [calls(*READS), _propose(25.0)])
        assert run.record.outcome is ReasoningOutcome.APPLIED
        assert run.record.tool_calls[-1] == "propose_setpoint"
        assert len(run.record.tool_calls) == 5

    def test_the_verdict_is_in_the_audit_record(self, room):
        run = self._run(room, [calls(*READS), _propose(5.0)])
        assert run.verdict.verdict is Verdict.CLAMPED
        assert "CLAMPED" in run.record.applied

    def test_every_run_is_published_to_the_audit_topic(self, room):
        self._run(room, [calls(*READS), _propose(25.0)])
        assert room.records()[-1].call_site is CallSite.SUPERVISOR

    def test_latency_and_tokens_are_recorded(self, room):
        run = self._run(room, [calls(*READS), _propose(25.0)])
        assert run.record.rounds == 2 and run.record.prompt_tokens > 0

    def test_a_server_that_is_down_proposes_nothing(self, room):
        """FR-47: the loop holds its setpoint and the record says why."""
        before = room.control.setpoint_c
        run = self._run(room, [ConnectionError("refused")])
        assert run.record.outcome is ReasoningOutcome.UNAVAILABLE
        assert room.proposals() == [] and room.control.setpoint_c == before

    def test_an_answer_in_words_changes_nothing(self, room):
        run = self._run(room, [text("The room seems fine.")])
        assert run.record.outcome is ReasoningOutcome.NO_ACTION
        assert room.proposals() == []

    def test_a_discarded_proposal_ends_the_run(self, room):
        """FR-44: discarded, not negotiated."""
        run = self._run(room, [_propose(500.0), _propose(24.0)])
        assert run.record.outcome is ReasoningOutcome.DISCARDED
        assert room.proposals() == []
        assert room.client.replies  # the second reply was never asked for

    def test_running_out_of_rounds_is_discarded(self, room, config):
        endless = [calls(("get_occupancy", {}))] * config.reasoning.supervisor_max_rounds
        run = self._run(room, endless)
        assert run.record.outcome is ReasoningOutcome.DISCARDED
        assert "rounds" in run.record.reason

    def test_a_mistaken_tool_call_can_be_recovered_from(self, room):
        run = self._run(
            room, [calls(("open_window", {})), calls(*READS), _propose(25.0)]
        )
        assert run.record.outcome is ReasoningOutcome.APPLIED

    def test_the_model_is_never_offered_an_assistance_tool(self, room):
        """Booking a flight is not the supervisor's business (section 5.7.1)."""
        self._run(room, [calls(*READS), _propose(25.0)])
        assert "book_travel" not in room.client.tool_names_offered(0)

    def test_the_policy_in_the_prompt_comes_from_config(self, config):
        prompt = supervisor_prompt(config)
        assert f"{config.controller.default_setpoint_c:.1f} C" in prompt
        assert f"{config.reasoning.vacancy_relax_c:.1f} C" in prompt

    def test_the_trigger_is_named_in_the_request_and_the_record(self, room):
        run = self._run(room, [text("fine")], trigger="tariff: normal -> peak")
        assert run.record.trigger == "tariff: normal -> peak"
        assert "tariff: normal -> peak" in room.client.requests[0]["messages"][1]["content"]


class TestSchedule:
    def test_the_first_run_is_at_startup(self, config):
        assert SupervisorSchedule(config, SimClock()).due() == "startup"

    def test_nothing_is_due_straight_after_a_run(self, config):
        schedule = SupervisorSchedule(config, SimClock())
        schedule.due()
        assert schedule.due() is None

    def test_the_cadence_comes_round(self, config):
        clock = SimClock()
        schedule = SupervisorSchedule(config, clock)
        schedule.due()
        clock.advance(config.reasoning.supervisor_period_s)
        assert schedule.due() == "cadence"

    def test_an_event_triggers_a_run_early(self, config):
        clock = SimClock()
        schedule = SupervisorSchedule(config, clock)
        schedule.due()
        clock.advance(config.reasoning.supervisor_min_interval_s)
        schedule.notice(["occupancy: the room became empty"])
        assert schedule.due() == "occupancy: the room became empty"

    def test_an_event_inside_the_spacing_waits_rather_than_vanishing(self, config):
        clock = SimClock()
        schedule = SupervisorSchedule(config, clock)
        schedule.due()
        schedule.notice(["tariff: normal -> peak"])
        assert schedule.due() is None
        clock.advance(config.reasoning.supervisor_min_interval_s)
        assert schedule.due() == "tariff: normal -> peak"

    def test_many_events_are_bounded(self, config):
        clock = SimClock()
        schedule = SupervisorSchedule(config, clock)
        schedule.due()
        schedule.notice([f"e{index}" for index in range(100)])
        clock.advance(config.reasoning.supervisor_min_interval_s)
        assert schedule.due().count(";") < 8
