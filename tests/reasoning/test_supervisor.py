"""Unit tests for the Environmental Supervisor (FR-40, FR-41, FR-44, FR-45)."""

import json
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AdaptationState,
    FaultEvent,
    Goal,
    Mode,
    ModeState,
    ReasoningRecord,
    SensorReading,
    ThermalEstimate,
    Unit,
)
from src.control.service import build_service as build_control
from src.reasoning.__main__ import build_supervisor
from tests.reasoning.stubs import UNAVAILABLE, ScriptedChat, call, calling, saying


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


class Room:
    def __init__(self, config, chat) -> None:
        self.config = config
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(3)]
        for board in boards:
            self.transport.attach(board)
        self.control = build_control(config, self.clock, boards[0])
        self.control.subscribe()
        self.supervisor = build_supervisor(config, self.clock, boards[1], chat)
        self.supervisor.subscribe()
        self.world = boards[2]

    def estimate(self, t_in=27.0):
        self.world.publish(
            topics.ESTIMATE_THERMAL,
            ThermalEstimate(
                ts=self.clock.now(), t_in=t_in, t_pred=t_in, residual=0.0,
                residual_sigma=0.1, model_confidence=0.9, adaptation=AdaptationState.ACTIVE,
            ),
        )

    def occupancy(self, occupied: bool):
        self.world.publish(
            topics.SENSOR_STATE,
            SensorReading(ts=self.clock.now(), sensor_id="pir_01", value=1.0 if occupied else 0.0, unit=Unit.BOOLEAN),
            sensor_id="pir_01",
        )

    def mode(self, mode=Mode.NORMAL, faults=()):
        self.world.publish(
            topics.SYSTEM_MODE,
            ModeState(ts=self.clock.now(), mode=mode, since_ts=self.clock.now(), active_fault_ids=faults),
        )

    def published(self, spec, schema):
        return [
            schema.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic == spec.pattern and payload
        ]


def _cycle(*proposal):
    return (
        calling(call("get_thermal_state"), call("get_occupancy")),
        calling(call("get_tariff_state"), call("get_active_faults")),
        calling(call("propose_setpoint", **dict(proposal))),
    )


class TestScheduling:
    def test_nothing_runs_before_the_room_is_known(self, config):
        room = Room(config, ScriptedChat())
        assert room.supervisor.due() is None

    def test_the_first_cycle_runs_once_the_room_is_known(self, config):
        room = Room(config, ScriptedChat())
        room.estimate()
        assert room.supervisor.due() == "startup"

    def test_the_next_cycle_waits_for_the_period(self, config):
        room = Room(config, ScriptedChat(saying("nothing")))
        room.estimate()
        room.supervisor.maybe_run()
        room.clock.advance(config.supervisor.period_s - 1.0)
        assert room.supervisor.due() is None
        room.clock.advance(1.0)
        assert room.supervisor.due() == "periodic"

    def test_an_occupancy_transition_triggers_a_cycle(self, config):
        """FR-41: event triggers, not only the cadence."""
        room = Room(config, ScriptedChat(saying("nothing")))
        room.estimate()
        room.occupancy(True)
        room.supervisor.maybe_run()
        room.clock.advance(config.supervisor.event_holdoff_s)
        room.occupancy(False)
        assert room.supervisor.due() == "occupancy transition"

    def test_a_fault_confirmation_triggers_a_cycle(self, config):
        room = Room(config, ScriptedChat(saying("nothing")))
        room.estimate()
        room.mode()
        room.supervisor.maybe_run()
        room.clock.advance(config.supervisor.event_holdoff_s)
        room.mode(Mode.DEGRADED_SENSOR, faults=("f_1",))
        assert room.supervisor.due() == "fault confirmation"

    def test_events_inside_the_holdoff_wait(self, config):
        room = Room(config, ScriptedChat(saying("nothing")))
        room.estimate()
        room.occupancy(True)
        room.supervisor.maybe_run()
        room.occupancy(False)
        assert room.supervisor.due() is None


class TestProposing:
    def test_a_proposal_reaches_the_goal_topic_not_an_actuator(self, config):
        """FR-45: its whole influence is a proposed goal."""
        chat = ScriptedChat(*_cycle(("setpoint_c", 25.0), ("mode", "NORMAL"), ("rationale", "occupied")))
        room = Room(config, chat)
        room.estimate()
        room.mode()
        room.supervisor.maybe_run()
        goals = room.published(topics.GOAL_PROPOSED, Goal)
        assert [goal.setpoint_c for goal in goals] == [25.0]
        assert goals[0].source.value == "supervisor"

    def test_the_validator_judges_it(self, config):
        chat = ScriptedChat(*_cycle(("setpoint_c", 25.0), ("mode", "NORMAL"), ("rationale", "occupied")))
        room = Room(config, chat)
        room.estimate()
        room.mode()
        room.supervisor.maybe_run()
        assert room.control.setpoint_c == 25.0

    def test_the_model_is_told_the_verdict(self, config):
        chat = ScriptedChat(*_cycle(("setpoint_c", 5.0), ("mode", "NORMAL"), ("rationale", "cold")))
        room = Room(config, chat)
        room.estimate()
        room.mode()
        record = room.supervisor.maybe_run()
        assert "CLAMPED" in record.verdict

    def test_a_proposal_in_another_mode_is_discarded(self, config):
        """FR-44: the supervisor may restate the mode, never change it."""
        chat = ScriptedChat(*_cycle(("setpoint_c", 25.0), ("mode", "SAFE_HOLD"), ("rationale", "x")))
        room = Room(config, chat)
        room.estimate()
        room.mode()
        record = room.supervisor.maybe_run()
        assert room.published(topics.GOAL_PROPOSED, Goal) == []
        assert "discarded" in record.verdict

    def test_a_proposal_with_no_reason_is_discarded(self, config):
        chat = ScriptedChat(*_cycle(("setpoint_c", 25.0), ("mode", "NORMAL"), ("rationale", "  ")))
        room = Room(config, chat)
        room.estimate()
        room.mode()
        room.supervisor.maybe_run()
        assert room.published(topics.GOAL_PROPOSED, Goal) == []

    def test_an_unreachable_model_leaves_the_goal_alone(self, config):
        room = Room(config, ScriptedChat(UNAVAILABLE))
        room.estimate()
        record = room.supervisor.maybe_run()
        assert room.published(topics.GOAL_PROPOSED, Goal) == []
        assert "previous goal retained" in record.verdict

    def test_the_loop_stops_at_its_step_bound(self, config):
        endless = [calling(call("get_thermal_state")) for _ in range(20)]
        chat = ScriptedChat(*endless)
        room = Room(config, chat)
        room.estimate()
        room.supervisor.maybe_run()
        assert len(chat.requests) == config.supervisor.max_steps


class TestTheReadTools:
    def _read(self, room, name):
        return json.loads(room.supervisor._state.read(name))

    def test_thermal_state_reports_what_the_blackboard_said(self, config):
        room = Room(config, ScriptedChat())
        room.estimate(t_in=26.5)
        assert self._read(room, "get_thermal_state")["t_in_c"] == 26.5

    def test_unknown_occupancy_counts_as_occupied(self, config):
        """Section 7.1: conservative for comfort, as for a failed PIR."""
        room = Room(config, ScriptedChat())
        assert self._read(room, "get_occupancy")["occupied"] is True

    def test_the_setback_applies_only_after_the_configured_vacancy(self, config):
        room = Room(config, ScriptedChat())
        room.occupancy(True)
        room.occupancy(False)
        assert self._read(room, "get_occupancy")["setback_applies"] is False
        room.clock.advance(config.supervisor.setback_after_s)
        assert self._read(room, "get_occupancy")["setback_applies"] is True

    def test_active_faults_follow_the_mode(self, config):
        room = Room(config, ScriptedChat())
        room.mode(Mode.DEGRADED_SENSOR, faults=("f_1",))
        room.world.publish(
            topics.FAULT,
            FaultEvent.model_validate({
                "fault_id": "f_1", "detector": "D1_DROPOUT", "subject": "temp_01",
                "class": "sensor", "confidence": 1.0, "detected_ts": room.clock.now(),
                "evidence": {}, "mode_impact": "DEGRADED_SENSOR",
            }),
            fault_id="f_1",
        )
        faults = self._read(room, "get_active_faults")
        assert faults["mode"] == "DEGRADED_SENSOR"
        assert [fault["fault_id"] for fault in faults["faults"]] == ["f_1"]

    def test_the_tariff_is_unknown_until_published(self, config):
        room = Room(config, ScriptedChat())
        assert self._read(room, "get_tariff_state")["band"] == "unknown"


class TestTheAudit:
    def test_every_cycle_is_recorded(self, config):
        """FR-46, FR-63."""
        chat = ScriptedChat(*_cycle(("setpoint_c", 25.0), ("mode", "NORMAL"), ("rationale", "occupied")))
        room = Room(config, chat)
        room.estimate()
        room.mode()
        room.supervisor.maybe_run()
        record = room.published(topics.AUDIT_REASONING, ReasoningRecord)[-1]
        assert record.caller.value == "supervisor"
        assert record.trigger == "startup"
        assert record.tool_calls[-1] == "propose_setpoint"
        assert record.latency_s == 3.0
