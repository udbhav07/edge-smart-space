"""Weeks 5 and 6, as one running system.

Every process on one in-process bus, in lockstep with a simulated clock: the
room and its instruments (power meter included), the estimator, the detector
bank, the control loop with its tariff, the assistance executor, and the
reasoning process. Only the model is scripted -- by call site, so each of the
three answers its own questions.

What is asserted is ROADMAP.md, sentence by sentence:

* the reasoning layer proposes goals in plain language, and the gate refuses
  the unsafe ones -- demonstrably, on the audit topic;
* asking for a meeting on Thursday puts it in the calendar, and it says so;
* asking about Thursday reads it back;
* asking for a flight gets a question, not a booking -- and only a person's
  confirmation reaches the mock, which says it is a mock;
* a fault changes the mode first and is explained after (FR-26);
* killing the reasoning process costs nothing but reasoning (FR-47);
* the power meter publishes on the topic hardware will use (Week 5).
"""

import json
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service as build_executor
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Comfort,
    Command,
    FaultDiagnosis,
    Goal,
    GoalSource,
    Intent,
    Mode,
    ModeState,
    PreferenceHint,
    ReasoningOutcome,
    ReasoningRecord,
    SensorReading,
    TariffState,
    Utterance,
    UtteranceSource,
    ValidationVerdict,
    Verdict,
)
from src.common.tools import ToolResult, ToolStatus
from src.control.service import build_service as build_control
from src.control.tariff import TariffPublisher, TariffSchedule
from src.estimation.__main__ import build_service as build_estimator
from src.faults.injector import FaultInjector
from src.faults.service import build_service as build_bank
from src.reasoning.service import build_service as build_reasoning
from sim.run_sim import build_simulator
from tests.reasoning.fakes import calls, text
from tools.confirm import ConfirmationDesk

READS = (
    ("get_thermal_state", {}),
    ("get_occupancy", {}),
    ("get_tariff_state", {}),
    ("get_active_faults", {}),
)


class ModelByCallSite:
    """One scripted model, answering each call site from its own queue.

    The call site is told apart by its system prompt, which is how the real
    server would see them too: one model, three prompts (section 5.7.5).
    """

    def __init__(self) -> None:
        self.supervisor: list = []
        self.personal: list = []
        self.diagnosis: list = []
        self.requests: list[dict] = []
        from types import SimpleNamespace

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        prompt = str(request["messages"][0]["content"])
        if prompt.startswith("You are the supervisor"):
            queue = self.supervisor
        elif prompt.startswith("You explain a fault"):
            queue = self.diagnosis
        else:
            queue = self.personal
        if not queue:
            raise ConnectionError("nothing scripted for this call")
        reply = queue.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _hint(spoken, intent="service", subject="calendar"):
    return text(
        json.dumps(
            {
                "intent": intent,
                "subject": subject,
                "comfort": "unchanged",
                "target_c": None,
                "rationale": "",
                "spoken_reply": spoken,
            }
        )
    )


class System:
    def __init__(self, config) -> None:
        self.config = config
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        self.model = ModelByCallSite()

        def board():
            return Blackboard(config.mqtt, self.transport)

        boards = [board() for _ in range(9)]
        (plant, estimator, bank, control, executor, reasoning, operator, desk, tariff) = boards
        self.simulator = build_simulator(config, self.clock, plant)
        self.estimator = build_estimator(config, self.clock, estimator)
        self.bank = build_bank(config, self.clock, bank)
        self.control = build_control(config, self.clock, control)
        self.tariff = TariffPublisher(
            TariffSchedule(config.tariff, config.site.utc_offset_h), self.clock, tariff
        )
        self.executor = build_executor(config, self.clock, executor)
        self.reasoning = build_reasoning(config, self.clock, reasoning, client=self.model)
        self.desk = ConfirmationDesk(self.clock, desk)
        self.operator = operator
        self.injector = FaultInjector(operator, self.clock)
        for service in (
            self.simulator, self.estimator, self.bank, self.control,
            self.executor, self.reasoning, self.desk,
        ):
            service.subscribe()
        for each in boards:
            self.transport.attach(each)
        self.reasoning_alive = True

    def run_for(self, seconds: float) -> None:
        period = self.config.loop.sensor_period_s
        elapsed = 0.0
        while elapsed < seconds:
            self.simulator.step()
            self.estimator.tick()
            self.bank.tick()
            self.tariff.tick()
            self.control.tick()
            if self.reasoning_alive:
                self.reasoning.tick()
            self.clock.advance(period)
            elapsed += period

    def say(self, words: str) -> None:
        self.operator.publish(
            topics.CONTEXT_UTTERANCE,
            Utterance(ts=self.clock.now(), text=words, source=UtteranceSource.SPEECH),
        )
        self.run_for(self.config.loop.sensor_period_s)

    # --- what crossed the bus -----------------------------------------

    def decoded(self, topic: str, schema):
        return [
            schema.model_validate_json(payload)
            for name, payload, _, _ in self.transport.published
            if name == topic and payload
        ]

    def index_of(self, predicate) -> int:
        for index, (name, payload, _, _) in enumerate(self.transport.published):
            if predicate(name, payload):
                return index
        return -1

    def replies(self) -> list[str]:
        return [h.spoken_reply for h in self.decoded("space/context/preference", PreferenceHint)]


@pytest.fixture(name="system")
def _system(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(
        update={"calendar_path": str(tmp_path / "calendar.json")}
    )
    persistence = config.persistence.model_copy(
        update={"path": str(tmp_path / "coefficients.json")}
    )
    config = config.model_copy(update={"assistance": assistance, "persistence": persistence})
    system = System(config)
    # The startup supervisor run answers "nothing to do", so each test starts
    # from a system that has already been through one cycle.
    system.model.supervisor.append(text("Nothing to change yet."))
    system.run_for(60.0)
    return system


class TestTheGateRefusesTheUnsafe:
    """Week 6: it proposes goals in plain language and the gate refuses the
    unsafe ones -- demonstrably, not by trust."""

    def _propose(self, system, setpoint_c, rationale):
        system.model.supervisor += [
            calls(*READS),
            calls(
                (
                    "propose_setpoint",
                    {"setpoint_c": setpoint_c, "mode": "NORMAL", "rationale": rationale},
                )
            ),
        ]
        system.run_for(system.config.reasoning.supervisor_period_s)

    def test_an_unsafe_proposal_is_clamped_on_the_audit_topic(self, system):
        self._propose(system, 5.0, "Occupied and warm; cool it hard.")
        verdicts = [
            v for v in system.decoded("space/audit/validation", ValidationVerdict)
            if v.proposed.get("setpoint_c") == 5.0
        ]
        assert verdicts and verdicts[-1].verdict is Verdict.CLAMPED
        assert system.control.setpoint_c > 5.0

    def test_the_proposal_is_in_plain_language(self, system):
        self._propose(system, 25.0, "Occupied, normal tariff: 25 C keeps it comfortable.")
        goal = [g for g in system.decoded("space/goal/proposed", Goal) if g.source is GoalSource.SUPERVISOR][-1]
        assert goal.rationale.startswith("Occupied")

    def test_the_reasoning_record_carries_the_verdict(self, system):
        self._propose(system, 5.0, "cool it hard")
        record = [
            r for r in system.decoded("space/audit/reasoning", ReasoningRecord)
            if r.call_site.value == "supervisor"
        ][-1]
        assert record.outcome is ReasoningOutcome.APPLIED
        assert "CLAMPED" in record.applied

    def test_the_room_follows_the_proposal_the_gate_admitted(self, system):
        self._propose(system, 25.0, "Occupied: 25 C.")
        assert system.control.setpoint_c == 25.0
        commands = system.decoded("space/actuator/ac/command", Command)
        assert commands and all(
            c.setpoint_c in (None, 25.0) for c in commands[-5:]
        )


class TestTheCalendar:
    """Week 6: a meeting on Thursday goes in; asking about Thursday reads it."""

    def test_a_meeting_goes_in_and_it_says_so(self, system):
        # SimClock's start is Sunday 2025-08-24; Thursday is the 28th.
        system.model.personal += [
            calls(("schedule_event", {"starts_at": "2025-08-28T15:00:00", "subject": "design review"})),
            _hint("I've put the design review in for Thursday at three."),
        ]
        system.say("put the design review in on Thursday at three")
        stored = json.loads(Path(system.config.assistance.calendar_path).read_text())
        assert any(entry.get("subject") == "design review" for entry in stored)
        assert "Thursday" in system.replies()[-1]

    def test_asking_about_thursday_reads_it_back(self, system):
        system.model.personal += [
            calls(("schedule_event", {"starts_at": "2025-08-28T15:00:00", "subject": "design review"})),
            _hint("Added."),
            calls(("get_events", {"from_time": "2025-08-28T00:00:00", "to_time": "2025-08-28T23:59:00"})),
            _hint("On Thursday you have the design review at 15:00."),
        ]
        system.say("put the design review in on Thursday at three")
        system.say("what have I got on Thursday?")
        read = [r for r in system.decoded("space/assist/result", ToolResult) if r.tool == "get_events"]
        assert read[-1].status is ToolStatus.OK and "design review" in read[-1].message


class TestAFlight:
    """Week 6: asking for a flight gets you a question, not a booking."""

    FLIGHT = {
        "kind": "flight",
        "destination": "Delhi",
        "origin": "Hyderabad",
        "depart_on": "2025-08-29T09:00:00",
    }

    def _ask(self, system):
        system.model.personal += [
            calls(("book_travel", self.FLIGHT)),
            _hint("Shall I book the flight to Delhi? Please confirm.", subject="booking"),
        ]
        system.say("book me a flight to Delhi on Friday")

    def test_the_answer_is_a_question(self, system):
        self._ask(system)
        assert "confirm" in system.replies()[-1].lower()
        results = system.decoded("space/assist/result", ToolResult)
        assert results[-1].status is ToolStatus.CONFIRMATION_REQUIRED
        assert not any(r.tool == "book_travel" and r.status is ToolStatus.OK for r in results)

    def test_only_a_persons_yes_reaches_the_mock_and_it_says_so(self, system):
        self._ask(system)
        waiting = system.desk.waiting()
        assert [i.tool for i in waiting] == ["book_travel"]
        system.desk.confirm(waiting[0].invocation_id)
        booked = system.decoded("space/assist/result", ToolResult)[-1]
        assert booked.status is ToolStatus.OK and booked.simulated


class TestFaults:
    def test_the_mode_changes_before_the_explanation_arrives(self, system):
        """FR-26: the transition never waits for the model (section 5.9.2)."""
        system.model.diagnosis.append(
            text(
                json.dumps(
                    {
                        "primary_hypothesis": "sensor_dropout",
                        "confidence": "high",
                        "supporting_evidence": ["silence"],
                        "recommended_mode": "DEGRADED_SENSOR",
                        "user_message": "The temperature sensor has gone quiet.",
                    }
                )
            )
        )
        system.injector.inject("temp_01", InjectedFault.DROPOUT)
        system.run_for(120.0)
        degraded = system.index_of(
            lambda name, payload: name == "space/system/mode"
            and payload and ModeState.model_validate_json(payload).mode is Mode.DEGRADED_SENSOR
        )
        explained = system.index_of(lambda name, payload: name == "space/diagnosis")
        assert 0 <= degraded < explained
        diagnosis = system.decoded("space/diagnosis", FaultDiagnosis)[-1]
        assert diagnosis.generated and "quiet" in diagnosis.user_message


class TestKillingReasoning:
    def test_control_carries_on_without_it(self, system):
        """FR-47: the demonstration kills the reasoning process."""
        system.reasoning_alive = False
        before = len(system.decoded("space/actuator/ac/command", Command))
        system.run_for(600.0)
        after = len(system.decoded("space/actuator/ac/command", Command))
        assert after - before == pytest.approx(600.0 / system.config.loop.sensor_period_s, abs=2)

    def test_the_last_valid_setpoint_is_held_after_the_goal_expires(self, system):
        """FR-11, literally: the loop holds the last valid setpoint. The
        supervisor's goal expiring withdraws it from arbitration; with nothing
        else live there is no new winner to gate, so what the gate last
        admitted stays in force and the loop keeps commanding to it."""
        system.model.supervisor += [
            calls(*READS),
            calls(("propose_setpoint", {"setpoint_c": 25.0, "mode": "NORMAL", "rationale": "x"})),
        ]
        system.run_for(system.config.reasoning.supervisor_period_s)
        assert system.control.setpoint_c == 25.0
        system.reasoning_alive = False
        before = len(system.decoded("space/actuator/ac/command", Command))
        system.run_for(system.config.reasoning.supervisor_goal_lifetime_s + 60.0)
        assert system.control.setpoint_c == 25.0
        assert len(system.decoded("space/actuator/ac/command", Command)) > before

    def test_an_occupant_is_still_heard_with_the_supervisor_dead(self, system):
        """Arbitration and the gate live in the control process, so a spoken
        preference reaches the room with no reasoning process at all -- here
        published directly, as Personal Context would have."""
        system.reasoning_alive = False
        system.operator.publish(
            topics.CONTEXT_PREFERENCE,
            PreferenceHint(
                ts=system.clock.now(),
                intent=Intent.ENVIRONMENT,
                comfort=Comfort.COOLER,
                target_c=23.0,
                rationale="too warm",
            ),
        )
        assert system.control.setpoint_c == 23.0


class TestWeekFiveInstruments:
    def test_the_power_meter_publishes_where_hardware_will(self, system):
        readings = system.decoded("space/sensor/pwr_01/state", SensorReading)
        assert readings and readings[-1].unit.value == "W"

    def test_the_tariff_is_on_the_blackboard(self, system):
        assert system.decoded("space/context/tariff", TariffState)
