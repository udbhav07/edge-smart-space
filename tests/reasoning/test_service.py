"""The reasoning process as a whole: what it does on each tick, and what it never does."""

import json
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    CallSite,
    DetectorId,
    FaultClass,
    FaultDiagnosis,
    FaultEvent,
    Mode,
    ModeState,
    PreferenceHint,
    ReasoningRecord,
    Utterance,
    UtteranceSource,
)
from src.reasoning.__main__ import run
from src.reasoning.service import MAX_WAITING_UTTERANCES, build_service
from tests.reasoning.fakes import ScriptedClient, text

HINT = json.dumps(
    {
        "intent": "environment",
        "subject": "temperature",
        "comfort": "cooler",
        "target_c": None,
        "rationale": "too warm",
        "spoken_reply": "I have passed that on.",
    }
)


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


class Process:
    def __init__(self, config, replies) -> None:
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        board = Blackboard(config.mqtt, self.transport)
        self.client = ScriptedClient(list(replies))
        self.service = build_service(config, self.clock, board, client=self.client)
        self.service.subscribe()
        self.transport.attach(board)
        self.operator = Blackboard(config.mqtt, self.transport)
        self.transport.attach(self.operator)
        # Past the startup run, so a tick does only what a test asks of it.
        self.service.schedule.due()
        # A running system: section 5.6 has no path from INIT to a fault mode.
        self.operator.publish(
            topics.SYSTEM_MODE,
            ModeState(ts=self.clock.now(), mode=Mode.NORMAL, since_ts=self.clock.now()),
        )

    def say(self, words: str) -> None:
        self.operator.publish(
            topics.CONTEXT_UTTERANCE,
            Utterance(ts=self.clock.now(), text=words, source=UtteranceSource.OPERATOR),
        )

    def fault(self, fault_id="f_1") -> None:
        self.operator.publish(
            topics.FAULT,
            FaultEvent(
                fault_id=fault_id,
                detector=DetectorId.D1_DROPOUT,
                subject="temp_01",
                fault_class=FaultClass.SENSOR,
                confidence=0.95,
                detected_ts=self.clock.now(),
                evidence={"silence_s": 20.0},
                mode_impact=Mode.DEGRADED_SENSOR,
            ),
            fault_id=fault_id,
        )

    def decoded(self, topic, schema):
        return [
            schema.model_validate_json(payload)
            for name, payload, _, _ in self.transport.published
            if name == topic and payload
        ]


class TestUtterances:
    def test_nothing_is_answered_inside_the_callback(self, config):
        """The MQTT thread only records; a model call there could deadlock
        against the answers it is itself meant to deliver."""
        process = Process(config, [text(HINT)])
        process.say("it is too warm")
        assert process.client.requests == []

    def test_an_utterance_is_answered_on_the_tick(self, config):
        process = Process(config, [text(HINT)])
        process.say("it is too warm")
        process.service.tick()
        hints = process.decoded("space/context/preference", PreferenceHint)
        assert hints and hints[-1].spoken_reply == "I have passed that on."

    def test_the_audit_names_where_the_words_came_from(self, config):
        process = Process(config, [text(HINT)])
        process.say("it is too warm")
        process.service.tick()
        record = process.decoded("space/audit/reasoning", ReasoningRecord)[-1]
        assert record.trigger == "utterance (operator)"

    def test_a_backlog_is_bounded(self, config):
        process = Process(config, [text(HINT)] * (MAX_WAITING_UTTERANCES + 5))
        for index in range(MAX_WAITING_UTTERANCES + 5):
            process.say(f"request {index}")
        process.service.tick()
        assert len(process.client.requests) == MAX_WAITING_UTTERANCES


class TestFaults:
    def test_a_confirmed_fault_is_explained(self, config):
        diagnosis = {
            "primary_hypothesis": "sensor_dropout",
            "confidence": "high",
            "supporting_evidence": ["silence_s"],
            "recommended_mode": "DEGRADED_SENSOR",
            "user_message": "The temperature sensor went quiet.",
        }
        process = Process(config, [text(json.dumps(diagnosis))])
        process.fault()
        process.service.tick()
        published = process.decoded("space/diagnosis", FaultDiagnosis)
        assert published[-1].generated is True
        assert published[-1].fault_id == "f_1"

    def test_with_no_server_the_generic_explanation_is_still_published(self, config):
        process = Process(config, [ConnectionError("refused")])
        process.fault()
        process.service.tick()
        assert process.decoded("space/diagnosis", FaultDiagnosis)[-1].generated is False

    def test_a_fault_is_explained_once(self, config):
        process = Process(config, [ConnectionError("x")] * 3)
        process.fault()
        process.service.tick()
        process.service.tick()
        assert len(process.decoded("space/diagnosis", FaultDiagnosis)) == 1


class TestTheSupervisor:
    def test_it_runs_at_startup(self, config):
        process = Process(config, [text("nothing to do")])
        process.service.schedule = type(process.service.schedule)(config, process.clock)
        process.service.tick()
        records = process.decoded("space/audit/reasoning", ReasoningRecord)
        assert records[-1].call_site is CallSite.SUPERVISOR
        assert records[-1].trigger == "startup"

    def test_a_fault_wakes_it_early(self, config):
        process = Process(config, [ConnectionError("x"), text("fine")])
        process.clock.advance(config.reasoning.supervisor_min_interval_s)
        process.fault()
        process.service.tick()
        triggers = [
            r.trigger
            for r in process.decoded("space/audit/reasoning", ReasoningRecord)
            if r.call_site is CallSite.SUPERVISOR
        ]
        assert triggers and triggers[-1].startswith("fault confirmed")


class TestContainment:
    def test_one_broken_call_site_does_not_silence_the_others(self, config, monkeypatch):
        process = Process(config, [text(HINT)])

        def broken(*_args, **_kwargs):
            raise RuntimeError("a bug in the diagnosis prompt")

        monkeypatch.setattr(process.service.diagnoser, "diagnose", broken)
        process.fault()
        process.say("it is too warm")
        process.service.tick()
        assert process.decoded("space/context/preference", PreferenceHint)

    def test_it_never_publishes_to_an_actuator(self, config):
        """FR-45, for the whole process."""
        process = Process(config, [text(HINT), ConnectionError("x")])
        process.say("make it freezing")
        process.fault()
        process.service.tick()
        assert not any("/actuator/" in name for name, _, _, _ in process.transport.published)


class TestRunLoop:
    def test_it_ticks_the_requested_number_of_times(self, config):
        process = Process(config, [])
        assert run(process.service, process.clock, period_s=0.5, ticks=3) == 3

    def test_it_sleeps_between_ticks(self, config):
        process = Process(config, [])
        started = process.clock.now()
        run(process.service, process.clock, period_s=0.5, ticks=4)
        assert process.clock.now() - started == pytest.approx(2.0)
