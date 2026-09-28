"""Unit tests for Fault Diagnosis (FR-25, FR-43, FR-44; section 6.3)."""

import json
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    FaultDiagnosis,
    FaultEvent,
    Mode,
    ModeState,
    ReasoningRecord,
)
from src.reasoning.diagnosis import FaultDiagnoser
from tests.reasoning.stubs import UNAVAILABLE, ScriptedChat, saying

STUCK = {
    "fault_id": "f_temp01_stuck_1",
    "detector": "D2_STUCK_AT",
    "subject": "temp_01",
    "class": "sensor",
    "confidence": 1.0,
    "detected_ts": 1756032300.0,
    "evidence": {"variance": 0.0, "window_s": 300.0},
    "mode_impact": "DEGRADED_SENSOR",
}

GOOD = json.dumps(
    {
        "primary_hypothesis": "sensor_stuck",
        "confidence": "high",
        "supporting_evidence": ["variance_collapse"],
        "recommended_mode": "DEGRADED_SENSOR",
        "user_message": "The temperature sensor looks stuck; the room is running on its model.",
    }
)


class Room:
    def __init__(self, chat) -> None:
        config = load_config(Path("config/default.yaml"))
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(2)]
        for board in boards:
            self.transport.attach(board)
        self.diagnoser = FaultDiagnoser(self.clock, boards[0], chat)
        self.diagnoser.subscribe()
        self.world = boards[1]
        self.world.publish(
            topics.SYSTEM_MODE,
            ModeState(ts=self.clock.now(), mode=Mode.DEGRADED_SENSOR, since_ts=self.clock.now()),
        )

    def fault(self, **overrides):
        event = FaultEvent.model_validate({**STUCK, **overrides})
        self.world.publish(topics.FAULT, event, fault_id=event.fault_id)
        return self.diagnoser.process_pending()

    def published(self, prefix, schema):
        return [
            schema.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic.startswith(prefix) and payload
        ]


def test_a_confirmed_fault_is_explained():
    room = Room(ScriptedChat(saying(GOOD)))
    diagnosis = room.fault()[0]
    assert diagnosis.primary_hypothesis.value == "sensor_stuck"
    assert diagnosis.generic is False


def test_the_explanation_is_published_for_that_fault():
    room = Room(ScriptedChat(saying(GOOD)))
    room.fault()
    published = room.published("space/diagnosis/", FaultDiagnosis)
    assert [d.fault_id for d in published] == ["f_temp01_stuck_1"]


def test_the_model_gets_the_detectors_evidence_and_no_tools():
    chat = ScriptedChat(saying(GOOD))
    room = Room(chat)
    room.fault()
    messages, tools = chat.requests[0]
    assert tools == ()
    assert json.loads(messages[1]["content"])["evidence"]["variance"] == 0.0


def test_an_illegal_recommendation_is_marked_and_changes_nothing():
    """Section 6.3: the detector-derived mode stands."""
    illegal = json.loads(GOOD) | {"recommended_mode": "DEGRADED_ACTUATOR"}
    room = Room(ScriptedChat(saying(json.dumps(illegal))))
    diagnosis = room.fault()[0]
    assert diagnosis.recommendation_legal is False
    assert room.published("space/system/mode", ModeState)[-1].mode is Mode.DEGRADED_SENSOR


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        json.dumps(json.loads(GOOD) | {"primary_hypothesis": "gremlins"}),
        json.dumps(json.loads(GOOD) | {"confidence": "certain"}),
        json.dumps({k: v for k, v in json.loads(GOOD).items() if k != "user_message"}),
    ],
    ids=["unparseable", "unknown-hypothesis", "unknown-confidence", "no-message"],
)
def test_unusable_output_falls_back_to_the_generic_notification(raw):
    """FR-44, section 5.7.1."""
    room = Room(ScriptedChat(saying(raw)))
    diagnosis = room.fault()[0]
    assert diagnosis.generic is True
    assert "stuck" in diagnosis.user_message


def test_an_unreachable_model_still_gets_the_occupant_told():
    room = Room(ScriptedChat(UNAVAILABLE))
    diagnosis = room.fault()[0]
    assert diagnosis.generic is True


def test_a_fault_is_explained_once():
    """A retained fault redelivered on reconnect is not explained again."""
    chat = ScriptedChat(saying(GOOD), saying(GOOD))
    room = Room(chat)
    room.fault()
    room.fault()
    assert len(chat.requests) == 1


def test_every_diagnosis_is_audited():
    """FR-46, FR-63."""
    room = Room(ScriptedChat(saying(GOOD, latency_s=2.0)))
    room.fault()
    record = room.published("space/audit/reasoning", ReasoningRecord)[-1]
    assert record.caller.value == "diagnosis"
    assert record.latency_s == 2.0
