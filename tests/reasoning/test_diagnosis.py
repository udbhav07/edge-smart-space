"""Unit tests for Fault Diagnosis (FR-25, FR-43, section 6.3)."""

import json
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    DetectorId,
    DiagnosisConfidence,
    FaultClass,
    FaultEvent,
    Hypothesis,
    Mode,
    ReasoningOutcome,
    ReasoningRecord,
)
from src.reasoning.audit import ReasoningAudit
from src.reasoning.diagnosis import FaultDiagnoser
from src.reasoning.endpoint import ChatEndpoint
from tests.reasoning.fakes import ScriptedClient, text

GOOD = {
    "primary_hypothesis": "sensor_stuck",
    "confidence": "high",
    "supporting_evidence": ["variance_collapse"],
    "recommended_mode": "DEGRADED_SENSOR",
    "user_message": "The temperature sensor is stuck. Running on the room model.",
}


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


def _fault(detector=DetectorId.D2_STUCK_AT, confidence=0.94, impact=Mode.DEGRADED_SENSOR):
    return FaultEvent(
        fault_id="f_temp01_stuck_1",
        detector=detector,
        subject="temp_01",
        fault_class=FaultClass.SENSOR,
        confidence=confidence,
        detected_ts=1756032300.0,
        evidence={"variance": 0.0002, "window_s": 300.0},
        mode_impact=impact,
    )


def _diagnoser(config, replies):
    clock = SimClock()
    transport = LoopbackTransport()
    board = Blackboard(config.mqtt, transport)
    transport.attach(board)
    client = ScriptedClient(list(replies))
    diagnoser = FaultDiagnoser(
        clock,
        ChatEndpoint(config.reasoning, clock, client),
        ReasoningAudit(clock, board, config.reasoning.max_audit_chars),
    )
    return diagnoser, client, transport


def _records(transport) -> list[ReasoningRecord]:
    return [
        ReasoningRecord.model_validate_json(payload)
        for topic, payload, _, _ in transport.published
        if topic == "space/audit/reasoning"
    ]


class TestAGoodAnswer:
    def test_it_is_published_as_generated(self, config):
        diagnoser, _, _ = _diagnoser(config, [text(json.dumps(GOOD))])
        diagnosis = diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert diagnosis.generated is True
        assert diagnosis.primary_hypothesis is Hypothesis.SENSOR_STUCK

    def test_it_is_audited_as_applied(self, config):
        diagnoser, _, transport = _diagnoser(config, [text(json.dumps(GOOD))])
        diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert _records(transport)[-1].outcome is ReasoningOutcome.APPLIED

    def test_the_model_is_given_no_tools(self, config):
        """FR-43: single-shot, tool-less."""
        diagnoser, client, _ = _diagnoser(config, [text(json.dumps(GOOD))])
        diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert "tools" not in client.requests[-1]

    def test_the_decode_is_constrained_to_json(self, config):
        diagnoser, client, _ = _diagnoser(config, [text(json.dumps(GOOD))])
        diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert client.requests[-1]["response_format"] == {"type": "json_object"}

    def test_the_model_sees_the_detectors_evidence(self, config):
        diagnoser, client, _ = _diagnoser(config, [text(json.dumps(GOOD))])
        diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        features = json.loads(client.requests[-1]["messages"][1]["content"])
        assert features["evidence"]["variance"] == 0.0002
        assert features["detector"] == "D2_STUCK_AT"

    def test_evidence_lists_are_bounded(self, config):
        long = dict(GOOD, supporting_evidence=["x" * 200] * 50)
        diagnoser, _, _ = _diagnoser(config, [text(json.dumps(long))])
        diagnosis = diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert len(diagnosis.supporting_evidence) == 8
        assert all(len(item) == 64 for item in diagnosis.supporting_evidence)


class TestABadAnswerIsReplaced:
    @pytest.mark.parametrize(
        "raw",
        [
            "not json",
            "[1, 2]",
            json.dumps(dict(GOOD, primary_hypothesis="gremlins")),
            json.dumps(dict(GOOD, confidence="certain")),
            json.dumps(dict(GOOD, user_message="")),
            json.dumps(dict(GOOD, supporting_evidence="variance")),
        ],
        ids=["not-json", "not-object", "hypothesis", "confidence", "empty", "evidence"],
    )
    def test_it_falls_back_to_the_generic_notification(self, config, raw):
        diagnoser, _, transport = _diagnoser(config, [text(raw)])
        diagnosis = diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert diagnosis.generated is False
        assert _records(transport)[-1].outcome is ReasoningOutcome.DISCARDED

    def test_an_illegal_recommended_mode_is_discarded(self, config):
        """Section 6.3: checked against the state machine before use."""
        illegal = dict(GOOD, recommended_mode="DEGRADED_ACTUATOR")
        diagnoser, _, transport = _diagnoser(config, [text(json.dumps(illegal))])
        diagnosis = diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert diagnosis.generated is False
        assert "section 5.6" in _records(transport)[-1].reason

    def test_a_server_that_is_down_still_yields_a_notification(self, config):
        diagnoser, _, transport = _diagnoser(config, [ConnectionError("refused")])
        diagnosis = diagnoser.diagnose(_fault(), Mode.DEGRADED_SENSOR)
        assert diagnosis.generated is False
        assert _records(transport)[-1].outcome is ReasoningOutcome.UNAVAILABLE


class TestTheGenericNotification:
    @pytest.mark.parametrize(
        ("detector", "hypothesis"),
        [
            (DetectorId.D1_DROPOUT, Hypothesis.SENSOR_DROPOUT),
            (DetectorId.D2_STUCK_AT, Hypothesis.SENSOR_STUCK),
            (DetectorId.D3_OUT_OF_RANGE, Hypothesis.SENSOR_OUT_OF_RANGE),
            (DetectorId.D4_DRIFT, Hypothesis.SENSOR_DRIFT),
            (DetectorId.D5_ACTUATOR_NO_RESPONSE, Hypothesis.ACTUATOR_NO_RESPONSE),
            (DetectorId.MODEL_DIVERGENCE, Hypothesis.MODEL_DIVERGENCE),
        ],
    )
    def test_it_names_what_the_detector_found(self, config, detector, hypothesis):
        diagnoser, _, _ = _diagnoser(config, [])
        assert (
            diagnoser.generic(_fault(detector), Mode.NORMAL).primary_hypothesis
            is hypothesis
        )

    def test_it_names_the_subject_in_plain_words(self, config):
        diagnoser, _, _ = _diagnoser(config, [])
        assert "temp_01" in diagnoser.generic(_fault(), Mode.NORMAL).user_message

    @pytest.mark.parametrize(
        ("confidence", "coarse"),
        [
            (0.95, DiagnosisConfidence.HIGH),
            (0.9, DiagnosisConfidence.HIGH),
            (0.7, DiagnosisConfidence.MEDIUM),
            (0.3, DiagnosisConfidence.LOW),
        ],
    )
    def test_confidence_is_coarsened(self, config, confidence, coarse):
        diagnoser, _, _ = _diagnoser(config, [])
        assert diagnoser.generic(_fault(confidence=confidence), Mode.NORMAL).confidence is coarse

    def test_it_never_recommends_an_illegal_mode(self, config):
        """From SAFE_HOLD only NORMAL is reachable; a sensor fault's
        DEGRADED_SENSOR impact is not recommended from there."""
        diagnoser, _, _ = _diagnoser(config, [])
        diagnosis = diagnoser.generic(_fault(), Mode.SAFE_HOLD)
        assert diagnosis.recommended_mode is Mode.SAFE_HOLD
