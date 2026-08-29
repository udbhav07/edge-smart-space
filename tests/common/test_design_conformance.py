"""Conformance to DESIGN.md, asserted against the document itself.

Two things are checked here that no other test covers:

* Every example payload in section 6.2 must validate, verbatim. If the
  document publishes a message shape, the code must accept exactly that
  shape. A schema that rejects the specification's own example is a
  deviation regardless of how reasonable the extra field seemed.
* The topic table in section 6.1 must be exactly what ``topics.py``
  defines: no missing entries, and no additions either. An extra topic is
  as much a deviation as a missing one, because a component publishing
  somewhere the document does not describe is invisible to anyone reading
  it.

These payloads are transcribed from the document. Do not "fix" one to make
a test pass: change the code, or change the document and then this file.
"""

import pytest

from src.common import topics
from src.common.schemas import (
    Coefficients,
    PreferenceHint,
    FaultEvent,
    Goal,
    SensorReading,
    ThermalEstimate,
    ValidationVerdict,
)

# --- Section 6.2 payloads, verbatim ----------------------------------------

SENSOR_READING_PAYLOAD = {
    "sensor_id": "temp_01",
    "ts": 1756032000.123,
    "value": 27.4,
    "unit": "C",
    "quality": "ok",
}

THERMAL_ESTIMATE_PAYLOAD = {
    "ts": 1756032000.123,
    "t_in": 27.4,
    "t_pred": 27.31,
    "residual": 0.09,
    "residual_sigma": 0.12,
    "model_confidence": 0.87,
    "adaptation": "active",
}

COEFFICIENTS_PAYLOAD = {
    "ts": 1756032000.123,
    "a1": 0.9812,
    "a2": 0.0173,
    "a3": -0.0421,
    "a4": 0.0094,
    "trace_p": 0.0031,
    "steady_state_residual": 0.0015,
    "samples_since_reset": 14203,
}

FAULT_EVENT_PAYLOAD = {
    "fault_id": "f_temp01_stuck_1756032",
    "detector": "D2_STUCK_AT",
    "subject": "temp_01",
    "class": "sensor",
    "confidence": 0.94,
    "detected_ts": 1756032300.0,
    "evidence": {"window_s": 300, "variance": 0.0002},
    "mode_impact": "DEGRADED_SENSOR",
}

GOAL_PAYLOAD = {
    "ts": 1756032000.0,
    "source": "supervisor",
    "setpoint_c": 25.5,
    "mode": "NORMAL",
    "rationale": "Occupied 42 min, peak tariff until 22:00, band shifted +1.0 C",
    "expires_ts": 1756032600.0,
}

VALIDATION_VERDICT_PAYLOAD = {
    "ts": 1756032000.4,
    "proposed": {"setpoint_c": 23.0},
    "verdict": "CLAMPED",
    "reason": "RATE_LIMIT",
    "applied": {"setpoint_c": 25.5},
}

PREFERENCE_HINT_PAYLOAD = {
    "ts": 1756032000.0,
    "intent": "environment",
    "comfort": "cooler",
    "subject": "temperature",
    "target_c": 24.0,
    "rationale": "it is too warm in here",
    "spoken_reply": "I have passed that on.",
}

DOCUMENTED_PAYLOADS = [
    (SensorReading, SENSOR_READING_PAYLOAD),
    (ThermalEstimate, THERMAL_ESTIMATE_PAYLOAD),
    (Coefficients, COEFFICIENTS_PAYLOAD),
    (FaultEvent, FAULT_EVENT_PAYLOAD),
    (Goal, GOAL_PAYLOAD),
    (ValidationVerdict, VALIDATION_VERDICT_PAYLOAD),
    (PreferenceHint, PREFERENCE_HINT_PAYLOAD),
]

# --- Section 6.1 topic table, verbatim -------------------------------------

DOCUMENTED_TOPICS = frozenset(
    {
        "space/sensor/{id}/state",
        "space/sensor/{id}/health",
        "space/estimate/thermal",
        "space/estimate/coefficients",
        "space/fault/{fault_id}",
        "space/system/mode",
        "space/goal/proposed",
        "space/goal/active",
        "space/actuator/{actuator_id}/command",
        "space/actuator/{actuator_id}/state",
        "space/context/preference",
        "space/audit/validation",
        "space/audit/reasoning",
    }
)


def _normalise(pattern: str) -> str:
    """Section 6.1 writes the sensor parameter as ``{id}``; the code names it
    ``{sensor_id}``. The wire format is identical, so compare on that."""
    return pattern.replace("{sensor_id}", "{id}")


def _defined_topics() -> frozenset[str]:
    return frozenset(
        _normalise(value.pattern)
        for value in vars(topics).values()
        if isinstance(value, topics.TopicSpec)
    )


class TestSection62Payloads:
    @pytest.mark.parametrize(
        ("model", "payload"),
        DOCUMENTED_PAYLOADS,
        ids=[model.__name__ for model, _ in DOCUMENTED_PAYLOADS],
    )
    def test_the_documented_payload_validates(self, model, payload):
        assert model.model_validate(payload) is not None

    @pytest.mark.parametrize(
        ("model", "payload"),
        DOCUMENTED_PAYLOADS,
        ids=[model.__name__ for model, _ in DOCUMENTED_PAYLOADS],
    )
    def test_the_documented_payload_survives_a_round_trip(self, model, payload):
        message = model.model_validate(payload)
        restored = model.model_validate_json(message.model_dump_json(by_alias=True))
        assert restored == message

    def test_a_fault_event_times_itself_with_detected_ts_not_ts(self):
        """Section 6.2's FaultEvent has no ``ts`` key; requiring one would
        reject the document's own payload."""
        assert "ts" not in FaultEvent.model_fields

    def test_coefficients_carry_only_the_documented_fields(self):
        assert set(Coefficients.model_fields) == set(COEFFICIENTS_PAYLOAD)

    def test_a_preference_hint_carries_more_than_a_temperature(self):
        """Section 6.4: a request about anything else must be representable."""
        hint = PreferenceHint.model_validate(PREFERENCE_HINT_PAYLOAD)
        assert hint.subject == "temperature"
        assert hint.spoken_reply

    def test_a_verdict_nests_its_decision_objects(self):
        """The nesting is what lets one verdict schema carry both the setpoint
        rules (V-1, V-2, V-6) and the command rules (V-3, V-4, V-5), which
        section 5.4 sends to the same topic."""
        verdict = ValidationVerdict.model_validate(VALIDATION_VERDICT_PAYLOAD)
        assert verdict.proposed["setpoint_c"] == 23.0
        assert verdict.applied["setpoint_c"] == 25.5


class TestSection61Topics:
    def test_no_documented_topic_is_missing(self):
        assert DOCUMENTED_TOPICS - _defined_topics() == frozenset()

    def test_no_topic_is_defined_beyond_the_documented_table(self):
        """An extra topic is a deviation too: a component publishing where the
        document does not describe is invisible to anyone reading it."""
        assert _defined_topics() - DOCUMENTED_TOPICS == frozenset()

    def test_the_real_actuator_topics_resolve_to_the_documented_strings(self):
        assert (
            topics.ACTUATOR_COMMAND.format(actuator_id=topics.AIR_CONDITIONER_ID)
            == "space/actuator/ac/command"
        )
        assert (
            topics.ACTUATOR_STATE.format(actuator_id=topics.AIR_CONDITIONER_ID)
            == "space/actuator/ac/state"
        )
