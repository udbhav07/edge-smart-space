"""Unit tests for the Week 6 contracts: tariff, utterance, audit, diagnosis.

Each is a boundary the reasoning layer crosses, so each is tested at its
edges: what it must refuse, and what it must still be able to carry.
"""

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    LEGAL_TRANSITIONS,
    CallSite,
    DiagnosisConfidence,
    FaultDiagnosis,
    Hypothesis,
    Mode,
    ReasoningOutcome,
    ReasoningRecord,
    TariffBand,
    TariffState,
    Utterance,
    UtteranceSource,
    is_legal_transition,
)

TS = 1756032000.0


class TestLegalTransitions:
    """Section 5.6, as a table something can be checked against."""

    @pytest.mark.parametrize("mode", list(Mode))
    def test_staying_put_is_always_legal(self, mode):
        assert is_legal_transition(mode, mode)

    def test_every_mode_is_in_the_table(self):
        assert set(LEGAL_TRANSITIONS) == set(Mode)

    def test_a_sensor_fault_may_degrade_a_normal_system(self):
        assert is_legal_transition(Mode.NORMAL, Mode.DEGRADED_SENSOR)

    def test_safe_hold_cannot_be_left_for_a_degraded_mode(self):
        """Only a manual reset leaves SAFE_HOLD, and it goes to NORMAL."""
        assert not is_legal_transition(Mode.SAFE_HOLD, Mode.DEGRADED_SENSOR)

    def test_nothing_returns_to_init(self):
        assert not any(
            is_legal_transition(mode, Mode.INIT) for mode in Mode if mode is not Mode.INIT
        )

    def test_a_sensor_fault_cannot_jump_straight_to_an_actuator_fault(self):
        assert not is_legal_transition(Mode.DEGRADED_SENSOR, Mode.DEGRADED_ACTUATOR)


class TestTariffState:
    def _state(self, **overrides):
        return TariffState(
            **{
                "ts": TS,
                "band": TariffBand.PEAK,
                "since_ts": TS - 600.0,
                "next_transition_ts": TS + 3600.0,
                "offset_c": 1.0,
                **overrides,
            }
        )

    def test_a_well_formed_state_validates(self):
        assert self._state().band is TariffBand.PEAK

    def test_the_next_transition_must_follow_the_current_one(self):
        with pytest.raises(ValidationError):
            self._state(next_transition_ts=TS - 600.0)

    def test_a_negative_offset_is_refused(self):
        """Peak shifts the band up; a negative shift would cool harder at
        the expensive time."""
        with pytest.raises(ValidationError):
            self._state(offset_c=-1.0)

    def test_a_zero_offset_is_allowed(self):
        assert self._state(offset_c=0.0).offset_c == 0.0


class TestUtterance:
    def test_empty_text_is_refused(self):
        with pytest.raises(ValidationError):
            Utterance(ts=TS, text="", source=UtteranceSource.OPERATOR)

    def test_a_long_monologue_is_refused_at_the_boundary(self):
        """Bounded, so one pasted document cannot become a prompt of any size."""
        with pytest.raises(ValidationError):
            Utterance(ts=TS, text="x" * 2001, source=UtteranceSource.CONSOLE)

    def test_text_at_the_limit_is_accepted(self):
        assert Utterance(ts=TS, text="x" * 2000, source=UtteranceSource.SPEECH)


class TestReasoningRecord:
    def _record(self, **overrides):
        return ReasoningRecord(
            **{
                "ts": TS,
                "invocation_id": "rsn_1",
                "call_site": CallSite.SUPERVISOR,
                "trigger": "cadence",
                "rounds": 1,
                "outcome": ReasoningOutcome.UNAVAILABLE,
                "latency_s": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                **overrides,
            }
        )

    def test_a_call_that_never_reached_the_model_is_still_recordable(self):
        """FR-46 records every invocation, the failed ones included."""
        assert self._record().outcome is ReasoningOutcome.UNAVAILABLE

    def test_negative_latency_is_refused(self):
        with pytest.raises(ValidationError):
            self._record(latency_s=-0.1)

    def test_negative_token_counts_are_refused(self):
        with pytest.raises(ValidationError):
            self._record(prompt_tokens=-1)

    def test_a_record_must_say_what_triggered_it(self):
        with pytest.raises(ValidationError):
            self._record(trigger="")


class TestFaultDiagnosis:
    def _diagnosis(self, **overrides):
        return FaultDiagnosis(
            **{
                "ts": TS,
                "fault_id": "f_1",
                "primary_hypothesis": Hypothesis.SENSOR_STUCK,
                "confidence": DiagnosisConfidence.HIGH,
                "recommended_mode": Mode.DEGRADED_SENSOR,
                "user_message": "The temperature sensor appears stuck.",
                "generated": True,
                **overrides,
            }
        )

    def test_a_hypothesis_outside_the_fixed_set_is_refused(self):
        with pytest.raises(ValidationError):
            self._diagnosis(primary_hypothesis="gremlins")

    def test_an_empty_message_is_refused(self):
        """A diagnosis exists to be read; one with nothing to read is not one."""
        with pytest.raises(ValidationError):
            self._diagnosis(user_message="")

    def test_a_message_longer_than_a_notification_is_refused(self):
        with pytest.raises(ValidationError):
            self._diagnosis(user_message="x" * 501)

    def test_a_generic_diagnosis_says_it_was_not_generated(self):
        assert self._diagnosis(generated=False).generated is False
