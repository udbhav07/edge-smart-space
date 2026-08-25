"""Unit tests for the blackboard message contracts.

Field names and payload shapes asserted here are the interface frozen at
Week 2 (R-06). A failure is an interface change, not a test to edit.
"""

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    AckStatus,
    ActuatorState,
    AdaptationState,
    BlackboardMessage,
    Coefficients,
    Command,
    CommandKind,
    DetectorId,
    FaultClass,
    FaultEvent,
    Goal,
    GoalSource,
    Mode,
    ModeState,
    Quality,
    ReasonCode,
    SensorHealth,
    SensorReading,
    ThermalEstimate,
    Unit,
    ValidationVerdict,
    Verdict,
)

TS = 1756032000.123
SENSOR_ID = "temp_01"
SETPOINT_C = 25.5


def _reading(**overrides) -> SensorReading:
    return SensorReading(
        **{
            "ts": TS,
            "sensor_id": SENSOR_ID,
            "value": 27.4,
            "unit": Unit.CELSIUS,
            **overrides,
        }
    )


class TestBaseContract:
    """Properties every blackboard message shares."""

    def test_a_message_is_immutable(self):
        with pytest.raises(ValidationError):
            _reading().value = 30.0

    def test_an_unknown_field_is_rejected_rather_than_silently_dropped(self):
        with pytest.raises(ValidationError):
            _reading(temperature=27.4)

    def test_a_non_positive_timestamp_is_rejected(self):
        with pytest.raises(ValidationError):
            _reading(ts=0.0)

    def test_every_message_type_derives_from_the_base_contract(self):
        for model in (
            SensorReading,
            SensorHealth,
            ThermalEstimate,
            Coefficients,
            FaultEvent,
            ModeState,
            Goal,
            ValidationVerdict,
            Command,
            ActuatorState,
        ):
            assert issubclass(model, BlackboardMessage)


class TestSensorReading:
    def test_serialises_to_the_documented_payload(self):
        assert _reading().model_dump() == {
            "ts": TS,
            "sensor_id": SENSOR_ID,
            "value": 27.4,
            "unit": Unit.CELSIUS,
            "quality": Quality.OK,
        }

    def test_defaults_to_trusted_quality(self):
        assert _reading().quality is Quality.OK

    def test_rejects_an_empty_sensor_id(self):
        with pytest.raises(ValidationError):
            _reading(sensor_id="")

    def test_rejects_an_unrecognised_unit(self):
        with pytest.raises(ValidationError):
            _reading(unit="kelvin")

    def test_a_boolean_reading_carrying_a_temperature_is_rejected(self):
        """A PIR reporting 27.4 is a malformed message, not a sensor fault."""
        with pytest.raises(ValidationError):
            _reading(sensor_id="pir_01", unit=Unit.BOOLEAN, value=27.4)

    @pytest.mark.parametrize("value", [0.0, 1.0])
    def test_a_boolean_reading_accepts_both_occupancy_states(self, value):
        assert _reading(sensor_id="pir_01", unit=Unit.BOOLEAN, value=value).value == value

    @pytest.mark.parametrize("value", [-40.0, 150.0])
    def test_an_out_of_range_temperature_stays_representable_for_d3(self, value):
        """FR-22 detects out-of-range readings, so they must reach the detector."""
        assert _reading(value=value).value == value

    def test_an_out_of_range_humidity_stays_representable_for_d3(self):
        assert _reading(unit=Unit.PERCENT_RH, value=120.0).value == 120.0


class TestThermalEstimate:
    def test_model_confidence_is_bounded_to_the_unit_interval(self):
        with pytest.raises(ValidationError):
            ThermalEstimate(
                ts=TS,
                t_in=27.4,
                t_pred=27.31,
                residual=0.09,
                residual_sigma=0.12,
                model_confidence=1.2,
                adaptation=AdaptationState.ACTIVE,
            )

    def test_residual_sigma_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            ThermalEstimate(
                ts=TS,
                t_in=27.4,
                t_pred=27.31,
                residual=0.09,
                residual_sigma=-0.01,
                model_confidence=0.87,
                adaptation=AdaptationState.ACTIVE,
            )

    def test_a_residual_contradicting_its_inputs_is_rejected(self):
        """An inconsistent residual would corrupt D4's CUSUM silently."""
        with pytest.raises(ValidationError):
            ThermalEstimate(
                ts=TS,
                t_in=27.4,
                t_pred=27.31,
                residual=99.0,
                residual_sigma=0.12,
                model_confidence=0.87,
                adaptation=AdaptationState.ACTIVE,
            )

    def test_the_documented_payload_survives_float_round_off(self):
        """27.4 - 27.31 is not exactly 0.09 in binary floating point."""
        estimate = ThermalEstimate(
            ts=TS,
            t_in=27.4,
            t_pred=27.31,
            residual=0.09,
            residual_sigma=0.12,
            model_confidence=0.87,
            adaptation=AdaptationState.ACTIVE,
        )
        assert estimate.residual == 0.09

    def test_a_zero_residual_is_valid_when_prediction_matches_measurement(self):
        estimate = ThermalEstimate(
            ts=TS,
            t_in=27.4,
            t_pred=27.4,
            residual=0.0,
            residual_sigma=0.12,
            model_confidence=0.87,
            adaptation=AdaptationState.FROZEN,
        )
        assert estimate.residual == 0.0


class TestCoefficients:
    def test_accepts_an_implausible_estimate_so_a_rejection_can_be_logged(self):
        """FR-06 logs rejected estimates; the message must be able to carry one.

        a3 positive means the identifier believes the air conditioner heats the
        room. That is exactly what the estimator's projection step must reject
        and publish, so the schema must not block it first.
        """
        coefficients = Coefficients(
            ts=TS,
            a1=1.4,
            a2=-0.3,
            a3=0.9,
            a4=0.0,
            trace_p=0.0031,
            steady_state_residual=0.1,
            samples_since_reset=14203,
            adaptation=AdaptationState.FROZEN,
        )
        assert coefficients.a3 == 0.9

    def test_covariance_trace_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            Coefficients(
                ts=TS,
                a1=0.98,
                a2=0.02,
                a3=-0.04,
                a4=0.01,
                trace_p=-1.0,
                steady_state_residual=0.0,
                samples_since_reset=0,
                adaptation=AdaptationState.ACTIVE,
            )

    def test_sample_count_cannot_be_negative(self):
        with pytest.raises(ValidationError):
            Coefficients(
                ts=TS,
                a1=0.98,
                a2=0.02,
                a3=-0.04,
                a4=0.01,
                trace_p=0.1,
                steady_state_residual=0.0,
                samples_since_reset=-1,
                adaptation=AdaptationState.ACTIVE,
            )

    def test_a_steady_state_residual_contradicting_the_coefficients_is_rejected(self):
        """Section 5.2.1 makes drift from a1 + a2 = 1 a diagnostic signal."""
        with pytest.raises(ValidationError):
            Coefficients(
                ts=TS,
                a1=0.98,
                a2=0.02,
                a3=-0.04,
                a4=0.01,
                trace_p=0.1,
                steady_state_residual=42.0,
                samples_since_reset=1,
                adaptation=AdaptationState.ACTIVE,
            )

    def test_the_documented_payload_is_internally_consistent(self):
        """The example in DESIGN.md section 6.2 must satisfy its own invariant."""
        coefficients = Coefficients(
            ts=TS,
            a1=0.9812,
            a2=0.0173,
            a3=-0.0421,
            a4=0.0094,
            trace_p=0.0031,
            steady_state_residual=0.0015,
            samples_since_reset=14203,
            adaptation=AdaptationState.ACTIVE,
        )
        assert coefficients.steady_state_residual == 0.0015


class TestFaultEvent:
    """The payload in DESIGN.md section 6.2 uses the reserved word 'class'."""

    def _event(self, **overrides) -> FaultEvent:
        return FaultEvent.model_validate(
            {
                "ts": 1756032300.0,
                "fault_id": "f_temp01_stuck_1756032",
                "detector": DetectorId.D2_STUCK_AT,
                "subject": SENSOR_ID,
                "class": FaultClass.SENSOR,
                "confidence": 0.94,
                "detected_ts": 1756032300.0,
                "evidence": {"window_s": 300, "variance": 0.0002},
                "mode_impact": Mode.DEGRADED_SENSOR,
                **overrides,
            }
        )

    def test_accepts_the_reserved_word_class_as_an_alias(self):
        assert self._event().fault_class is FaultClass.SENSOR

    def test_serialises_back_to_the_documented_class_key(self):
        assert "class" in self._event().model_dump(by_alias=True)

    def test_confidence_is_bounded_to_the_unit_interval(self):
        with pytest.raises(ValidationError):
            self._event(confidence=1.01)

    def test_evidence_defaults_to_empty_rather_than_null(self):
        event = FaultEvent.model_validate(
            {
                "ts": 1756032300.0,
                "fault_id": "f",
                "detector": DetectorId.D1_DROPOUT,
                "subject": SENSOR_ID,
                "class": FaultClass.SENSOR,
                "confidence": 1.0,
                "detected_ts": 1756032300.0,
                "mode_impact": Mode.DEGRADED_SENSOR,
            }
        )
        assert event.evidence == {}

    def test_evidence_cannot_be_mutated_after_construction(self):
        """Freezing the model alone would still leave the mapping writable."""
        event = self._event()
        with pytest.raises(TypeError):
            event.evidence["injected"] = 999.0

    def test_evidence_still_serialises_to_a_plain_json_object(self):
        assert '"evidence":{"window_s":300.0' in self._event().model_dump_json()

    def test_evidence_survives_a_json_round_trip(self):
        original = self._event()
        restored = FaultEvent.model_validate_json(original.model_dump_json())
        assert dict(restored.evidence) == dict(original.evidence)

    def test_a_restored_event_is_also_immutable(self):
        restored = FaultEvent.model_validate_json(self._event().model_dump_json())
        with pytest.raises(TypeError):
            restored.evidence["injected"] = 999.0


class TestModeState:
    def test_carries_no_active_faults_by_default(self):
        state = ModeState(ts=TS, mode=Mode.NORMAL, since_ts=TS)
        assert state.active_fault_ids == ()

    def test_active_faults_are_an_immutable_sequence(self):
        state = ModeState(
            ts=TS, mode=Mode.SAFE_HOLD, since_ts=TS, active_fault_ids=("f1", "f2")
        )
        assert isinstance(state.active_fault_ids, tuple)


class TestValidationVerdict:
    def _verdict(self, **overrides) -> ValidationVerdict:
        return ValidationVerdict(
            **{
                "ts": TS,
                "proposed_setpoint_c": SETPOINT_C,
                "verdict": Verdict.ACCEPTED,
                "reason": ReasonCode.NONE,
                "applied_setpoint_c": SETPOINT_C,
                **overrides,
            }
        )

    def test_an_accepted_proposal_passes_through_unchanged(self):
        assert self._verdict().applied_setpoint_c == SETPOINT_C

    def test_an_accepted_verdict_cannot_carry_a_reason_code(self):
        with pytest.raises(ValidationError):
            self._verdict(reason=ReasonCode.RATE_LIMIT)

    def test_an_accepted_verdict_cannot_alter_the_proposal(self):
        with pytest.raises(ValidationError):
            self._verdict(applied_setpoint_c=23.0)

    def test_a_clamped_verdict_records_why(self):
        verdict = self._verdict(
            verdict=Verdict.CLAMPED,
            reason=ReasonCode.RATE_LIMIT,
            proposed_setpoint_c=23.0,
        )
        assert verdict.reason is ReasonCode.RATE_LIMIT

    def test_a_clamped_verdict_without_a_reason_is_rejected(self):
        with pytest.raises(ValidationError):
            self._verdict(verdict=Verdict.CLAMPED, reason=ReasonCode.NONE)

    def test_a_blocked_verdict_without_a_reason_is_rejected(self):
        with pytest.raises(ValidationError):
            self._verdict(verdict=Verdict.BLOCKED, reason=ReasonCode.NONE)


class TestCommand:
    def test_a_cool_command_carries_its_setpoint(self):
        command = Command(
            ts=TS, actuator_id="ac", kind=CommandKind.COOL, setpoint_c=SETPOINT_C
        )
        assert command.setpoint_c == SETPOINT_C

    def test_a_cool_command_without_a_setpoint_is_rejected(self):
        with pytest.raises(ValidationError):
            Command(ts=TS, actuator_id="ac", kind=CommandKind.COOL)

    @pytest.mark.parametrize(
        "kind", [CommandKind.OFF, CommandKind.MAINTAIN, CommandKind.HOLD]
    )
    def test_a_non_cool_command_must_not_carry_a_setpoint(self, kind):
        with pytest.raises(ValidationError):
            Command(ts=TS, actuator_id="ac", kind=kind, setpoint_c=SETPOINT_C)

    @pytest.mark.parametrize(
        "kind", [CommandKind.OFF, CommandKind.MAINTAIN, CommandKind.HOLD]
    )
    def test_a_non_cool_command_is_valid_without_a_setpoint(self, kind):
        assert Command(ts=TS, actuator_id="ac", kind=kind).setpoint_c is None


class TestActuatorState:
    def test_simulation_labelling_has_no_default_and_must_be_stated(self):
        """FR-15: every simulated actuator is labelled in every state message."""
        with pytest.raises(ValidationError):
            ActuatorState(
                ts=TS, actuator_id="light_01", kind=CommandKind.OFF, ack=AckStatus.UNKNOWN
            )

    def test_acknowledgement_has_no_default_and_must_be_stated(self):
        """R-02: an open-loop IR path must say UNKNOWN, not inherit success."""
        with pytest.raises(ValidationError):
            ActuatorState(
                ts=TS, actuator_id="ac", simulated=False, kind=CommandKind.OFF
            )

    def test_unknown_is_a_representable_acknowledgement(self):
        state = ActuatorState(
            ts=TS,
            actuator_id="ac",
            simulated=False,
            kind=CommandKind.COOL,
            setpoint_c=SETPOINT_C,
            ack=AckStatus.UNKNOWN,
        )
        assert state.ack is AckStatus.UNKNOWN


class TestGoal:
    def test_records_the_rationale_that_produced_it(self):
        goal = Goal(
            ts=TS,
            source=GoalSource.SUPERVISOR,
            setpoint_c=SETPOINT_C,
            mode=Mode.NORMAL,
            rationale="Occupied 42 min, peak tariff until 22:00",
            expires_ts=TS + 600.0,
        )
        assert goal.rationale.startswith("Occupied")

    def test_an_out_of_policy_setpoint_is_representable_for_the_validator_to_clamp(self):
        """Bounds are validator policy (V-1), not schema truth."""
        goal = Goal(
            ts=TS,
            source=GoalSource.SUPERVISOR,
            setpoint_c=5.0,
            mode=Mode.NORMAL,
            expires_ts=TS + 600.0,
        )
        assert goal.setpoint_c == 5.0
