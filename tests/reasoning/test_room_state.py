"""Unit tests for what the assistant may say about the room (FR-28, U-03, U-04, U-06)."""

from src.common.schemas import (
    AdaptationState,
    DiagnosisConfidence,
    FaultDiagnosis,
    Hypothesis,
    Mode,
    ModeState,
    TariffBand,
    ThermalEstimate,
    TrackingState,
)
from src.reasoning.room_state import RoomState


def _mode(room, mode, faults=()):
    room._on_mode("", ModeState(ts=1.0, mode=mode, since_ts=1.0, active_fault_ids=faults))


def _estimate(room, t_in=27.0, t_pred=26.4):
    room._on_estimate(
        "",
        ThermalEstimate(ts=1.0, t_in=t_in, t_pred=t_pred, residual=t_in - t_pred, residual_sigma=0.2,
                        model_confidence=0.9, adaptation=AdaptationState.ACTIVE),
    )


def _diagnosis(fault_id, message):
    return FaultDiagnosis(
        ts=1.0, fault_id=fault_id, primary_hypothesis=Hypothesis.SENSOR_STUCK,
        confidence=DiagnosisConfidence.HIGH, recommended_mode=Mode.DEGRADED_SENSOR,
        user_message=message, recommendation_legal=True,
    )


def test_an_unknown_state_says_so():
    assert "not known" in RoomState().status()


def test_a_healthy_room_has_no_caveat():
    room = RoomState()
    _mode(room, Mode.NORMAL)
    assert room.caveat(22.0) == ""


def test_the_reading_is_the_temperature_when_the_sensor_works():
    room = RoomState()
    _mode(room, Mode.NORMAL)
    _estimate(room)
    assert room.temperature_c() == 27.0


def test_the_estimate_is_the_temperature_when_the_sensor_is_out():
    """The reading is the fault; the estimate is what the room is believed to be."""
    room = RoomState()
    _mode(room, Mode.DEGRADED_SENSOR, ("f1",))
    _estimate(room)
    assert room.temperature_c() == 26.4


def test_an_explanation_is_given_while_its_fault_is_active():
    room = RoomState()
    _mode(room, Mode.DEGRADED_SENSOR, ("f1",))
    room._on_diagnosis("", _diagnosis("f1", "The sensor appears stuck."))
    assert "The sensor appears stuck." in room.status()


def test_an_explanation_goes_when_its_fault_does():
    room = RoomState()
    _mode(room, Mode.DEGRADED_SENSOR, ("f1",))
    room._on_diagnosis("", _diagnosis("f1", "The sensor appears stuck."))
    _mode(room, Mode.NORMAL)
    assert room.explanations() == ()


def test_on_hold_the_caveat_says_nothing_will_cool():
    room = RoomState()
    _mode(room, Mode.SAFE_HOLD, ("f1",))
    assert "will not be cooled" in room.caveat(20.0)


def test_during_peak_the_held_temperature_is_named():
    room = RoomState()
    _mode(room, Mode.NORMAL)
    room._on_tracking(
        "", TrackingState(ts=1.0, setpoint_c=24.0, tracked_c=25.0, band=TariffBand.PEAK,
                          reason="peak tariff: held +1 C until 22:00")
    )
    assert room.held_c(22.0) == (23.0, "peak tariff: held +1 C until 22:00")
    assert "held at 25 C" in room.summary()
