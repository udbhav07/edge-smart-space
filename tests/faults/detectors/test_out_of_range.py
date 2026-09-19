"""Unit tests for D3, the out-of-range detector (FR-22).

The boundaries are the bounds themselves -- inclusive -- and the debounce that
separates one corrupted packet from a faulted sensor.
"""

import pytest

from src.common.config import Bounds
from src.common.schemas import DetectorId, SensorReading, Unit
from src.faults.detectors.base import Judgment
from src.faults.detectors.out_of_range import OutOfRangeDetector

SUBJECT = "temp_01"
LOW_C = -10.0
HIGH_C = 60.0
DEBOUNCE = 2
TS = 1756032000.0


def _bounds() -> Bounds:
    return Bounds(low=LOW_C, high=HIGH_C)


@pytest.fixture(name="detector")
def _detector() -> OutOfRangeDetector:
    return OutOfRangeDetector(
        subject=SUBJECT,
        unit=Unit.CELSIUS,
        bounds=_bounds(),
        debounce_samples=DEBOUNCE,
    )


def _observe(detector, *values, sensor_id: str = SUBJECT) -> None:
    for value in values:
        detector.observe(
            SensorReading(
                ts=TS, sensor_id=sensor_id, value=value, unit=Unit.CELSIUS
            )
        )


class TestBeforeAnythingArrives:
    def test_a_sensor_never_heard_from_is_unknown(self, detector):
        """It has reported nothing implausible, but nothing plausible either.
        Silence is D1's subject."""
        assert detector.evaluate().judgment is Judgment.UNKNOWN


class TestTheBounds:
    def test_a_reading_inside_the_bounds_is_clear(self, detector):
        _observe(detector, 27.4)
        assert detector.evaluate().judgment is Judgment.CLEAR

    @pytest.mark.parametrize("value", [LOW_C, HIGH_C])
    def test_the_bounds_themselves_are_inside(self, detector, value):
        _observe(detector, value, value)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_reading_just_above_the_high_bound_is_outside(self, detector):
        _observe(detector, HIGH_C + 0.01, HIGH_C + 0.01)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_a_reading_just_below_the_low_bound_is_outside(self, detector):
        _observe(detector, LOW_C - 0.01, LOW_C - 0.01)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_the_bounds_in_force_are_reported(self, detector):
        assert (detector.bounds.low, detector.bounds.high) == (LOW_C, HIGH_C)


class TestDebounce:
    def test_one_implausible_reading_is_not_yet_a_fault(self, detector):
        """One corrupted packet looks exactly like this."""
        _observe(detector, 999.0)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_the_debounce_count_is_what_raises_it(self, detector):
        _observe(detector, 999.0, 999.0)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_a_plausible_reading_in_between_resets_the_count(self, detector):
        _observe(detector, 999.0, 27.4, 999.0)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_debounce_of_one_raises_on_the_first_reading(self):
        detector = OutOfRangeDetector(
            subject=SUBJECT,
            unit=Unit.CELSIUS,
            bounds=_bounds(),
            debounce_samples=1,
        )
        _observe(detector, 999.0)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_detection_lands_inside_the_documented_latency_target(self):
        """Section 5.5 targets under 10 s for D3: two samples at 5 s."""
        assert DEBOUNCE * 5.0 <= 10.0


class TestRecovery:
    def test_a_plausible_reading_clears_the_fault(self, detector):
        _observe(detector, 999.0, 999.0)
        assert detector.evaluate().judgment is Judgment.FAULTED

        _observe(detector, 27.4)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_sensor_stuck_outside_the_range_stays_faulted(self, detector):
        _observe(detector, 999.0, 999.0, 999.0, 999.0)
        assert detector.evaluate().judgment is Judgment.FAULTED


class TestEvidence:
    def test_the_finding_names_the_detector(self, detector):
        _observe(detector, 27.4)
        assert detector.evaluate().detector is DetectorId.D3_OUT_OF_RANGE

    def test_the_evidence_carries_the_value_and_the_bounds_it_left(self, detector):
        _observe(detector, 999.0, 999.0)
        evidence = detector.evaluate().evidence
        assert evidence["value"] == 999.0
        assert (evidence["low"], evidence["high"]) == (LOW_C, HIGH_C)

    def test_the_evidence_carries_the_debounce_progress(self, detector):
        _observe(detector, 999.0)
        evidence = detector.evaluate().evidence
        assert evidence["consecutive_outside"] == 1.0
        assert evidence["debounce_samples"] == float(DEBOUNCE)

    def test_an_implausible_reading_is_reported_with_full_confidence(self, detector):
        """A room is not at 300 degrees; the debounce carries the doubt about
        whether a packet was corrupted."""
        _observe(detector, 999.0, 999.0)
        assert detector.evaluate().confidence == 1.0


class TestConstruction:
    def test_a_boolean_sensor_is_refused(self):
        """Both its legal values are inside any range, and SensorReading
        already rejects a third."""
        with pytest.raises(ValueError):
            OutOfRangeDetector(
                subject="pir_01",
                unit=Unit.BOOLEAN,
                bounds=_bounds(),
                debounce_samples=DEBOUNCE,
            )

    def test_an_unnamed_subject_is_refused(self):
        with pytest.raises(ValueError):
            OutOfRangeDetector(
                subject="",
                unit=Unit.CELSIUS,
                bounds=_bounds(),
                debounce_samples=DEBOUNCE,
            )

    @pytest.mark.parametrize("debounce", [0, -1])
    def test_a_debounce_below_one_sample_is_refused(self, debounce):
        """Zero would raise a fault before any reading had been seen."""
        with pytest.raises(ValueError):
            OutOfRangeDetector(
                subject=SUBJECT,
                unit=Unit.CELSIUS,
                bounds=_bounds(),
                debounce_samples=debounce,
            )

    def test_a_reading_from_another_sensor_is_refused(self, detector):
        with pytest.raises(ValueError):
            _observe(detector, 31.0, sensor_id="outdoor_01")


class TestHumidity:
    """The same detector, different bounds: nothing about it is temperature."""

    def test_humidity_above_one_hundred_percent_is_a_fault(self):
        detector = OutOfRangeDetector(
            subject="hum_01",
            unit=Unit.PERCENT_RH,
            bounds=Bounds(low=0.0, high=100.0),
            debounce_samples=1,
        )
        detector.observe(
            SensorReading(
                ts=TS, sensor_id="hum_01", value=105.0, unit=Unit.PERCENT_RH
            )
        )
        assert detector.evaluate().judgment is Judgment.FAULTED
