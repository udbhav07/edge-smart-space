"""Unit tests for D1, the dropout detector (FR-20).

The boundary that matters is the one between "has not reported yet" and "has
stopped reporting", and the one between the timeout and just inside it.
"""

import pytest

from src.common.clock import SimClock
from src.common.config import DropoutDetectorConfig
from src.common.schemas import DetectorId, SensorReading, Unit
from src.faults.detectors.base import Judgment
from src.faults.detectors.dropout import DropoutDetector

SUBJECT = "temp_01"
PERIOD_S = 5.0
TIMEOUT_PERIODS = 3.0
TIMEOUT_S = TIMEOUT_PERIODS * PERIOD_S


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="detector")
def _detector(clock) -> DropoutDetector:
    return DropoutDetector(
        subject=SUBJECT,
        config=DropoutDetectorConfig(timeout_periods=TIMEOUT_PERIODS),
        sensor_period_s=PERIOD_S,
        clock=clock,
    )


def _reading(clock, value: float = 27.4, sensor_id: str = SUBJECT) -> SensorReading:
    return SensorReading(
        ts=clock.now(), sensor_id=sensor_id, value=value, unit=Unit.CELSIUS
    )


class TestBeforeAnythingArrives:
    def test_a_sensor_never_heard_from_is_unknown_not_clear(self, detector):
        """A sensor that has never reported is not one that has stopped, and
        it is certainly not one observed to be working."""
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_it_stays_unknown_however_long_the_silence(self, detector, clock):
        clock.advance(TIMEOUT_S * 10)
        assert detector.evaluate().judgment is Judgment.UNKNOWN


class TestTheTimeout:
    def test_a_reading_that_just_arrived_is_clear(self, detector, clock):
        detector.observe(_reading(clock))
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_silence_just_inside_the_timeout_is_still_clear(self, detector, clock):
        detector.observe(_reading(clock))
        clock.advance(TIMEOUT_S)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_silence_past_the_timeout_is_a_fault(self, detector, clock):
        detector.observe(_reading(clock))
        clock.advance(TIMEOUT_S + 0.001)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_the_timeout_comes_from_config_and_the_period(self, detector):
        assert detector.timeout_s == TIMEOUT_S

    def test_detection_lands_inside_the_documented_latency_target(self, detector):
        """Section 5.5 targets under 20 s for D1."""
        assert detector.timeout_s < 20.0


class TestRecovery:
    def test_a_reading_after_the_timeout_clears_the_fault(self, detector, clock):
        detector.observe(_reading(clock))
        clock.advance(TIMEOUT_S + 1.0)
        assert detector.evaluate().judgment is Judgment.FAULTED

        detector.observe(_reading(clock))
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_an_implausible_value_still_counts_as_having_reported(
        self, detector, clock
    ):
        """A message arrived, which is all D1 asks. Whether the number makes
        sense is D3's business."""
        detector.observe(_reading(clock, value=999.0))
        assert detector.evaluate().judgment is Judgment.CLEAR


class TestArrivalNotTimestamp:
    def test_a_reading_with_an_ancient_timestamp_counts_as_an_arrival(
        self, detector, clock
    ):
        """An ESP32 with no NTP reports 1970. Keyed on that timestamp, a
        healthy sensor would be declared dropped forever."""
        stale = SensorReading(
            ts=1.0, sensor_id=SUBJECT, value=27.4, unit=Unit.CELSIUS
        )
        detector.observe(stale)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_reconnect_burst_of_stale_readings_counts_as_talking_again(
        self, detector, clock
    ):
        """Five stale readings at once means the sensor is back, which is the
        truth D1 is reporting."""
        detector.observe(_reading(clock))
        clock.advance(TIMEOUT_S + 5.0)
        assert detector.evaluate().judgment is Judgment.FAULTED

        for _ in range(5):
            detector.observe(
                SensorReading(
                    ts=clock.now() - 60.0,
                    sensor_id=SUBJECT,
                    value=27.4,
                    unit=Unit.CELSIUS,
                )
            )
        assert detector.evaluate().judgment is Judgment.CLEAR


class TestEvidenceAndIdentity:
    def test_the_finding_names_the_detector(self, detector, clock):
        detector.observe(_reading(clock))
        assert detector.evaluate().detector is DetectorId.D1_DROPOUT

    def test_the_evidence_carries_the_silence_and_the_threshold(
        self, detector, clock
    ):
        detector.observe(_reading(clock))
        clock.advance(20.0)
        evidence = detector.evaluate().evidence
        assert evidence["silence_s"] == pytest.approx(20.0)
        assert evidence["timeout_s"] == TIMEOUT_S

    def test_a_timeout_is_reported_with_full_confidence(self, detector, clock):
        """Either a message arrived inside the window or it did not; there is
        no borderline case to hedge about."""
        detector.observe(_reading(clock))
        clock.advance(TIMEOUT_S + 1.0)
        assert detector.evaluate().confidence == 1.0


class TestConstruction:
    def test_a_reading_from_another_sensor_is_refused(self, detector, clock):
        """One instance per sensor: a shared one would let the last sensor to
        report mask every other one's silence."""
        with pytest.raises(ValueError):
            detector.observe(_reading(clock, sensor_id="outdoor_01"))

    def test_an_unnamed_subject_is_refused(self, clock):
        with pytest.raises(ValueError):
            DropoutDetector(
                subject="",
                config=DropoutDetectorConfig(timeout_periods=TIMEOUT_PERIODS),
                sensor_period_s=PERIOD_S,
                clock=clock,
            )

    @pytest.mark.parametrize("period_s", [0.0, -5.0])
    def test_a_non_positive_period_is_refused(self, clock, period_s):
        with pytest.raises(ValueError):
            DropoutDetector(
                subject=SUBJECT,
                config=DropoutDetectorConfig(timeout_periods=TIMEOUT_PERIODS),
                sensor_period_s=period_s,
                clock=clock,
            )
