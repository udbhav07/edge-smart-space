"""Unit tests for D2, the stuck-at detector (FR-21).

A small window is used throughout so the tests state the boundary rather than
loop sixty times to reach it. The boundaries that matter: an unfilled window,
variance either side of epsilon, and the consecutive-window requirement.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import StuckAtDetectorConfig, load_config
from src.common.schemas import DetectorId, SensorReading, Unit
from src.faults.detectors.base import Judgment
from src.faults.detectors.stuck_at import StuckAtDetector

SUBJECT = "temp_01"
WINDOW = 4
EPSILON = 0.001
CONSECUTIVE = 2
STUCK_VALUE = 27.0
PERIOD_S = 5.0


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


def _config(**overrides) -> StuckAtDetectorConfig:
    return StuckAtDetectorConfig(
        **{
            "window_samples": WINDOW,
            "variance_epsilon": EPSILON,
            "consecutive_windows": CONSECUTIVE,
            **overrides,
        }
    )


@pytest.fixture(name="detector")
def _detector(clock) -> StuckAtDetector:
    return StuckAtDetector(
        subject=SUBJECT, unit=Unit.CELSIUS, config=_config(), clock=clock
    )


def _feed(detector, clock, values) -> None:
    """Deliver readings one sampling period apart."""
    for value in values:
        detector.observe(
            SensorReading(
                ts=clock.now(), sensor_id=SUBJECT, value=value, unit=Unit.CELSIUS
            )
        )
        clock.advance(PERIOD_S)


class TestBeforeTheWindowIsFull:
    def test_an_empty_window_is_unknown(self, detector):
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_a_partly_filled_window_is_still_unknown(self, detector, clock):
        """Its variance is over a shorter span than the threshold was chosen
        for, and a room that happens to be settling would trip it."""
        _feed(detector, clock, [STUCK_VALUE] * (WINDOW - 1))
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_the_window_filling_exactly_permits_a_judgment(self, detector, clock):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        assert detector.evaluate().judgment is not Judgment.UNKNOWN


class TestVarianceAgainstEpsilon:
    def test_a_frozen_signal_is_a_fault_after_the_required_windows(
        self, detector, clock
    ):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        for _ in range(CONSECUTIVE):
            judgment = detector.evaluate().judgment
        assert judgment is Judgment.FAULTED

    def test_a_moving_signal_is_clear(self, detector, clock):
        _feed(detector, clock, [26.8, 27.1, 26.9, 27.3])
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_variance_just_below_epsilon_counts_as_stuck(self, clock):
        """Two values 0.05 apart over four samples give a variance of
        0.000625, under the 0.001 threshold."""
        detector = StuckAtDetector(
            subject=SUBJECT, unit=Unit.CELSIUS, config=_config(), clock=clock
        )
        _feed(detector, clock, [27.0, 27.05, 27.0, 27.05])
        for _ in range(CONSECUTIVE):
            judgment = detector.evaluate().judgment
        assert judgment is Judgment.FAULTED

    def test_variance_above_epsilon_is_clear(self, detector, clock):
        """The same shape with values 0.1 apart gives 0.0025, over it."""
        _feed(detector, clock, [27.0, 27.1, 27.0, 27.1])
        assert detector.evaluate().judgment is Judgment.CLEAR


class TestConsecutiveWindows:
    def test_one_evaluation_below_the_threshold_is_not_enough(self, detector, clock):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_the_count_resets_when_variance_returns(self, detector, clock):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        detector.evaluate()

        _feed(detector, clock, [26.0, 28.0, 26.5, 27.5])
        assert detector.evaluate().judgment is Judgment.CLEAR

        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_single_window_configuration_trips_at_once(self, clock):
        detector = StuckAtDetector(
            subject=SUBJECT,
            unit=Unit.CELSIUS,
            config=_config(consecutive_windows=1),
            clock=clock,
        )
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        assert detector.evaluate().judgment is Judgment.FAULTED


class TestRecovery:
    def test_a_sensor_that_starts_moving_again_clears(self, detector, clock):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        for _ in range(CONSECUTIVE):
            detector.evaluate()
        assert detector.evaluate().judgment is Judgment.FAULTED

        _feed(detector, clock, [26.0, 28.0, 26.5, 27.5])
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_the_window_holds_only_the_most_recent_samples(self, detector, clock):
        """Bounded by construction: the window is the whole history kept."""
        _feed(detector, clock, [26.0, 28.0, 26.5, 27.5])
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        for _ in range(CONSECUTIVE - 1):
            detector.evaluate()
        assert detector.evaluate().judgment is Judgment.FAULTED


class TestEvidence:
    def test_the_finding_names_the_detector(self, detector, clock):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        assert detector.evaluate().detector is DetectorId.D2_STUCK_AT

    def test_the_evidence_carries_the_variance_and_the_threshold(
        self, detector, clock
    ):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        evidence = detector.evaluate().evidence
        assert evidence["variance"] == pytest.approx(0.0)
        assert evidence["variance_epsilon"] == EPSILON

    def test_the_evidence_reports_the_span_the_window_covers(self, detector, clock):
        """Section 6.2's example evidence carries window_s."""
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        span_s = detector.evaluate().evidence["window_s"]
        assert span_s == pytest.approx(PERIOD_S * (WINDOW - 1))

    def test_the_span_is_measured_on_arrival_not_on_the_reading(
        self, detector, clock
    ):
        """A node with an unset clock reports 1970, and the evidence would
        otherwise describe a window fifty years wide."""
        for _ in range(WINDOW):
            detector.observe(
                SensorReading(
                    ts=1.0, sensor_id=SUBJECT, value=STUCK_VALUE, unit=Unit.CELSIUS
                )
            )
            clock.advance(PERIOD_S)
        assert detector.evaluate().evidence["window_s"] == pytest.approx(
            PERIOD_S * (WINDOW - 1)
        )

    def test_a_frozen_signal_is_reported_with_full_confidence(self, detector, clock):
        _feed(detector, clock, [STUCK_VALUE] * WINDOW)
        for _ in range(CONSECUTIVE - 1):
            detector.evaluate()
        assert detector.evaluate().confidence == pytest.approx(1.0)

    def test_variance_near_the_threshold_is_reported_with_little_confidence(
        self, detector, clock
    ):
        _feed(detector, clock, [27.0, 27.06, 27.0, 27.06])
        for _ in range(CONSECUTIVE - 1):
            detector.evaluate()
        finding = detector.evaluate()
        assert finding.judgment is Judgment.FAULTED
        assert 0.0 < finding.confidence < 0.2


class TestConstruction:
    def test_a_boolean_sensor_is_refused(self, clock):
        """An empty room reports a constant legitimately; accepting one would
        raise a fault every quiet night and teach everybody to ignore D2."""
        with pytest.raises(ValueError):
            StuckAtDetector(
                subject="pir_01", unit=Unit.BOOLEAN, config=_config(), clock=clock
            )

    def test_an_unnamed_subject_is_refused(self, clock):
        with pytest.raises(ValueError):
            StuckAtDetector(
                subject="", unit=Unit.CELSIUS, config=_config(), clock=clock
            )

    def test_a_reading_from_another_sensor_is_refused(self, detector, clock):
        """Mixing two signals would produce variance from their difference."""
        with pytest.raises(ValueError):
            detector.observe(
                SensorReading(
                    ts=clock.now(),
                    sensor_id="outdoor_01",
                    value=31.0,
                    unit=Unit.CELSIUS,
                )
            )

    def test_the_configured_window_is_reported(self, detector):
        assert detector.window_samples == WINDOW


class TestDocumentedDefaults:
    def test_the_shipped_window_cannot_detect_inside_sixty_seconds(self):
        """Sixty samples at 5 s is 300 s. The section 5.5 latency target of
        under 60 s is unreachable by construction, which is why the document
        now records 305 s: an honest limit, not a tuning problem."""
        config = load_config(Path("config/default.yaml"))
        window_s = config.detectors.stuck_at.window_samples * (
            config.loop.sensor_period_s
        )
        assert window_s > 60.0
