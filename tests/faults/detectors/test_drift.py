"""Unit tests for D4, the drift detector (FR-23).

The two properties that matter are the two the CUSUM exists for: noise inside
the slack must never accumulate to a fault however long it runs, and a drift
smaller than the noise must accumulate to one anyway.
"""

import random
from pathlib import Path

import pytest

from src.common.config import DriftDetectorConfig, load_config
from src.common.schemas import AdaptationState, DetectorId, ThermalEstimate
from src.faults.detectors.base import Judgment
from src.faults.detectors.drift import DriftDetector

SUBJECT = "temp_01"
SIGMA_C = 0.15
SLACK = 0.5
THRESHOLD = 5.0
MAX_SAMPLE = 4.0
TS = 1756032000.0


def _config(**overrides) -> DriftDetectorConfig:
    return DriftDetectorConfig(
        **{
            "slack_sigma": SLACK,
            "threshold_sigma": THRESHOLD,
            "min_residual_sigma_c": 0.02,
            "max_sample_sigma": MAX_SAMPLE,
            "warmup_samples": 0,
            **overrides,
        }
    )


@pytest.fixture(name="detector")
def _detector() -> DriftDetector:
    return DriftDetector(subject=SUBJECT, config=_config())


def _estimate(residual_c: float, sigma_c: float = SIGMA_C) -> ThermalEstimate:
    """An estimate carrying a given prediction error.

    t_in and t_pred have to agree with the residual: the schema checks it,
    because an inconsistent triple would corrupt exactly this detector.
    """
    t_pred = 27.0
    return ThermalEstimate(
        ts=TS,
        t_in=t_pred + residual_c,
        t_pred=t_pred,
        residual=residual_c,
        residual_sigma=sigma_c,
        model_confidence=0.9,
        adaptation=AdaptationState.FROZEN,
    )


def _feed(detector, residual_c: float, samples: int, sigma_c: float = SIGMA_C):
    for _ in range(samples):
        detector.observe(_estimate(residual_c, sigma_c))


class TestBeforeAnythingIsMeasurable:
    def test_a_detector_with_no_estimates_is_unknown(self, detector):
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_an_uncharacterised_noise_floor_is_not_accumulated(self, detector):
        """Dividing by a sigma the estimator has not measured yet would make
        the first samples of any run enormous and declare drift on a healthy
        sensor seconds after boot."""
        _feed(detector, residual_c=0.5, samples=50, sigma_c=0.0)
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_the_sums_stay_empty_while_the_floor_is_uncharacterised(self, detector):
        _feed(detector, residual_c=0.5, samples=50, sigma_c=0.0)
        assert (detector.cusum_high, detector.cusum_low) == (0.0, 0.0)

    def test_a_measurable_sigma_starts_the_test(self, detector):
        _feed(detector, residual_c=0.0, samples=1)
        assert detector.evaluate().judgment is Judgment.CLEAR


class TestNoiseDoesNotAccumulate:
    def test_a_residual_inside_the_slack_never_accumulates(self, detector):
        """This is what the slack term is for."""
        _feed(detector, residual_c=SIGMA_C * 0.4, samples=1000)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_zero_mean_noise_does_not_drift_the_sums(self, detector):
        """A thousand samples of symmetric noise must not raise a fault. The
        seed is fixed so a failure here is a real regression, not a run of
        bad luck."""
        rng = random.Random(20260922)
        for _ in range(1000):
            detector.observe(_estimate(rng.gauss(0.0, SIGMA_C)))
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_a_signal_that_wanders_up_then_down_is_not_drifting(self, detector):
        """The two sums are kept apart for exactly this case; one combined
        sum could not tell it from real drift."""
        _feed(detector, residual_c=SIGMA_C * 2.0, samples=3)
        _feed(detector, residual_c=-SIGMA_C * 2.0, samples=3)
        assert detector.evaluate().judgment is Judgment.CLEAR


class TestDriftAccumulates:
    def test_a_consistent_upward_drift_is_detected(self, detector):
        _feed(detector, residual_c=SIGMA_C * 1.0, samples=20)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_a_consistent_downward_drift_is_detected(self, detector):
        """A sensor can drift either way."""
        _feed(detector, residual_c=-SIGMA_C * 1.0, samples=20)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_drift_below_the_noise_is_still_detected_given_enough_samples(
        self, detector
    ):
        """The whole reason for a cumulative test: a bias of 0.6 sigma is
        invisible at any single sample and unmistakable over fifty."""
        _feed(detector, residual_c=SIGMA_C * 0.6, samples=60)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_a_larger_drift_is_detected_in_fewer_samples(self, detector):
        fast = DriftDetector(subject=SUBJECT, config=_config())
        _feed(fast, residual_c=SIGMA_C * 3.0, samples=2)
        assert fast.evaluate().judgment is Judgment.FAULTED
        slow = DriftDetector(subject=SUBJECT, config=_config())
        _feed(slow, residual_c=SIGMA_C * 0.6, samples=2)
        assert slow.evaluate().judgment is Judgment.CLEAR

    def test_exactly_at_the_threshold_counts_as_drift(self, detector):
        """Two samples of z = 3.0, each contributing 2.5 after slack."""
        _feed(detector, residual_c=SIGMA_C * 3.0, samples=2)
        assert detector.cusum_high == pytest.approx(THRESHOLD)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_just_under_the_threshold_does_not(self, detector):
        _feed(detector, residual_c=SIGMA_C * 2.9, samples=2)
        assert detector.evaluate().judgment is Judgment.CLEAR


class TestNoSingleSampleCarriesTheTest:
    """A cumulative test is tripped by persistence, not by magnitude."""

    def test_one_enormous_residual_is_not_enough(self, detector):
        """A repaired sensor produces exactly this: the reading steps back to
        the truth while the frozen model is still predicting from where it
        was. Uncapped, fixing a sensor would immediately declare it broken."""
        _feed(detector, residual_c=SIGMA_C * 50.0, samples=1)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_the_contribution_is_capped_not_discarded(self, detector):
        """A genuinely large error is still evidence, just not the whole
        case."""
        _feed(detector, residual_c=SIGMA_C * 50.0, samples=1)
        assert detector.cusum_high == pytest.approx(MAX_SAMPLE - SLACK)

    def test_a_sustained_large_error_still_accumulates(self, detector):
        _feed(detector, residual_c=SIGMA_C * 50.0, samples=2)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_the_cap_applies_downwards_too(self, detector):
        _feed(detector, residual_c=-SIGMA_C * 50.0, samples=1)
        assert detector.cusum_low == pytest.approx(MAX_SAMPLE - SLACK)

    def test_the_evidence_reports_the_cap_in_force(self, detector):
        _feed(detector, residual_c=SIGMA_C, samples=1)
        assert detector.evaluate().evidence["max_sample_sigma"] == MAX_SAMPLE

    def test_a_cap_that_could_carry_the_test_alone_is_refused(self):
        with pytest.raises(ValueError):
            _config(max_sample_sigma=6.0)


class TestResetting:
    def test_resetting_clears_the_accumulated_evidence(self, detector):
        _feed(detector, residual_c=SIGMA_C * 2.0, samples=20)
        assert detector.evaluate().judgment is Judgment.FAULTED

        detector.reset()
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_without_a_reset_a_recalibrated_sensor_would_stay_faulted(
        self, detector
    ):
        """The sums sit above the threshold until something empties them."""
        _feed(detector, residual_c=SIGMA_C * 2.0, samples=20)
        _feed(detector, residual_c=0.0, samples=20)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_a_reset_detector_can_detect_a_second_drift(self, detector):
        _feed(detector, residual_c=SIGMA_C * 2.0, samples=20)
        detector.reset()
        _feed(detector, residual_c=SIGMA_C * 2.0, samples=20)
        assert detector.evaluate().judgment is Judgment.FAULTED


class TestEvidence:
    def test_the_finding_names_the_detector(self, detector):
        _feed(detector, residual_c=0.0, samples=1)
        assert detector.evaluate().detector is DetectorId.D4_DRIFT

    def test_the_evidence_carries_both_sums_and_the_threshold(self, detector):
        _feed(detector, residual_c=SIGMA_C * 2.0, samples=20)
        evidence = detector.evaluate().evidence
        assert evidence["cusum_high"] >= evidence["threshold_sigma"]
        assert evidence["cusum_low"] == 0.0

    def test_the_evidence_carries_the_sigma_it_normalised_by(self, detector):
        """R-04 re-derives the thresholds from measured sigma, so the sigma in
        force is the number that makes a finding auditable."""
        _feed(detector, residual_c=0.0, samples=1)
        assert detector.evaluate().evidence["residual_sigma_c"] == SIGMA_C

    def test_confidence_reports_how_far_past_the_threshold_it_went(self, detector):
        _feed(detector, residual_c=SIGMA_C * 2.0, samples=20)
        assert detector.evaluate().confidence == 1.0


class TestConstruction:
    def test_an_unnamed_subject_is_refused(self):
        with pytest.raises(ValueError):
            DriftDetector(subject="", config=_config())

    def test_a_slack_at_or_above_the_threshold_is_refused(self):
        """It would make a single sample the whole test and turn the detector
        into a noisy threshold alarm, silently."""
        with pytest.raises(ValueError):
            _config(slack_sigma=5.0, threshold_sigma=5.0)


class TestWarmUp:
    """A residual is the joint error of the sensor and the model."""

    def _warming(self, samples: int = 10) -> DriftDetector:
        return DriftDetector(
            subject=SUBJECT, config=_config(warmup_samples=samples)
        )

    def test_nothing_is_judged_while_the_model_is_converging(self):
        """During the first approach to setpoint the estimator's prediction
        lags the real cooling, which looks exactly like a drifting sensor."""
        detector = self._warming()
        _feed(detector, residual_c=SIGMA_C * 3.0, samples=10)
        assert detector.evaluate().judgment is Judgment.UNKNOWN

    def test_the_warm_up_samples_do_not_accumulate(self):
        detector = self._warming()
        _feed(detector, residual_c=SIGMA_C * 3.0, samples=10)
        assert (detector.cusum_high, detector.cusum_low) == (0.0, 0.0)

    def test_the_test_starts_once_the_warm_up_has_passed(self):
        detector = self._warming()
        _feed(detector, residual_c=0.0, samples=11)
        assert detector.evaluate().judgment is Judgment.CLEAR

    def test_drift_after_the_warm_up_is_still_caught(self):
        """The warm-up applies once, at start; it does not blunt the test."""
        detector = self._warming()
        _feed(detector, residual_c=0.0, samples=10)
        _feed(detector, residual_c=SIGMA_C * 3.0, samples=2)
        assert detector.evaluate().judgment is Judgment.FAULTED

    def test_the_evidence_reports_the_warm_up_in_force(self):
        detector = self._warming()
        _feed(detector, residual_c=0.0, samples=11)
        assert detector.evaluate().evidence["warmup_samples"] == 10.0

    def test_the_shipped_warm_up_is_the_measured_one(self):
        """Measured over a healthy run, not chosen: the cumulative sum peaks
        at 3.05 after 120 samples and stops improving beyond it."""
        config = load_config(Path("config/default.yaml")).detectors.drift
        assert config.warmup_samples == 120
