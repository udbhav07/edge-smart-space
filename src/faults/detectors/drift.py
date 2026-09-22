"""D4: the sensor is wrong, and only the model can tell (FR-23).

A drifting sensor reports plausible numbers that move plausibly. Nothing in
the signal itself is out of range, frozen, or missing, so D1 to D3 are all
silent and a threshold controller has no way to know. What gives it away is
the *model's expectation*: the room stops behaving the way its own identified
thermal model says it should, in one direction, persistently.

That asymmetry is the fault-tolerance argument in one sentence. D4 and D5 are
the two detectors a thermostat cannot implement, because a thermostat has no
expectation to compare against.

**Why CUSUM rather than a threshold on the residual.** Drift is small and
slow: an instrument going off by 0.01 degrees a minute produces a residual far
inside the noise at every individual sample, and a threshold large enough not
to fire on noise would never fire on the drift either. A cumulative sum
integrates the small consistent part and cancels the large random part, which
is exactly the shape of this fault. The slack term ``k`` is what stops noise
from accumulating: only the part of each sample beyond k adds to the sum.

**The test runs on the residual divided by its own standard deviation**, so
both thresholds are in sigma. A sensor with different noise needs no new
numbers, which is what makes R-04's re-derivation on hardware a config edit.

**What this detector cannot see.** RLS with a forgetting factor will slowly
absorb a slow enough drift into the coefficients themselves, and a model that
has adapted to the drift produces no residual to accumulate. The detector is
therefore blind to drift substantially slower than the estimator's memory
(about 17 minutes at the configured forgetting factor). That is a real limit
of the pairing, not of the implementation, and it is the reason FR-29 freezes
adaptation once *any* fault is confirmed: the freeze stops the model from
chasing the fault while the system decides what to do about it.
"""

from __future__ import annotations

from src.common.config import DriftDetectorConfig
from src.common.schemas import DetectorId, ThermalEstimate
from src.faults.detectors.base import Finding, Judgment

#: A cumulative sum cannot go negative: the test is "has evidence accumulated",
#: and letting a quiet stretch bank credit against future drift would postpone
#: detection by however long the room happened to behave.
_SUM_FLOOR = 0.0


class DriftDetector:
    """Two-sided CUSUM on one subject's normalised model residual.

    Two-sided because a sensor can drift either way, and the two sums are kept
    separately: a signal that wanders up and then down is not drifting, and one
    combined sum could not tell the difference.
    """

    def __init__(self, subject: str, config: DriftDetectorConfig) -> None:
        if not subject:
            raise ValueError("a detector must name the subject it watches")
        self._subject = subject
        self._config = config
        self._high = _SUM_FLOOR
        self._low = _SUM_FLOOR
        self._samples = 0
        self._normalised = 0.0
        self._sigma_c = 0.0

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def cusum_high(self) -> float:
        """Accumulated evidence of upward drift, in sigma."""
        return self._high

    @property
    def cusum_low(self) -> float:
        """Accumulated evidence of downward drift, in sigma."""
        return self._low

    def observe(self, estimate: ThermalEstimate) -> None:
        """Fold one prediction error into the cumulative sums.

        An estimate whose residual is not yet normalisable is counted as seen
        but not accumulated: dividing by a standard deviation the estimator has
        not measured yet would turn the first few samples of any run into
        enormous normalised residuals and declare drift on a healthy sensor
        within seconds of boot.
        """
        self._sigma_c = estimate.residual_sigma
        if estimate.residual_sigma < self._config.min_residual_sigma_c:
            return

        self._samples += 1
        self._normalised = estimate.residual / estimate.residual_sigma
        slack = self._config.slack_sigma
        self._high = max(_SUM_FLOOR, self._high + self._normalised - slack)
        self._low = max(_SUM_FLOOR, self._low - self._normalised - slack)

    def reset(self) -> None:
        """Forget the accumulated evidence.

        Called when the fault is retired. Without it the sums stay above the
        threshold and the detector re-raises on its next sample, so a sensor
        that was recalibrated would be permanently faulted.
        """
        self._high = _SUM_FLOOR
        self._low = _SUM_FLOOR

    def evaluate(self) -> Finding:
        """Judge the subject on the evidence accumulated so far.

        UNKNOWN until at least one residual has been normalisable. Before that
        the sums are zero because nothing was measured, which must not be
        reported as a sensor observed to be drift-free.
        """
        if self._samples == 0:
            return self._finding(Judgment.UNKNOWN)

        worst = max(self._high, self._low)
        drifting = worst >= self._config.threshold_sigma
        return self._finding(Judgment.FAULTED if drifting else Judgment.CLEAR)

    def _confidence(self) -> float:
        """How much of the threshold has accumulated, capped at one.

        Reported, never acted on: no mode transition is gated on a confidence
        (FR-26). On a clear finding it reads as progress towards detection,
        which is the number worth watching during a demonstration.
        """
        worst = max(self._high, self._low)
        return min(1.0, worst / self._config.threshold_sigma)

    def _finding(self, judgment: Judgment) -> Finding:
        return Finding(
            detector=DetectorId.D4_DRIFT,
            subject=self._subject,
            judgment=judgment,
            confidence=self._confidence() if judgment is Judgment.FAULTED else 0.0,
            evidence={
                "cusum_high": self._high,
                "cusum_low": self._low,
                "threshold_sigma": self._config.threshold_sigma,
                "slack_sigma": self._config.slack_sigma,
                "normalised_residual": self._normalised,
                "residual_sigma_c": self._sigma_c,
                "samples": float(self._samples),
            },
        )
