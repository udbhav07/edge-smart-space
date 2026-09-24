"""D2: a sensor is still reporting, but has stopped measuring (FR-21).

A stuck sensor is the fault a threshold controller is blindest to. The
readings keep arriving, they look entirely plausible, and the controller acts
on them; nothing is wrong except that the number no longer has anything to do
with the room. The signature is the one thing a real measurement always has
and a frozen one never does: variance.

Three decisions worth stating.

**The window slides, and it is evaluated on every sample once full.** Section
5.5 asks for variance below epsilon over N consecutive windows. Disjoint
blocks would make detection cost two full windows -- ten minutes at the
configured size -- to gain almost nothing, because a sliding window that is
still below epsilon one sample later is the same evidence. What
``consecutive_windows`` buys here is a guard against a single evaluation
tipping under the threshold on quantisation alone; the real debounce lives in
the aggregator.

**Latency is the window, and the window is long.** Sixty samples at 5 s is
300 s, so D2 cannot detect faster than that, whatever the target in section
5.5 says -- an honest limit rather than a tuning problem. Shortening the window
to detect faster costs false positives on a settled room, which is the trade
R-04 will make against measured noise rather than assumed noise.

**A boolean sensor is refused, not handled.** An unoccupied room legitimately
reports zero all night, and its variance is legitimately zero: variance says
nothing about whether a PIR is stuck. Accepting one here would produce a fault
event every quiet night and teach everybody to ignore D2.
"""

from __future__ import annotations

import statistics
from collections import deque

from src.common.clock import Clock
from src.common.config import StuckAtDetectorConfig
from src.common.schemas import DetectorId, SensorReading, Unit
from src.faults.detectors.base import Finding, Judgment

#: Variance of a genuinely frozen signal is exactly zero, which maps to full
#: confidence. A variance just under the threshold maps to nearly none. The
#: value is reported, never acted on: no mode transition is gated on it.
_NO_VARIANCE = 0.0


class StuckAtDetector:
    """Watches one subject's variance over a sliding window.

    One instance per sensor: the window holds that sensor's samples and
    nothing else, since mixing two signals would produce variance from their
    difference rather than from either one.
    """

    def __init__(
        self,
        subject: str,
        unit: Unit,
        config: StuckAtDetectorConfig,
        clock: Clock,
    ) -> None:
        if not subject:
            raise ValueError("a detector must name the subject it watches")
        if unit is Unit.BOOLEAN:
            raise ValueError(
                f"variance says nothing about a {Unit.BOOLEAN.value} sensor: an "
                f"empty room reports a constant legitimately, so {subject!r} "
                f"cannot be watched for stuck-at this way"
            )
        self._subject = subject
        self._config = config
        self._clock = clock
        # Bounded by construction: the window is the only history kept, and
        # its size is the only memory this detector ever uses (NFR-05).
        self._values: deque[float] = deque(maxlen=config.window_samples)
        self._arrivals_s: deque[float] = deque(maxlen=config.window_samples)
        self._consecutive_below = 0

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def window_samples(self) -> int:
        """How many samples the variance is computed over."""
        return self._config.window_samples

    def observe(self, reading: SensorReading) -> None:
        """Add a sample to the window.

        The reading's own timestamp is not used. Span is measured on arrival
        for the same reason D1 is: a node with an unset clock reports a
        timestamp from 1970, and the evidence would then describe a window
        fifty years wide.
        """
        if reading.sensor_id != self._subject:
            raise ValueError(
                f"{type(self).__name__} watches {self._subject!r}, "
                f"got a reading from {reading.sensor_id!r}"
            )
        self._values.append(reading.value)
        self._arrivals_s.append(self._clock.monotonic())

    def evaluate(self) -> Finding:
        """Judge the subject as of the samples collected so far.

        Returns UNKNOWN until the window is full. A partly filled window has a
        variance, but it is a variance over a shorter span than the threshold
        was chosen for, and a room that happens to be settling would trip it.
        """
        if len(self._values) < self._config.window_samples:
            return self._finding(Judgment.UNKNOWN, variance=_NO_VARIANCE)

        variance = statistics.pvariance(self._values)
        if variance < self._config.variance_epsilon:
            self._consecutive_below += 1
        else:
            self._consecutive_below = 0

        stuck = self._consecutive_below >= self._config.consecutive_windows
        return self._finding(
            Judgment.FAULTED if stuck else Judgment.CLEAR, variance=variance
        )

    def _window_span_s(self) -> float:
        """Elapsed time the window covers, for the evidence record."""
        if len(self._arrivals_s) < 2:
            return 0.0
        return self._arrivals_s[-1] - self._arrivals_s[0]

    def _confidence(self, variance: float) -> float:
        """How far below the threshold the variance sits, as a fraction.

        Zero variance is full confidence; variance at the threshold is none.
        """
        epsilon = self._config.variance_epsilon
        return max(0.0, min(1.0, 1.0 - (variance / epsilon)))

    def _finding(self, judgment: Judgment, variance: float) -> Finding:
        faulted = judgment is Judgment.FAULTED
        return Finding(
            detector=DetectorId.D2_STUCK_AT,
            subject=self._subject,
            judgment=judgment,
            confidence=self._confidence(variance) if faulted else 0.0,
            evidence={
                "variance": variance,
                "variance_epsilon": self._config.variance_epsilon,
                "window_s": self._window_span_s(),
                "window_samples": float(len(self._values)),
                "consecutive_windows": float(self._consecutive_below),
            },
        )
