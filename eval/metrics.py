"""What the experiments measure (DESIGN.md section 8.3).

Pure functions over a run's samples. Nothing here touches the blackboard,
holds a clock, or knows which system produced the numbers, so the same metric
computes identically for this system and for the baseline -- which is the only
way a comparison between them means anything.

**Every metric is computed against the room, not against the sensor.** A
stuck sensor reports a comfortable room while the real one bakes, so a comfort
metric read off the sensor would score a broken system perfectly. The
simulator's true temperature is the ground truth, and it is the one number no
component above Layer 1 is allowed to see (section 5.10) -- which is exactly
why the evaluation may.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

#: A sample contributes to comfort statistics only if the system was supposed
#: to be controlling. Samples before the first command are start-up, not
#: performance.
_NO_SAMPLES = 0


@dataclass(frozen=True)
class TrackingMetrics:
    """How well a run held the setpoint (E2)."""

    rms_error_c: float
    mean_error_c: float
    worst_error_c: float
    overshoot_c: float
    samples: int

    def report(self) -> str:
        return (
            f"RMS error {self.rms_error_c:.3f} C, "
            f"mean {self.mean_error_c:+.3f} C, "
            f"worst {self.worst_error_c:.3f} C, "
            f"overshoot {self.overshoot_c:.3f} C "
            f"over {self.samples} samples"
        )


@dataclass(frozen=True)
class ComfortMetrics:
    """How often a run left the comfort band (E5)."""

    violations: int
    samples: int
    worst_excursion_c: float
    band_c: float

    @property
    def violation_fraction(self) -> float:
        """Share of samples outside the band. Zero for an empty run."""
        if self.samples == _NO_SAMPLES:
            return 0.0
        return self.violations / self.samples

    @property
    def held_the_bound(self) -> bool:
        """Whether the run stayed inside the band throughout."""
        return self.violations == _NO_SAMPLES

    def report(self) -> str:
        return (
            f"{self.violations}/{self.samples} samples outside "
            f"+/-{self.band_c:.1f} C ({self.violation_fraction:.1%}), "
            f"worst excursion {self.worst_excursion_c:.2f} C"
        )


def tracking(
    temperatures_c: Sequence[float], setpoints_c: Sequence[float]
) -> TrackingMetrics:
    """Setpoint-tracking error over a run.

    :param temperatures_c: the room's true temperature at each sample.
    :param setpoints_c: the setpoint in force at the same samples.
    :raises ValueError: if the two series are different lengths, which would
        silently pair each temperature with the wrong setpoint.
    """
    if len(temperatures_c) != len(setpoints_c):
        raise ValueError(
            f"{len(temperatures_c)} temperatures against "
            f"{len(setpoints_c)} setpoints"
        )
    if not temperatures_c:
        return TrackingMetrics(0.0, 0.0, 0.0, 0.0, _NO_SAMPLES)

    errors = [
        temperature - setpoint
        for temperature, setpoint in zip(temperatures_c, setpoints_c)
    ]
    squared = sum(error * error for error in errors)
    return TrackingMetrics(
        rms_error_c=math.sqrt(squared / len(errors)),
        mean_error_c=sum(errors) / len(errors),
        worst_error_c=max(abs(error) for error in errors),
        # Overshoot is cooling past the target: the room ending up colder than
        # asked for, which costs energy and comfort in the other direction.
        overshoot_c=max(0.0, -min(errors)),
        samples=len(errors),
    )


def comfort(
    temperatures_c: Sequence[float],
    setpoints_c: Sequence[float],
    band_c: float,
) -> ComfortMetrics:
    """How far and how often the room left the comfort band.

    The band is symmetric about the setpoint in force at each sample, so a
    setpoint that moves during a run does not make the metric meaningless.

    :raises ValueError: if the series disagree in length, or the band is not
        positive -- a band of zero would count every sample as a violation and
        report a number that looks like a result.
    """
    if len(temperatures_c) != len(setpoints_c):
        raise ValueError(
            f"{len(temperatures_c)} temperatures against "
            f"{len(setpoints_c)} setpoints"
        )
    if band_c <= 0.0:
        raise ValueError(f"comfort band must be positive, got {band_c!r}")

    violations = 0
    worst = 0.0
    for temperature, setpoint in zip(temperatures_c, setpoints_c):
        excursion = abs(temperature - setpoint) - band_c
        if excursion > 0.0:
            violations += 1
            worst = max(worst, excursion)
    return ComfortMetrics(
        violations=violations,
        samples=len(temperatures_c),
        worst_excursion_c=worst,
        band_c=band_c,
    )


def detection_latency_s(injected_ts: float, detected_ts: float | None) -> float | None:
    """How long a fault took to find.

    :returns: the delay, or None when the fault was never detected. None
        rather than a large number: a miss and a slow detection are different
        outcomes, and averaging a sentinel into a latency would report a
        detector as slow when it was actually blind.
    :raises ValueError: if detection precedes injection, which means the
        trial's bookkeeping is wrong rather than the detector being fast.
    """
    if detected_ts is None:
        return None
    if detected_ts < injected_ts:
        raise ValueError(
            f"detected at {detected_ts!r} before injection at {injected_ts!r}"
        )
    return detected_ts - injected_ts


def rate(successes: int, trials: int) -> float:
    """A success rate, with an empty run reported as zero rather than dividing.

    :raises ValueError: if there were more successes than trials.
    """
    if successes > trials:
        raise ValueError(f"{successes} successes in {trials} trials")
    if trials == _NO_SAMPLES:
        return 0.0
    return successes / trials


def mean(values: Sequence[float]) -> float | None:
    """Average, or None for nothing to average.

    None rather than zero, because a detector that never fired has no latency
    and reporting zero would make it look instantaneous.
    """
    if not values:
        return None
    return sum(values) / len(values)
