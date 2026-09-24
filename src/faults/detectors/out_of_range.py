"""D3: a sensor is reporting something physically impossible (FR-22).

The cheapest detector and the fastest: a comparison against two numbers, with
a short debounce so one corrupted packet does not raise a fault. Latency is
the debounce, which at two samples is 10 s against section 5.5's target.

**Why the reading reaches here at all.** A Layer 1 adapter marks a reading
outside its instrument's limits as suspect and publishes it anyway (section
5.1), and ``SensorReading`` deliberately leaves temperature unbounded. Both are
this detector's doing: a reading filtered at Layer 1 could never reach the
detector that exists to find it, and a fault suppressed at the boundary is a
fault nobody can see. Fail fast applies to *malformed* messages, not to
implausible measurements -- an implausible measurement is data.

**The bounds here are detection policy, not the instrument's.** Config carries
both: ``sensors.adapters[].limits`` is what an instrument can physically
report, used by the adapter for the suspect flag, and
``detectors.out_of_range`` is what counts as a fault. They are separate because
they answer different questions and are retuned at different times -- the
first when the hardware changes, the second when R-04 characterises the room.
The detector takes its bounds as an argument rather than reading either, so
neither becomes a hidden dependency.
"""

from __future__ import annotations

from src.common.config import Bounds
from src.common.schemas import DetectorId, SensorReading, Unit
from src.faults.detectors.base import Finding, Judgment

#: A reading outside physical limits is not a judgment call: a room is not at
#: 300 degrees. Confidence reflects that, and the debounce carries the doubt
#: about whether one packet was corrupted.
_IMPLAUSIBLE_CONFIDENCE = 1.0

#: Units with no range to leave. A binary sensor's only two values are both
#: legal by construction, and SensorReading already rejects a third.
_UNBOUNDED_UNITS = frozenset({Unit.BOOLEAN})


class OutOfRangeDetector:
    """Checks one subject's readings against configured physical bounds.

    One instance per sensor, because the bounds and the debounce count both
    belong to the sensor rather than to the bank.
    """

    def __init__(
        self,
        subject: str,
        unit: Unit,
        bounds: Bounds,
        debounce_samples: int,
    ) -> None:
        if not subject:
            raise ValueError("a detector must name the subject it watches")
        if unit in _UNBOUNDED_UNITS:
            raise ValueError(
                f"a {unit.value} reading has no range to leave, so {subject!r} "
                f"cannot be watched for out-of-range"
            )
        if debounce_samples < 1:
            raise ValueError(
                f"debounce must be at least one sample, got {debounce_samples!r}"
            )
        self._subject = subject
        self._bounds = bounds
        self._debounce_samples = debounce_samples
        self._consecutive_outside = 0
        self._latest_value: float | None = None

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def bounds(self) -> Bounds:
        """What counts as physically possible here. Config, so R-04 can move it."""
        return self._bounds

    def observe(self, reading: SensorReading) -> None:
        """Count this reading towards or against the debounce.

        A single reading outside the bounds is not yet a fault: one corrupted
        packet looks exactly like one, and D3 is the detector most likely to
        see corruption because it is the only one looking at absolute values.
        """
        if reading.sensor_id != self._subject:
            raise ValueError(
                f"{type(self).__name__} watches {self._subject!r}, "
                f"got a reading from {reading.sensor_id!r}"
            )
        self._latest_value = reading.value
        if self._bounds.contains(reading.value):
            self._consecutive_outside = 0
        else:
            self._consecutive_outside += 1

    def evaluate(self) -> Finding:
        """Judge the subject on the readings seen so far.

        Returns UNKNOWN before any reading has arrived: a sensor that has not
        reported has not reported anything implausible, but neither has it
        reported anything plausible. D1 is the detector that has something to
        say about silence.
        """
        if self._latest_value is None:
            return self._finding(Judgment.UNKNOWN)

        outside = self._consecutive_outside >= self._debounce_samples
        return self._finding(Judgment.FAULTED if outside else Judgment.CLEAR)

    def _finding(self, judgment: Judgment) -> Finding:
        faulted = judgment is Judgment.FAULTED
        return Finding(
            detector=DetectorId.D3_OUT_OF_RANGE,
            subject=self._subject,
            judgment=judgment,
            confidence=_IMPLAUSIBLE_CONFIDENCE if faulted else 0.0,
            evidence={
                "value": self._latest_value if self._latest_value is not None else 0.0,
                "low": self._bounds.low,
                "high": self._bounds.high,
                "consecutive_outside": float(self._consecutive_outside),
                "debounce_samples": float(self._debounce_samples),
            },
        )
