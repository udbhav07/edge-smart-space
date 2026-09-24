"""D1: a sensor has gone quiet (FR-20).

The simplest detector in the bank, and the one most exposed to A-02 being
wrong. Its whole content is a timeout, and the only real decision is which
clock the timeout runs on.

**It measures arrival, not the timestamp inside the reading.** A reading
carries the moment it describes; D1 asks when it turned up. The two differ
whenever the network delays a batch, and they differ permanently if a node's
clock is wrong -- an ESP32 with no NTP will report a timestamp from 1970, and
a detector keyed on that would declare a perfectly healthy sensor dropped
forever. Keying on arrival also means a reconnect burst delivering five stale
readings at once counts as five arrivals, which is the truth: the sensor is
talking again.

Latency is the timeout itself, so the default of three sampling intervals puts
detection at 15 s against the section 5.5 target of under 20 s.
"""

from __future__ import annotations

from src.common.clock import Clock
from src.common.config import DropoutDetectorConfig
from src.common.schemas import DetectorId, SensorReading
from src.faults.detectors.base import Finding, Judgment

#: A timeout is unambiguous: either a message arrived inside the window or it
#: did not. There is no borderline case to hedge about, unlike a variance that
#: sits just under a threshold.
_TIMEOUT_CONFIDENCE = 1.0


class DropoutDetector:
    """Watches one subject's message arrivals against a timeout.

    One instance per sensor. Sharing an instance would make the last sensor to
    report mask every other one's silence.
    """

    def __init__(
        self,
        subject: str,
        config: DropoutDetectorConfig,
        sensor_period_s: float,
        clock: Clock,
    ) -> None:
        if not subject:
            raise ValueError("a detector must name the subject it watches")
        if sensor_period_s <= 0.0:
            raise ValueError(f"period must be positive, got {sensor_period_s!r}")
        self._subject = subject
        self._timeout_s = config.timeout_periods * sensor_period_s
        self._clock = clock
        self._last_arrival_s: float | None = None

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def timeout_s(self) -> float:
        """How long silence is tolerated. Config, so R-04 can retune it."""
        return self._timeout_s

    def observe(self, reading: SensorReading) -> None:
        """Note that this subject just reported.

        The reading's own fields are not consulted. A dropout is about whether
        a message arrived, and any value at all is proof that one did -- even
        an implausible one, which is D3's business rather than D1's.
        """
        if reading.sensor_id != self._subject:
            raise ValueError(
                f"{type(self).__name__} watches {self._subject!r}, "
                f"got a reading from {reading.sensor_id!r}"
            )
        self._last_arrival_s = self._clock.monotonic()

    def evaluate(self) -> Finding:
        """Judge the subject as of now.

        Returns UNKNOWN until the first reading arrives. Before that there is
        no silence to measure: a sensor that has never reported is not a sensor
        that has stopped, and treating it as either faulted or clear would
        state something unobserved. It is the mode manager's business that the
        system does not leave INIT until every sensor has been heard from.
        """
        if self._last_arrival_s is None:
            return self._finding(Judgment.UNKNOWN, silence_s=0.0)

        silence_s = self._clock.monotonic() - self._last_arrival_s
        judgment = (
            Judgment.FAULTED if silence_s > self._timeout_s else Judgment.CLEAR
        )
        return self._finding(judgment, silence_s=silence_s)

    def _finding(self, judgment: Judgment, silence_s: float) -> Finding:
        return Finding(
            detector=DetectorId.D1_DROPOUT,
            subject=self._subject,
            judgment=judgment,
            confidence=_TIMEOUT_CONFIDENCE if judgment is Judgment.FAULTED else 0.0,
            evidence={"silence_s": silence_s, "timeout_s": self._timeout_s},
        )
