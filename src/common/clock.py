"""Injectable time source.

Only this module may call the standard library ``time`` module. Every other
component receives a :class:`Clock` through its constructor.

Two structural reasons, not stylistic ones:

* NFR-01 measures regulatory-loop period and jitter against wall-clock time,
  so the demo and hardware-in-loop paths need a real clock.
* E1 is a 24 h experiment and E2-E5 are batch runs. Those drive a
  :class:`SimClock`, so simulated time advances far faster than wall-clock
  time without any component above Layer 1 knowing which clock it holds.

Retrofitting this after components call ``time.time()`` directly is painful,
which is why it is the first module in the project.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol, runtime_checkable

# Matches the worked timestamps in DESIGN.md section 6.2, so simulated runs
# and the documented example payloads line up.
DEFAULT_SIM_EPOCH_S: float = 1756032000.0

_MONOTONIC_ORIGIN_S: float = 0.0


def _require_non_negative(seconds: float) -> None:
    """Reject a negative duration at the boundary rather than propagating it.

    :raises ValueError: if ``seconds`` is negative.
    """
    if seconds < 0.0:
        raise ValueError(f"duration must be non-negative, got {seconds!r}")


@runtime_checkable
class Clock(Protocol):
    """A source of time.

    Postconditions every implementation must satisfy:

    * ``monotonic()`` never decreases between successive calls.
    * ``now()`` may jump backwards under NTP correction and must therefore
      never be used to measure a duration. Use ``monotonic()`` for that.
    * ``sleep()`` advances the clock's notion of time by at least the
      requested amount.
    """

    def now(self) -> float:
        """Wall-clock time as unix epoch seconds. For timestamping only."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary origin. For measuring durations."""
        ...

    def sleep(self, seconds: float) -> None:
        """Advance time by ``seconds``.

        :raises ValueError: if ``seconds`` is negative.
        """
        ...


class RealClock:
    """Wall-clock time source, for live demonstration and hardware runs.

    ``sleep()`` blocks the calling thread. This is the clock used whenever a
    measurement must be comparable to real elapsed time (NFR-01 to NFR-04).
    """

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        _require_non_negative(seconds)
        time.sleep(seconds)


class SimClock:
    """Virtual time source, advanced explicitly by its driver.

    ``sleep()`` does not block; it advances virtual time immediately. That is
    what lets a 24 h experiment (E1) complete in seconds while every component
    above Layer 1 believes a day has passed.

    Thread-safety: the counters are guarded by a lock so a simulation driver
    on one thread and a component on another observe consistent time. No I/O
    is performed while the lock is held.
    """

    def __init__(self, start_epoch_s: float = DEFAULT_SIM_EPOCH_S) -> None:
        self._epoch_s = start_epoch_s
        self._monotonic_s = _MONOTONIC_ORIGIN_S
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._epoch_s

    def monotonic(self) -> float:
        with self._lock:
            return self._monotonic_s

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Move virtual time forward.

        :raises ValueError: if ``seconds`` is negative. Time never runs
            backwards, so a negative advance is a caller defect.
        """
        _require_non_negative(seconds)
        with self._lock:
            self._epoch_s += seconds
            self._monotonic_s += seconds
