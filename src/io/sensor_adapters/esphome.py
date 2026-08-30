"""ESPHome-backed sensor source (Week 6-7 integration point).

The ESP32 nodes publish their own readings; this source holds the most
recent value one of them reported, so a :class:`SensorAdapter` can turn it
into a schema-valid message on the blackboard.

Deliberately a *stub* in the sense DESIGN.md section 9.1 uses: the shape is
here and tested, and what is missing is the part that cannot be written
honestly until the hardware exists. Two things wait for Week 6-7:

* The device topic each node publishes under. Guessing an ESPHome entity
  naming convention now would produce code that looks finished and silently
  matches nothing.
* Whether the payload is a bare number or a JSON object. That depends on how
  the nodes are configured, which is a bring-up decision.

Both are device details, which is why this file is the only place they may
appear (Layer 1 owns hardware knowledge). ``accept`` is the seam: whatever
subscribes to the device topics calls it, and nothing above Layer 1 changes.
"""

from __future__ import annotations

import logging

from src.common.clock import Clock

LOGGER = logging.getLogger(__name__)


class EsphomeSource:
    """Holds the last value an ESP32 node reported.

    A value goes stale rather than being repeated forever: a node that has
    stopped publishing must look like silence to the adapter above, because
    silence is what D1 detects. Repeating the last value would turn a dropped
    node into a stuck sensor and send D1 looking for a fault that D2 would
    then find in the wrong place.
    """

    def __init__(self, clock: Clock, stale_after_s: float) -> None:
        if stale_after_s <= 0.0:
            raise ValueError(f"staleness horizon must be positive, got {stale_after_s!r}")
        self._clock = clock
        self._stale_after_s = stale_after_s
        self._value: float | None = None
        self._received_ts: float | None = None

    @property
    def stale_after_s(self) -> float:
        return self._stale_after_s

    @property
    def is_stale(self) -> bool:
        """Whether the held value has aged past its horizon."""
        if self._received_ts is None:
            return True
        return self._clock.now() - self._received_ts > self._stale_after_s

    def accept(self, value: float) -> None:
        """Record a value reported by the device.

        This is the seam the Week 6-7 MQTT bridge calls. Everything above it
        is already written and tested.
        """
        self._value = value
        self._received_ts = self._clock.now()

    def read(self) -> float | None:
        """The held value, or None once it has gone stale."""
        if self._value is None or self.is_stale:
            return None
        return self._value
