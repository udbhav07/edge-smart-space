"""Sensor adapters: the Layer 1 bridge from an instrument to the blackboard.

An adapter owns one sensor. It reads a raw value from a source, wraps it in
the shared schema, and publishes it (DESIGN.md section 5.1). Nothing above
Layer 1 knows whether the source behind it is an ESP32 over WiFi or a
simulator, which is what makes the Week 6 hardware transfer a config change
rather than a rewrite.

Three decisions are worth stating because they are easy to get backwards:

* **An adapter never suppresses a reading.** A value outside the
  instrument's limits is published, flagged suspect. D3 exists to detect
  out-of-range readings (FR-22), and one filtered here could never reach the
  detector that exists to find it. The adapter's flag is an instantaneous
  hint; D3's finding is debounced, carries evidence, and drives a mode.

* **A lost sample is published as nothing at all.** Absence is precisely
  what D1 detects, so a dropout returns ``None`` rather than a sentinel
  value or a stale repeat.

* **Health is retained, readings are not.** A late-joining subscriber must
  be able to learn what is currently trusted without waiting for the next
  sample (FR-61).

Fault injection lives here rather than in the source, because section 5.9.2
injects into the adapter and FR-31 requires the same mechanism on hardware
as in simulation.
"""

from __future__ import annotations

import logging
from typing import Protocol

from src.common.clock import Clock
from src.common.config import SensorConfig
from src.common.injection import NO_FAULT, FaultInjection, InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import Quality, SensorHealth, SensorReading
from src.common.topics import SENSOR_HEALTH, SENSOR_STATE

LOGGER = logging.getLogger(__name__)

#: A source with nothing to report yields no reading, which is what D1 sees.
NO_READING: None = None

_NO_DRIFT = 0.0


class SensorSource(Protocol):
    """Where an adapter's raw values come from.

    Deliberately tiny. A source knows how to obtain one number and nothing
    about schemas, topics or faults, so a hardware integration is a class
    with one method rather than a rewrite of the publishing path.
    """

    def read(self) -> float | None:
        """The latest value, or None if the instrument has not reported."""
        ...


class SensorAdapter:
    """One sensor, from instrument to topic."""

    def __init__(
        self,
        config: SensorConfig,
        source: SensorSource,
        clock: Clock,
        blackboard: Blackboard,
    ) -> None:
        self._config = config
        self._source = source
        self._clock = clock
        self._blackboard = blackboard
        self._injection = NO_FAULT
        self._injected_at_ts = clock.now()
        self._last_reading_ts: float | None = None
        self._active_fault_id: str | None = None

    @property
    def sensor_id(self) -> str:
        return self._config.sensor_id

    @property
    def unit(self):
        return self._config.unit

    @property
    def limits(self):
        """What the instrument can physically report (section 5.1)."""
        return self._config.limits

    @property
    def injected_fault(self) -> InjectedFault:
        """What is currently being injected, for the injector's own audit."""
        return self._injection.kind

    @property
    def last_reading_ts(self) -> float | None:
        """When this adapter last produced a reading. None before the first."""
        return self._last_reading_ts

    def inject(self, injection: FaultInjection) -> None:
        """Begin injecting a fault (FR-31). Replaces any fault already active."""
        self._injection = injection
        self._injected_at_ts = self._clock.now()
        LOGGER.info("injecting %s on %s", injection.kind.value, self.sensor_id)

    def clear(self) -> None:
        """Stop injecting. The instrument's own behaviour resumes."""
        self._injection = NO_FAULT
        LOGGER.info("cleared injection on %s", self.sensor_id)

    def attach_fault(self, fault_id: str | None) -> None:
        """Record which fault, if any, the detectors have raised against this
        sensor, so the retained health topic reflects it (FR-61)."""
        self._active_fault_id = fault_id

    def read(self) -> SensorReading | None:
        """Obtain one reading, applying any injected fault.

        :returns: the reading, or None when nothing was reported. None is an
            ordinary outcome: it is what a dropout looks like and what D1
            detects.
        """
        observed = self._observe()
        if observed is None:
            return NO_READING
        return SensorReading(
            ts=self._clock.now(),
            sensor_id=self.sensor_id,
            value=observed,
            unit=self.unit,
            quality=self._quality(observed),
        )

    def _observe(self) -> float | None:
        """Apply the injected fault to whatever the source reported."""
        kind = self._injection.kind
        if kind is InjectedFault.DROPOUT:
            return NO_READING
        if kind in (InjectedFault.STUCK_AT, InjectedFault.OUT_OF_RANGE):
            return self._injection.magnitude

        raw = self._source.read()
        if raw is None:
            return NO_READING
        return raw + self._drift()

    def _drift(self) -> float:
        if self._injection.kind is not InjectedFault.DRIFT:
            return _NO_DRIFT
        elapsed_s = self._clock.now() - self._injected_at_ts
        return self._injection.magnitude * elapsed_s

    def _quality(self, value: float) -> Quality:
        """An instantaneous flag, not a detection.

        A reading outside the instrument's limits is still published: D3
        decides, with debounce and evidence, whether it is a fault.
        """
        if not self.limits.contains(value):
            return Quality.SUSPECT
        return Quality.OK

    def publish(self, reading: SensorReading) -> None:
        """Put one reading on the blackboard."""
        self._last_reading_ts = reading.ts
        self._blackboard.publish(
            SENSOR_STATE, reading, sensor_id=reading.sensor_id
        )

    def publish_health(self) -> None:
        """Publish retained health, so a late subscriber knows what is trusted."""
        health = SensorHealth(
            ts=self._clock.now(),
            sensor_id=self.sensor_id,
            quality=self._health_quality(),
            last_reading_ts=self._last_reading_ts,
            active_fault_id=self._active_fault_id,
        )
        self._blackboard.publish(SENSOR_HEALTH, health, sensor_id=self.sensor_id)

    def _health_quality(self) -> Quality:
        if self._active_fault_id is not None:
            return Quality.FAULTED
        if self._last_reading_ts is None:
            return Quality.SUSPECT
        return Quality.OK

    def poll(self) -> SensorReading | None:
        """Read, publish if there was anything, and refresh health.

        Health is published every poll, including when the reading was lost:
        that is exactly the case a subscriber most needs to see.
        """
        reading = self.read()
        if reading is not None:
            self.publish(reading)
        else:
            LOGGER.debug("no reading from %s", self.sensor_id)
        self.publish_health()
        return reading
