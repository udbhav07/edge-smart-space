"""Simulated sensors, imperfect by default.

A sensor here is a lossy, noisy, delayed view of the plant's true state. The
imperfections are not optional extras: A-02's uniform sampling will not
survive WiFi, and a simulator that delivers clean 5 s samples teaches the
rest of the system to depend on something hardware will not provide.

The same object carries the fault-injection path (FR-31). Injection is a
property of Layer 1, so the identical mechanism works when this class is
replaced by a real ESP32 adapter in Week 6: nothing above Layer 1 knows a
fault was injected rather than suffered.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from src.common.clock import Clock
from src.common.config import SensorNoiseConfig
from src.common.schemas import Quality, SensorReading, Unit

#: A dropped sample produces no reading at all, which is what D1 detects.
NO_READING: None = None

_NO_DRIFT_C = 0.0


class InjectedFault(str, Enum):
    """Fault classes injectable at Layer 1 (FR-31, matching D1 to D3)."""

    NONE = "NONE"
    STUCK_AT = "STUCK_AT"
    DROPOUT = "DROPOUT"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    DRIFT = "DRIFT"


@dataclass(frozen=True)
class FaultInjection:
    """What to inject and how hard.

    ``magnitude`` is interpreted per fault: the frozen reading for STUCK_AT,
    the reported value for OUT_OF_RANGE, and degrees per second for DRIFT.
    It is ignored for NONE and DROPOUT.
    """

    kind: InjectedFault
    magnitude: float = 0.0


NO_FAULT = FaultInjection(kind=InjectedFault.NONE)


class SimulatedSensor:
    """One sensor: noise, quantisation, dropout, and injectable faults.

    Randomness comes from an injected generator so a scenario replays
    identically from the same seed (FR-62).
    """

    def __init__(
        self,
        sensor_id: str,
        unit: Unit,
        config: SensorNoiseConfig,
        rng: random.Random,
        clock: Clock,
    ) -> None:
        self._sensor_id = sensor_id
        self._unit = unit
        self._config = config
        self._rng = rng
        self._clock = clock
        self._injection = NO_FAULT
        self._injected_at_ts = clock.now()

    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    @property
    def injected_fault(self) -> InjectedFault:
        """What is currently being injected. For the injector's own audit."""
        return self._injection.kind

    def inject(self, injection: FaultInjection) -> None:
        """Begin injecting a fault. Replaces any fault already active."""
        self._injection = injection
        self._injected_at_ts = self._clock.now()

    def clear(self) -> None:
        """Stop injecting. The sensor returns to nominal imperfection."""
        self._injection = NO_FAULT

    def sampling_delay_s(self, nominal_period_s: float) -> float:
        """Interval until the next sample, with jitter applied.

        Never returns a non-positive delay: a sensor that reports twice at the
        same instant is not a jitter model, it is a bug.
        """
        if nominal_period_s <= 0.0:
            raise ValueError(f"period must be positive, got {nominal_period_s!r}")
        jitter = self._config.jitter_s
        if jitter <= 0.0:
            return nominal_period_s
        offset = self._rng.uniform(-jitter, jitter)
        return max(nominal_period_s + offset, nominal_period_s / 2.0)

    def _drift_c(self) -> float:
        if self._injection.kind is not InjectedFault.DRIFT:
            return _NO_DRIFT_C
        elapsed_s = self._clock.now() - self._injected_at_ts
        return self._injection.magnitude * elapsed_s

    def _quantise(self, value: float) -> float:
        step = self._config.quantisation_c
        if step <= 0.0:
            return value
        return round(value / step) * step

    def sample(self, true_value: float) -> SensorReading | None:
        """Observe the plant.

        Returns ``None`` when the sample is lost, either to the configured
        dropout probability or to an injected DROPOUT. A lost sample is
        genuinely the absence of a reading, which is exactly what D1 detects,
        so it is represented as absence rather than as a sentinel value.
        """
        if self._injection.kind is InjectedFault.DROPOUT:
            return NO_READING
        if self._rng.random() < self._config.dropout_probability:
            return NO_READING

        if self._injection.kind is InjectedFault.STUCK_AT:
            observed = self._injection.magnitude
        elif self._injection.kind is InjectedFault.OUT_OF_RANGE:
            observed = self._injection.magnitude
        else:
            observed = true_value + self._config.bias_c + self._drift_c()
            observed += self._rng.gauss(0.0, self._config.sigma_c)
            observed = self._quantise(observed)

        return SensorReading(
            ts=self._clock.now(),
            sensor_id=self._sensor_id,
            value=observed,
            unit=self._unit,
            quality=Quality.OK,
        )


class BinarySensor:
    """A PIR or reed switch: no noise model, but it can still drop or stick.

    Occupancy is binary by design (ADR-0003), so the noise, quantisation and
    drift models above are meaningless here and are deliberately absent
    rather than applied and rounded away.
    """

    def __init__(
        self,
        sensor_id: str,
        config: SensorNoiseConfig,
        rng: random.Random,
        clock: Clock,
    ) -> None:
        self._sensor_id = sensor_id
        self._config = config
        self._rng = rng
        self._clock = clock
        self._injection = NO_FAULT

    @property
    def sensor_id(self) -> str:
        return self._sensor_id

    @property
    def injected_fault(self) -> InjectedFault:
        return self._injection.kind

    def inject(self, injection: FaultInjection) -> None:
        self._injection = injection

    def clear(self) -> None:
        self._injection = NO_FAULT

    def sample(self, occupied: bool) -> SensorReading | None:
        if self._injection.kind is InjectedFault.DROPOUT:
            return NO_READING
        if self._rng.random() < self._config.dropout_probability:
            return NO_READING

        if self._injection.kind is InjectedFault.STUCK_AT:
            value = 1.0 if self._injection.magnitude >= 0.5 else 0.0
        else:
            value = 1.0 if occupied else 0.0

        return SensorReading(
            ts=self._clock.now(),
            sensor_id=self._sensor_id,
            value=value,
            unit=Unit.BOOLEAN,
            quality=Quality.OK,
        )
