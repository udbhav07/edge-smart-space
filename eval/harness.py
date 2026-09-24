"""Running a whole system for an experiment, with nothing faked.

Both systems under comparison are assembled here so that everything they
share is provably shared: the same plant, the same seed, the same sensors with
the same noise and dropouts, the same control law, the same setpoint. The only
difference between :func:`full_system` and :func:`baseline_system` is the two
contributions -- the identified model and the fault layer -- which is what
makes E5's result attributable to them rather than to a tuning accident.

Every component talks over the loopback transport, so an experiment exercises
the real topics, the real schemas and the real sample-interval checks. A result
obtained by calling components directly would say nothing about the system that
actually runs.

Ground truth is recorded alongside what the system believed. The simulator's
true temperature is the only honest basis for a comfort metric: a stuck sensor
reports a pleasant room while the real one bakes, and a metric read off the
sensor would score the broken system perfectly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

from eval.baseline_thermostat import BaselineThermostat
from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import Clock, SimClock
from src.common.config import Config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    CommandKind,
    FaultEvent,
    Mode,
    ModeState,
    SensorReading,
)
from src.control.service import build_service as build_control
from src.estimation.__main__ import build_service as build_estimator
from src.faults.injector import FaultInjector
from src.faults.service import build_service as build_bank
from sim.run_sim import INDOOR_TEMPERATURE_ID, build_simulator

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    """One instant of a run, from both sides of the sensor."""

    ts: float
    true_c: float
    setpoint_c: float
    mode: Mode
    command: CommandKind | None


@dataclass
class RunLog:
    """What a run produced, for the metrics to read."""

    samples: list[Sample] = field(default_factory=list)
    faults: list[FaultEvent] = field(default_factory=list)

    @property
    def temperatures_c(self) -> list[float]:
        return [sample.true_c for sample in self.samples]

    @property
    def setpoints_c(self) -> list[float]:
        return [sample.setpoint_c for sample in self.samples]

    def after(self, ts: float) -> RunLog:
        """The part of the run from an instant onwards.

        A fault's effect is measured from when it was injected, not from the
        start: averaging in the healthy hour before it would dilute exactly
        the thing under test.
        """
        return RunLog(
            samples=[sample for sample in self.samples if sample.ts >= ts],
            faults=self.faults,
        )

    def first_fault(self, detector=None) -> FaultEvent | None:
        """The earliest fault, optionally of one detector."""
        for event in sorted(self.faults, key=lambda event: event.detected_ts):
            if detector is None or event.detector is detector:
                return event
        return None


class RunnableSystem(Protocol):
    """What an experiment needs from either system."""

    log: RunLog

    def run_for(self, seconds: float) -> None: ...


class FullSystem:
    """Plant, estimator, detector bank and regulatory loop."""

    def __init__(self, config: Config, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self.transport = LoopbackTransport()
        self.log = RunLog()

        boards = [Blackboard(config.mqtt, self.transport) for _ in range(5)]
        plant, estimator, bank, control, operator = boards
        self.simulator = build_simulator(config, clock, plant)
        self.estimator = build_estimator(config, clock, estimator)
        self.bank = build_bank(config, clock, bank)
        self.control = build_control(config, clock, control)
        self.injector = FaultInjector(operator, clock)

        self._mode = Mode.INIT
        self._last_command: CommandKind | None = None
        watcher = Blackboard(config.mqtt, self.transport)
        watcher.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)
        watcher.subscribe(topics.FAULT, FaultEvent, self._on_fault)

        for service in (self.simulator, self.estimator, self.bank, self.control):
            service.subscribe()
        for board in [*boards, watcher]:
            self.transport.attach(board)

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        self._mode = state.mode

    def _on_fault(self, _topic: str, event: FaultEvent) -> None:
        self.log.faults.append(event)

    def run_for(self, seconds: float) -> None:
        period_s = self.config.loop.sensor_period_s
        elapsed = 0.0
        while elapsed < seconds:
            self.simulator.step()
            self.estimator.tick()
            self.bank.tick()
            command = self.control.tick()
            self._last_command = command.kind if command is not None else None
            self.log.samples.append(
                Sample(
                    ts=self.clock.now(),
                    true_c=self.simulator.room_temperature_c,
                    setpoint_c=self.control.setpoint_c,
                    mode=self._mode,
                    command=self._last_command,
                )
            )
            self.clock.advance(period_s)
            elapsed += period_s

    def inject(self, kind: InjectedFault, magnitude: float | None = None) -> float:
        """Break the indoor sensor. Returns the instant it happened."""
        self.injector.inject(INDOOR_TEMPERATURE_ID, kind, magnitude)
        return self.clock.now()

    def clear_injection(self) -> None:
        self.injector.clear(INDOOR_TEMPERATURE_ID)


class BaselineSystem:
    """Plant and a fixed-deadband thermostat. No model, no fault layer."""

    def __init__(self, config: Config, clock: Clock) -> None:
        self.config = config
        self.clock = clock
        self.transport = LoopbackTransport()
        self.log = RunLog()

        plant = Blackboard(config.mqtt, self.transport)
        controller_board = Blackboard(config.mqtt, self.transport)
        operator = Blackboard(config.mqtt, self.transport)
        self.simulator = build_simulator(config, clock, plant)
        self.injector = FaultInjector(operator, clock)
        self.thermostat = BaselineThermostat(
            config=config.controller,
            clock=clock,
            actuator_id=topics.AIR_CONDITIONER_ID,
            limits=config.detectors.out_of_range.temperature_c,
        )

        self._controller_board = controller_board
        controller_board.subscribe(
            topics.SENSOR_STATE, SensorReading, self._on_reading
        )
        self.simulator.subscribe()
        for board in (plant, controller_board, operator):
            self.transport.attach(board)

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        if reading.sensor_id != self.config.estimator.indoor_sensor_id:
            return
        self.thermostat.observe(reading)

    def run_for(self, seconds: float) -> None:
        period_s = self.config.loop.sensor_period_s
        elapsed = 0.0
        while elapsed < seconds:
            self.simulator.step()
            command = self.thermostat.tick()
            if command is not None:
                # Published for the plant to act on, exactly as the real
                # controller publishes: the baseline drives the same actuator
                # through the same topic.
                self._controller_board.publish(
                    topics.ACTUATOR_COMMAND,
                    command,
                    actuator_id=command.actuator_id,
                )
            self.log.samples.append(
                Sample(
                    ts=self.clock.now(),
                    true_c=self.simulator.room_temperature_c,
                    setpoint_c=self.thermostat.setpoint_c,
                    mode=Mode.NORMAL,
                    command=command.kind if command is not None else None,
                )
            )
            self.clock.advance(period_s)
            elapsed += period_s

    def inject(self, kind: InjectedFault, magnitude: float | None = None) -> float:
        self.injector.inject(INDOOR_TEMPERATURE_ID, kind, magnitude)
        return self.clock.now()

    def clear_injection(self) -> None:
        self.injector.clear(INDOOR_TEMPERATURE_ID)



def for_experiment(config: Config, state_dir) -> Config:
    """Point the estimator's state file somewhere disposable.

    An experiment must not adopt coefficients left by the last one, or the
    second run of any comparison starts from an answer the first one found.
    """
    persistence = config.persistence.model_copy(
        update={"path": str(state_dir / "coefficients.json")}
    )
    return config.model_copy(update={"persistence": persistence})


def full_system(config: Config, clock: Clock | None = None) -> FullSystem:
    return FullSystem(config, clock or SimClock())


def baseline_system(config: Config, clock: Clock | None = None) -> BaselineSystem:
    return BaselineSystem(config, clock or SimClock())
