"""The real room's Layer 1: ESPHome nodes in, the air conditioner out (Week 5).

What the simulator is for simulation, this is for hardware, and everything
above Layer 1 is identical for both (section 4.2). The bridge listens to the
nodes' own MQTT topics -- plain values such as ``27.4`` or ``ON`` -- and
publishes schema-valid readings through the existing :class:`SensorAdapter`,
so injection, quality flags and "absence is a dropout" behave exactly as the
adapter was written and tested to. Commands admitted by the validator go out
to the unit's climate topics.

Three things are deliberate:

* **Readings are published when a node reports, not on a timer.** A silent
  node must look silent, because silence is what D1 detects; a timer
  republishing the last value would turn a dead node into a stuck sensor.
  Occupancy is the exception: it is derived from motion and the door through
  the vacancy hold-off (FR-02) and published every period like the simulator.
* **Health is not published here.** The detector bank owns
  ``space/sensor/{id}/health``; two publishers would fight over it.
* **A dead unit cannot be injected into real hardware**, so ``ac dead`` is
  emulated by withholding commands: the unit keeps doing whatever it was
  doing and the room stops responding to what the system asks, which is the
  symptom D5 exists to find. The log says it is an emulation.
"""

from __future__ import annotations

import logging

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config, SensorConfig
from src.common.injection import ACTUATOR_SUPPORTED_FAULTS, FaultInjection, InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.occupancy import OccupancyTracker
from src.common.schemas import (
    AckStatus,
    ActuatorState,
    Command,
    CommandKind,
    InjectionCommand,
)
from src.io.sensor_adapters.base import SensorAdapter
from src.io.sensor_adapters.esphome import EsphomeSource

LOGGER = logging.getLogger(__name__)

#: What ESPHome's binary sensors publish.
_ON = "ON"

#: ESPHome climate modes for the two commands the control law issues.
_CLIMATE_MODES = {CommandKind.COOL: "cool", CommandKind.OFF: "off"}

_OCCUPIED = 1.0
_VACANT = 0.0


class _OccupancySource:
    """Occupancy as a sensor source: presence through the hold-off (FR-02)."""

    def __init__(self, tracker: OccupancyTracker) -> None:
        self._tracker = tracker

    def read(self) -> float | None:
        return _OCCUPIED if self._tracker.occupied else _VACANT


class DeviceBridge:
    """ESPHome nodes to the blackboard, and admitted commands to the unit."""

    def __init__(self, config: Config, clock: Clock, blackboard: Blackboard) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        devices = config.devices
        self._occupancy_id = config.estimator.occupancy_sensor_id
        self._tracker = OccupancyTracker(
            hold_off_s=config.sensors.vacancy_hold_off_s, clock=clock
        )
        self._sources: dict[str, EsphomeSource] = {}
        self._adapters: dict[str, SensorAdapter] = {}
        self._by_topic: dict[str, str] = {}
        for sensor in config.sensors.adapters:
            topic = devices.sensor_topics.get(sensor.sensor_id, "")
            if sensor.sensor_id == self._occupancy_id:
                self._adapters[sensor.sensor_id] = self._adapter(
                    sensor, _OccupancySource(self._tracker)
                )
            elif topic:
                source = EsphomeSource(clock, devices.stale_after_s)
                self._sources[sensor.sensor_id] = source
                self._adapters[sensor.sensor_id] = self._adapter(sensor, source)
            if topic:
                self._by_topic[topic] = sensor.sensor_id
        self._last_kind = CommandKind.OFF
        self._last_command_ts: float | None = None
        self._ack = AckStatus.UNKNOWN
        self._readback_mode: str | None = None
        self._unit_dead = False

    def _adapter(self, sensor: SensorConfig, source) -> SensorAdapter:
        return SensorAdapter(sensor, source, self._clock, self._blackboard)

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        for topic in self._by_topic:
            self._blackboard.subscribe_device(topic, self._on_device)
        if self._config.devices.door_topic:
            self._blackboard.subscribe_device(self._config.devices.door_topic, self._on_door)
        if self._config.devices.ac_mode_state_topic:
            self._blackboard.subscribe_device(
                self._config.devices.ac_mode_state_topic, self._on_unit_mode
            )
        self._blackboard.subscribe(topics.ACTUATOR_COMMAND, Command, self._on_command)
        self._blackboard.subscribe(topics.INJECT, InjectionCommand, self._on_injection)

    @property
    def sensor_ids(self) -> tuple[str, ...]:
        return tuple(self._adapters)

    # --- sensors ------------------------------------------------------

    def _on_device(self, topic: str, text: str) -> None:
        sensor_id = self._by_topic[topic]
        if sensor_id == self._occupancy_id:
            if text.upper() == _ON:
                self._tracker.motion()
            return
        try:
            value = float(text)
        except ValueError:
            LOGGER.warning("unreadable value %r from %s ignored", text, topic)
            return
        self._sources[sensor_id].accept(value)
        self._publish(sensor_id)

    def _on_door(self, _topic: str, _text: str) -> None:
        # Any transition, open or close, says someone was at the door (FR-02).
        self._tracker.door_transition()

    def _publish(self, sensor_id: str) -> None:
        reading = self._adapters[sensor_id].read()
        if reading is not None:
            self._adapters[sensor_id].publish(reading)

    def tick(self) -> None:
        """Once per sensor period: occupancy, and the unit's reported state."""
        if self._occupancy_id in self._adapters:
            self._publish(self._occupancy_id)
        self._blackboard.publish(
            topics.ACTUATOR_STATE,
            ActuatorState(
                ts=self._clock.now(),
                actuator_id=topics.AIR_CONDITIONER_ID,
                simulated=False,
                kind=self._last_kind,
                setpoint_c=None,
                ack=self._ack,
                last_command_ts=self._last_command_ts,
            ),
            actuator_id=topics.AIR_CONDITIONER_ID,
        )

    # --- the unit -----------------------------------------------------

    def _on_command(self, _topic: str, command: Command) -> None:
        """Carry an admitted command to the unit, one at a time (FR-12)."""
        if command.actuator_id != topics.AIR_CONDITIONER_ID:
            return
        mode = _CLIMATE_MODES.get(command.kind)
        if mode is None:
            return  # MAINTAIN and HOLD ask for nothing new
        self._last_kind = command.kind
        self._last_command_ts = command.ts
        if self._unit_dead:
            LOGGER.warning("emulated dead unit: %s withheld", command.kind.value)
            return
        devices = self._config.devices
        if devices.ac_mode_command_topic:
            self._blackboard.publish_device(devices.ac_mode_command_topic, mode)
        if command.setpoint_c is not None and devices.ac_target_command_topic:
            self._blackboard.publish_device(
                devices.ac_target_command_topic, f"{command.setpoint_c:.1f}"
            )
        self._ack = AckStatus.UNKNOWN
        self._reconcile()

    def _on_unit_mode(self, _topic: str, text: str) -> None:
        self._readback_mode = text.strip().lower()
        self._reconcile()

    def _reconcile(self) -> None:
        """Acknowledge only what the unit itself reported (R-02)."""
        if self._readback_mode is None:
            return
        expected = _CLIMATE_MODES[self._last_kind]
        self._ack = (
            AckStatus.ACKNOWLEDGED if self._readback_mode == expected else AckStatus.FAILED
        )

    # --- injection (FR-31) --------------------------------------------

    def _on_injection(self, _topic: str, command: InjectionCommand) -> None:
        if command.subject == topics.AIR_CONDITIONER_ID:
            if command.kind in ACTUATOR_SUPPORTED_FAULTS:
                self._unit_dead = command.kind is InjectedFault.NO_RESPONSE
                LOGGER.info("unit %s", "emulated dead" if self._unit_dead else "restored")
            return
        adapter = self._adapters.get(command.subject)
        if adapter is None:
            LOGGER.warning("injection for unknown subject %s ignored", command.subject)
            return
        if command.kind is InjectedFault.NONE:
            adapter.clear()
        elif command.kind is InjectedFault.NO_RESPONSE:
            LOGGER.warning("injection on %s refused: a sensor cannot be unresponsive", command.subject)
        else:
            adapter.inject(FaultInjection(kind=command.kind, magnitude=command.magnitude))
