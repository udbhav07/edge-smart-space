"""The room simulator process.

Stands in for Layer 1 (DESIGN.md section 4.4). It publishes to exactly the
topics the ESP32 nodes will publish to and consumes exactly the topic the
actuator driver will consume, so nothing above Layer 1 can tell which is
running. That is what lets the Weeks 1-4 work carry over to hardware
unchanged.

This module must not import from ``src.estimation``: the plant and the
estimator's model have to stay independently parameterised or the evaluation
degenerates into the model predicting itself (section 5.10).

Everything is injected -- clock, blackboard, plant, sensors, actuator -- so
the whole loop is testable with no broker and no wall-clock waiting.
"""

from __future__ import annotations

import argparse
import logging
import random
import math
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import (
    AckStatus,
    ActuatorState,
    Command,
    CommandKind,
    SensorReading,
    Unit,
)
from sim.actuator import SimulatedActuator
from sim.room_model import RoomModel
from sim.sensors import BinarySensor, SimulatedSensor

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: Sensor identifiers. These are the ids the ESP32 nodes will report under,
#: so nothing above Layer 1 has to change when the hardware arrives.
INDOOR_TEMPERATURE_ID = "temp_01"
OUTDOOR_TEMPERATURE_ID = "outdoor_01"
OCCUPANCY_ID = "pir_01"

#: The simulated air conditioner is not a real actuator, but it is the stand
#: -in for the one that is, so it publishes under the real id and is *not*
#: labelled simulated: the topic contract is what hardware will inherit.
_SIMULATED_PLANT = True

_FULL_CYCLE_RADIANS = 2.0 * math.pi


class RoomSimulator:
    """Drives the plant and speaks the blackboard's language.

    One ``step`` advances simulated time by the sensor period, moves the
    plant, samples every sensor, and publishes whatever survived. A dropped
    sample is published as nothing at all, because that is what D1 detects.
    """

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        room: RoomModel,
        actuator: SimulatedActuator,
        indoor: SimulatedSensor,
        outdoor: SimulatedSensor,
        occupancy: BinarySensor,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._room = room
        self._actuator = actuator
        self._indoor = indoor
        self._outdoor = outdoor
        self._occupancy = occupancy
        self._occupied = True
        self._started_ts = clock.now()
        self._last_outdoor_publish_ts: float | None = None
        self._last_ack = AckStatus.UNKNOWN
        self._last_kind = CommandKind.OFF

    @property
    def occupied(self) -> bool:
        """Whether someone is in the room.

        Defaults to occupied, which is the conservative assumption for
        comfort and matches how the system treats a failed PIR (section 7.1).
        Scenario files will drive this.
        """
        return self._occupied

    @occupied.setter
    def occupied(self, value: bool) -> None:
        self._occupied = value

    def subscribe(self) -> None:
        """Listen for actuator commands, exactly as a real driver would."""
        self._blackboard.subscribe(
            topics.ACTUATOR_COMMAND, Command, self._on_command
        )

    def _on_command(self, topic: str, command: Command) -> None:
        LOGGER.debug("command on %s: %s", topic, command.kind.value)
        self._last_kind = command.kind
        self._last_ack = self._actuator.command(command.kind, command.setpoint_c)

    def outdoor_temperature_c(self) -> float:
        """Ambient, as a daily cycle around the configured mean."""
        elapsed_s = self._clock.now() - self._started_ts
        phase = _FULL_CYCLE_RADIANS * elapsed_s / self._config.sim.outdoor_period_s
        return self._config.sim.outdoor_mean_c + (
            self._config.sim.outdoor_amplitude_c * math.sin(phase)
        )

    def step(self) -> None:
        """Advance the plant one sensor period and publish what came out."""
        period_s = self._config.loop.sensor_period_s
        self._actuator.apply_due_commands()
        outdoor_c = self.outdoor_temperature_c()

        self._room.step(
            duration_s=period_s,
            cooling_fraction=self._actuator.cooling_fraction,
            occupied=self._occupied,
            outdoor_c=outdoor_c,
        )

        self._publish_indoor()
        self._publish_outdoor(outdoor_c)
        self._publish_occupancy()
        self._publish_actuator_state()

    def _publish_indoor(self) -> None:
        reading = self._indoor.sample(self._room.temperature_c)
        self._publish_reading(reading, INDOOR_TEMPERATURE_ID)

    def _publish_outdoor(self, outdoor_c: float) -> None:
        """Ambient updates at its own, slower cadence (FR-03, A-04)."""
        now = self._clock.now()
        due = (
            self._last_outdoor_publish_ts is None
            or now - self._last_outdoor_publish_ts >= self._config.loop.outdoor_period_s
        )
        if not due:
            return
        self._last_outdoor_publish_ts = now
        self._publish_reading(self._outdoor.sample(outdoor_c), OUTDOOR_TEMPERATURE_ID)

    def _publish_occupancy(self) -> None:
        self._publish_reading(self._occupancy.sample(self._occupied), OCCUPANCY_ID)

    def _publish_reading(self, reading: SensorReading | None, sensor_id: str) -> None:
        if reading is None:
            # A lost sample is the absence of a message, which is exactly
            # what D1 detects. Publishing a placeholder would hide it.
            LOGGER.debug("sample lost for %s", sensor_id)
            return
        self._blackboard.publish(
            topics.SENSOR_STATE, reading, sensor_id=reading.sensor_id
        )

    def _publish_actuator_state(self) -> None:
        state = ActuatorState(
            ts=self._clock.now(),
            actuator_id=topics.AIR_CONDITIONER_ID,
            simulated=_SIMULATED_PLANT,
            kind=self._last_kind,
            setpoint_c=None,
            ack=self._last_ack,
            last_command_ts=self._actuator.last_command_ts,
        )
        self._blackboard.publish(
            topics.ACTUATOR_STATE, state, actuator_id=topics.AIR_CONDITIONER_ID
        )

    def run(self, steps: int | None = None) -> None:
        """Run the plant. ``steps=None`` runs until interrupted."""
        period_s = self._config.loop.sensor_period_s
        completed = 0
        while steps is None or completed < steps:
            self.step()
            self._clock.sleep(period_s)
            completed += 1


def build_simulator(
    config: Config, clock: Clock, blackboard: Blackboard
) -> RoomSimulator:
    """Assemble the plant and its imperfect sensors from configuration."""
    rng = random.Random(config.sim.random_seed)
    return RoomSimulator(
        config=config,
        clock=clock,
        blackboard=blackboard,
        room=RoomModel(config.sim.room, clock),
        actuator=SimulatedActuator(config.sim.actuator, rng, clock),
        indoor=SimulatedSensor(
            INDOOR_TEMPERATURE_ID, Unit.CELSIUS, config.sim.sensor_noise, rng, clock
        ),
        outdoor=SimulatedSensor(
            OUTDOOR_TEMPERATURE_ID, Unit.CELSIUS, config.sim.sensor_noise, rng, clock
        ),
        occupancy=BinarySensor(
            OCCUPANCY_ID, config.sim.sensor_noise, rng, clock
        ),
    )


def main(argv: list[str] | None = None) -> int:
    """Run the simulator against a live broker."""
    parser = argparse.ArgumentParser(description="Run the room simulator.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--steps", type=int, default=None, help="Stop after N steps; default is forever."
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(arguments.config)

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, "room-simulator", holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)

    simulator = build_simulator(config, clock, blackboard)
    simulator.subscribe()
    blackboard.start()
    LOGGER.info(
        "room simulator publishing under %s; subscribe with: mosquitto_sub -t '%s' -v",
        topics.TOPIC_ROOT,
        topics.ALL_TOPICS,
    )
    try:
        simulator.run(steps=arguments.steps)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
