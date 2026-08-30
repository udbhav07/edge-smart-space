"""Unit tests for the room simulator process.

Runs the whole Layer 1 loop with no broker and no wall-clock waiting. What
is asserted is the thing the Week 1-2 gate asks for: the plant runs and the
topics are observable, under exactly the names hardware will publish.
"""

from pathlib import Path

import pytest

from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    ActuatorState,
    Command,
    CommandKind,
    SensorReading,
    Unit,
)
from sim.run_sim import (
    INDOOR_TEMPERATURE_ID,
    OCCUPANCY_ID,
    OUTDOOR_TEMPERATURE_ID,
    build_simulator,
)


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="quiet_config")
def _quiet_config(config):
    """Imperfection off, for tests that need every sample to arrive."""
    noise = config.sim.sensor_noise.model_copy(
        update={"dropout_probability": 0.0, "jitter_s": 0.0}
    )
    actuator = config.sim.actuator.model_copy(
        update={"command_loss_probability": 0.0}
    )
    return config.model_copy(
        update={"sim": config.sim.model_copy(update={"sensor_noise": noise, "actuator": actuator})}
    )


class RecordingTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []

    def connect(self, host, port, keepalive): ...
    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))
    def loop_start(self): ...
    def loop_stop(self): ...
    def disconnect(self): ...

    def topics_seen(self) -> set[str]:
        return {topic for topic, _, _, _ in self.published}

    def payloads_on(self, topic: str) -> list[bytes]:
        return [payload for name, payload, _, _ in self.published if name == topic]


def _running(config):
    clock = SimClock()
    transport = RecordingTransport()
    blackboard = Blackboard(config.mqtt, transport)
    simulator = build_simulator(config, clock, blackboard)
    simulator.subscribe()
    return simulator, transport, clock, blackboard


class TestTopicsAreObservable:
    """The Week 1-2 gate: the plant runs and its topics can be watched."""

    def test_indoor_temperature_is_published(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        assert f"space/sensor/{INDOOR_TEMPERATURE_ID}/state" in transport.topics_seen()

    def test_occupancy_is_published(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        assert f"space/sensor/{OCCUPANCY_ID}/state" in transport.topics_seen()

    def test_outdoor_temperature_is_published(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        assert f"space/sensor/{OUTDOOR_TEMPERATURE_ID}/state" in transport.topics_seen()

    def test_actuator_state_is_published_every_step(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        assert "space/actuator/ac/state" in transport.topics_seen()

    def test_the_simulator_listens_for_commands_like_a_real_driver(self, quiet_config):
        _, transport, _, blackboard = _running(quiet_config)
        blackboard.on_connected()
        assert ("space/actuator/+/command", topics.ACTUATOR_COMMAND.qos.value) in (
            transport.subscribed
        )

    def test_every_published_topic_lives_under_the_blackboard_root(self, quiet_config):
        simulator, transport, clock, _ = _running(quiet_config)
        for _ in range(5):
            simulator.step()
            clock.advance(quiet_config.loop.sensor_period_s)
        assert all(
            name.startswith(f"{topics.TOPIC_ROOT}/") for name in transport.topics_seen()
        )


class TestPublishedPayloads:
    def test_readings_decode_against_the_shared_schema(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        payload = transport.payloads_on(
            f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"
        )[0]
        assert SensorReading.model_validate_json(payload).unit is Unit.CELSIUS

    def test_occupancy_is_reported_as_a_boolean_reading(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        payload = transport.payloads_on(f"space/sensor/{OCCUPANCY_ID}/state")[0]
        assert SensorReading.model_validate_json(payload).unit is Unit.BOOLEAN

    def test_the_actuator_reports_it_cannot_confirm(self, quiet_config):
        """R-02: the shipped configuration is open-loop."""
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        payload = transport.payloads_on("space/actuator/ac/state")[0]
        assert ActuatorState.model_validate_json(payload).ack.value == "UNKNOWN"

    def test_readings_carry_the_simulated_clock_time(self, quiet_config):
        simulator, transport, clock, _ = _running(quiet_config)
        clock.advance(1234.0)
        simulator.step()
        payload = transport.payloads_on(
            f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"
        )[0]
        assert SensorReading.model_validate_json(payload).ts == clock.now()


class TestCadence:
    def test_ambient_updates_less_often_than_the_indoor_sensor(self, quiet_config):
        """FR-03 and A-04: ambient arrives at 60 s or slower."""
        simulator, transport, clock, _ = _running(quiet_config)
        steps = int(quiet_config.loop.outdoor_period_s / quiet_config.loop.sensor_period_s)
        for _ in range(steps):
            simulator.step()
            clock.advance(quiet_config.loop.sensor_period_s)

        indoor = len(transport.payloads_on(f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"))
        outdoor = len(
            transport.payloads_on(f"space/sensor/{OUTDOOR_TEMPERATURE_ID}/state")
        )
        assert outdoor < indoor

    def test_a_lost_sample_is_published_as_nothing_at_all(self, config):
        """D1 detects absence, so absence is what must reach the bus."""
        noise = config.sim.sensor_noise.model_copy(
            update={"dropout_probability": 1.0}
        )
        certain_loss = config.model_copy(
            update={"sim": config.sim.model_copy(update={"sensor_noise": noise})}
        )
        simulator, transport, _, _ = _running(certain_loss)
        simulator.step()
        assert transport.payloads_on(
            f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"
        ) == []


class TestClosedLoop:
    def test_a_cool_command_eventually_lowers_the_room(self, quiet_config):
        simulator, _, clock, blackboard = _running(quiet_config)
        start_c = simulator._room.temperature_c

        command = Command(
            ts=clock.now(),
            actuator_id=topics.AIR_CONDITIONER_ID,
            kind=CommandKind.COOL,
            setpoint_c=22.0,
        )
        blackboard.dispatch(
            "space/actuator/ac/command", command.model_dump_json().encode()
        )

        for _ in range(200):
            simulator.step()
            clock.advance(quiet_config.loop.sensor_period_s)

        assert simulator._room.temperature_c < start_c

    def test_the_room_warms_toward_ambient_with_no_cooling(self, quiet_config):
        cool_start = quiet_config.sim.room.model_copy(
            update={"initial_temperature_c": 20.0, "solar_gain_amplitude_w": 0.0}
        )
        warmer = quiet_config.model_copy(
            update={"sim": quiet_config.sim.model_copy(update={"room": cool_start})}
        )
        simulator, _, clock, _ = _running(warmer)
        for _ in range(200):
            simulator.step()
            clock.advance(warmer.loop.sensor_period_s)
        assert simulator._room.temperature_c > 20.0

    def test_the_command_acknowledgement_is_reported_back(self, quiet_config):
        simulator, transport, clock, blackboard = _running(quiet_config)
        command = Command(
            ts=clock.now(),
            actuator_id=topics.AIR_CONDITIONER_ID,
            kind=CommandKind.COOL,
            setpoint_c=22.0,
        )
        blackboard.dispatch(
            "space/actuator/ac/command", command.model_dump_json().encode()
        )
        simulator.step()
        payload = transport.payloads_on("space/actuator/ac/state")[-1]
        assert ActuatorState.model_validate_json(payload).kind is CommandKind.COOL


class TestRunLoop:
    def test_run_executes_the_requested_number_of_steps(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.run(steps=3)
        assert len(transport.payloads_on("space/actuator/ac/state")) == 3

    def test_run_advances_simulated_time(self, quiet_config):
        simulator, _, clock, _ = _running(quiet_config)
        before = clock.now()
        simulator.run(steps=4)
        assert clock.now() == before + 4 * quiet_config.loop.sensor_period_s


class TestOccupancy:
    def test_occupancy_defaults_to_present(self, quiet_config):
        """Conservative for comfort, matching how a failed PIR is treated."""
        simulator, _, _, _ = _running(quiet_config)
        assert simulator.occupied is True

    def test_vacancy_is_reported_when_set(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.occupied = False
        simulator.step()
        payload = transport.payloads_on(f"space/sensor/{OCCUPANCY_ID}/state")[0]
        assert SensorReading.model_validate_json(payload).value == 0.0
