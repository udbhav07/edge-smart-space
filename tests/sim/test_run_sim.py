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
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    ActuatorState,
    Command,
    CommandKind,
    InjectionCommand,
    SensorReading,
    Unit,
)
from sim.run_sim import (
    INDOOR_TEMPERATURE_ID,
    OCCUPANCY_ID,
    OUTDOOR_TEMPERATURE_ID,
    POWER_ID,
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


class TestInjectionAtLayerOne:
    """FR-31: the adapter obeys, and nothing above can tell the difference."""

    def _inject(self, blackboard, subject, kind, magnitude=0.0):
        command = InjectionCommand(
            ts=1756032000.0,
            subject=subject,
            kind=kind,
            magnitude=magnitude,
            requester="operator",
        )
        blackboard.dispatch(
            f"space/inject/{subject}", command.model_dump_json().encode()
        )

    def test_the_simulator_listens_for_injections(self, config):
        simulator, transport, _, blackboard = _running(config)
        blackboard.on_connected()
        assert ("space/inject/+", topics.INJECT.qos.value) in transport.subscribed

    def test_a_stuck_injection_freezes_the_reported_value(self, quiet_config):
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(
            blackboard, INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, 27.0
        )
        for _ in range(3):
            simulator.step()
        values = [
            SensorReading.model_validate_json(payload).value
            for payload in transport.payloads_on(
                f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"
            )
        ]
        assert values == [27.0, 27.0, 27.0]

    def test_a_dropout_injection_publishes_nothing_at_all(self, quiet_config):
        """A lost sample is the absence of a message, which is what D1
        detects. A placeholder would hide it."""
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(blackboard, INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        for _ in range(3):
            simulator.step()
        assert (
            transport.payloads_on(f"space/sensor/{INDOOR_TEMPERATURE_ID}/state")
            == []
        )

    def test_an_out_of_range_injection_reports_the_implausible_value(
        self, quiet_config
    ):
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(
            blackboard, INDOOR_TEMPERATURE_ID, InjectedFault.OUT_OF_RANGE, 999.0
        )
        simulator.step()
        reading = SensorReading.model_validate_json(
            transport.payloads_on(f"space/sensor/{INDOOR_TEMPERATURE_ID}/state")[0]
        )
        assert reading.value == 999.0

    def test_clearing_returns_the_sensor_to_the_plant(self, quiet_config):
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(
            blackboard, INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, 27.0
        )
        simulator.step()
        self._inject(blackboard, INDOOR_TEMPERATURE_ID, InjectedFault.NONE)
        simulator.step()
        values = [
            SensorReading.model_validate_json(payload).value
            for payload in transport.payloads_on(
                f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"
            )
        ]
        assert values[0] == 27.0 and values[1] != 27.0

    def test_injecting_one_sensor_leaves_the_others_alone(self, quiet_config):
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(blackboard, INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        simulator.step()
        assert transport.payloads_on(f"space/sensor/{OCCUPANCY_ID}/state")

    def test_a_fault_a_binary_sensor_cannot_honour_is_refused(self, quiet_config):
        """An injection that appears to work and does nothing turns a
        detection trial into a phantom missed detection."""
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(blackboard, OCCUPANCY_ID, InjectedFault.DRIFT, 0.01)
        simulator.step()
        reading = SensorReading.model_validate_json(
            transport.payloads_on(f"space/sensor/{OCCUPANCY_ID}/state")[0]
        )
        assert reading.value in (0.0, 1.0)

    def test_an_injection_for_an_unknown_subject_does_not_stop_the_plant(
        self, quiet_config
    ):
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(blackboard, "ghost_99", InjectedFault.DROPOUT)
        simulator.step()
        assert transport.payloads_on(
            f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"
        )

    def test_a_dropout_can_be_injected_on_the_binary_sensor(self, quiet_config):
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(blackboard, OCCUPANCY_ID, InjectedFault.DROPOUT)
        simulator.step()
        assert transport.payloads_on(f"space/sensor/{OCCUPANCY_ID}/state") == []


def _quiet_meter(config):
    meter = config.sim.power_meter.model_copy(
        update={"dropout_probability": 0.0, "sigma_w": 0.0, "resolution_w": 0.0}
    )
    return config.model_copy(
        update={"sim": config.sim.model_copy(update={"power_meter": meter})}
    )


def _power_readings(transport) -> list[float]:
    return [
        SensorReading.model_validate_json(payload).value
        for payload in transport.payloads_on(f"space/sensor/{POWER_ID}/state")
    ]


def _cool(blackboard, clock) -> None:
    command = Command(
        ts=clock.now(),
        actuator_id=topics.AIR_CONDITIONER_ID,
        kind=CommandKind.COOL,
        setpoint_c=22.0,
    )
    blackboard.dispatch(
        "space/actuator/ac/command", command.model_dump_json().encode()
    )


class TestPowerMeter:
    """Week 5: the air conditioner's draw, on the topic hardware will use."""

    def test_power_is_published_in_watts(self, quiet_config):
        simulator, transport, _, _ = _running(_quiet_meter(quiet_config))
        simulator.step()
        payload = transport.payloads_on(f"space/sensor/{POWER_ID}/state")[-1]
        assert SensorReading.model_validate_json(payload).unit is Unit.WATT

    def test_an_idle_unit_draws_standby(self, quiet_config):
        config = _quiet_meter(quiet_config)
        simulator, transport, _, _ = _running(config)
        simulator.step()
        assert _power_readings(transport)[-1] == pytest.approx(
            config.sim.power_meter.standby_power_w
        )

    def test_a_running_unit_draws_its_rated_power(self, quiet_config):
        config = _quiet_meter(quiet_config)
        simulator, transport, clock, blackboard = _running(config)
        _cool(blackboard, clock)
        for _ in range(20):  # past the dead time
            simulator.step()
            clock.advance(config.loop.sensor_period_s)
        assert _power_readings(transport)[-1] == pytest.approx(
            config.sim.power_meter.rated_power_w
        )

    def test_a_dead_unit_draws_standby_while_commanded_to_cool(self, quiet_config):
        """What makes the meter evidence about the actuator rather than an echo
        of the command (R-02)."""
        config = _quiet_meter(quiet_config)
        simulator, transport, clock, blackboard = _running(config)
        TestInjectionAtLayerOne()._inject(
            blackboard, topics.AIR_CONDITIONER_ID, InjectedFault.STUCK_OFF
        )
        _cool(blackboard, clock)
        for _ in range(20):
            simulator.step()
            clock.advance(config.loop.sensor_period_s)
        assert _power_readings(transport)[-1] == pytest.approx(
            config.sim.power_meter.standby_power_w
        )

    def test_a_noisy_meter_never_reports_a_negative_draw(self, config):
        """Noise around standby must not read as a meter wired backwards."""
        simulator, transport, clock, _ = _running(config)
        for _ in range(200):
            simulator.step()
            clock.advance(config.loop.sensor_period_s)
        assert min(_power_readings(transport)) >= 0.0

    def test_an_injected_negative_reading_is_left_negative(self, quiet_config):
        """The floor is the meter's physics, not a filter on injected faults."""
        simulator, transport, _, blackboard = _running(_quiet_meter(quiet_config))
        TestInjectionAtLayerOne()._inject(
            blackboard, POWER_ID, InjectedFault.OUT_OF_RANGE, -50.0
        )
        simulator.step()
        assert _power_readings(transport)[-1] == -50.0

    def test_the_meter_can_be_broken_like_any_sensor(self, quiet_config):
        """FR-31 covers every instrument, not only the ones that feed the model."""
        simulator, transport, _, blackboard = _running(_quiet_meter(quiet_config))
        TestInjectionAtLayerOne()._inject(blackboard, POWER_ID, InjectedFault.DROPOUT)
        simulator.step()
        assert _power_readings(transport) == []

    def test_metering_leaves_every_other_noise_sequence_alone(self, config):
        """Seeded results already measured must not move because a meter was
        added: the meter draws from its own random stream."""
        without = config.model_copy(
            update={
                "sensors": config.sensors.model_copy(
                    update={
                        "adapters": tuple(
                            a for a in config.sensors.adapters
                            if a.sensor_id != POWER_ID
                        )
                    }
                )
            }
        )
        runs = []
        for variant in (config, without):
            simulator, transport, clock, _ = _running(variant)
            for _ in range(30):
                simulator.step()
                clock.advance(variant.loop.sensor_period_s)
            runs.append(
                transport.payloads_on(f"space/sensor/{INDOOR_TEMPERATURE_ID}/state")
            )
        assert runs[0] == runs[1]

    def test_no_meter_is_simulated_when_none_is_configured(self, config):
        without = config.model_copy(
            update={
                "sensors": config.sensors.model_copy(
                    update={
                        "adapters": tuple(
                            a for a in config.sensors.adapters
                            if a.sensor_id != POWER_ID
                        )
                    }
                )
            }
        )
        simulator, transport, _, _ = _running(without)
        simulator.step()
        assert _power_readings(transport) == []


class TestLayerOneIsExclusive:
    def test_the_simulator_refuses_to_run_beside_real_hardware(self, tmp_path):
        """With io.source esphome, real sensors own the topics (section 9.1);
        a simulator beside them would be a second room contradicting the
        first."""
        from sim.run_sim import main

        text = Path("config/default.yaml").read_text(encoding="utf-8")
        hardware = tmp_path / "hardware.yaml"
        hardware.write_text(
            text.replace("  source: simulated", "  source: esphome", 1), encoding="utf-8"
        )
        assert main(["--config", str(hardware)]) == 2
