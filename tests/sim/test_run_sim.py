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
    AckStatus,
    ActuatorState,
    Command,
    CommandKind,
    InjectionCommand,
    Quality,
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

    def _command(self, blackboard, clock, kind):
        command = Command(
            ts=clock.now(),
            actuator_id=topics.AIR_CONDITIONER_ID,
            kind=kind,
            setpoint_c=22.0 if kind is CommandKind.COOL else None,
        )
        blackboard.dispatch(
            "space/actuator/ac/command", command.model_dump_json().encode()
        )

    def _published_kind(self, transport):
        payload = transport.payloads_on("space/actuator/ac/state")[-1]
        return ActuatorState.model_validate_json(payload).kind

    @pytest.mark.parametrize("carry_on", [CommandKind.MAINTAIN, CommandKind.HOLD])
    def test_carrying_on_leaves_the_published_state_as_it_was(
        self, quiet_config, carry_on
    ):
        """Regression: the state echoed MAINTAIN, so a subscriber that missed
        the one COOL -- a restarted estimator -- believed cooling was off."""
        simulator, transport, clock, blackboard = _running(quiet_config)
        self._command(blackboard, clock, CommandKind.COOL)
        simulator.step()
        self._command(blackboard, clock, carry_on)
        simulator.step()
        assert self._published_kind(transport) is CommandKind.COOL

    def test_switching_off_is_published_as_off(self, quiet_config):
        simulator, transport, clock, blackboard = _running(quiet_config)
        self._command(blackboard, clock, CommandKind.COOL)
        simulator.step()
        self._command(blackboard, clock, CommandKind.OFF)
        simulator.step()
        assert self._published_kind(transport) is CommandKind.OFF


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

    def test_a_dead_unit_cools_nothing(self, quiet_config):
        """FR-24, FR-31: the actuator fault is injectable like any other."""
        simulator, transport, clock, blackboard = _running(quiet_config)
        self._inject(blackboard, "ac", InjectedFault.NO_RESPONSE)
        command = Command(
            ts=clock.now(), actuator_id="ac", kind=CommandKind.COOL, setpoint_c=22.0
        )
        blackboard.dispatch("space/actuator/ac/command", command.model_dump_json().encode())
        start = simulator.room_temperature_c
        for _ in range(120):
            simulator.step()
            clock.advance(quiet_config.loop.sensor_period_s)
        assert simulator.room_temperature_c >= start

    def test_a_dead_unit_still_reports_what_it_was_told(self, quiet_config):
        """No acknowledgement betrays it (R-02): D5 must find it from the room."""
        simulator, transport, clock, blackboard = _running(quiet_config)
        self._inject(blackboard, "ac", InjectedFault.NO_RESPONSE)
        command = Command(
            ts=clock.now(), actuator_id="ac", kind=CommandKind.COOL, setpoint_c=22.0
        )
        blackboard.dispatch("space/actuator/ac/command", command.model_dump_json().encode())
        simulator.step()
        payload = transport.payloads_on("space/actuator/ac/state")[-1]
        assert ActuatorState.model_validate_json(payload).kind is CommandKind.COOL

    def test_a_sensor_fault_on_the_unit_is_refused(self, quiet_config):
        simulator, _, clock, blackboard = _running(quiet_config)
        self._inject(blackboard, "ac", InjectedFault.STUCK_AT, 27.0)
        assert simulator._actuator.is_failed is False

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

    def test_an_out_of_range_reading_is_flagged_by_the_configured_limits(
        self, quiet_config
    ):
        """The limits come from config.sensors, the same ones D3 uses."""
        simulator, transport, _, blackboard = _running(quiet_config)
        self._inject(
            blackboard, INDOOR_TEMPERATURE_ID, InjectedFault.OUT_OF_RANGE, 999.0
        )
        simulator.step()
        reading = SensorReading.model_validate_json(
            transport.payloads_on(f"space/sensor/{INDOOR_TEMPERATURE_ID}/state")[0]
        )
        assert reading.quality is Quality.SUSPECT

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


class TestAmbientFollowsTheLocalDay:
    """FR-03: the hottest hour is a time of day, not an offset from launch."""

    def _ambient_at(self, config, hour: float) -> float:
        from datetime import datetime, timedelta

        from src.common.localtime import epoch_of

        local = datetime(2026, 10, 1) + timedelta(hours=hour)
        clock = SimClock(start_epoch_s=epoch_of(local, config.site.utc_offset_h))
        blackboard = Blackboard(config.mqtt, RecordingTransport())
        return build_simulator(config, clock, blackboard).outdoor_temperature_c()

    def test_the_peak_hour_is_the_hottest(self, config):
        peak = config.sim.outdoor_peak_hour
        assert self._ambient_at(config, peak) == pytest.approx(
            config.sim.outdoor_mean_c + config.sim.outdoor_amplitude_c
        )

    def test_twelve_hours_later_is_the_coolest(self, config):
        night = (config.sim.outdoor_peak_hour + 12.0) % 24.0
        assert self._ambient_at(config, night) == pytest.approx(
            config.sim.outdoor_mean_c - config.sim.outdoor_amplitude_c
        )

    def test_when_the_run_starts_does_not_move_the_day(self, config):
        """A run launched at 20:00 meets the same 20:00 as one launched at 11:00."""
        from datetime import datetime

        from src.common.localtime import epoch_of

        offset = config.site.utc_offset_h
        evening = epoch_of(datetime(2026, 10, 1, 20, 0), offset)
        late = SimClock(start_epoch_s=evening)
        early = SimClock(start_epoch_s=epoch_of(datetime(2026, 10, 1, 11, 0), offset))
        board = Blackboard(config.mqtt, RecordingTransport())
        launched_early = build_simulator(config, early, board)
        early.advance(evening - early.now())
        assert launched_early.outdoor_temperature_c() == pytest.approx(
            build_simulator(config, late, board).outdoor_temperature_c()
        )


class TestPowerMeter:
    """R-02: the unit's draw follows what the room receives, not the command."""

    def _power(self, transport) -> list[float]:
        return [
            SensorReading.model_validate_json(payload).value
            for payload in transport.payloads_on(f"space/sensor/{POWER_ID}/state")
        ]

    def _cool(self, simulator, clock, blackboard, steps: int) -> None:
        blackboard.dispatch(
            topics.ACTUATOR_COMMAND.format(actuator_id="ac"),
            Command(ts=clock.now(), actuator_id="ac", kind=CommandKind.COOL, setpoint_c=24.0)
            .model_dump_json().encode(),
        )
        for _ in range(steps):
            simulator.step()
            clock.advance(5.0)

    def test_the_meter_is_published_as_watts(self, quiet_config):
        simulator, transport, _, _ = _running(quiet_config)
        simulator.step()
        reading = SensorReading.model_validate_json(
            transport.payloads_on(f"space/sensor/{POWER_ID}/state")[-1]
        )
        assert reading.unit is Unit.WATT

    def test_a_running_unit_draws_its_rated_power(self, quiet_config):
        simulator, transport, clock, board = _running(quiet_config)
        self._cool(simulator, clock, board, 20)
        rated = quiet_config.sim.power_meter.rated_power_w
        assert self._power(transport)[-1] == pytest.approx(rated, abs=50.0)

    def test_a_dead_unit_draws_standby_while_told_to_cool(self, quiet_config):
        simulator, transport, clock, board = _running(quiet_config)
        simulator._actuator.inject_failure(True)
        self._cool(simulator, clock, board, 20)
        assert self._power(transport)[-1] < 50.0

    def test_an_idle_meter_never_reads_negative(self, quiet_config):
        simulator, transport, clock, _ = _running(quiet_config)
        for _ in range(200):
            simulator.step()
            clock.advance(5.0)
        assert min(self._power(transport)) >= 0.0

    def test_adding_the_meter_moves_no_other_instrument(self, quiet_config):
        """Its own random stream: every seeded result already measured stands."""
        with_meter, transport_a, clock_a, _ = _running(quiet_config)
        adapters = tuple(s for s in quiet_config.sensors.adapters if s.sensor_id != POWER_ID)
        without = quiet_config.model_copy(
            update={"sensors": quiet_config.sensors.model_copy(update={"adapters": adapters})}
        )
        without_meter, transport_b, clock_b, _ = _running(without)
        for _ in range(50):
            with_meter.step(); clock_a.advance(5.0)
            without_meter.step(); clock_b.advance(5.0)
        topic = f"space/sensor/{INDOOR_TEMPERATURE_ID}/state"
        assert transport_a.payloads_on(topic) == transport_b.payloads_on(topic)


class TestAcknowledgingFromTheMeter:
    """U-21: with no readback, the simulator acknowledges on the draw, as the bridge does."""

    def _ack(self, transport):
        payload = transport.payloads_on("space/actuator/ac/state")[-1]
        return ActuatorState.model_validate_json(payload).ack

    def _cool_for(self, config, simulator, clock, board, seconds):
        TestPowerMeter()._cool(simulator, clock, board, int(seconds / 5.0) + 1)

    def test_a_working_unit_is_acknowledged_after_the_grace_period(self, quiet_config):
        simulator, transport, clock, board = _running(quiet_config)
        self._cool_for(quiet_config, simulator, clock, board, quiet_config.devices.ac_ack_grace_s + 30.0)
        assert self._ack(transport) is AckStatus.ACKNOWLEDGED

    def test_a_dead_unit_is_reported_failed(self, quiet_config):
        simulator, transport, clock, board = _running(quiet_config)
        simulator._actuator.inject_failure(True)
        self._cool_for(quiet_config, simulator, clock, board, quiet_config.devices.ac_ack_grace_s + 30.0)
        assert self._ack(transport) is AckStatus.FAILED

    def test_within_the_grace_period_it_stays_unknown(self, quiet_config):
        """A compressor takes time to start; judging sooner would cry wolf."""
        simulator, transport, clock, board = _running(quiet_config)
        TestPowerMeter()._cool(simulator, clock, board, 2)
        assert self._ack(transport) is AckStatus.UNKNOWN


class TestInjectionsLeftFromEarlier:
    """An injection older than the simulator is obeyed, and said out loud."""

    def _inject(self, board, ts, kind="NO_RESPONSE"):
        from src.common.schemas import InjectionCommand
        command = InjectionCommand(ts=ts, subject="ac", kind=kind, magnitude=0.0, requester="operator")
        board.dispatch("space/inject/ac", command.model_dump_json().encode())

    def test_an_old_injection_is_a_warning(self, quiet_config, caplog):
        _, _, clock, board = _running(quiet_config)
        with caplog.at_level("WARNING"):
            self._inject(board, clock.now() - 86400.0)
        assert "left on the broker by an earlier session" in caplog.text

    def test_it_is_still_obeyed(self, quiet_config):
        simulator, _, clock, board = _running(quiet_config)
        self._inject(board, clock.now() - 86400.0)
        assert simulator._actuator._failed is True

    def test_a_fresh_injection_is_not_a_warning(self, quiet_config, caplog):
        _, _, clock, board = _running(quiet_config)
        with caplog.at_level("WARNING"):
            self._inject(board, clock.now())
        assert "earlier session" not in caplog.text
