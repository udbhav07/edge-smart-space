"""Unit tests for the real room's Layer 1 bridge (Week 5).

No hardware: device messages are delivered as the broker would deliver them,
as plain text on the nodes' own topics, and what the bridge publishes is read
back off the transport.
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
)
from src.io.devices import DeviceBridge


class Transport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []

    def connect(self, host, port, keepalive): ...
    def loop_start(self): ...
    def loop_stop(self): ...
    def disconnect(self): ...

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))

    def on(self, topic):
        return [payload for name, payload, _, _ in self.published if name == topic]


@pytest.fixture(name="config")
def _config():
    config = load_config(Path("config/default.yaml"))
    devices = config.devices.model_copy(update={"ac_mode_state_topic": "ac-node/climate/air_conditioner/mode/state"})
    return config.model_copy(update={"devices": devices})


@pytest.fixture(name="wired")
def _wired(config):
    clock = SimClock()
    transport = Transport()
    board = Blackboard(config.mqtt, transport)
    bridge = DeviceBridge(config, clock, board)
    bridge.subscribe()
    board.on_connected()
    return bridge, transport, board, clock


def _device(board, config, sensor_id, text):
    board.dispatch(config.devices.sensor_topics[sensor_id], text.encode())


def _readings(transport, sensor_id):
    return [SensorReading.model_validate_json(p) for p in transport.on(f"space/sensor/{sensor_id}/state")]


class TestSensors:
    def test_it_listens_on_the_configured_device_topics(self, wired, config):
        _, transport, _, _ = wired
        subscribed = {topic for topic, _ in transport.subscribed}
        assert config.devices.sensor_topics["temp_01"] in subscribed

    def test_a_device_value_becomes_a_schema_valid_reading(self, wired, config):
        _, transport, board, _ = wired
        _device(board, config, "temp_01", "27.4")
        reading = _readings(transport, "temp_01")[-1]
        assert reading.value == 27.4 and reading.quality is Quality.OK

    def test_a_reading_is_published_only_when_the_node_reports(self, wired, config):
        """Silence must look like silence, because that is what D1 detects."""
        bridge, transport, _, _ = wired
        bridge.tick()
        assert _readings(transport, "temp_01") == []

    def test_an_impossible_value_is_published_flagged_suspect(self, wired, config):
        _, transport, board, _ = wired
        _device(board, config, "temp_01", "999")
        assert _readings(transport, "temp_01")[-1].quality is Quality.SUSPECT

    def test_an_unreadable_value_is_dropped(self, wired, config):
        _, transport, board, _ = wired
        _device(board, config, "temp_01", "nan-ish")
        assert _readings(transport, "temp_01") == []

    def test_no_health_is_published_here(self, wired, config):
        """The detector bank owns health; two publishers would fight."""
        _, transport, board, _ = wired
        _device(board, config, "temp_01", "27.4")
        assert not [t for t, *_ in transport.published if t.endswith("/health")]


class TestOccupancy:
    def test_motion_makes_the_room_occupied(self, wired, config):
        bridge, transport, board, _ = wired
        _device(board, config, "pir_01", "ON")
        bridge.tick()
        assert _readings(transport, "pir_01")[-1].value == 1.0

    def test_occupancy_is_published_every_period(self, wired):
        bridge, transport, _, _ = wired
        bridge.tick()
        bridge.tick()
        assert len(_readings(transport, "pir_01")) == 2

    def test_the_hold_off_keeps_a_quiet_room_occupied(self, wired, config):
        """FR-02: somebody reading quietly sets off nothing."""
        bridge, transport, board, clock = wired
        _device(board, config, "pir_01", "ON")
        clock.advance(config.sensors.vacancy_hold_off_s / 2)
        bridge.tick()
        assert _readings(transport, "pir_01")[-1].value == 1.0


class TestTheUnit:
    def _command(self, board, clock, kind, setpoint=None):
        command = Command(ts=clock.now(), actuator_id="ac", kind=kind, setpoint_c=setpoint)
        board.dispatch("space/actuator/ac/command", command.model_dump_json().encode())

    def test_cool_goes_to_the_climate_mode_topic(self, wired, config):
        _, transport, board, clock = wired
        self._command(board, clock, CommandKind.COOL, 23.0)
        assert transport.on(config.devices.ac_mode_command_topic)[-1] == b"cool"
        assert transport.on(config.devices.ac_target_command_topic)[-1] == b"23.0"

    def test_off_goes_as_off(self, wired, config):
        _, transport, board, clock = wired
        self._command(board, clock, CommandKind.OFF)
        assert transport.on(config.devices.ac_mode_command_topic)[-1] == b"off"

    def test_maintain_sends_the_unit_nothing(self, wired, config):
        _, transport, board, clock = wired
        self._command(board, clock, CommandKind.MAINTAIN)
        assert transport.on(config.devices.ac_mode_command_topic) == []

    def test_the_state_says_it_is_real_and_unacknowledged(self, wired):
        """FR-15 and R-02: not simulated, and no acknowledgement assumed."""
        bridge, transport, board, clock = wired
        self._command(board, clock, CommandKind.COOL, 23.0)
        bridge.tick()
        state = ActuatorState.model_validate_json(transport.on("space/actuator/ac/state")[-1])
        assert state.simulated is False and state.ack is AckStatus.UNKNOWN

    def test_a_readback_that_matches_acknowledges(self, wired, config):
        bridge, transport, board, clock = wired
        self._command(board, clock, CommandKind.COOL, 23.0)
        board.dispatch(config.devices.ac_mode_state_topic, b"cool")
        bridge.tick()
        state = ActuatorState.model_validate_json(transport.on("space/actuator/ac/state")[-1])
        assert state.ack is AckStatus.ACKNOWLEDGED

    def test_a_readback_that_disagrees_is_a_failure(self, wired, config):
        bridge, transport, board, clock = wired
        self._command(board, clock, CommandKind.COOL, 23.0)
        board.dispatch(config.devices.ac_mode_state_topic, b"off")
        bridge.tick()
        state = ActuatorState.model_validate_json(transport.on("space/actuator/ac/state")[-1])
        assert state.ack is AckStatus.FAILED


class TestInjection:
    def _inject(self, board, clock, subject, kind, magnitude=0.0):
        command = InjectionCommand(ts=clock.now(), subject=subject, kind=kind, magnitude=magnitude, requester="operator")
        board.dispatch(f"space/inject/{subject}", command.model_dump_json().encode())

    def test_a_stuck_injection_freezes_a_real_sensor(self, wired, config):
        """FR-31: the same channel works on hardware as in simulation."""
        _, transport, board, clock = wired
        self._inject(board, clock, "temp_01", InjectedFault.STUCK_AT, 27.0)
        _device(board, config, "temp_01", "24.2")
        assert _readings(transport, "temp_01")[-1].value == 27.0

    def test_a_dead_unit_is_emulated_by_withholding_commands(self, wired, config):
        _, transport, board, clock = wired
        self._inject(board, clock, "ac", InjectedFault.NO_RESPONSE)
        TestTheUnit()._command(board, clock, CommandKind.COOL, 23.0)
        assert transport.on(config.devices.ac_mode_command_topic) == []

    def test_clearing_restores_the_unit(self, wired, config):
        _, transport, board, clock = wired
        self._inject(board, clock, "ac", InjectedFault.NO_RESPONSE)
        self._inject(board, clock, "ac", InjectedFault.NONE)
        TestTheUnit()._command(board, clock, CommandKind.COOL, 23.0)
        assert transport.on(config.devices.ac_mode_command_topic)[-1] == b"cool"
