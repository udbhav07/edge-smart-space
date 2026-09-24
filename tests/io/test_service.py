"""Unit tests for Layer 1 on hardware.

There is no hardware, so what is tested is everything up to the wire: that a
device payload in any shape ESPHome produces becomes a reading, that a node
which stops publishing becomes silence rather than a repeat, that an injected
fault works the same way it does in simulation, and that an acknowledgement is
never invented.

A fake device publisher stands in for the nodes. It publishes on the topics
configuration names, so the part that changes at bring-up -- the topics
themselves -- is the part these tests deliberately do not hard-code.
"""

from pathlib import Path

import pytest

from src.common import topics
from src.common.clock import SimClock
from src.common.config import Layer1Source, load_config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AckStatus,
    ActuatorState,
    Command,
    CommandKind,
    InjectionCommand,
    Quality,
    SensorHealth,
    SensorReading,
)
from src.io.service import (
    DevicePayloadError,
    build_layer,
    parse_device_payload,
)

INDOOR = "temp_01"
OCCUPANCY = "pir_01"


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []

    def connect(self, host, port, keepalive):
        pass

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def readings(self) -> list[SensorReading]:
        return [
            SensorReading.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic.startswith("space/sensor/") and topic.endswith("/state")
        ]

    def health(self) -> list[SensorHealth]:
        return [
            SensorHealth.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic.endswith("/health")
        ]

    def actuator_states(self) -> list[ActuatorState]:
        return [
            ActuatorState.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/actuator/ac/state"
        ]

    def device_commands(self, topic: str) -> list[bytes]:
        return [
            payload for name, payload, _, _ in self.published if name == topic
        ]


@pytest.fixture(name="config")
def _config():
    base = load_config(Path("config/default.yaml"))
    return base.model_copy(
        update={"io": base.io.model_copy(update={"source": Layer1Source.ESPHOME})}
    )


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="wired")
def _wired(config, clock):
    transport = FakeTransport()
    blackboard = Blackboard(config.mqtt, transport)
    layer = build_layer(config, clock, blackboard, device_bus=transport)
    layer.subscribe()
    blackboard.on_connected()
    return layer, transport, blackboard


def _device_says(blackboard, config, sensor_id: str, payload: bytes) -> None:
    blackboard.dispatch(config.io.topic_for(sensor_id), payload)


class TestDevicePayloads:
    """Every shape an ESPHome node actually produces."""

    def test_a_bare_number(self):
        assert parse_device_payload(b"24.31") == pytest.approx(24.31)

    def test_a_number_with_whitespace(self):
        assert parse_device_payload(b"  24.31\n") == pytest.approx(24.31)

    def test_an_integer(self):
        assert parse_device_payload(b"24") == 24.0

    def test_an_object_carrying_a_value(self):
        assert parse_device_payload(b'{"value": 24.31}') == pytest.approx(24.31)

    @pytest.mark.parametrize("word", [b"ON", b"on", b"TRUE", b"1"])
    def test_a_binary_sensor_reporting_presence(self, word):
        assert parse_device_payload(word) == 1.0

    @pytest.mark.parametrize("word", [b"OFF", b"off", b"FALSE", b"0"])
    def test_a_binary_sensor_reporting_absence(self, word):
        assert parse_device_payload(word) == 0.0

    def test_an_empty_payload_is_refused(self):
        with pytest.raises(DevicePayloadError):
            parse_device_payload(b"")

    def test_nonsense_is_refused_rather_than_guessed(self):
        """A device nobody understands must be noticed. Inventing a zero
        would publish a reading the instrument never took."""
        with pytest.raises(DevicePayloadError):
            parse_device_payload(b"unavailable")

    def test_an_object_without_a_value_is_refused(self):
        with pytest.raises(DevicePayloadError):
            parse_device_payload(b'{"state": "ok"}')

    def test_undecodable_bytes_are_refused(self):
        with pytest.raises(DevicePayloadError):
            parse_device_payload(b"\xff\xfe")


class TestBinding:
    def test_every_configured_device_is_bound(self, wired, config):
        layer, _, _ = wired
        assert layer.bound_sensors == {
            binding.sensor_id for binding in config.io.devices
        }

    def test_it_subscribes_to_each_device_topic(self, wired, config):
        _, transport, _ = wired
        subscribed = {topic for topic, _ in transport.subscribed}
        for binding in config.io.devices:
            assert binding.topic in subscribed

    def test_a_sensor_with_no_device_is_left_out(self, config, clock):
        """An adapter with no device publishes silence, which D1 reports as a
        dropped sensor when it is really a configuration gap."""
        io = config.io.model_copy(update={"devices": config.io.devices[:1]})
        transport = FakeTransport()
        blackboard = Blackboard(config.mqtt, transport)
        layer = build_layer(
            config.model_copy(update={"io": io}),
            clock,
            blackboard,
            device_bus=transport,
        )
        assert layer.bound_sensors == {config.io.devices[0].sensor_id}


class TestReadings:
    def test_a_device_value_becomes_a_reading(self, wired, config, clock):
        layer, transport, blackboard = wired
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        indoor = [r for r in transport.readings() if r.sensor_id == INDOOR]
        assert indoor and indoor[-1].value == pytest.approx(24.5)

    def test_the_reading_carries_the_configured_unit(self, wired, config, clock):
        layer, transport, blackboard = wired
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        indoor = [r for r in transport.readings() if r.sensor_id == INDOOR][-1]
        assert indoor.unit is config.sensors.by_id(INDOOR).unit

    def test_a_silent_device_produces_no_reading(self, wired, config, clock):
        """Absence is what D1 detects. A repeat would turn a dropped node into
        a stuck one and send D1 after a fault D2 would find in the wrong
        place."""
        layer, transport, blackboard = wired
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        before = len(transport.readings())

        clock.advance(config.io.stale_after_s + 1.0)
        layer.poll()
        assert len(transport.readings()) == before

    def test_health_is_published_even_when_the_reading_was_lost(
        self, wired, config, clock
    ):
        """The case a subscriber most needs to see."""
        layer, transport, _ = wired
        layer.poll()
        assert transport.health()

    def test_an_unparseable_payload_does_not_stop_the_bridge(
        self, wired, config, clock
    ):
        layer, transport, blackboard = wired
        _device_says(blackboard, config, INDOOR, b"unavailable")
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        assert [r for r in transport.readings() if r.sensor_id == INDOOR]

    def test_a_binary_device_reads_as_occupancy(self, wired, config, clock):
        layer, transport, blackboard = wired
        _device_says(blackboard, config, OCCUPANCY, b"ON")
        layer.poll()
        pir = [r for r in transport.readings() if r.sensor_id == OCCUPANCY]
        assert pir and pir[-1].value == 1.0


class TestTheActuator:
    def test_a_command_reaches_the_device_topic(self, wired, config, clock):
        layer, transport, blackboard = wired
        blackboard.dispatch(
            "space/actuator/ac/command",
            Command(
                ts=clock.now(),
                actuator_id="ac",
                kind=CommandKind.COOL,
                setpoint_c=24.0,
            ).model_dump_json().encode(),
        )
        sent = transport.device_commands(config.io.actuator.command_topic)
        assert sent and sent[-1] == b"COOL"

    def test_a_device_command_is_not_retained(self, wired, config, clock):
        """A retained command would be redelivered to the unit on every
        reconnect."""
        layer, transport, blackboard = wired
        blackboard.dispatch(
            "space/actuator/ac/command",
            Command(
                ts=clock.now(), actuator_id="ac", kind=CommandKind.OFF
            ).model_dump_json().encode(),
        )
        entry = [
            e for e in transport.published
            if e[0] == config.io.actuator.command_topic
        ][-1]
        assert entry[3] is False

    def test_the_acknowledgement_is_unknown(self, wired, clock):
        """R-02: there is nothing to acknowledge with. UNKNOWN is the correct
        answer, not a missing one."""
        layer, transport, _ = wired
        layer.poll()
        assert transport.actuator_states()[-1].ack is AckStatus.UNKNOWN

    def test_the_state_is_not_labelled_simulated(self, wired):
        """FR-15 is about simulated actuators saying so. This one is real."""
        layer, transport, _ = wired
        layer.poll()
        assert transport.actuator_states()[-1].simulated is False

    def test_the_state_reports_what_was_last_asked(self, wired, clock):
        layer, transport, blackboard = wired
        blackboard.dispatch(
            "space/actuator/ac/command",
            Command(
                ts=clock.now(),
                actuator_id="ac",
                kind=CommandKind.COOL,
                setpoint_c=24.0,
            ).model_dump_json().encode(),
        )
        layer.poll()
        assert transport.actuator_states()[-1].kind is CommandKind.COOL


class TestInjectionOnHardware:
    """FR-31 asks for the same mechanism here as in simulation."""

    def _inject(self, blackboard, clock, kind, magnitude=0.0, subject=INDOOR):
        blackboard.dispatch(
            f"space/inject/{subject}",
            InjectionCommand(
                ts=clock.now(),
                subject=subject,
                kind=kind,
                magnitude=magnitude,
                requester="operator",
            ).model_dump_json().encode(),
        )

    def test_a_stuck_injection_freezes_the_published_value(
        self, wired, config, clock
    ):
        layer, transport, blackboard = wired
        self._inject(blackboard, clock, InjectedFault.STUCK_AT, 27.0)
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        indoor = [r for r in transport.readings() if r.sensor_id == INDOOR][-1]
        assert indoor.value == pytest.approx(27.0)

    def test_a_dropout_injection_publishes_nothing(self, wired, config, clock):
        layer, transport, blackboard = wired
        self._inject(blackboard, clock, InjectedFault.DROPOUT)
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        assert [r for r in transport.readings() if r.sensor_id == INDOOR] == []

    def test_clearing_returns_the_device_value(self, wired, config, clock):
        layer, transport, blackboard = wired
        self._inject(blackboard, clock, InjectedFault.STUCK_AT, 27.0)
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()

        self._inject(blackboard, clock, InjectedFault.NONE)
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        indoor = [r for r in transport.readings() if r.sensor_id == INDOOR][-1]
        assert indoor.value == pytest.approx(24.5)

    def test_an_injection_for_an_unbound_sensor_is_ignored(
        self, wired, config, clock
    ):
        layer, transport, blackboard = wired
        self._inject(blackboard, clock, InjectedFault.DROPOUT, subject="ghost")
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        assert [r for r in transport.readings() if r.sensor_id == INDOOR]


class TestTheSameTopicsAsTheSimulator:
    """Section 9.1: nothing above Layer 1 may be able to tell the difference."""

    def test_readings_go_to_the_sensor_state_topic(self, wired, config, clock):
        layer, transport, blackboard = wired
        _device_says(blackboard, config, INDOOR, b"24.5")
        layer.poll()
        assert any(
            topic == topics.SENSOR_STATE.format(sensor_id=INDOOR)
            for topic, _, _, _ in transport.published
        )

    def test_health_goes_to_the_sensor_health_topic(self, wired, clock):
        layer, transport, _ = wired
        layer.poll()
        assert any(
            topic == topics.SENSOR_HEALTH.format(sensor_id=INDOOR)
            for topic, _, _, _ in transport.published
        )

    def test_the_quality_flag_is_the_adapters_not_a_detectors(
        self, wired, config, clock
    ):
        """An adapter flags a suspect reading and publishes it anyway; D3 is
        what turns that into a fault."""
        layer, transport, blackboard = wired
        _device_says(blackboard, config, INDOOR, b"999.0")
        layer.poll()
        indoor = [r for r in transport.readings() if r.sensor_id == INDOOR][-1]
        assert indoor.value == pytest.approx(999.0)
        assert indoor.quality is Quality.SUSPECT
