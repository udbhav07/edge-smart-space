"""The device contract (FR-87, FR-88): manifests, availability, state, and bounds.

What matters: a device says what it can do in a form the registry can check
arguments against, a capability always says what it can do to the room, and
the bounds a device declares are enforced by the one shared argument check.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.config import McpToolConfig, load_config
from src.common.devices import (
    DeviceAvailability,
    DeviceCapability,
    DeviceClass,
    DeviceKind,
    DeviceManifest,
    DeviceState,
)
from src.common.tools import ParameterType, ToolArgumentError, ToolEffect, ToolParameter, ToolSpec

SPEED = ToolParameter(name="speed", type=ParameterType.INTEGER, description="0 is off", minimum=0, maximum=3)


def _capability(name="set_speed", device_class=DeviceClass.COMFORT, parameters=(SPEED,)):
    return DeviceCapability(name=name, purpose="Set the fan speed.", effect=ToolEffect.WRITE, device_class=device_class, parameters=parameters)


def _manifest(*capabilities, device_id="fan"):
    return DeviceManifest(ts=1.0, device_id=device_id, name="Ceiling fan", kind=DeviceKind.FAN, simulated=True, capabilities=capabilities or (_capability(),))


class TestBounds:
    def test_a_value_inside_the_bounds_passes(self):
        spec = ToolSpec(name="set_speed", purpose="x", effect=ToolEffect.WRITE, parameters=(SPEED,))
        assert spec.validate_arguments({"speed": 2})["speed"] == 2

    @pytest.mark.parametrize("value", [0, 3])
    def test_the_bounds_themselves_pass(self, value):
        spec = ToolSpec(name="set_speed", purpose="x", effect=ToolEffect.WRITE, parameters=(SPEED,))
        assert spec.validate_arguments({"speed": value})["speed"] == value

    @pytest.mark.parametrize("value", [-1, 4, 5.0])
    def test_a_value_outside_is_refused_saying_the_range(self, value):
        spec = ToolSpec(name="set_speed", purpose="x", effect=ToolEffect.WRITE, parameters=(SPEED,))
        with pytest.raises(ToolArgumentError, match="from 0 to 3"):
            spec.validate_arguments({"speed": value})

    def test_a_one_sided_bound_says_which_side(self):
        volume = ToolParameter(name="volume", type=ParameterType.NUMBER, description="x", maximum=100)
        spec = ToolSpec(name="set_volume", purpose="x", effect=ToolEffect.WRITE, parameters=(volume,))
        with pytest.raises(ToolArgumentError, match="at most 100"):
            spec.validate_arguments({"volume": 101})

    def test_bounds_on_a_string_are_refused(self):
        with pytest.raises(ValidationError):
            ToolParameter(name="track", type=ParameterType.STRING, description="x", minimum=0)

    def test_a_minimum_above_the_maximum_is_refused(self):
        with pytest.raises(ValidationError):
            ToolParameter(name="speed", type=ParameterType.INTEGER, description="x", minimum=3, maximum=1)

    def test_the_model_is_shown_the_bounds(self):
        properties = ToolSpec(name="set_speed", purpose="x", effect=ToolEffect.WRITE, parameters=(SPEED,)).as_schema()["function"]["parameters"]["properties"]
        assert properties["speed"]["minimum"] == 0 and properties["speed"]["maximum"] == 3


class TestManifest:
    def test_a_capability_must_say_what_it_does_to_the_room(self):
        with pytest.raises(ValidationError):
            DeviceCapability(name="turn_on", purpose="x", effect=ToolEffect.WRITE)

    def test_a_capability_is_found_by_name(self):
        assert _manifest().capability("set_speed").device_class is DeviceClass.COMFORT

    def test_an_unknown_capability_is_none(self):
        assert _manifest().capability("explode") is None

    def test_the_tool_name_joins_device_and_capability(self):
        assert _manifest().tool_name("set_speed") == "fan_set_speed"

    def test_a_capability_declared_twice_is_refused(self):
        with pytest.raises(ValidationError):
            _manifest(_capability(), _capability())

    @pytest.mark.parametrize("device_id", ["Fan", "fan-1", "1fan", ""])
    def test_a_device_id_must_fit_a_tool_name(self, device_id):
        with pytest.raises(ValidationError):
            _manifest(device_id=device_id)

    def test_simulated_has_no_default(self):
        with pytest.raises(ValidationError):
            DeviceManifest(ts=1.0, device_id="fan", name="Fan", kind=DeviceKind.FAN)

    def test_a_manifest_survives_the_wire(self):
        manifest = _manifest()
        assert DeviceManifest.model_validate_json(manifest.model_dump_json()) == manifest


class TestStateAndAvailability:
    def test_state_values_are_frozen(self):
        state = DeviceState(ts=1.0, device_id="fan", values={"on": True, "speed": 2}, simulated=True)
        with pytest.raises(TypeError):
            state.values["speed"] = 3

    def test_state_survives_the_wire(self):
        state = DeviceState(ts=1.0, device_id="fan", values={"on": True, "speed": 2}, simulated=True)
        assert DeviceState.model_validate_json(state.model_dump_json()).values == {"on": True, "speed": 2}

    def test_availability_names_the_device(self):
        assert DeviceAvailability(ts=1.0, device_id="fan", online=False).online is False


class TestConfig:
    def test_the_reference_devices_are_configured(self):
        actuators = load_config(Path("config/default.yaml")).actuators
        assert actuators.fan.enabled and actuators.music.enabled and actuators.mcp_servers[0].device_id == "lamp"

    def test_a_bridged_tool_cannot_be_plant(self):
        with pytest.raises(ValidationError):
            McpToolConfig(effect=ToolEffect.WRITE, device_class=DeviceClass.PLANT)

    def test_a_bridged_tool_must_declare_its_effect(self):
        with pytest.raises(ValidationError):
            McpToolConfig(device_class=DeviceClass.COMFORT)


class TestLastWill:
    def test_a_will_is_registered_before_connecting(self):
        from src.common.mqtt_client import build_transport

        config = load_config(Path("config/default.yaml")).mqtt
        client = build_transport(config, "test-will", [None], will=("space/device/fan/availability", b'{"online": false}', True))
        assert client._will is True and client._will_topic == b"space/device/fan/availability" and client._will_retain is True

    def test_without_a_will_none_is_registered(self):
        from src.common.mqtt_client import build_transport

        client = build_transport(load_config(Path("config/default.yaml")).mqtt, "test-no-will", [None])
        assert client._will is False


class TestTheAirConditioner:
    """FR-87, FR-88, FR-45: the AC is seen on the device contract, and none of
    it is a tool."""

    def _state(self, ts=10.0):
        from src.common.schemas import AckStatus, ActuatorState, CommandKind

        return ActuatorState(ts=ts, actuator_id="ac", simulated=True, kind=CommandKind.COOL, ack=AckStatus.ACKNOWLEDGED, last_command_ts=5.0)

    def _announcer(self):
        from eval.loopback import LoopbackTransport
        from src.common.clock import SimClock
        from src.common.devices import PlantAnnouncer
        from src.common.mqtt_client import Blackboard

        transport = LoopbackTransport()
        board = Blackboard(load_config(Path("config/default.yaml")).mqtt, transport)
        transport.attach(board)
        return PlantAnnouncer(board, SimClock(start_epoch_s=100.0), simulated=True), board, transport

    def test_every_capability_is_plant(self):
        from src.common.devices import air_conditioner_manifest

        assert {c.device_class for c in air_conditioner_manifest(1.0, False).capabilities} == {DeviceClass.PLANT}

    def test_its_state_mirrors_the_actuator(self):
        from src.common.devices import air_conditioner_state

        assert air_conditioner_state(self._state()).values == {"command": "COOL", "ack": "ACKNOWLEDGED", "last_command_ts": 5.0}

    def test_announced_once_then_only_mirrored(self):
        announcer, _, transport = self._announcer()
        announcer.mirror(self._state())
        announcer.mirror(self._state(11.0))
        assert transport.count_on("space/device/ac/manifest") == 1 and transport.count_on("space/device/ac/state") == 2

    def test_announced_again_after_the_link_comes_back(self):
        announcer, board, transport = self._announcer()
        announcer.mirror(self._state())
        board.on_disconnected()
        announcer.mirror(self._state(11.0))
        board.on_connected()
        announcer.mirror(self._state(12.0))
        assert transport.count_on("space/device/ac/manifest") == 2

    def test_nothing_is_published_while_the_link_is_down(self):
        announcer, board, transport = self._announcer()
        board.on_disconnected()
        announcer.mirror(self._state())
        assert transport.count_on("space/device/ac/state") == 0

    def test_the_simulator_publishes_the_ac_as_a_device(self):
        from eval.loopback import LoopbackTransport
        from sim.run_sim import build_simulator
        from src.common.clock import SimClock
        from src.common.mqtt_client import Blackboard

        config = load_config(Path("config/default.yaml"))
        transport = LoopbackTransport()
        board = Blackboard(config.mqtt, transport)
        simulator = build_simulator(config, SimClock(), board)
        transport.attach(board)
        simulator.step()
        manifest = DeviceManifest.model_validate_json([p for t, p, _, _ in transport.published if t == "space/device/ac/manifest"][0])
        assert manifest.kind is DeviceKind.AC and manifest.simulated is True
