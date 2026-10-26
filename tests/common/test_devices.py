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
