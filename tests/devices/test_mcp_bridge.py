"""The MCP bridge (FR-90): any server, mounted as a device, on the room's terms.

What matters: only the tools the configuration names are exposed, with the
effect and class the configuration gives them; their arguments are checked
by the room before the server sees them; and a server that will not start
is reported, not waited on forever. The end-to-end tests launch the real
example server over stdio, as the bridge does in service.
"""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.clock import SimClock
from src.common.config import McpServerConfig, McpToolConfig, load_config
from src.common.devices import DeviceClass, DeviceKind
from src.common.tools import ParameterType, ToolEffect, ToolInvocation, ToolRequester, ToolStatus
from src.devices.mcp_bridge import BridgeError, McpConnection, build, mount, outcome_from_result, parameters_from_schema
from src.devices.service import DeviceService

CONFIG = load_config(Path("config/default.yaml"))


class TestSchemas:
    def test_flat_parameters_with_bounds_and_requirement(self):
        schema = {"properties": {"level": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Brightness"}}, "required": ["level"]}
        (level,) = parameters_from_schema("set", schema)
        assert (level.type, level.minimum, level.maximum, level.required, level.description) == (ParameterType.INTEGER, 0, 100, True, "Brightness")

    def test_an_optional_value_is_not_required(self):
        schema = {"properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}], "title": "Note"}}, "required": ["note"]}
        (note,) = parameters_from_schema("t", schema)
        assert note.type is ParameterType.STRING and note.required is False

    def test_a_string_enum_becomes_choices(self):
        (mode,) = parameters_from_schema("t", {"properties": {"mode": {"type": "string", "enum": ["warm", "cool"]}}})
        assert mode.choices == ("warm", "cool")

    def test_a_nested_object_is_refused(self):
        with pytest.raises(BridgeError, match="only flat values"):
            parameters_from_schema("t", {"properties": {"colour": {"type": "object"}}})

    def test_several_types_are_refused(self):
        with pytest.raises(BridgeError, match="several types"):
            parameters_from_schema("t", {"properties": {"x": {"anyOf": [{"type": "string"}, {"type": "integer"}]}}})

    def test_no_parameters(self):
        assert parameters_from_schema("t", {"type": "object", "properties": {}}) == ()


def _result(text="", structured=None, error=False):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)] if text else [], structured_content=structured, is_error=error)


class TestResults:
    def test_a_json_object_gives_its_message_and_flat_values(self):
        outcome = outcome_from_result(_result('{"message": "On.", "on": true, "colour": {"r": 1}}'))
        assert outcome.message == "On." and dict(outcome.detail) == {"on": True}

    def test_structured_content_is_preferred(self):
        outcome = outcome_from_result(_result("ignored", structured={"message": "Off.", "on": False}))
        assert outcome.message == "Off." and outcome.detail["on"] is False

    def test_plain_text_is_the_message(self):
        assert outcome_from_result(_result("Lamp toggled.")).message == "Lamp toggled."

    def test_nothing_said_is_done(self):
        assert outcome_from_result(_result()).message == "Done."

    def test_an_error_is_a_sentence_for_the_occupant(self):
        with pytest.raises(ValueError, match="reported a problem: bulb blown"):
            outcome_from_result(_result("bulb blown\nTraceback ...", error=True))


def _server(**tools):
    return McpServerConfig(device_id="lamp", name="Desk lamp", kind=DeviceKind.LIGHT, simulated=True, command="python", args=["examples/mcp_lamp/server.py"], tools=tools or {"turn_on": McpToolConfig(effect=ToolEffect.WRITE, device_class=DeviceClass.COMFORT)})


class TestMounting:
    def _tools(self, *names):
        return [SimpleNamespace(name=name, description=f"{name} it", input_schema={"properties": {}}) for name in names]

    def test_only_configured_tools_are_capabilities(self):
        manifest = mount(_server(), self._tools("turn_on", "factory_reset"), 1.0)
        assert [c.name for c in manifest.capabilities] == ["turn_on"]

    def test_the_configured_effect_and_class_are_used(self):
        server = _server(turn_on=McpToolConfig(effect=ToolEffect.COMMIT, device_class=DeviceClass.MEDIA, purpose="Light up."))
        capability = mount(server, self._tools("turn_on"), 1.0).capability("turn_on")
        assert (capability.effect, capability.device_class, capability.purpose) == (ToolEffect.COMMIT, DeviceClass.MEDIA, "Light up.")

    def test_the_servers_description_is_used_when_none_is_configured(self):
        assert mount(_server(), self._tools("turn_on"), 1.0).capability("turn_on").purpose == "turn_on it"

    def test_a_configured_tool_the_server_lacks_is_found_at_startup(self):
        with pytest.raises(BridgeError, match="offers no"):
            mount(_server(), self._tools("turn_off"), 1.0)

    def test_the_device_says_it_is_behind_the_bridge(self):
        assert mount(_server(), self._tools("turn_on"), 1.0).provider == "mcp:lamp"


@pytest.fixture(name="lamp", scope="module")
def _lamp():
    """The real example server, started over stdio by the bridge."""
    (parts,) = build(CONFIG, SimClock())
    yield parts
    parts.close()


def _ask(service, clock, tool, **arguments):
    return service.handle(ToolInvocation(ts=clock.now(), invocation_id=f"{tool}{clock.now()}", tool=tool, arguments=arguments, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 60))


class TestEndToEnd:
    def _service(self, lamp):
        from eval.loopback import LoopbackTransport
        from src.common.mqtt_client import Blackboard

        clock = SimClock()
        return DeviceService(lamp.manifest, lamp.handlers, lamp.state, clock, Blackboard(CONFIG.mqtt, LoopbackTransport())), clock

    def test_the_unlisted_tool_is_never_exposed(self, lamp):
        assert lamp.manifest.capability("factory_reset") is None

    def test_a_command_reaches_the_server_and_comes_back(self, lamp):
        service, clock = self._service(lamp)
        result = _ask(service, clock, "set_brightness", level=40)
        assert result.status is ToolStatus.OK and "40%" in result.message and lamp.state()["brightness"] == 40

    def test_out_of_bounds_is_refused_before_the_server_sees_it(self, lamp):
        service, clock = self._service(lamp)
        result = _ask(service, clock, "set_brightness", level=400)
        assert result.status is ToolStatus.BAD_ARGUMENTS and "from 0 to 100" in result.message

    def test_a_bridged_device_reports_simulated_as_configured(self, lamp):
        service, clock = self._service(lamp)
        assert _ask(service, clock, "turn_off").simulated is True


class TestAServerThatWillNotStart:
    def test_it_is_left_out_quickly_with_a_reason(self, caplog):
        broken = _server().model_copy(update={"command": "python", "args": ["-c", "import sys; sys.exit(3)"]})
        config = CONFIG.model_copy(update={"actuators": CONFIG.actuators.model_copy(update={"mcp_servers": [broken]})})
        started = time.monotonic()
        assert build(config, SimClock()) == []
        assert time.monotonic() - started < 20 and "did not start" in caplog.text

    def test_a_call_before_connecting_is_said_plainly(self):
        with pytest.raises(ValueError, match="not connected"):
            McpConnection(_server(), 5.0).call("turn_on", {})
