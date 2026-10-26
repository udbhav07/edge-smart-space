"""The device-service base (FR-87, FR-88), over an in-process blackboard.

What matters: a device announces itself, answers only its own commands,
refuses whatever its manifest does not allow even when the command skipped
the registry, never runs a plant capability, and never lets a failing
handler take its service down.
"""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceAvailability, DeviceCapability, DeviceClass, DeviceKind, DeviceManifest, DeviceState
from src.common.mqtt_client import Blackboard
from src.common.tools import ParameterType, ProviderOutcome, ToolEffect, ToolInvocation, ToolParameter, ToolRequester, ToolResult, ToolStatus
from src.devices.service import DeviceService, last_will

CONFIG = load_config(Path("config/default.yaml"))
SPEED = ToolParameter(name="speed", type=ParameterType.INTEGER, description="0 to 3", minimum=0, maximum=3)


def _manifest(min_interval_s=0.0, extra=()):
    return DeviceManifest(
        ts=1.0,
        device_id="fan",
        name="Ceiling fan",
        kind=DeviceKind.FAN,
        simulated=True,
        capabilities=(
            DeviceCapability(name="set_speed", purpose="Set the speed.", effect=ToolEffect.WRITE, device_class=DeviceClass.COMFORT, parameters=(SPEED,), min_interval_s=min_interval_s),
            DeviceCapability(name="explode", purpose="Fails.", effect=ToolEffect.WRITE, device_class=DeviceClass.COMFORT),
            *extra,
        ),
    )


class _Fan:
    def __init__(self):
        self.speed = 0

    def set_speed(self, arguments):
        self.speed = arguments["speed"]
        return ProviderOutcome(message=f"The fan is at speed {self.speed}.", detail={"speed": self.speed})

    @staticmethod
    def explode(_arguments):
        raise RuntimeError("bearing seized")


@pytest.fixture(name="rig")
def _rig():
    clock = SimClock(start_epoch_s=1000.0)
    transport = LoopbackTransport()
    board = Blackboard(CONFIG.mqtt, transport)
    transport.attach(board)
    fan = _Fan()
    service = DeviceService(_manifest(), {"set_speed": fan.set_speed, "explode": fan.explode}, lambda: {"speed": fan.speed}, clock, board)
    service.start()
    return service, fan, transport, clock


def _command(clock, tool="set_speed", device_id="fan", expires_in=60.0, **arguments):
    return ToolInvocation(ts=clock.now(), invocation_id=f"i{clock.now()}", tool=tool, arguments=arguments, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + expires_in)


def _send(transport, clock, device_id="fan", **kwargs):
    transport.publish(topics.DEVICE_COMMAND.format(device_id=device_id), _command(clock, **kwargs).model_dump_json().encode(), 1, False)


def _last(transport, spec, model, device_id="fan"):
    topic = spec.format(device_id=device_id)
    payloads = [payload for name, payload, _, _ in transport.published if name == topic]
    return model.model_validate_json(payloads[-1]) if payloads else None


class TestAnnouncing:
    def test_the_manifest_is_published_retained(self, rig):
        _, _, transport, _ = rig
        topic = topics.DEVICE_MANIFEST.format(device_id="fan")
        assert [retain for name, _, _, retain in transport.published if name == topic] == [True]

    def test_it_says_it_is_online(self, rig):
        _, _, transport, _ = rig
        assert _last(transport, topics.DEVICE_AVAILABILITY, DeviceAvailability).online is True

    def test_its_state_is_on_the_board_from_the_start(self, rig):
        _, _, transport, _ = rig
        assert _last(transport, topics.DEVICE_STATE, DeviceState).values == {"speed": 0}

    def test_stopping_says_offline(self, rig):
        service, _, transport, _ = rig
        service.stop()
        assert _last(transport, topics.DEVICE_AVAILABILITY, DeviceAvailability).online is False

    def test_the_last_will_is_offline_and_retained(self):
        topic, payload, retain = last_will("fan", SimClock(start_epoch_s=5.0))
        assert topic == "space/device/fan/availability" and retain is True
        assert DeviceAvailability.model_validate_json(payload).online is False


class TestCommands:
    def test_a_command_runs_and_answers(self, rig):
        _, fan, transport, clock = rig
        _send(transport, clock, speed=2)
        result = _last(transport, topics.DEVICE_RESULT, ToolResult)
        assert result.status is ToolStatus.OK and fan.speed == 2 and result.provider == "fan" and result.simulated

    def test_the_new_state_follows_a_command(self, rig):
        _, _, transport, clock = rig
        _send(transport, clock, speed=3)
        assert _last(transport, topics.DEVICE_STATE, DeviceState).values == {"speed": 3}

    def test_another_devices_command_is_ignored(self, rig):
        _, fan, transport, clock = rig
        _send(transport, clock, device_id="lamp", speed=2)
        assert fan.speed == 0 and _last(transport, topics.DEVICE_RESULT, ToolResult, device_id="fan") is None

    def test_out_of_bounds_is_refused_even_without_the_registry(self, rig):
        _, fan, transport, clock = rig
        _send(transport, clock, speed=9)
        result = _last(transport, topics.DEVICE_RESULT, ToolResult)
        assert result.status is ToolStatus.BAD_ARGUMENTS and "from 0 to 3" in result.message and fan.speed == 0

    def test_an_unknown_capability_is_refused(self, rig):
        _, _, transport, clock = rig
        _send(transport, clock, tool="fly")
        assert _last(transport, topics.DEVICE_RESULT, ToolResult).status is ToolStatus.UNKNOWN_TOOL

    def test_an_expired_command_is_not_run(self, rig):
        _, fan, transport, clock = rig
        _send(transport, clock, expires_in=-1.0, speed=2)
        assert _last(transport, topics.DEVICE_RESULT, ToolResult).status is ToolStatus.EXPIRED and fan.speed == 0

    def test_a_failing_handler_is_a_result_not_a_crash(self, rig):
        _, _, transport, clock = rig
        _send(transport, clock, tool="explode")
        result = _last(transport, topics.DEVICE_RESULT, ToolResult)
        assert result.status is ToolStatus.FAILED and "bearing" not in result.message

    def test_a_handlers_own_sentence_is_passed_on(self, rig):
        service, _, _, clock = rig
        service._handlers["explode"] = lambda _a: (_ for _ in ()).throw(ValueError("The fan is already off."))
        assert service.handle(_command(clock, tool="explode")).message == "The fan is already off."

    def test_no_state_is_published_after_a_refusal(self, rig):
        _, _, transport, clock = rig
        before = transport.count_on("space/device/fan/state")
        _send(transport, clock, speed=9)
        assert transport.count_on("space/device/fan/state") == before


class TestRateLimit:
    def _service(self, clock, board):
        fan = _Fan()
        return DeviceService(_manifest(min_interval_s=5.0), {"set_speed": fan.set_speed, "explode": fan.explode}, lambda: {}, clock, board), fan

    def test_asked_again_too_soon_is_refused(self):
        clock, board = SimClock(start_epoch_s=1000.0), Blackboard(CONFIG.mqtt, LoopbackTransport())
        service, _ = self._service(clock, board)
        service.handle(_command(clock, speed=1))
        clock.advance(2.0)
        result = service.handle(_command(clock, speed=2))
        assert result.status is ToolStatus.UNAVAILABLE and "try in 3 s" in result.message

    def test_after_the_interval_it_runs(self):
        clock, board = SimClock(start_epoch_s=1000.0), Blackboard(CONFIG.mqtt, LoopbackTransport())
        service, fan = self._service(clock, board)
        service.handle(_command(clock, speed=1))
        clock.advance(5.0)
        assert service.handle(_command(clock, speed=2)).status is ToolStatus.OK and fan.speed == 2

    def test_a_refusal_does_not_restart_the_interval(self):
        clock, board = SimClock(start_epoch_s=1000.0), Blackboard(CONFIG.mqtt, LoopbackTransport())
        service, _ = self._service(clock, board)
        service.handle(_command(clock, speed=1))
        clock.advance(4.0)
        service.handle(_command(clock, speed=2))
        clock.advance(1.0)
        assert service.handle(_command(clock, speed=2)).status is ToolStatus.OK


class TestPlant:
    def test_a_plant_capability_needs_no_handler_and_is_never_run(self):
        clock, board = SimClock(start_epoch_s=1000.0), Blackboard(CONFIG.mqtt, LoopbackTransport())
        plant = DeviceCapability(name="set_temperature", purpose="x", effect=ToolEffect.WRITE, device_class=DeviceClass.PLANT)
        fan = _Fan()
        service = DeviceService(_manifest(extra=(plant,)), {"set_speed": fan.set_speed, "explode": fan.explode}, lambda: {}, clock, board)
        result = service.handle(_command(clock, tool="set_temperature"))
        assert result.status is ToolStatus.UNAVAILABLE and "temperature setting" in result.message


class TestConstruction:
    def test_a_capability_without_a_handler_is_found_at_startup(self):
        with pytest.raises(ValueError, match="without a handler"):
            DeviceService(_manifest(), {"set_speed": _Fan().set_speed}, lambda: {}, SimClock(), Blackboard(CONFIG.mqtt, LoopbackTransport()))

    def test_a_handler_without_a_capability_is_found_at_startup(self):
        fan = _Fan()
        with pytest.raises(ValueError, match="without a capability"):
            DeviceService(_manifest(), {"set_speed": fan.set_speed, "explode": fan.explode, "fly": fan.explode}, lambda: {}, SimClock(), Blackboard(CONFIG.mqtt, LoopbackTransport()))
