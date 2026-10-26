"""The simulated fan (FR-87, FR-88, FR-15), run through the device-service base."""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceClass
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolEffect, ToolInvocation, ToolRequester, ToolStatus
from src.devices.fan import build
from src.devices.service import DeviceService

CONFIG = load_config(Path("config/default.yaml"))


@pytest.fixture(name="fan")
def _fan():
    clock = SimClock(start_epoch_s=1000.0)
    parts = build(CONFIG, clock)
    service = DeviceService(parts.manifest, parts.handlers, parts.state, clock, Blackboard(CONFIG.mqtt, LoopbackTransport()))
    return service, parts, clock


def _ask(fan, tool, **arguments):
    service, _, clock = fan
    clock.advance(CONFIG.actuators.fan.min_interval_s)
    return service.handle(ToolInvocation(ts=clock.now(), invocation_id=f"i{clock.now()}", tool=tool, arguments=arguments, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 60))


class TestManifest:
    def test_it_is_simulated(self, fan):
        assert fan[1].manifest.simulated is True

    def test_nothing_it_does_is_plant(self, fan):
        assert all(capability.device_class is DeviceClass.COMFORT for capability in fan[1].manifest.capabilities)

    def test_status_is_a_read(self, fan):
        assert fan[1].manifest.capability("status").effect is ToolEffect.READ

    def test_the_speed_is_bounded_by_config(self, fan):
        speed = fan[1].manifest.capability("set_speed").parameters[0]
        assert (speed.minimum, speed.maximum) == (0, CONFIG.actuators.fan.max_speed)


class TestBehaviour:
    def test_on_starts_at_the_lowest_speed(self, fan):
        _ask(fan, "turn_on")
        assert fan[1].state() == {"on": True, "speed": 1, "max_speed": 3}

    def test_on_twice_says_so(self, fan):
        _ask(fan, "turn_on")
        assert "already on" in _ask(fan, "turn_on").message

    def test_speed_turns_it_on(self, fan):
        _ask(fan, "set_speed", speed=3)
        assert fan[1].state()["on"] is True and fan[1].state()["speed"] == 3

    def test_speed_zero_is_off(self, fan):
        _ask(fan, "set_speed", speed=2)
        _ask(fan, "set_speed", speed=0)
        assert fan[1].state()["on"] is False

    def test_too_fast_is_refused(self, fan):
        assert _ask(fan, "set_speed", speed=4).status is ToolStatus.BAD_ARGUMENTS

    def test_off_when_off_says_so(self, fan):
        assert "already off" in _ask(fan, "turn_off").message

    def test_off_remembers_nothing_running(self, fan):
        _ask(fan, "set_speed", speed=2)
        _ask(fan, "turn_off")
        assert fan[1].state()["speed"] == 0

    def test_every_answer_says_simulated(self, fan):
        assert "simulated" in _ask(fan, "set_speed", speed=2).message

    def test_switching_too_fast_is_refused(self, fan):
        service, _, clock = fan
        _ask(fan, "turn_on")
        result = service.handle(ToolInvocation(ts=clock.now(), invocation_id="again", tool="turn_off", arguments={}, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 60))
        assert result.status is ToolStatus.OK  # a different capability has its own interval
        result = service.handle(ToolInvocation(ts=clock.now(), invocation_id="again2", tool="turn_off", arguments={}, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 60))
        assert result.status is ToolStatus.UNAVAILABLE
