"""The simulated house's fans, lights and TV (FR-99, FR-15)."""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceClass, DeviceKind
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolEffect, ToolInvocation, ToolRequester, ToolStatus
from src.devices.home import build
from src.devices.service import DeviceService

CONFIG = load_config(Path("config/default.yaml"))


@pytest.fixture(name="house")
def _house():
    clock = SimClock(start_epoch_s=1000.0)
    services = {}
    for parts in build(CONFIG, clock):
        services[parts.manifest.device_id] = (DeviceService(parts.manifest, parts.handlers, parts.state, clock, Blackboard(CONFIG.mqtt, LoopbackTransport())), parts)
    return services, clock


def _ask(house, device_id, tool, **arguments):
    services, clock = house
    clock.advance(5.0)
    service = services[device_id][0]
    return service.handle(ToolInvocation(ts=clock.now(), invocation_id=f"i{clock.now()}", tool=tool, arguments=arguments, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 60))


def _state(house, device_id):
    return house[0][device_id][1].state()


class TestWhatIsInTheHouse:
    def test_one_device_per_entry(self, house):
        assert set(house[0]) == {device.device_id for device in CONFIG.actuators.home.devices}

    def test_every_device_stands_in_a_room_of_the_house(self, house):
        rooms = {room.room_id for room in CONFIG.house.rooms}
        assert all(parts.manifest.room in rooms for _, parts in house[0].values())

    def test_all_are_simulated_and_none_is_plant(self, house):
        manifests = [parts.manifest for _, parts in house[0].values()]
        assert all(manifest.simulated for manifest in manifests)
        assert all(capability.device_class is not DeviceClass.PLANT for manifest in manifests for capability in manifest.capabilities)

    def test_status_is_a_read_everywhere(self, house):
        assert all(parts.manifest.capability("status").effect is ToolEffect.READ for _, parts in house[0].values())

    def test_only_the_living_room_fan_answers_to_the_fan(self, house):
        holders = [device_id for device_id, (_, parts) in house[0].items() if "the fan" in parts.manifest.aliases or "fan" in parts.manifest.aliases]
        assert holders == ["fan"]

    def test_switched_off_there_is_nothing(self):
        off = CONFIG.model_copy(update={"actuators": CONFIG.actuators.model_copy(update={"home": CONFIG.actuators.home.model_copy(update={"enabled": False})})})
        assert build(off, SimClock(start_epoch_s=0.0)) == []


class TestTheFans:
    def test_the_bedroom_fan_is_named_as_itself(self, house):
        result = _ask(house, "bedroom_fan", "turn_on")
        assert result.message.startswith("The bedroom fan is on")

    def test_the_living_room_fan_keeps_its_old_name(self, house):
        assert _ask(house, "fan", "turn_on").message.startswith("The fan is on")


class TestALight:
    def test_on_at_full_brightness(self, house):
        _ask(house, "kitchen_light", "turn_on")
        assert _state(house, "kitchen_light") == {"on": True, "brightness": 100}

    def test_dimmed(self, house):
        result = _ask(house, "kitchen_light", "set_brightness", level=30)
        assert result.status is ToolStatus.OK and _state(house, "kitchen_light") == {"on": True, "brightness": 30}

    def test_dimmed_to_nothing_is_off(self, house):
        _ask(house, "kitchen_light", "set_brightness", level=0)
        assert _state(house, "kitchen_light")["on"] is False

    def test_too_bright_is_refused(self, house):
        assert _ask(house, "kitchen_light", "set_brightness", level=150).status is ToolStatus.BAD_ARGUMENTS

    def test_off_when_off_says_so(self, house):
        assert "already off" in _ask(house, "bathroom_light", "turn_off").message

    def test_it_is_a_light(self, house):
        assert house[0]["bedroom_light"][1].manifest.kind is DeviceKind.LIGHT


class TestTheTv:
    def test_on_at_channel_one(self, house):
        result = _ask(house, "tv", "turn_on")
        assert result.message == "The TV is on, channel 1 (simulated)." and _state(house, "tv")["on"] is True

    def test_on_twice_says_it_already_is(self, house):
        _ask(house, "tv", "turn_on")
        assert "already on" in _ask(house, "tv", "turn_on").message

    def test_a_channel_turns_it_on(self, house):
        _ask(house, "tv", "set_channel", channel=4)
        assert _state(house, "tv")["on"] is True and _state(house, "tv")["channel"] == 4

    def test_a_channel_it_lacks_is_refused(self, house):
        assert _ask(house, "tv", "set_channel", channel=99).status is ToolStatus.BAD_ARGUMENTS

    def test_it_is_a_tv(self, house):
        assert house[0]["tv"][1].manifest.kind is DeviceKind.TV
