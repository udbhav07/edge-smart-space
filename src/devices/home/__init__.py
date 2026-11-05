"""The simulated house's fans, lights and TV, as one service (FR-99, FR-15).

Each device is its own device on the board -- its own manifest, connection
and last will -- and stands in a room of ``house``, so the console draws it
there and "the bedroom fan" is a thing Jarvis can name. They move no air and
light nothing; every manifest and every sentence says so (FR-15). The fans
are :class:`src.devices.fan.SimulatedFan`, the smallest device, reused.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.common.clock import Clock
from src.common.config import Config, FanConfig, HomeDeviceConfig
from src.common.devices import DeviceCapability, DeviceClass, DeviceKind, DeviceManifest
from src.common.tools import ArgumentValue, ParameterType, ProviderOutcome, ToolEffect, ToolParameter
from src.devices import fan
from src.devices.runner import DeviceParts

#: A light's brightness runs from off to full, in percent.
FULL_BRIGHTNESS = 100


class SimulatedLight:
    """A dimmable light's switch and brightness, kept in memory."""

    def __init__(self, name: str) -> None:
        self._name = name.lower()
        self.on = False
        self.brightness = FULL_BRIGHTNESS

    def turn_on(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if self.on:
            return ProviderOutcome(message=f"The {self._name} is already on, at {self.brightness}%.", detail=self.state())
        self.on = True
        self.brightness = self.brightness or FULL_BRIGHTNESS
        return ProviderOutcome(message=f"The {self._name} is on, at {self.brightness}% (simulated).", detail=self.state())

    def turn_off(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        was_on, self.on = self.on, False
        said = f"The {self._name} is off (simulated)." if was_on else f"The {self._name} was already off."
        return ProviderOutcome(message=said, detail=self.state())

    def set_brightness(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        level = int(arguments["level"])
        if level == 0:
            return self.turn_off(arguments)
        self.on, self.brightness = True, level
        return ProviderOutcome(message=f"The {self._name} is at {level}% (simulated).", detail=self.state())

    def status(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        said = f"on, at {self.brightness}%" if self.on else "off"
        return ProviderOutcome(message=f"The {self._name} is {said} (simulated).", detail=self.state())

    def state(self) -> dict[str, ArgumentValue]:
        return {"on": self.on, "brightness": self.brightness}


class SimulatedTv:
    """A TV's power and channel, kept in memory."""

    def __init__(self, name: str, channels: int) -> None:
        self._name = "TV" if name.lower() == "tv" else name.lower()
        self._channels = channels
        self.on = False
        self.channel = 1

    def turn_on(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if self.on:
            return ProviderOutcome(message=f"The {self._name} is already on, channel {self.channel}.", detail=self.state())
        self.on = True
        return ProviderOutcome(message=f"The {self._name} is on, channel {self.channel} (simulated).", detail=self.state())

    def turn_off(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        was_on, self.on = self.on, False
        said = f"The {self._name} is off (simulated)." if was_on else f"The {self._name} was already off."
        return ProviderOutcome(message=said, detail=self.state())

    def set_channel(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        self.on, self.channel = True, int(arguments["channel"])
        return ProviderOutcome(message=f"The {self._name} is on channel {self.channel} (simulated).", detail=self.state())

    def status(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        said = f"on, channel {self.channel}" if self.on else "off"
        return ProviderOutcome(message=f"The {self._name} is {said} (simulated).", detail=self.state())

    def state(self) -> dict[str, ArgumentValue]:
        return {"on": self.on, "channel": self.channel, "channels": self._channels}


def _switch(name: str, purpose: str, device: HomeDeviceConfig) -> DeviceCapability:
    return DeviceCapability(name=name, purpose=purpose, effect=ToolEffect.WRITE, device_class=DeviceClass.COMFORT, min_interval_s=device.min_interval_s)


def _manifest(device: HomeDeviceConfig, kind: DeviceKind, now: float, *capabilities: DeviceCapability) -> DeviceManifest:
    return DeviceManifest(
        ts=now,
        device_id=device.device_id,
        name=device.name,
        kind=kind,
        room=device.room,
        aliases=tuple(device.aliases),
        simulated=True,
        capabilities=capabilities,
    )


def _light(device: HomeDeviceConfig, now: float) -> DeviceParts:
    light = SimulatedLight(device.name)
    level = ToolParameter(name="level", type=ParameterType.INTEGER, description="Brightness from 0 (off) to 100 percent", minimum=0, maximum=FULL_BRIGHTNESS)
    manifest = _manifest(
        device,
        DeviceKind.LIGHT,
        now,
        _switch("turn_on", f"Turn the {device.name.lower()} on.", device),
        _switch("turn_off", f"Turn the {device.name.lower()} off.", device),
        DeviceCapability(
            name="set_brightness",
            purpose=f"Set how bright the {device.name.lower()} is.",
            effect=ToolEffect.WRITE,
            device_class=DeviceClass.COMFORT,
            parameters=(level,),
            min_interval_s=device.min_interval_s,
        ),
        DeviceCapability(name="status", purpose=f"Say whether the {device.name.lower()} is on.", effect=ToolEffect.READ, device_class=DeviceClass.COMFORT),
    )
    handlers = {"turn_on": light.turn_on, "turn_off": light.turn_off, "set_brightness": light.set_brightness, "status": light.status}
    return DeviceParts(manifest=manifest, handlers=handlers, state=light.state)


def _tv(device: HomeDeviceConfig, now: float) -> DeviceParts:
    tv = SimulatedTv(device.name, device.channels)
    channel = ToolParameter(name="channel", type=ParameterType.INTEGER, description=f"Channel from 1 to {device.channels}", minimum=1, maximum=device.channels)
    manifest = _manifest(
        device,
        DeviceKind.TV,
        now,
        _switch("turn_on", "Turn the TV on.", device),
        _switch("turn_off", "Turn the TV off.", device),
        DeviceCapability(
            name="set_channel",
            purpose="Put the TV on a channel.",
            effect=ToolEffect.WRITE,
            device_class=DeviceClass.MEDIA,
            parameters=(channel,),
            min_interval_s=device.min_interval_s,
        ),
        DeviceCapability(name="status", purpose="Say whether the TV is on, and on which channel.", effect=ToolEffect.READ, device_class=DeviceClass.MEDIA),
    )
    handlers = {"turn_on": tv.turn_on, "turn_off": tv.turn_off, "set_channel": tv.set_channel, "status": tv.status}
    return DeviceParts(manifest=manifest, handlers=handlers, state=tv.state)


def _fan(device: HomeDeviceConfig, now: float) -> DeviceParts:
    config = FanConfig(
        device_id=device.device_id,
        name=device.name,
        aliases=device.aliases,
        room=device.room,
        max_speed=device.max_speed,
        min_interval_s=device.min_interval_s,
    )
    simulated = fan.SimulatedFan(config.max_speed, "fan" if device.device_id == "fan" else device.name)
    handlers = {"turn_on": simulated.turn_on, "turn_off": simulated.turn_off, "set_speed": simulated.set_speed, "status": simulated.status}
    return DeviceParts(manifest=fan.manifest(config, now), handlers=handlers, state=simulated.state)


_BUILDERS = {"fan": _fan, "light": _light, "tv": _tv}


def build(config: Config, clock: Clock) -> list[DeviceParts]:
    """One device per entry in ``actuators.home.devices``; none if the house is switched off."""
    if not config.actuators.home.enabled:
        return []
    return [_BUILDERS[device.kind](device, clock.now()) for device in config.actuators.home.devices]
