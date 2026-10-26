"""A simulated ceiling fan: the smallest complete device service (FR-87, FR-15).

Everything a device needs is here -- a manifest, a handler per capability,
and its state -- and nothing else; :mod:`src.devices.runner` runs it. It is
simulated: it moves no air, and no thermal effect is modelled or claimed for
it, which its manifest and every state message say (FR-15).
"""

from __future__ import annotations

from collections.abc import Mapping

from src.common.clock import Clock
from src.common.config import Config, FanConfig
from src.common.devices import DeviceCapability, DeviceClass, DeviceKind, DeviceManifest
from src.common.tools import ArgumentValue, ParameterType, ProviderOutcome, ToolEffect, ToolParameter
from src.devices.runner import DeviceParts


class SimulatedFan:
    """A fan's switch and speed, kept in memory."""

    def __init__(self, max_speed: int) -> None:
        self._max_speed = max_speed
        self.on = False
        self.speed = 0

    def turn_on(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if self.on:
            return ProviderOutcome(message=f"The fan is already on, at speed {self.speed}.", detail=self.state())
        self.on = True
        self.speed = self.speed or 1
        return ProviderOutcome(message=f"The fan is on, at speed {self.speed} (simulated).", detail=self.state())

    def turn_off(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        was_on = self.on
        self.on = False
        return ProviderOutcome(message="The fan is off (simulated)." if was_on else "The fan was already off.", detail=self.state())

    def set_speed(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        speed = int(arguments["speed"])
        if speed == 0:
            return self.turn_off(arguments)
        self.on, self.speed = True, speed
        return ProviderOutcome(message=f"The fan is at speed {speed} of {self._max_speed} (simulated).", detail=self.state())

    def status(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        said = f"on, at speed {self.speed} of {self._max_speed}" if self.on else "off"
        return ProviderOutcome(message=f"The fan is {said} (simulated).", detail=self.state())

    def state(self) -> dict[str, ArgumentValue]:
        return {"on": self.on, "speed": self.speed if self.on else 0, "max_speed": self._max_speed}


def manifest(config: FanConfig, now: float) -> DeviceManifest:
    """What the fan is and what it can do."""

    def switch(name: str, purpose: str) -> DeviceCapability:
        return DeviceCapability(name=name, purpose=purpose, effect=ToolEffect.WRITE, device_class=DeviceClass.COMFORT, min_interval_s=config.min_interval_s)

    speed = ToolParameter(
        name="speed",
        type=ParameterType.INTEGER,
        description=f"Fan speed from 0 (off) to {config.max_speed} (fastest)",
        minimum=0,
        maximum=config.max_speed,
    )
    return DeviceManifest(
        ts=now,
        device_id=config.device_id,
        name=config.name,
        kind=DeviceKind.FAN,
        aliases=tuple(config.aliases),
        simulated=True,
        capabilities=(
            switch("turn_on", "Turn the fan on."),
            switch("turn_off", "Turn the fan off."),
            DeviceCapability(
                name="set_speed",
                purpose="Set how fast the fan runs.",
                effect=ToolEffect.WRITE,
                device_class=DeviceClass.COMFORT,
                parameters=(speed,),
                min_interval_s=config.min_interval_s,
            ),
            DeviceCapability(name="status", purpose="Say whether the fan is on and how fast.", effect=ToolEffect.READ, device_class=DeviceClass.COMFORT),
        ),
    )


def build(config: Config, clock: Clock) -> DeviceParts:
    fan = SimulatedFan(config.actuators.fan.max_speed)
    handlers = {"turn_on": fan.turn_on, "turn_off": fan.turn_off, "set_speed": fan.set_speed, "status": fan.status}
    return DeviceParts(manifest=manifest(config.actuators.fan, clock.now()), handlers=handlers, state=fan.state)
