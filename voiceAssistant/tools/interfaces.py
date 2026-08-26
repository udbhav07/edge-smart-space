from enum import Enum
from typing import Protocol


class ThermostatMode(str, Enum):
    AUTO = "auto"
    COOL = "cool"
    HEAT = "heat"
    DRY = "dry"
    FAN_ONLY = "fan_only"


class PowerSwitch(Protocol):
    def turn_on(self) -> None:
        """Turn the device on."""

    def turn_off(self) -> None:
        """Turn the device off."""


class Fan(PowerSwitch, Protocol):
    pass


class AirConditioner(PowerSwitch, Protocol):
    def set_temperature(self, temperature_celsius: float) -> None:
        """Set the requested temperature in degrees Celsius."""

    def increase_temperature(self, degrees: float = 1.0) -> None:
        """Raise the requested temperature."""

    def decrease_temperature(self, degrees: float = 1.0) -> None:
        """Lower the requested temperature."""

    def set_mode(self, mode: ThermostatMode) -> None:
        """Set the operating mode."""