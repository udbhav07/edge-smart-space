"""Layer 1 actuator contracts.

These Protocols describe what a device driver can be asked to do. They are
the interface the regulatory controller's commands are ultimately carried
out through, and they live at Layer 1 because this is the only layer allowed
to know a device detail (an IR protocol, an ESPHome entity, a GPIO pin).

Who may call these matters as much as what they say. The regulatory
controller drives them, through the actuator driver, after the safety
validator has passed a command. The reasoning layer never does: its entire
influence is a proposed setpoint that the validator gates (FR-45). Exposing
these methods as a model's tool surface would bypass the setpoint bounds,
the rate limit, the compressor dwell timer and the mode interlocks in one
step, which is the arrangement the supervisory architecture exists to
prevent.

Only the air conditioner is real (DESIGN.md section 2.1). Fan and switch
implementations are simulated, are labelled ``simulated: true`` in every
state message they publish (FR-15), and no thermal outcome is ever
attributed to them.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class ThermostatMode(str, Enum):
    """Operating mode of an air conditioner.

    Only COOL is exercised by the regulatory controller: the deadband law in
    section 5.3 emits COOL or OFF and nothing else. The rest exist because
    real units expose them and a driver has to be able to report what it
    found the unit in.
    """

    AUTO = "auto"
    COOL = "cool"
    HEAT = "heat"
    DRY = "dry"
    FAN_ONLY = "fan_only"


class PowerSwitch(Protocol):
    """Anything that can be switched on or off."""

    def turn_on(self) -> None:
        """Turn the device on."""

    def turn_off(self) -> None:
        """Turn the device off."""


class Fan(PowerSwitch, Protocol):
    """A fan. Simulated: it has no modelled effect on room temperature."""


class AirConditioner(PowerSwitch, Protocol):
    """The one physical actuator.

    ``set_temperature`` sets the unit's own target. It is not the system's
    setpoint: the system's setpoint is decided by the goal path and gated by
    the validator, and reaches the unit only as a COOL command carrying an
    already-admitted value.
    """

    def set_temperature(self, temperature_celsius: float) -> None:
        """Set the requested temperature in degrees Celsius."""

    def increase_temperature(self, degrees: float = 1.0) -> None:
        """Raise the requested temperature."""

    def decrease_temperature(self, degrees: float = 1.0) -> None:
        """Lower the requested temperature."""

    def set_mode(self, mode: ThermostatMode) -> None:
        """Set the operating mode."""
