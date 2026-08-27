"""Unit tests for the Layer 1 actuator contracts.

These assert the shape of the driver interface. The more important property
they guard is who is allowed to hold it: nothing in the reasoning layer may
import this module, because a model able to call ``turn_on`` bypasses the
setpoint bounds, the rate limit, the compressor dwell timer and the mode
interlocks in one step (FR-45).
"""

import inspect

from src.io.actuator_driver import AirConditioner, Fan, PowerSwitch, ThermostatMode


def declared_methods(interface) -> set[str]:
    return {
        name
        for name, _ in inspect.getmembers(interface, inspect.isfunction)
        if not name.startswith("__")
    }


class TestThermostatMode:
    def test_the_declared_modes_are_stable(self):
        assert {mode.value for mode in ThermostatMode} == {
            "auto",
            "cool",
            "heat",
            "dry",
            "fan_only",
        }

    def test_cool_is_the_mode_the_control_law_uses(self):
        """Section 5.3's deadband law emits COOL or OFF and nothing else."""
        assert ThermostatMode.COOL.value == "cool"


class TestInterfaces:
    def test_a_power_switch_declares_only_on_and_off(self):
        assert declared_methods(PowerSwitch) == {"turn_on", "turn_off"}

    def test_a_fan_is_a_power_switch_and_nothing_more(self):
        """Simulated: no thermal outcome is ever attributed to it."""
        assert declared_methods(Fan) == {"turn_on", "turn_off"}

    def test_an_air_conditioner_declares_its_full_surface(self):
        assert declared_methods(AirConditioner) == {
            "turn_on",
            "turn_off",
            "set_temperature",
            "increase_temperature",
            "decrease_temperature",
            "set_mode",
        }

    def test_an_air_conditioner_is_substitutable_for_a_power_switch(self):
        """Substitutability for a Protocol is structural, not nominal: what
        matters is that the wider interface declares everything the narrower
        one promises, so anything driving a PowerSwitch can drive this."""
        assert declared_methods(PowerSwitch) <= declared_methods(AirConditioner)


class TestLayerBoundary:
    def test_the_reasoning_layer_does_not_import_the_driver_contract(self):
        """FR-45: all influence is exerted through the setpoint goal."""
        from pathlib import Path

        for module in Path("src/reasoning").rglob("*.py"):
            source = module.read_text(encoding="utf-8")
            assert "actuator_driver" not in source, module

    def test_the_speech_layer_does_not_import_the_driver_contract(self):
        from pathlib import Path

        for module in Path("src/speech").rglob("*.py"):
            source = module.read_text(encoding="utf-8")
            assert "actuator_driver" not in source, module
