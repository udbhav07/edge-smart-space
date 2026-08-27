import inspect

from voiceAssistant.tools import AirConditioner, Fan, PowerSwitch, ThermostatMode


def declared_methods(interface):
    return {
        name
        for name, _ in inspect.getmembers(interface, inspect.isfunction)
        if not name.startswith("__")
    }


def test_thermostat_modes_are_declared():
    assert {mode.value for mode in ThermostatMode} == {
        "auto",
        "cool",
        "heat",
        "dry",
        "fan_only",
    }


def test_power_and_appliance_interfaces_declare_expected_operations():
    assert declared_methods(PowerSwitch) == {
        "turn_on",
        "turn_off",
    }
    assert declared_methods(Fan) == {
        "turn_on",
        "turn_off",
    }
    assert declared_methods(AirConditioner) == {
        "turn_on",
        "turn_off",
        "set_temperature",
        "increase_temperature",
        "decrease_temperature",
        "set_mode",
    }