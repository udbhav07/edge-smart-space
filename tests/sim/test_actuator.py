"""Unit tests for the simulated air conditioner."""

import random
from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import SimActuatorConfig, load_config
from src.common.schemas import AckStatus, CommandKind
from sim.actuator import COOLING_OFF, COOLING_ON, SimulatedActuator

SETPOINT_C = 25.5
SEED = 20260826


@pytest.fixture(name="actuator_config")
def _actuator_config() -> SimActuatorConfig:
    return load_config(Path("config/default.yaml")).sim.actuator


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


def _reliable(config: SimActuatorConfig) -> SimActuatorConfig:
    """No command loss, so a test can isolate one behaviour at a time."""
    return config.model_copy(update={"command_loss_probability": 0.0})


def _unit(config: SimActuatorConfig, clock: SimClock) -> SimulatedActuator:
    return SimulatedActuator(config, random.Random(SEED), clock)


def _settle(unit: SimulatedActuator, clock: SimClock, config) -> None:
    clock.advance(config.dead_time_s)
    unit.apply_due_commands()


class TestInitialState:
    def test_starts_switched_off(self, actuator_config, clock):
        assert _unit(actuator_config, clock).cooling_fraction == COOLING_OFF

    def test_starts_with_no_commands_in_flight(self, actuator_config, clock):
        assert _unit(actuator_config, clock).pending_count == 0

    def test_has_no_command_timestamp_before_the_first_command(
        self, actuator_config, clock
    ):
        assert _unit(actuator_config, clock).last_command_ts is None

    def test_starts_healthy(self, actuator_config, clock):
        assert _unit(actuator_config, clock).is_failed is False


class TestDeadTime:
    def test_a_command_has_no_effect_before_its_dead_time_elapses(
        self, actuator_config, clock
    ):
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        unit.apply_due_commands()
        assert unit.cooling_fraction == COOLING_OFF

    def test_a_command_takes_effect_once_its_dead_time_elapses(
        self, actuator_config, clock
    ):
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        _settle(unit, clock, config)
        assert unit.cooling_fraction == COOLING_ON

    def test_reading_the_drive_never_advances_time_by_itself(
        self, actuator_config, clock
    ):
        """A getter that promoted commands would make behaviour depend on
        how many times it happened to be called."""
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        clock.advance(config.dead_time_s)
        assert unit.cooling_fraction == COOLING_OFF
        assert unit.cooling_fraction == COOLING_OFF
        unit.apply_due_commands()
        assert unit.cooling_fraction == COOLING_ON

    def test_two_commands_inside_one_dead_time_both_take_effect_in_order(
        self, actuator_config, clock
    ):
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        clock.advance(config.dead_time_s / 2.0)
        unit.command(CommandKind.OFF)
        assert unit.pending_count == 2

        clock.advance(config.dead_time_s / 2.0)
        unit.apply_due_commands()
        assert unit.cooling_fraction == COOLING_ON

        clock.advance(config.dead_time_s / 2.0)
        unit.apply_due_commands()
        assert unit.cooling_fraction == COOLING_OFF

    def test_in_flight_commands_are_bounded(self, actuator_config, clock):
        unit = _unit(_reliable(actuator_config), clock)
        for _ in range(100):
            unit.command(CommandKind.COOL, SETPOINT_C)
        assert unit.pending_count <= 16


class TestCommandKinds:
    def test_off_stops_the_drive(self, actuator_config, clock):
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        _settle(unit, clock, config)
        unit.command(CommandKind.OFF)
        _settle(unit, clock, config)
        assert unit.cooling_fraction == COOLING_OFF

    @pytest.mark.parametrize("kind", [CommandKind.MAINTAIN, CommandKind.HOLD])
    def test_a_no_change_command_leaves_the_drive_alone(
        self, actuator_config, clock, kind
    ):
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        _settle(unit, clock, config)
        unit.command(kind)
        _settle(unit, clock, config)
        assert unit.cooling_fraction == COOLING_ON

    @pytest.mark.parametrize("kind", [CommandKind.MAINTAIN, CommandKind.HOLD])
    def test_a_no_change_command_queues_nothing(self, actuator_config, clock, kind):
        unit = _unit(_reliable(actuator_config), clock)
        unit.command(kind)
        assert unit.pending_count == 0

    def test_every_command_records_when_it_was_sent(self, actuator_config, clock):
        unit = _unit(_reliable(actuator_config), clock)
        clock.advance(10.0)
        unit.command(CommandKind.OFF)
        assert unit.last_command_ts == clock.now()


class TestAcknowledgement:
    def test_an_open_loop_unit_never_confirms_anything(self, actuator_config, clock):
        """R-02: with no readback, UNKNOWN is the only honest answer."""
        config = actuator_config.model_copy(update={"acknowledges": False})
        unit = _unit(config, clock)
        assert unit.command(CommandKind.COOL, SETPOINT_C) is AckStatus.UNKNOWN

    def test_the_shipped_configuration_is_open_loop(self, actuator_config, clock):
        assert actuator_config.acknowledges is False

    def test_a_unit_with_readback_confirms_a_delivered_command(
        self, actuator_config, clock
    ):
        config = actuator_config.model_copy(
            update={"acknowledges": True, "command_loss_probability": 0.0}
        )
        unit = _unit(config, clock)
        assert unit.command(CommandKind.COOL, SETPOINT_C) is AckStatus.ACKNOWLEDGED

    def test_a_unit_with_readback_reports_a_lost_command(self, actuator_config, clock):
        config = actuator_config.model_copy(
            update={"acknowledges": True, "command_loss_probability": 1.0}
        )
        unit = _unit(config, clock)
        assert unit.command(CommandKind.COOL, SETPOINT_C) is AckStatus.FAILED

    def test_an_open_loop_unit_reports_unknown_even_when_the_command_is_lost(
        self, actuator_config, clock
    ):
        config = actuator_config.model_copy(
            update={"acknowledges": False, "command_loss_probability": 1.0}
        )
        unit = _unit(config, clock)
        assert unit.command(CommandKind.COOL, SETPOINT_C) is AckStatus.UNKNOWN


class TestCommandLoss:
    def test_a_lost_command_never_reaches_the_drive(self, actuator_config, clock):
        config = actuator_config.model_copy(update={"command_loss_probability": 1.0})
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        _settle(unit, clock, config)
        assert unit.cooling_fraction == COOLING_OFF

    def test_a_lost_command_queues_nothing(self, actuator_config, clock):
        config = actuator_config.model_copy(update={"command_loss_probability": 1.0})
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        assert unit.pending_count == 0


class TestInjectedFailure:
    def test_a_failed_unit_stops_affecting_the_room(self, actuator_config, clock):
        """FR-24: commands keep being accepted, the room stops responding."""
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        _settle(unit, clock, config)
        unit.inject_failure(True)
        assert unit.cooling_fraction == COOLING_OFF

    def test_a_failed_unit_still_accepts_commands(self, actuator_config, clock):
        config = actuator_config.model_copy(
            update={"acknowledges": True, "command_loss_probability": 0.0}
        )
        unit = _unit(config, clock)
        unit.inject_failure(True)
        assert unit.command(CommandKind.COOL, SETPOINT_C) is AckStatus.ACKNOWLEDGED

    def test_clearing_the_failure_restores_the_commanded_drive(
        self, actuator_config, clock
    ):
        config = _reliable(actuator_config)
        unit = _unit(config, clock)
        unit.command(CommandKind.COOL, SETPOINT_C)
        _settle(unit, clock, config)
        unit.inject_failure(True)
        unit.inject_failure(False)
        assert unit.cooling_fraction == COOLING_ON

    def test_the_failure_state_is_reportable(self, actuator_config, clock):
        unit = _unit(actuator_config, clock)
        unit.inject_failure(True)
        assert unit.is_failed is True


class TestDeterminism:
    def test_the_same_seed_reproduces_the_same_acknowledgements(
        self, actuator_config
    ):
        config = actuator_config.model_copy(
            update={"acknowledges": True, "command_loss_probability": 0.5}
        )

        def run() -> list[AckStatus]:
            unit = SimulatedActuator(config, random.Random(SEED), SimClock())
            return [unit.command(CommandKind.COOL, SETPOINT_C) for _ in range(50)]

        assert run() == run()
