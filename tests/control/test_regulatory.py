"""Unit tests for the regulatory controller."""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import ControllerConfig, load_config
from src.common.schemas import CommandKind, Mode
from src.control.regulatory import RegulatoryController

ACTUATOR_ID = "ac"
SETPOINT_C = 25.0
UNUSED_PREDICTION_C = -999.0


@pytest.fixture(name="config")
def _config() -> ControllerConfig:
    return load_config(Path("config/default.yaml")).controller


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="controller")
def _controller(config: ControllerConfig, clock: SimClock) -> RegulatoryController:
    return RegulatoryController(config, clock, ACTUATOR_ID)


def _start_cooling(controller, clock, config) -> None:
    controller.tick(SETPOINT_C + 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL)
    clock.advance(config.min_off_s)


class TestDeadband:
    def test_cools_when_the_room_is_above_the_deadband(self, controller):
        command = controller.tick(
            SETPOINT_C + 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.kind is CommandKind.COOL

    def test_a_cool_command_carries_the_setpoint(self, controller):
        command = controller.tick(
            SETPOINT_C + 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.setpoint_c == SETPOINT_C

    def test_holds_inside_the_deadband(self, controller, config):
        command = controller.tick(
            SETPOINT_C + config.deadband_c / 2.0,
            UNUSED_PREDICTION_C,
            SETPOINT_C,
            Mode.NORMAL,
        )
        assert command.kind is CommandKind.MAINTAIN

    def test_stops_when_the_room_falls_below_the_deadband(
        self, controller, clock, config
    ):
        _start_cooling(controller, clock, config)
        command = controller.tick(
            SETPOINT_C - 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.kind is CommandKind.OFF

    def test_keeps_cooling_inside_the_deadband_once_running(
        self, controller, clock, config
    ):
        """Asymmetry is the point: the band prevents chatter around the
        setpoint, so a running compressor is not stopped by a small error."""
        _start_cooling(controller, clock, config)
        command = controller.tick(
            SETPOINT_C, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.kind is CommandKind.MAINTAIN

    def test_the_deadband_edge_does_not_trigger_cooling(self, controller, config):
        command = controller.tick(
            SETPOINT_C + config.deadband_c,
            UNUSED_PREDICTION_C,
            SETPOINT_C,
            Mode.NORMAL,
        )
        assert command.kind is CommandKind.MAINTAIN


class TestCompressorDwell:
    def test_a_restart_inside_the_dwell_window_is_not_commanded(
        self, controller, clock, config
    ):
        _start_cooling(controller, clock, config)
        controller.tick(SETPOINT_C - 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL)
        clock.advance(config.min_off_s / 2.0)
        command = controller.tick(
            SETPOINT_C + 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.kind is CommandKind.MAINTAIN

    def test_a_restart_after_the_dwell_window_is_commanded(
        self, controller, clock, config
    ):
        _start_cooling(controller, clock, config)
        controller.tick(SETPOINT_C - 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL)
        clock.advance(config.min_off_s + 1.0)
        command = controller.tick(
            SETPOINT_C + 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.kind is CommandKind.COOL

    def test_the_first_command_is_not_delayed_by_dwell(self, controller):
        command = controller.tick(
            SETPOINT_C + 2.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.kind is CommandKind.COOL


class TestDegradedSensorSubstitution:
    def test_control_continues_on_the_prediction(self, controller):
        """FR-27: a stuck sensor reading 27.0 must not drive the loop."""
        command = controller.tick(
            SETPOINT_C, SETPOINT_C + 3.0, SETPOINT_C, Mode.DEGRADED_SENSOR
        )
        assert command.kind is CommandKind.COOL

    def test_the_measurement_is_ignored_while_substituting(self, controller):
        command = controller.tick(
            SETPOINT_C + 10.0, SETPOINT_C, SETPOINT_C, Mode.DEGRADED_SENSOR
        )
        assert command.kind is CommandKind.MAINTAIN

    @pytest.mark.parametrize("mode", [Mode.NORMAL, Mode.INIT])
    def test_the_measurement_is_used_in_every_other_mode(self, controller, mode):
        """A model is a worse source of truth than a working sensor."""
        assert controller.effective_temperature_c(20.0, 30.0, mode) == 20.0

    def test_the_prediction_is_used_only_when_degraded(self, controller):
        assert (
            controller.effective_temperature_c(20.0, 30.0, Mode.DEGRADED_SENSOR) == 30.0
        )


class TestNonActuatingModes:
    @pytest.mark.parametrize("mode", [Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD])
    def test_the_controller_holds(self, controller, mode):
        command = controller.tick(
            SETPOINT_C + 10.0, UNUSED_PREDICTION_C, SETPOINT_C, mode
        )
        assert command.kind is CommandKind.HOLD

    @pytest.mark.parametrize("mode", [Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD])
    def test_a_hold_command_carries_no_setpoint(self, controller, mode):
        command = controller.tick(
            SETPOINT_C + 10.0, UNUSED_PREDICTION_C, SETPOINT_C, mode
        )
        assert command.setpoint_c is None

    def test_holding_does_not_flip_the_compressor_state(
        self, controller, clock, config
    ):
        _start_cooling(controller, clock, config)
        controller.tick(SETPOINT_C + 10.0, UNUSED_PREDICTION_C, SETPOINT_C, Mode.SAFE_HOLD)
        assert controller.compressor_on is True


class TestDeterminism:
    def test_the_same_inputs_always_produce_the_same_command(self, config):
        def run() -> list[CommandKind]:
            clock = SimClock()
            controller = RegulatoryController(config, clock, ACTUATOR_ID)
            kinds = []
            for step in range(60):
                measured = SETPOINT_C + 2.0 - step * 0.1
                kinds.append(
                    controller.tick(
                        measured, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
                    ).kind
                )
                clock.advance(5.0)
            return kinds

        assert run() == run()

    def test_the_controller_always_emits_something(self, controller, clock):
        """Silence would be indistinguishable from a crashed loop."""
        for _ in range(20):
            command = controller.tick(
                SETPOINT_C, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
            )
            assert command.kind in set(CommandKind)
            clock.advance(5.0)

    def test_commands_carry_the_clock_timestamp(self, controller, clock):
        clock.advance(123.0)
        command = controller.tick(
            SETPOINT_C, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.ts == clock.now()

    def test_commands_name_the_actuator(self, controller):
        command = controller.tick(
            SETPOINT_C, UNUSED_PREDICTION_C, SETPOINT_C, Mode.NORMAL
        )
        assert command.actuator_id == ACTUATOR_ID
