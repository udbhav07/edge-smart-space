"""Unit tests for the scripted demonstration.

The behaviour it demonstrates is tested in tests/test_system_end_to_end.py;
what is checked here is that the script itself runs, narrates, and does not
quietly soften the thing it claims to be demonstrating.
"""

from pathlib import Path

import pytest

from src.common.config import load_config
from tools.demo import DEMO_BUDGET_S, SCENARIOS, _configure, main


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


class TestScenarioConfiguration:
    def test_the_budget_is_shortened_so_the_story_fits_one_run(self, config):
        adjusted = _configure(config, "stuck")
        assert adjusted.mode.degraded_sensor_budget_s == DEMO_BUDGET_S

    def test_the_shipped_budget_is_longer_than_the_demonstration_one(self, config):
        """The demo shortens it deliberately; section 7.2's value is 1800 s."""
        assert config.mode.degraded_sensor_budget_s > DEMO_BUDGET_S

    def test_the_sensors_are_not_softened(self, config):
        """A demonstration against a kind simulator shows nothing. Noise,
        quantisation, jitter and dropout all stay as shipped."""
        adjusted = _configure(config, "stuck")
        assert adjusted.sim.sensor_noise == config.sim.sensor_noise

    def test_the_actuator_keeps_its_dead_time_and_command_loss(self, config):
        adjusted = _configure(config, "stuck")
        assert adjusted.sim.actuator == config.sim.actuator

    def test_the_stuck_scenario_leaves_the_plant_able_to_cool(self, config):
        adjusted = _configure(config, "stuck")
        assert adjusted.sim.room.cooling_power_w > 0.0

    def test_no_scenario_weakens_the_plant(self, config):
        """The actuator scenario breaks the unit by injecting a fault, not by
        configuring a room that could never be cooled. A weaker plant would
        also be one the model learns is weak, and D5 tests the room against
        what the model expects -- so a unit dead before identification began
        is a unit with no expectation left to violate."""
        for scenario in ("stuck", "actuator"):
            adjusted = _configure(config, scenario)
            assert adjusted.sim.room == config.sim.room


class TestRunning:
    def test_the_actuator_scenario_runs_and_reports_the_fault(self, capsys):
        assert main(["--scenario", "actuator", "--quiet"]) == 0
        printed = capsys.readouterr().out
        assert "D5_ACTUATOR_NO_RESPONSE" in printed
        assert "DEGRADED_ACTUATOR" in printed

    def test_it_says_the_fault_came_from_the_room_not_an_acknowledgement(
        self, capsys
    ):
        """R-02: there is no acknowledgement to have, and the demonstration
        should say so rather than let a viewer assume one."""
        main(["--scenario", "actuator", "--quiet"])
        assert "acknowledgement" in capsys.readouterr().out

    def test_quiet_mode_leaves_out_the_evidence_lines(self, capsys):
        main(["--scenario", "actuator", "--quiet"])
        quiet = capsys.readouterr().out
        main(["--scenario", "actuator"])
        verbose = capsys.readouterr().out
        assert len(verbose) > len(quiet)
        assert "cooled_c" in verbose

    def test_a_missing_config_file_exits_two(self):
        assert main(["--config", "config/nope.yaml"]) == 2

    def test_every_scenario_is_runnable_by_name(self):
        assert set(SCENARIOS) == {"stuck", "actuator"}


class TestTheStuckSensorStory:
    """The headline claim, asserted on the narration itself."""

    def test_the_run_shows_control_continuing_after_the_fault(self, capsys):
        assert main(["--quiet"]) == 0
        printed = capsys.readouterr().out
        assert "D2_STUCK_AT" in printed
        assert "DEGRADED_SENSOR" in printed
        assert "STILL   controlling" in printed

    def test_the_hold_is_reached_by_the_budget_not_by_a_second_fault(
        self, capsys
    ):
        """If a stuck sensor manufactures an actuator fault, the system holds
        for the wrong reason and the demonstration is a lie."""
        main(["--quiet"])
        printed = capsys.readouterr().out
        assert "SAFE_HOLD -- prediction budget" in printed
        assert "D5_ACTUATOR_NO_RESPONSE" not in printed
