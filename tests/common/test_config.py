"""Unit tests for configuration loading and validation."""

import textwrap
from pathlib import Path

import pytest

from src.common.config import Bounds, Config, ConfigError, load_config

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


@pytest.fixture(name="default_config")
def _default_config() -> Config:
    return load_config(DEFAULT_CONFIG_PATH)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


class TestBounds:
    def test_clamps_below_the_interval(self):
        assert Bounds(low=18.0, high=30.0).clamp(5.0) == 18.0

    def test_clamps_above_the_interval(self):
        assert Bounds(low=18.0, high=30.0).clamp(99.0) == 30.0

    def test_leaves_an_interior_value_alone(self):
        assert Bounds(low=18.0, high=30.0).clamp(25.5) == 25.5

    @pytest.mark.parametrize("value", [18.0, 30.0])
    def test_the_interval_is_inclusive(self, value):
        assert Bounds(low=18.0, high=30.0).contains(value)

    def test_an_inverted_interval_is_rejected(self):
        with pytest.raises(ValueError):
            Bounds(low=30.0, high=18.0)

    def test_a_degenerate_interval_is_rejected(self):
        with pytest.raises(ValueError):
            Bounds(low=18.0, high=18.0)


class TestDefaultConfig:
    """The shipped defaults must match the values DESIGN.md states."""

    def test_loads_without_error(self, default_config):
        assert isinstance(default_config, Config)

    def test_forgetting_factor_matches_section_5_2_2(self, default_config):
        assert default_config.estimator.forgetting_factor == 0.995

    def test_initial_theta_matches_section_5_2_2(self, default_config):
        """The identified vector is [a2, a3, a4]; a1 follows as 1 - a2."""
        assert default_config.estimator.initial_theta == (0.02, -0.05, 0.01)

    def test_the_prior_implies_the_documented_thermal_inertia(self, default_config):
        assert 1.0 - default_config.estimator.initial_theta[0] == pytest.approx(0.98)

    def test_regulatory_period_matches_nfr_01(self, default_config):
        assert default_config.loop.regulatory_period_s == 5.0

    def test_deadband_and_dwell_match_section_5_3(self, default_config):
        assert default_config.controller.deadband_c == 0.5
        assert default_config.controller.min_off_s == 180.0

    def test_setpoint_bounds_match_validator_rule_v1(self, default_config):
        assert default_config.validator.setpoint_bounds_c.low == 18.0
        assert default_config.validator.setpoint_bounds_c.high == 30.0

    def test_degradation_budget_matches_section_7_2(self, default_config):
        assert default_config.mode.degraded_sensor_budget_s == 1800.0

    def test_actuator_authority_bound_keeps_cooling_negative(self, default_config):
        """Section 5.2.1: a3 crossing zero claims the AC heats the room."""
        assert default_config.estimator.bounds_a3.high == 0.0


class TestAdversarialSimulationDefaults:
    """Every imperfection ships switched on, not off."""

    def test_sampling_jitter_is_enabled(self, default_config):
        assert default_config.sim.sensor_noise.jitter_s > 0.0

    def test_sensor_noise_is_enabled(self, default_config):
        assert default_config.sim.sensor_noise.sigma_c > 0.0

    def test_quantisation_is_enabled(self, default_config):
        assert default_config.sim.sensor_noise.quantisation_c > 0.0

    def test_dropouts_are_enabled(self, default_config):
        assert default_config.sim.sensor_noise.dropout_probability > 0.0

    def test_actuator_dead_time_is_enabled(self, default_config):
        assert default_config.sim.actuator.dead_time_s > 0.0

    def test_command_loss_is_enabled(self, default_config):
        assert default_config.sim.actuator.command_loss_probability > 0.0

    def test_the_actuator_cannot_acknowledge_by_default(self, default_config):
        """R-02: the IR path has no readback, so ack is always UNKNOWN."""
        assert default_config.sim.actuator.acknowledges is False

    def test_an_unmodelled_disturbance_is_present(self, default_config):
        """R-04: the estimator has no regressor for solar gain."""
        assert default_config.sim.room.solar_gain_amplitude_w > 0.0

    def test_the_seed_is_fixed_so_a_scenario_replays_identically(self, default_config):
        assert default_config.sim.random_seed >= 0


class TestCrossSectionConsistency:
    def test_controller_and_validator_must_agree_on_dwell(self, default_config):
        assert default_config.controller.min_off_s == default_config.validator.min_off_s

    def test_disagreeing_dwell_values_are_rejected(self, tmp_path):
        # The controller's dwell is the first of the two in the file, so one
        # replacement moves it and leaves the validator's alone. Anchored on
        # the value rather than on its neighbouring lines, which a comment
        # added above it would otherwise silently break -- as one did.
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "  min_off_s: 180.0", "  min_off_s: 120.0", 1
        )
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_initial_theta_outside_its_own_bounds_is_rejected(self, tmp_path):
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "initial_theta: [0.02, -0.05, 0.01]",
            "initial_theta: [0.02, 0.05, 0.01]",
        )
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)

    def test_a_prior_implying_an_impossible_a1_is_rejected(self, tmp_path):
        """a1 is derived, so a2 can look fine while what it implies does not."""
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "initial_theta: [0.02, -0.05, 0.01]",
            "initial_theta: [1.6, -0.05, 0.01]",
        )
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


class TestReanchorHistory:
    def test_a_history_inside_the_stuck_window_is_rejected(self, tmp_path):
        """It would only ever hold frozen readings to restart from (FR-27)."""
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace(
            "reanchor_history_samples: 180", "reanchor_history_samples: 60"
        )
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


class TestLoadFailures:
    def test_a_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(tmp_path / "absent.yaml")

    def test_malformed_yaml_is_reported_clearly(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, "mqtt: [unclosed\n"))

    def test_a_non_mapping_document_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, "- just\n- a\n- list\n"))

    def test_a_missing_section_is_rejected(self, tmp_path):
        with pytest.raises(ConfigError):
            load_config(_write(tmp_path, "mqtt: {host: localhost, port: 1883}\n"))

    def test_an_unknown_key_is_rejected_rather_than_ignored(self, tmp_path):
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8") + "\nnonsense: 1\n"
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


class TestImmutability:
    def test_configuration_cannot_be_changed_at_runtime(self, default_config):
        with pytest.raises(ValueError):
            default_config.controller.deadband_c = 1.0


class TestOverlays:
    """R-04: a calibration is a small file over the default, not an edit of it."""

    def _overlay(self, tmp_path, body: str) -> Path:
        path = tmp_path / "site.yaml"
        base = DEFAULT_CONFIG_PATH.resolve().as_posix()
        path.write_text(f"extends: {base}\n" + textwrap.dedent(body), encoding="utf-8")
        return path

    def test_an_overlay_changes_only_what_it_names(self, tmp_path, default_config):
        config = load_config(
            self._overlay(
                tmp_path,
                """
                detectors:
                  drift:
                    threshold_sigma: 9.5
                """,
            )
        )
        assert config.detectors.drift.threshold_sigma == 9.5
        assert config.detectors.drift.slack_sigma == default_config.detectors.drift.slack_sigma
        assert config.controller == default_config.controller

    def test_a_list_is_replaced_whole_not_merged(self, tmp_path):
        body = """
        sensors:
          adapters:
            - {sensor_id: temp_01, unit: C, limits: {low: -10.0, high: 60.0}}
        """
        config = load_config(self._overlay(tmp_path, body))
        assert [sensor.sensor_id for sensor in config.sensors.adapters] == ["temp_01"]

    def test_an_overlay_is_validated_like_any_config(self, tmp_path):
        body = """
        detectors:
          drift:
            threshold_sigma: -1
        """
        with pytest.raises(ConfigError):
            load_config(self._overlay(tmp_path, body))

    def test_a_relative_base_is_found_beside_the_overlay(self, tmp_path):
        (tmp_path / "base.yaml").write_text(
            DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8"
        )
        site = tmp_path / "site.yaml"
        site.write_text("extends: base.yaml\nsite: {utc_offset_h: 1.0}\n", encoding="utf-8")
        assert load_config(site).site.utc_offset_h == 1.0

    def test_a_cycle_is_refused(self, tmp_path):
        (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
        (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(tmp_path / "a.yaml")

    def test_a_missing_base_is_reported(self, tmp_path):
        site = tmp_path / "site.yaml"
        site.write_text("extends: absent.yaml\n", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(site)
