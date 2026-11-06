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

    def test_the_assistant_waits_longer_than_a_device_is_given(self, default_config):
        assert default_config.assistance.result_timeout_s > default_config.actuators.command_timeout_s

    def test_an_assistant_that_gives_up_before_a_device_is_rejected(self, tmp_path):
        # A song searched for online would play, and the occupant be told it did not (FR-96).
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("  result_timeout_s: 14.0", "  result_timeout_s: 5.0", 1)
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="must exceed actuators.command_timeout_s"):
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

    def test_a_partial_file_is_pointed_at_extends(self, tmp_path):
        """U-31: one problem, not eighteen "Field required" errors."""
        with pytest.raises(ConfigError, match="extends: default.yaml"):
            load_config(_write(tmp_path, "controller: {deadband_c: 0.5}\n"))

    def test_a_file_that_extends_gets_no_such_hint(self, tmp_path):
        base = tmp_path / "base.yaml"
        base.write_text("controller: {deadband_c: 0.5}\n", encoding="utf-8")
        override = tmp_path / "over.yaml"
        override.write_text("extends: base.yaml\n", encoding="utf-8")
        with pytest.raises(ConfigError) as caught:
            load_config(override)
        assert "start the file with" not in str(caught.value)

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


class TestAmbientListening:
    """FR-76: always listening is the shipped default, and it is configuration."""

    def test_the_room_listens_continuously_by_default(self):
        from src.common.schemas import ListenMode

        config = load_config(Path("config/default.yaml"))
        assert config.speech.listen_mode is ListenMode.CONTINUOUS

    def test_ambient_mining_is_bounded(self):
        config = load_config(Path("config/default.yaml"))
        assert config.ambient.max_calls_per_min > 0 and config.ambient.max_lines > 0


class TestJetsonConfigs:
    """The two boards' configs load, and keep the rules section 5.7.5 and 9.2 set."""

    @pytest.mark.parametrize("name", ["jetsonDefault.yaml", "jetson16.yaml"])
    def test_it_loads(self, name):
        load_config(Path("config") / name)

    @pytest.mark.parametrize("name", ["jetsonDefault.yaml", "jetson16.yaml"])
    def test_the_model_does_not_think_aloud(self, name):
        assert "instruct" in load_config(Path("config") / name).reasoning.model

    @pytest.mark.parametrize("name", ["jetsonDefault.yaml", "jetson16.yaml"])
    def test_whisper_demands_the_gpu(self, name):
        assert load_config(Path("config") / name).speech.asr_device == "cuda"

    def test_only_what_the_board_changes_differs_from_the_default(self, default_config):
        board, default = load_config(Path("config/jetsonDefault.yaml")), default_config
        assert board.controller == default.controller and board.validator == default.validator



def _extending(home_device: str) -> str:
    """A config file over the default whose house holds this one device."""
    return f"extends: {DEFAULT_CONFIG_PATH.resolve().as_posix()}\nactuators:\n  home:\n    devices:\n      - {home_device}\n"


class TestTheHouse:
    """The floor plan is whole, and holds what is said to be in it (FR-99, FR-100)."""

    @staticmethod
    def _house(default_config, **update):
        from src.common.config import HouseConfig

        return HouseConfig(**{**default_config.house.model_dump(), **update})

    def test_the_default_house_loads(self, default_config):
        assert {room.room_id for room in default_config.house.rooms} == {"living_room", "kitchen", "bathroom", "bedroom"}

    def test_a_door_off_the_shared_wall_is_refused(self, default_config):
        doors = [{"between": ["living_room", "kitchen"], "x": 3.0, "y": 2.0}]
        with pytest.raises(ValueError, match="not on a wall they share"):
            self._house(default_config, doors=doors)

    def test_a_room_no_door_leads_into_is_refused(self, default_config):
        doors = [door for door in default_config.house.model_dump()["doors"] if "bathroom" not in door["between"]]
        with pytest.raises(ValueError, match="no door leads into"):
            self._house(default_config, doors=doors)

    def test_a_device_placed_outside_its_room_is_refused(self, default_config):
        placements = {"tv": {"room": "kitchen", "x": 1.0, "y": 1.0, "z": 1.0}}
        with pytest.raises(ValueError, match="not inside the room"):
            self._house(default_config, placements=placements)

    def test_a_home_device_in_a_room_the_house_lacks_is_refused(self, tmp_path):
        path = _write(tmp_path, _extending("{device_id: garage_light, name: Garage light, kind: light, room: garage}"))
        with pytest.raises(ConfigError, match="no room for"):
            load_config(path)

    def test_a_device_id_used_twice_is_refused(self, tmp_path):
        path = _write(tmp_path, _extending("{device_id: music, name: Radio, kind: light, room: kitchen}"))
        with pytest.raises(ConfigError, match="used twice"):
            load_config(path)

    def test_the_robot_needs_its_spots(self, default_config):
        from src.common.config import Config

        spots = {name: spot for name, spot in default_config.house.model_dump()["spots"].items() if name != "dog_bowl"}
        data = default_config.model_dump()
        data["house"]["spots"] = spots
        with pytest.raises(ValueError, match="dog_bowl"):
            Config(**data)


class TestWebSearch:
    """FR-101 to FR-103: on by default, each switchable, each bounded in time."""

    def test_web_search_is_on_by_default_and_searches_india_in_english(self, default_config):
        search = default_config.web_search
        assert search.enabled is True
        assert search.region == "en-IN"

    def test_searxng_is_pinned_to_a_release_and_not_to_latest(self, default_config):
        image = default_config.web_search.searxng.image
        assert image.startswith("docker.io/searxng/searxng:")
        assert not image.endswith(":latest")

    def test_cricket_scores_are_on_by_default_from_cricinfo(self, default_config):
        cricket = default_config.web_search.cricket
        assert cricket.enabled is True
        assert cricket.provider == "cricinfo"

    def test_a_config_without_the_section_still_loads_with_it_on(self, tmp_path):
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        start = text.index("\nweb_search:")
        end = text.index("\n# ", start + 1)
        path = tmp_path / "config.yaml"
        path.write_text(text[:start] + text[end:], encoding="utf-8")
        assert load_config(path).web_search.enabled is True

    def test_a_search_that_outlasts_the_assistant_is_rejected(self, tmp_path):
        # The occupant would be told nothing came back while the answer was on its way.
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("  timeout_s: 6.0  ", "  timeout_s: 20.0  ", 1)
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="web_search.timeout_s"):
            load_config(path)

    def test_a_score_feed_that_outlasts_the_assistant_is_rejected(self, tmp_path):
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("    timeout_s: 5.0  ", "    timeout_s: 20.0  ", 1)
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="web_search.cricket.timeout_s"):
            load_config(path)


class TestDeepAnswers:
    """FR-104: reading the top pages and searching again, each bounded."""

    def test_deep_answers_are_on_and_read_three_pages(self, default_config):
        deep = default_config.web_search.deep
        assert deep.enabled is True and deep.pages == 3

    def test_a_second_search_is_allowed_by_default(self, default_config):
        assert default_config.web_search.deep.search_again is True

    def test_what_was_hard_coded_in_the_reader_is_config(self, default_config):
        deep = default_config.web_search.deep
        assert (deep.wait_grace_s, deep.min_page_chars, deep.min_paragraph_chars, deep.lead_paragraphs, deep.remembered_urls) == (
            1.0,
            200,
            30,
            2,
            50,
        )

    def test_a_page_read_whose_grace_outlasts_the_assistant_is_rejected(self, tmp_path):
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("    wait_grace_s: 1.0", "    wait_grace_s: 9.5", 1)
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="web_search.deep.page_timeout_s"):
            load_config(path)

    def test_a_page_read_that_outlasts_the_assistant_is_rejected(self, tmp_path):
        text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8").replace("    page_timeout_s: 5.0", "    page_timeout_s: 30.0", 1)
        path = tmp_path / "config.yaml"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ConfigError, match="web_search.deep.page_timeout_s"):
            load_config(path)
