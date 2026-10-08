"""Unit tests for what the experiment harness records.

A comparison is only as fair as what it scores against. These pin down that
both systems are judged against the setpoint they were actually tracking, and
that the baseline is given the same setpoint schedule as the real system
(section 8.2).
"""

from datetime import datetime
from pathlib import Path

import pytest

from eval import harness
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.localtime import epoch_of

#: Long enough for both systems to log a handful of samples.
SHORT_RUN_S = 30.0


@pytest.fixture(name="config")
def _config(tmp_path):
    return harness.for_experiment(load_config(Path("config/default.yaml")), tmp_path)


def _clock_at(config, hour: int) -> SimClock:
    local = datetime(2026, 10, 1, hour, 0)
    return SimClock(start_epoch_s=epoch_of(local, config.site.utc_offset_h))


def _peak_setpoint(config) -> float:
    return config.controller.default_setpoint_c + config.tariff.peak_offset_c


@pytest.mark.parametrize("build", [harness.full_system, harness.baseline_system])
def test_peak_samples_are_scored_against_the_shifted_setpoint(config, build):
    system = build(config, _clock_at(config, config.tariff.peak_start_hour))
    system.run_for(SHORT_RUN_S)
    assert set(system.log.setpoints_c) == {_peak_setpoint(config)}


@pytest.mark.parametrize("build", [harness.full_system, harness.baseline_system])
def test_off_peak_samples_are_scored_against_the_setpoint_as_set(config, build):
    system = build(config, _clock_at(config, config.tariff.peak_start_hour - 6))
    system.run_for(SHORT_RUN_S)
    assert set(system.log.setpoints_c) == {config.controller.default_setpoint_c}


def test_the_baseline_tracks_the_shift_it_is_scored_against(config):
    system = harness.baseline_system(config, _clock_at(config, config.tariff.peak_start_hour))
    system.run_for(SHORT_RUN_S)
    assert system.thermostat.setpoint_c == _peak_setpoint(config)
