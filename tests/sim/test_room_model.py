"""Unit tests for the ground-truth thermal plant.

These assert physics, not agreement with the estimator. The plant must never
be tuned to match the model: DESIGN.md section 5.10 requires them to be
independently parameterised.
"""

import math
from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import RoomConfig, load_config
from sim.room_model import RoomModel

OUTDOOR_C = 31.0
LONG_RUN_S = 100_000.0
STEP_S = 5.0
NO_COOLING = 0.0
FULL_COOLING = 1.0
SETTLING_TOLERANCE_C = 0.01


@pytest.fixture(name="room_config")
def _room_config() -> RoomConfig:
    return load_config(Path("config/default.yaml")).sim.room


@pytest.fixture(name="still_config")
def _still_config(room_config: RoomConfig) -> RoomConfig:
    """The same room with the solar disturbance switched off.

    Used only where a test needs a closed-form equilibrium to compare
    against. The default configuration keeps the disturbance on.
    """
    return room_config.model_copy(update={"solar_gain_amplitude_w": 0.0})


def _room(config: RoomConfig) -> tuple[RoomModel, SimClock]:
    clock = SimClock()
    return RoomModel(config, clock), clock


def _settle(room: RoomModel, clock: SimClock, cooling: float, occupied: bool) -> float:
    steps = int(LONG_RUN_S / STEP_S)
    for _ in range(steps):
        clock.advance(STEP_S)
        room.step(STEP_S, cooling, occupied, OUTDOOR_C)
    return room.temperature_c


class TestConstruction:
    def test_starts_at_the_configured_temperature(self, room_config):
        room, _ = _room(room_config)
        assert room.temperature_c == room_config.initial_temperature_c

    def test_time_constant_is_resistance_times_capacitance(self, room_config):
        room, _ = _room(room_config)
        expected = (
            room_config.thermal_resistance_k_per_w
            * room_config.thermal_capacitance_j_per_k
        )
        assert room.time_constant_s == expected


class TestEquilibrium:
    def test_an_undriven_room_settles_at_ambient(self, still_config):
        """With no cooling, occupancy or disturbance, T must reach T_out."""
        room, clock = _room(still_config)
        assert _settle(room, clock, NO_COOLING, occupied=False) == pytest.approx(
            OUTDOOR_C, abs=SETTLING_TOLERANCE_C
        )

    def test_full_cooling_settles_below_ambient_by_r_times_power(self, still_config):
        room, clock = _room(still_config)
        expected = OUTDOOR_C - (
            still_config.thermal_resistance_k_per_w * still_config.cooling_power_w
        )
        assert _settle(room, clock, FULL_COOLING, occupied=True) == pytest.approx(
            expected
            + still_config.thermal_resistance_k_per_w * still_config.occupant_gain_w,
            abs=SETTLING_TOLERANCE_C,
        )

    def test_occupancy_raises_the_equilibrium(self, still_config):
        room, clock = _room(still_config)
        expected = OUTDOOR_C + (
            still_config.thermal_resistance_k_per_w * still_config.occupant_gain_w
        )
        assert _settle(room, clock, NO_COOLING, occupied=True) == pytest.approx(
            expected, abs=SETTLING_TOLERANCE_C
        )


class TestDynamics:
    def test_cooling_lowers_the_temperature(self, room_config):
        room, clock = _room(room_config)
        before = room.temperature_c
        clock.advance(STEP_S)
        assert room.step(STEP_S, FULL_COOLING, False, OUTDOOR_C) < before

    def test_a_hot_ambient_raises_a_cool_room(self, room_config):
        cool_start = room_config.model_copy(update={"initial_temperature_c": 20.0})
        room, clock = _room(cool_start)
        clock.advance(STEP_S)
        assert room.step(STEP_S, NO_COOLING, False, OUTDOOR_C) > 20.0

    def test_one_time_constant_closes_the_expected_fraction_of_the_gap(
        self, still_config
    ):
        """After tau seconds a first-order lag has closed 1 - 1/e of the gap."""
        room, clock = _room(still_config)
        start = room.temperature_c
        tau = room.time_constant_s
        clock.advance(tau)
        room.step(tau, NO_COOLING, False, OUTDOOR_C)
        closed_fraction = (room.temperature_c - start) / (OUTDOOR_C - start)
        assert closed_fraction == pytest.approx(1.0 - 1.0 / math.e, abs=1e-6)

    def test_a_large_step_stays_stable_rather_than_diverging(self, still_config):
        """Exact integration, so a step far longer than tau is still bounded."""
        room, clock = _room(still_config)
        clock.advance(LONG_RUN_S)
        result = room.step(LONG_RUN_S, NO_COOLING, False, OUTDOOR_C)
        assert result == pytest.approx(OUTDOOR_C, abs=SETTLING_TOLERANCE_C)


class TestSolarDisturbance:
    def test_the_disturbance_is_present_by_default(self, room_config):
        room, clock = _room(room_config)
        clock.advance(room_config.solar_gain_period_s / 4.0)
        assert room.solar_gain_w() > 0.0

    def test_the_disturbance_never_cools(self, room_config):
        """Half-wave rectified: a daylight cycle adds heat or nothing."""
        room, clock = _room(room_config)
        samples = 200
        for _ in range(samples):
            clock.advance(room_config.solar_gain_period_s / samples)
            assert room.solar_gain_w() >= 0.0

    def test_the_disturbance_peaks_at_the_configured_amplitude(self, room_config):
        room, clock = _room(room_config)
        clock.advance(room_config.solar_gain_period_s / 4.0)
        assert room.solar_gain_w() == pytest.approx(
            room_config.solar_gain_amplitude_w
        )

    def test_the_disturbance_perturbs_the_equilibrium(self, room_config):
        """A converged estimator still cannot reach zero residual (R-04)."""
        room, clock = _room(room_config)
        settled = _settle(room, clock, NO_COOLING, occupied=False)
        assert settled != pytest.approx(OUTDOOR_C, abs=SETTLING_TOLERANCE_C)


class TestInputValidation:
    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_a_non_positive_duration_is_rejected(self, room_config, duration):
        room, _ = _room(room_config)
        with pytest.raises(ValueError):
            room.step(duration, NO_COOLING, False, OUTDOOR_C)

    @pytest.mark.parametrize("command", [-0.01, 1.01])
    def test_a_command_outside_the_normalised_range_is_rejected(
        self, room_config, command
    ):
        room, _ = _room(room_config)
        with pytest.raises(ValueError):
            room.step(STEP_S, command, False, OUTDOOR_C)

    @pytest.mark.parametrize("command", [0.0, 0.5, 1.0])
    def test_the_normalised_range_is_inclusive(self, room_config, command):
        room, _ = _room(room_config)
        assert isinstance(room.step(STEP_S, command, False, OUTDOOR_C), float)


class TestIndependenceFromTheEstimator:
    def test_the_plant_is_parameterised_physically_not_as_coefficients(
        self, room_config
    ):
        """Section 5.10: the plant must not share the estimator's parameters."""
        fields = set(RoomConfig.model_fields)
        assert not fields & {"a1", "a2", "a3", "a4", "initial_theta"}
        assert "thermal_resistance_k_per_w" in fields
        assert "thermal_capacitance_j_per_k" in fields
