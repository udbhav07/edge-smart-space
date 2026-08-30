"""Unit tests for the RC model's form.

Two forms are in play and the tests keep them straight. The model *means*
``T[k+1] = a1*T + a2*T_out + a3*u + a4*o``; it is *identified* as a change
against ``T_out - T`` with three parameters, and ``a1`` is recovered as
``1 - a2`` (section 5.2.1). Everything published, checked, or compared uses
the four; only the fit uses the three.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.estimation.rc_model import (
    COEFFICIENT_COUNT,
    COEFFICIENT_NAMES,
    IDENTIFIED_COUNT,
    Regressor,
    derive_thermal_inertia,
    full_coefficients,
    implausible_coefficients,
    is_plausible,
    predict,
    predict_change,
    steady_state_residual,
)

#: [a2, a3, a4]; implies a1 = 0.98.
FITTED = np.array([0.02, -0.05, 0.01])


@pytest.fixture(name="box")
def _box():
    return load_config(Path("config/default.yaml")).estimator.coefficient_bounds


class TestRegressor:
    def test_orders_the_entries_as_the_document_specifies(self):
        """phi[k] = [T_out[k] - T[k], u[k], o[k]] (section 5.2.2)."""
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=1.0, occupancy=0.0)
        assert list(phi.as_array()) == [4.0, 1.0, 0.0]

    def test_has_one_entry_per_identified_parameter(self):
        phi = Regressor(indoor_c=1.0, outdoor_c=2.0, command=0.0, occupancy=0.0)
        assert len(phi.as_array()) == IDENTIFIED_COUNT

    def test_the_ambient_gap_is_what_drives_the_room(self):
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        assert phi.ambient_gap_c == 4.0

    def test_the_gap_is_negative_when_the_room_is_warmer_than_outside(self):
        phi = Regressor(indoor_c=31.0, outdoor_c=27.0, command=0.0, occupancy=0.0)
        assert phi.ambient_gap_c == -4.0

    def test_magnitude_is_the_euclidean_norm(self):
        phi = Regressor(indoor_c=0.0, outdoor_c=3.0, command=4.0, occupancy=0.0)
        assert phi.magnitude == pytest.approx(5.0)

    def test_a_regressor_is_immutable(self):
        phi = Regressor(indoor_c=1.0, outdoor_c=2.0, command=0.0, occupancy=0.0)
        with pytest.raises(Exception):
            phi.indoor_c = 9.0

    def test_the_caller_still_names_the_physical_quantities(self):
        """The identification form is internal: a caller assembles the same
        four physical inputs whichever form is fitted."""
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=1.0, occupancy=1.0)
        assert (phi.indoor_c, phi.outdoor_c, phi.command, phi.occupancy) == (
            27.0,
            31.0,
            1.0,
            1.0,
        )


class TestDerivation:
    def test_thermal_inertia_is_one_minus_ambient_coupling(self):
        assert derive_thermal_inertia(FITTED) == pytest.approx(0.98)

    def test_the_four_coefficients_come_back_in_order(self):
        assert list(full_coefficients(FITTED)) == pytest.approx(
            [0.98, 0.02, -0.05, 0.01]
        )

    def test_there_are_four_of_them(self):
        assert len(full_coefficients(FITTED)) == COEFFICIENT_COUNT

    def test_a_wrongly_sized_fit_is_refused(self):
        with pytest.raises(ValueError):
            full_coefficients(np.array([0.02, -0.05]))


class TestSteadyStateConsistency:
    def test_the_identity_holds_by_construction(self):
        """a1 + a2 = 1 exactly, because a1 is derived rather than fitted."""
        assert steady_state_residual(FITTED) == 0.0

    @pytest.mark.parametrize("ambient_coupling", [0.0, 0.002, 0.2, 0.9])
    def test_it_holds_for_any_fit_at_all(self, ambient_coupling):
        theta = np.array([ambient_coupling, -0.05, 0.01])
        assert steady_state_residual(theta) == pytest.approx(0.0, abs=1e-12)

    def test_an_undriven_model_settles_at_ambient(self):
        """Which is what the identity guarantees (section 5.2.1)."""
        temperature = 20.0
        for _ in range(4000):
            phi = Regressor(
                indoor_c=temperature, outdoor_c=31.0, command=0.0, occupancy=0.0
            )
            temperature = predict(FITTED, phi)
        assert temperature == pytest.approx(31.0, abs=0.01)


class TestPrediction:
    def test_the_change_is_the_inner_product(self):
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=1.0, occupancy=1.0)
        expected = 0.02 * 4.0 - 0.05 * 1.0 + 0.01 * 1.0
        assert predict_change(FITTED, phi) == pytest.approx(expected)

    def test_the_prediction_adds_the_change_to_the_current_reading(self):
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        assert predict(FITTED, phi) == pytest.approx(27.0 + 0.02 * 4.0)

    def test_a_warmer_outside_pushes_the_prediction_up(self):
        cool = Regressor(indoor_c=27.0, outdoor_c=20.0, command=0.0, occupancy=0.0)
        warm = Regressor(indoor_c=27.0, outdoor_c=35.0, command=0.0, occupancy=0.0)
        assert predict(FITTED, warm) > predict(FITTED, cool)

    def test_cooling_lowers_the_prediction(self):
        idle = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        cooling = Regressor(indoor_c=27.0, outdoor_c=31.0, command=1.0, occupancy=0.0)
        assert predict(FITTED, cooling) < predict(FITTED, idle)

    def test_occupancy_raises_the_prediction(self):
        empty = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        occupied = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=1.0)
        assert predict(FITTED, occupied) > predict(FITTED, empty)

    def test_a_room_already_at_ambient_is_predicted_to_stay(self):
        phi = Regressor(indoor_c=31.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        assert predict(FITTED, phi) == pytest.approx(31.0)


class TestPlausibilityBox:
    def test_a_sensible_fit_is_plausible(self, box):
        assert is_plausible(FITTED, box)

    def test_a_positive_a3_is_implausible(self, box):
        """It says the air conditioner heats the room (FR-24)."""
        assert not is_plausible(np.array([0.02, 0.9, 0.01]), box)

    def test_an_a2_that_drags_a1_out_of_range_is_implausible(self, box):
        """a1 is derived, so a2 can be in range while what it implies is not."""
        assert not is_plausible(np.array([1.6, -0.05, 0.01]), box)

    @pytest.mark.parametrize(
        ("theta", "expected"),
        [
            (np.array([0.9, 0.9, 0.01]), ("a3",)),
            (np.array([0.02, -0.05, -0.9]), ("a4",)),
            (np.array([1.6, -0.05, 0.01]), ("a1", "a2")),
        ],
    )
    def test_names_the_coefficients_that_left_their_range(self, box, theta, expected):
        """FR-06 logs a rejection; 'a3 left its box' is a reason."""
        assert implausible_coefficients(theta, box) == expected

    def test_a_plausible_fit_names_nothing(self, box):
        assert implausible_coefficients(FITTED, box) == ()

    def test_small_negative_occupancy_gain_is_tolerated(self, box):
        """Section 5.2.1: a4 sits far below the noise floor and rejecting
        every dip below zero would raise MODEL_DIVERGENCE constantly."""
        assert is_plausible(np.array([0.02, -0.05, -0.01]), box)

    def test_a_large_negative_occupancy_gain_is_still_refused(self, box):
        assert not is_plausible(np.array([0.02, -0.05, -0.5]), box)

    def test_the_names_line_up_with_the_coefficient_order(self):
        assert COEFFICIENT_NAMES == ("a1", "a2", "a3", "a4")

    def test_a_box_of_the_wrong_size_is_refused(self, box):
        from src.estimation.rc_model import project

        with pytest.raises(ValueError):
            project(full_coefficients(FITTED), box[:2])
