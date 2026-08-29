"""Unit tests for the RC model's form.

These assert what the model *is*, independently of how it is fitted: the
regressor ordering, the steady-state identity, and the plausibility box.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.estimation.rc_model import (
    COEFFICIENT_COUNT,
    COEFFICIENT_NAMES,
    Regressor,
    implausible_coefficients,
    is_plausible,
    predict,
    project,
    steady_state_residual,
)

PLAUSIBLE = np.array([0.98, 0.02, -0.05, 0.01])


@pytest.fixture(name="box")
def _box():
    return load_config(Path("config/default.yaml")).estimator.coefficient_bounds


class TestRegressor:
    def test_orders_the_entries_as_the_document_specifies(self):
        """phi[k] = [T[k], T_out[k], u[k], o[k]] (section 5.2.2)."""
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=1.0, occupancy=0.0)
        assert list(phi.as_array()) == [27.0, 31.0, 1.0, 0.0]

    def test_has_one_entry_per_coefficient(self):
        phi = Regressor(indoor_c=1.0, outdoor_c=2.0, command=0.0, occupancy=0.0)
        assert len(phi.as_array()) == COEFFICIENT_COUNT

    def test_magnitude_is_the_euclidean_norm(self):
        phi = Regressor(indoor_c=3.0, outdoor_c=4.0, command=0.0, occupancy=0.0)
        assert phi.magnitude == pytest.approx(5.0)

    def test_a_regressor_is_immutable(self):
        phi = Regressor(indoor_c=1.0, outdoor_c=2.0, command=0.0, occupancy=0.0)
        with pytest.raises(Exception):
            phi.indoor_c = 9.0


class TestPrediction:
    def test_is_the_inner_product_of_theta_and_phi(self):
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=1.0, occupancy=1.0)
        expected = 0.98 * 27.0 + 0.02 * 31.0 - 0.05 * 1.0 + 0.01 * 1.0
        assert predict(PLAUSIBLE, phi) == pytest.approx(expected)

    def test_cooling_lowers_the_prediction(self):
        warm = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        cooling = Regressor(indoor_c=27.0, outdoor_c=31.0, command=1.0, occupancy=0.0)
        assert predict(PLAUSIBLE, cooling) < predict(PLAUSIBLE, warm)

    def test_occupancy_raises_the_prediction(self):
        empty = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        occupied = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=1.0)
        assert predict(PLAUSIBLE, occupied) > predict(PLAUSIBLE, empty)


class TestSteadyStateConsistency:
    def test_is_zero_when_a1_plus_a2_is_one(self):
        assert steady_state_residual(np.array([0.98, 0.02, -0.05, 0.01])) == 0.0

    def test_grows_with_departure_from_the_identity(self):
        assert steady_state_residual(np.array([0.9, 0.02, 0.0, 0.0])) == pytest.approx(
            0.08
        )

    def test_is_unsigned(self):
        """Departure in either direction is equally diagnostic."""
        below = steady_state_residual(np.array([0.90, 0.02, 0.0, 0.0]))
        above = steady_state_residual(np.array([0.98, 0.10, 0.0, 0.0]))
        assert below > 0.0 and above > 0.0

    def test_an_undriven_model_settles_at_ambient_when_consistent(self):
        """The identity is what makes that true (section 5.2.1)."""
        theta = np.array([0.98, 0.02, -0.05, 0.01])
        temperature = 20.0
        for _ in range(2000):
            phi = Regressor(
                indoor_c=temperature, outdoor_c=31.0, command=0.0, occupancy=0.0
            )
            temperature = predict(theta, phi)
        assert temperature == pytest.approx(31.0, abs=0.01)


class TestPlausibilityBox:
    def test_an_estimate_inside_the_box_is_plausible(self, box):
        assert is_plausible(PLAUSIBLE, box)

    def test_projection_leaves_a_plausible_estimate_alone(self, box):
        assert np.allclose(project(PLAUSIBLE, box), PLAUSIBLE)

    def test_a_positive_a3_is_implausible(self, box):
        """It says the air conditioner heats the room (FR-24)."""
        assert not is_plausible(np.array([0.98, 0.02, 0.9, 0.01]), box)

    def test_projection_pulls_a_positive_a3_back_to_zero(self, box):
        projected = project(np.array([0.98, 0.02, 0.9, 0.01]), box)
        assert projected[2] == 0.0

    @pytest.mark.parametrize(
        ("theta", "expected"),
        [
            (np.array([1.4, 0.02, -0.05, 0.01]), ("a1",)),
            (np.array([0.98, -0.3, -0.05, 0.01]), ("a2",)),
            (np.array([0.98, 0.02, 0.9, 0.01]), ("a3",)),
            (np.array([0.98, 0.02, -0.05, -0.4]), ("a4",)),
            (np.array([1.4, -0.3, -0.05, 0.01]), ("a1", "a2")),
        ],
    )
    def test_names_the_coefficients_that_left_their_range(self, box, theta, expected):
        """FR-06 logs a rejection; 'a3 left its box' is a reason, 'rejected'
        is not."""
        assert implausible_coefficients(theta, box) == expected

    def test_a_plausible_estimate_names_nothing(self, box):
        assert implausible_coefficients(PLAUSIBLE, box) == ()

    def test_the_names_line_up_with_the_regressor_order(self):
        assert COEFFICIENT_NAMES == ("a1", "a2", "a3", "a4")

    def test_a_box_of_the_wrong_size_is_refused(self, box):
        with pytest.raises(ValueError):
            project(PLAUSIBLE, box[:2])
