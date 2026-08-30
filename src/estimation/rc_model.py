"""The RC model's form: what the model *is*, separate from how it is fitted.

DESIGN.md section 5.2.1 discretises a single thermal capacitance coupled to
ambient into a form that is linear in the parameters:

    T[k+1] = a1*T[k] + a2*T_out[k] + a3*u[k] + a4*o[k]

That is what the model *means*. It is not the form it is identified in, and
the difference matters enough that section 5.2.1 spells it out.

``a1`` is close to 1 -- 0.998 for a 35 min room sampled at 5 s, because
almost nothing changes in five seconds. Fitting it needs the small part that
does change, about 0.019 C per step, and the sensor is accurate to 0.15 C.
Worse than imprecision: ``T[k]`` is the regressor *and* sits inside the
measurement being predicted, and a regressor carrying measurement error has
its coefficient pulled toward zero. Measured, ``a1`` settles near 0.78 and
``a2`` absorbs the difference.

Substituting the steady-state identity ``a2 = 1 - a1`` removes the problem
rather than mitigating it:

    T[k+1] - T[k] = a2*(T_out[k] - T[k]) + a3*u[k] + a4*o[k]

Three parameters are fitted; ``a1`` is recovered as ``1 - a2``. Same physics,
same coefficients, still linear in the parameters, still ordinary RLS. The
fit is now asked for a small number instead of one near 1, and small
coefficients survive a noisy regressor. Steady-state consistency becomes
structural: ``a1 + a2 == 1`` exactly, by construction.

Two consequences, recorded because they are easy to trip over:

* ``steady_state_residual`` is identically zero and is no longer a
  diagnostic. What it used to catch now shows up as ``a2`` leaving its
  plausible range, so the box does that job alone.
* ``a3`` must stay negative while cooling. An estimate crossing zero says
  the air conditioner heats the room, which is far likelier to be an
  actuator fault than a thermal property (FR-24).

This module holds no state. The parameter vector and its covariance belong
to the estimator (section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.config import Bounds

#: Coefficients the model has, in the order a1, a2, a3, a4.
COEFFICIENT_COUNT = 4

#: Coefficients actually fitted: a2, a3, a4. a1 is derived.
IDENTIFIED_COUNT = 3

COEFFICIENT_NAMES = ("a1", "a2", "a3", "a4")
IDENTIFIED_NAMES = ("a2", "a3", "a4")

#: Sum a1 + a2 must equal for the model to settle at ambient. Enforced by
#: construction rather than checked.
STEADY_STATE_SUM = 1.0

#: Index of a2 within the identified vector.
_AMBIENT_COUPLING = 0


@dataclass(frozen=True)
class Regressor:
    """One row of the identification problem.

    Named by the physical quantities it is built from rather than by the
    vector it becomes, so a caller assembles it the same way whatever form
    the identification uses. The ordering of :meth:`as_array` is a contract
    shared with the parameter vector, and a transposition would fit
    beautifully while meaning nothing.
    """

    indoor_c: float
    outdoor_c: float
    command: float
    occupancy: float

    @property
    def ambient_gap_c(self) -> float:
        """T_out[k] - T[k]: what actually drives the room toward ambient."""
        return self.outdoor_c - self.indoor_c

    def as_array(self) -> np.ndarray:
        """phi[k] = [T_out[k] - T[k], u[k], o[k]] (section 5.2.2)."""
        return np.array(
            [self.ambient_gap_c, self.command, self.occupancy], dtype=float
        )

    @property
    def magnitude(self) -> float:
        """Euclidean norm, used to judge whether the input excites the model."""
        return float(np.linalg.norm(self.as_array()))


def predict_change(theta: np.ndarray, regressor: Regressor) -> float:
    """The temperature change this model expects over one step."""
    return float(np.dot(theta, regressor.as_array()))


def predict(theta: np.ndarray, regressor: Regressor) -> float:
    """One-step-ahead temperature prediction (FR-05).

    The identification fits a change; what every consumer wants is the
    temperature, so the current reading is added back here rather than in
    each caller.
    """
    return regressor.indoor_c + predict_change(theta, regressor)


def derive_thermal_inertia(theta: np.ndarray) -> float:
    """a1 = 1 - a2. Recovered, never fitted."""
    return STEADY_STATE_SUM - float(theta[_AMBIENT_COUPLING])


def full_coefficients(theta: np.ndarray) -> np.ndarray:
    """The four coefficients the model is described by, from the three fitted.

    This is the vector to publish, to check against the plausibility box, and
    to compare against ground truth. The identified vector is an internal
    detail of how they were obtained.
    """
    if theta.shape != (IDENTIFIED_COUNT,):
        raise ValueError(f"expected {IDENTIFIED_COUNT} identified parameters")
    return np.array(
        [derive_thermal_inertia(theta), theta[0], theta[1], theta[2]], dtype=float
    )


def steady_state_residual(theta: np.ndarray) -> float:
    """|a1 + a2 - 1|.

    Identically zero since the identification form makes the identity
    structural. Retained because section 6.2 publishes it, and because a
    non-zero value would mean a1 was derived wrongly rather than that the
    model had drifted.
    """
    coefficients = full_coefficients(theta)
    return float(abs(coefficients[0] + coefficients[1] - STEADY_STATE_SUM))


def project(coefficients: np.ndarray, box: tuple[Bounds, ...]) -> np.ndarray:
    """Clamp each of the four coefficients into its plausible range.

    Used as the *test* for plausibility: a projection that changes anything
    means the estimate left the box.
    """
    if len(box) != COEFFICIENT_COUNT:
        raise ValueError(f"expected {COEFFICIENT_COUNT} bounds, got {len(box)}")
    if coefficients.shape != (COEFFICIENT_COUNT,):
        raise ValueError(f"expected {COEFFICIENT_COUNT} coefficients")
    return np.array(
        [bounds.clamp(float(value)) for value, bounds in zip(coefficients, box)],
        dtype=float,
    )


def is_plausible(theta: np.ndarray, box: tuple[Bounds, ...]) -> bool:
    """Whether the fitted vector implies four coefficients inside the box.

    Takes the *identified* vector and checks the derived one, so a1 is
    included: an a2 that drags a1 out of range is implausible even though a2
    itself might not be.
    """
    coefficients = full_coefficients(theta)
    return bool(np.allclose(coefficients, project(coefficients, box)))


def implausible_coefficients(
    theta: np.ndarray, box: tuple[Bounds, ...]
) -> tuple[str, ...]:
    """Names of the coefficients outside their range, for the rejection log.

    FR-06 requires a rejection to be logged with a reason. "a3 left its box"
    is a reason; "the update was rejected" is not.
    """
    coefficients = full_coefficients(theta)
    return tuple(
        name
        for name, value, bounds in zip(COEFFICIENT_NAMES, coefficients, box)
        if not bounds.contains(float(value))
    )
