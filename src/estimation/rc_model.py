"""The RC model's form: what the model *is*, separate from how it is fitted.

DESIGN.md section 5.2.1 discretises a single thermal capacitance coupled to
ambient into a form that is linear in the parameters:

    T[k+1] = a1*T[k] + a2*T_out[k] + a3*u[k] + a4*o[k]

Linearity is the whole reason recursive least squares applies without any
nonlinear optimisation, and it is why four coefficients with physical
meaning beat a network here (ADR-0002): they are inspectable, they converge
on hours of data rather than months, and an implausible estimate is
detectable *because* each one means something.

Two structural facts turn this from curve-fitting into identification, and
both live here rather than in the fitting code, because they are properties
of the model rather than of the algorithm:

* **Steady-state consistency.** With no cooling and no occupancy the model
  settles to ambient only if ``a1 + a2 == 1``. Drift away from that is
  itself a diagnostic signal, published on every coefficient update.
* **Sign-constrained authority.** ``a3`` must stay negative while cooling.
  An estimate crossing zero says the air conditioner heats the room, which
  is far likelier to be an actuator fault than a thermal property (FR-24).

This module holds no state. The parameter vector and its covariance belong
to the estimator (section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.config import Bounds

#: Coefficients in the model, in the order a1, a2, a3, a4.
COEFFICIENT_COUNT = 4

#: Coefficient names, for logging and for indexing the plausibility box.
COEFFICIENT_NAMES = ("a1", "a2", "a3", "a4")

#: Sum a1 + a2 must approach for the model to settle at ambient.
STEADY_STATE_SUM = 1.0


@dataclass(frozen=True)
class Regressor:
    """One row of the identification problem: phi[k].

    Named fields rather than a bare array, because the order is a contract
    shared with the coefficient vector and a transposition would fit
    beautifully while meaning nothing.
    """

    indoor_c: float
    outdoor_c: float
    command: float
    occupancy: float

    def as_array(self) -> np.ndarray:
        """phi[k] = [T[k], T_out[k], u[k], o[k]]."""
        return np.array(
            [self.indoor_c, self.outdoor_c, self.command, self.occupancy],
            dtype=float,
        )

    @property
    def magnitude(self) -> float:
        """Euclidean norm, used to judge whether the input excites the model."""
        return float(np.linalg.norm(self.as_array()))


def predict(theta: np.ndarray, regressor: Regressor) -> float:
    """One-step-ahead prediction, theta transpose times phi (FR-05)."""
    return float(np.dot(theta, regressor.as_array()))


def steady_state_residual(theta: np.ndarray) -> float:
    """|a1 + a2 - 1|.

    Zero when the model settles at ambient with no cooling and nobody in the
    room. Growing values mean the identified model no longer respects that,
    which is a diagnostic in its own right (section 5.2.1).
    """
    return float(abs(theta[0] + theta[1] - STEADY_STATE_SUM))


def project(theta: np.ndarray, box: tuple[Bounds, ...]) -> np.ndarray:
    """Clamp each coefficient into its plausible range (section 5.1).

    Returns a new array; the caller decides what to do with it. This is used
    as the *test* for plausibility -- a projection that changes anything
    means the estimate left the box.
    """
    if len(box) != COEFFICIENT_COUNT:
        raise ValueError(
            f"expected {COEFFICIENT_COUNT} bounds, got {len(box)}"
        )
    return np.array(
        [bounds.clamp(float(value)) for value, bounds in zip(theta, box)],
        dtype=float,
    )


def is_plausible(theta: np.ndarray, box: tuple[Bounds, ...]) -> bool:
    """Whether every coefficient already lies inside its range."""
    return bool(np.allclose(theta, project(theta, box)))


def implausible_coefficients(
    theta: np.ndarray, box: tuple[Bounds, ...]
) -> tuple[str, ...]:
    """Names of the coefficients outside their range, for the rejection log.

    FR-06 requires a rejection to be logged with a reason. "a3 left its box"
    is a reason; "the update was rejected" is not.
    """
    return tuple(
        name
        for name, value, bounds in zip(COEFFICIENT_NAMES, theta, box)
        if not bounds.contains(float(value))
    )
