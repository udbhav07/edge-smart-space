"""Recursive least squares with the safeguards that make it survive a room.

The update itself is four lines (DESIGN.md section 5.2.2). Everything else
here is section 5.2.3, and that is the point: the difference between "we ran
RLS" and "we ran RLS on a real system for twelve weeks" is entirely in what
happens when the data stops cooperating.

Five things can go wrong, and each has a specific answer:

* **Covariance windup.** With the air conditioner off overnight the
  regressor barely moves, so ``P`` grows without bound and the next real
  excitation produces a wild jump. Bounded by trace.
* **Symmetry loss.** ``P`` is a covariance and must stay symmetric;
  floating-point drift breaks that silently. Re-symmetrised every step.
* **Insufficient excitation.** A constant regressor carries no information.
  Updating on it only degrades ``P``, so the update is skipped.
* **Implausible parameters.** An estimate outside the physical box in
  section 5.2.1 is rejected, not accepted-and-clamped. Divergence is a
  separate judgement: a sustained rate of rejections over a window, not a
  run of consecutive ones. a2 and a4 both have true values near a box edge,
  so short runs of rejections are ordinary noise (FR-06).
* **Faulted inputs.** Adaptation freezes entirely while any sensor feeding
  the regressor is faulted (FR-29). Prediction continues; learning does not.
  Never adapt to bad data.

Three parameters are identified -- ``a2``, ``a3``, ``a4`` -- and ``a1`` is
recovered as ``1 - a2`` (section 5.2.2). That is what makes steady-state
consistency structural rather than something to check afterwards, and it is
why the plausibility test runs on the derived four rather than the fitted
three: an ``a2`` that drags ``a1`` out of range is implausible even when
``a2`` itself is not.

The estimator owns ``theta`` and ``P`` privately and hands out immutable
snapshots. That is the resolution of the one place where the coding standard
(prefer immutable state) and the algorithm (inherently recursive) pull
against each other.
"""

from __future__ import annotations

import logging
import statistics
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.common.clock import Clock
from src.common.config import EstimatorConfig
from src.common.schemas import AdaptationState, Coefficients
from src.estimation.rc_model import (
    IDENTIFIED_COUNT,
    Regressor,
    full_coefficients,
    implausible_coefficients,
    is_plausible,
    predict,
    project,
    steady_state_residual,
)

LOGGER = logging.getLogger(__name__)

#: Residual reported before enough samples exist to estimate a spread.
_NO_SPREAD = 0.0

#: Confidence when the covariance has reached its bound: the estimate is as
#: unsupported as this component is willing to represent.
_NO_CONFIDENCE = 0.0
_FULL_CONFIDENCE = 1.0

#: Rejections caused by nothing but a4 do not count toward divergence.
#: Occupancy gain is about 0.0008 for one person, far below the noise floor
#: (section 5.2.1), so noise pushes it across its bound routinely. That is
#: worth refusing the update over; it is not evidence the model is wrong.
_NON_DIAGNOSTIC_REJECTIONS = frozenset({("a4",)})


class UpdateStatus(str, Enum):
    """What happened to the parameter vector on one sample."""

    APPLIED = "APPLIED"
    REJECTED = "REJECTED"
    FROZEN = "FROZEN"
    INSUFFICIENT_EXCITATION = "INSUFFICIENT_EXCITATION"


@dataclass(frozen=True)
class UpdateResult:
    """The outcome of one step, including the parts nothing else can see.

    ``prediction_c`` and ``residual_c`` are produced whatever the status:
    prediction continues while adaptation is frozen, which is what makes
    DEGRADED_SENSOR control possible at all (FR-27).
    """

    status: UpdateStatus
    prediction_c: float
    residual_c: float
    consecutive_rejections: int
    diverged: bool
    rejected_coefficients: tuple[str, ...] = ()

    @property
    def adapted(self) -> bool:
        return self.status is UpdateStatus.APPLIED


class ThermalEstimator:
    """Identifies the four RC coefficients online, from operating data (FR-04)."""

    def __init__(self, config: EstimatorConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._theta = np.array(config.initial_theta, dtype=float)
        self._covariance = np.eye(IDENTIFIED_COUNT) * config.initial_covariance
        self._frozen = False
        self._consecutive_rejections = 0
        self._samples_since_reset = 0
        self._magnitudes: deque[float] = deque(
            maxlen=config.excitation_window_samples
        )
        self._residuals: deque[float] = deque(
            maxlen=config.residual_sigma_window_samples
        )
        self._recent_rejections: deque[bool] = deque(
            maxlen=config.divergence_window_samples
        )

    # --- state a reader may see ---------------------------------------

    @property
    def theta(self) -> np.ndarray:
        """The identified vector [a2, a3, a4], as a copy.

        A copy because the caller must not be able to move the estimate.
        """
        return self._theta.copy()

    @property
    def coefficients(self) -> np.ndarray:
        """The four coefficients the model is described by, a1 derived.

        This is what to compare against ground truth and against the
        plausibility box. The three fitted parameters are an internal detail
        of how they were obtained.
        """
        return full_coefficients(self._theta)

    @property
    def covariance(self) -> np.ndarray:
        """A copy of P, for persistence. The caller must not be able to move it."""
        return self._covariance.copy()

    @property
    def trace(self) -> float:
        """trace(P): how unsupported the current estimate is."""
        return float(np.trace(self._covariance))

    @property
    def adaptation(self) -> AdaptationState:
        return AdaptationState.FROZEN if self._frozen else AdaptationState.ACTIVE

    @property
    def samples_since_reset(self) -> int:
        return self._samples_since_reset

    @property
    def consecutive_rejections(self) -> int:
        return self._consecutive_rejections

    @property
    def residual_sigma(self) -> float:
        """Spread of recent residuals, which D4's CUSUM is scaled against."""
        if len(self._residuals) < 2:
            return _NO_SPREAD
        return float(statistics.pstdev(self._residuals))

    @property
    def model_confidence(self) -> float:
        """Derived from trace(P). Not a probability, and documented as such.

        One at a fully supported estimate, zero once the covariance has run
        to its configured bound.
        """
        ratio = self.trace / self._config.max_covariance_trace
        return max(_NO_CONFIDENCE, min(_FULL_CONFIDENCE, _FULL_CONFIDENCE - ratio))

    # --- adaptation control -------------------------------------------

    def freeze(self) -> None:
        """Stop learning. Prediction continues (FR-29)."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Resume learning."""
        self._frozen = False

    def reset(self) -> None:
        """Return to the configured prior.

        Used when a persisted estimate is too old to trust and when
        MODEL_DIVERGENCE forces a fresh start (section 7.1).
        """
        self._theta = np.array(self._config.initial_theta, dtype=float)
        self._covariance = np.eye(IDENTIFIED_COUNT) * self._config.initial_covariance
        self._consecutive_rejections = 0
        self._samples_since_reset = 0
        self._magnitudes.clear()
        self._residuals.clear()
        self._recent_rejections.clear()

    # --- the model ----------------------------------------------------

    def predict(self, regressor: Regressor) -> float:
        """One-step-ahead prediction (FR-05)."""
        return predict(self._theta, regressor)

    def project(self, coefficients: np.ndarray) -> np.ndarray:
        """Clamp the four coefficients into the plausible box (section 5.1)."""
        return project(coefficients, self._config.coefficient_bounds)

    def update(self, regressor: Regressor, measured_c: float) -> UpdateResult:
        """Fold one observation into the estimate.

        The prediction and residual are computed first and returned whatever
        else happens, because a frozen or skipped update still has to feed
        the detectors and the controller.
        """
        prediction_c = self.predict(regressor)
        residual_c = measured_c - prediction_c
        self._samples_since_reset += 1
        self._magnitudes.append(regressor.magnitude)
        self._residuals.append(residual_c)

        if self._frozen:
            return self._result(UpdateStatus.FROZEN, prediction_c, residual_c)

        if not self._sufficiently_excited():
            return self._result(
                UpdateStatus.INSUFFICIENT_EXCITATION, prediction_c, residual_c
            )

        return self._apply(regressor, residual_c, prediction_c)

    def _apply(
        self, regressor: Regressor, residual_c: float, prediction_c: float
    ) -> UpdateResult:
        previous_theta = self._theta.copy()
        previous_covariance = self._covariance.copy()

        phi = regressor.as_array()
        forgetting = self._config.forgetting_factor
        covariance_phi = self._covariance @ phi
        denominator = forgetting + float(phi @ covariance_phi)
        gain = covariance_phi / denominator

        candidate = self._theta + gain * residual_c
        self._covariance = (
            self._covariance - np.outer(gain, phi) @ self._covariance
        ) / forgetting
        self._symmetrise()
        self._bound_trace()

        if not is_plausible(candidate, self._config.coefficient_bounds):
            return self._reject(
                candidate, previous_theta, previous_covariance, prediction_c, residual_c
            )

        self._theta = candidate
        self._consecutive_rejections = 0
        return self._result(UpdateStatus.APPLIED, prediction_c, residual_c)

    def _reject(
        self,
        candidate: np.ndarray,
        previous_theta: np.ndarray,
        previous_covariance: np.ndarray,
        prediction_c: float,
        residual_c: float,
    ) -> UpdateResult:
        """Discard an implausible update and say which coefficient broke.

        Section 5.2.3's table says to project the estimate back into the box
        and its flowchart says to revert. Reverting is the stricter of the
        two and is what happens here: a projected vector is a point the data
        never actually supported, and adopting it would let a bad update move
        the estimate to the box edge and stay there. The covariance is
        reverted with it, so the discarded step leaves no trace at all.
        """
        offenders = implausible_coefficients(
            candidate, self._config.coefficient_bounds
        )
        self._theta = previous_theta
        self._covariance = previous_covariance
        self._consecutive_rejections += 1

        LOGGER.warning(
            "rejected implausible estimate: %s outside its range (%d in a row)",
            ", ".join(offenders),
            self._consecutive_rejections,
        )
        return self._result(
            UpdateStatus.REJECTED,
            prediction_c,
            residual_c,
            rejected_coefficients=offenders,
        )

    # --- safeguards ---------------------------------------------------

    def _sufficiently_excited(self) -> bool:
        """Whether the regressor has moved enough to carry information.

        A constant input identifies nothing; updating on it only degrades the
        covariance. The window has to fill before this can be judged, and
        until then the update proceeds: refusing to learn at startup would
        leave the first minutes of control on the prior alone.
        """
        if len(self._magnitudes) < self._config.excitation_window_samples:
            return True
        spread = max(self._magnitudes) - min(self._magnitudes)
        return spread >= self._config.min_excitation

    def _symmetrise(self) -> None:
        """P must stay symmetric; floating point erodes that quietly."""
        self._covariance = (self._covariance + self._covariance.T) / 2.0

    def _bound_trace(self) -> None:
        """Cap trace(P) so a quiet night cannot wind the covariance up."""
        trace = float(np.trace(self._covariance))
        limit = self._config.max_covariance_trace
        if trace > limit:
            self._covariance *= limit / trace

    def _record_outcome(
        self, status: UpdateStatus, rejected_coefficients: tuple[str, ...]
    ) -> None:
        """Remember whether this update told us anything about divergence.

        Frozen and skipped updates say nothing either way and are left out
        entirely, so a quiet night cannot dilute the window into silence.
        """
        if status in (UpdateStatus.FROZEN, UpdateStatus.INSUFFICIENT_EXCITATION):
            return
        diagnostic = (
            status is UpdateStatus.REJECTED
            and rejected_coefficients not in _NON_DIAGNOSTIC_REJECTIONS
        )
        self._recent_rejections.append(diagnostic)

    @property
    def rejection_rate(self) -> float:
        """Share of recent updates rejected for a reason that matters."""
        if not self._recent_rejections:
            return 0.0
        return sum(self._recent_rejections) / len(self._recent_rejections)

    @property
    def diverged(self) -> bool:
        """Whether the model has genuinely stopped tracking the room.

        Judged on a sustained rate over a full window rather than on a run of
        consecutive rejections. Two of the four coefficients have true values
        sitting essentially on a box edge, so short runs happen by chance
        constantly; a rate this high for this long does not.
        """
        window = self._recent_rejections
        if len(window) < window.maxlen:
            return False
        return self.rejection_rate >= self._config.divergence_rejection_fraction

    def _result(
        self,
        status: UpdateStatus,
        prediction_c: float,
        residual_c: float,
        rejected_coefficients: tuple[str, ...] = (),
    ) -> UpdateResult:
        self._record_outcome(status, rejected_coefficients)
        return UpdateResult(
            status=status,
            prediction_c=prediction_c,
            residual_c=residual_c,
            consecutive_rejections=self._consecutive_rejections,
            diverged=self.diverged,
            rejected_coefficients=rejected_coefficients,
        )

    # --- publishing ---------------------------------------------------

    def snapshot(self) -> Coefficients:
        """An immutable view of the estimate, ready for the blackboard."""
        coefficients = self.coefficients
        return Coefficients(
            ts=self._clock.now(),
            a1=float(coefficients[0]),
            a2=float(coefficients[1]),
            a3=float(coefficients[2]),
            a4=float(coefficients[3]),
            trace_p=self.trace,
            steady_state_residual=steady_state_residual(self._theta),
            samples_since_reset=self._samples_since_reset,
        )

    def restore(self, theta: np.ndarray, covariance: np.ndarray) -> None:
        """Adopt a persisted estimate (FR-07).

        :raises ValueError: if the shapes are wrong or the estimate is
            outside the plausible box. A stored vector that would be rejected
            on its first update must not be adopted on startup either.
        """
        if theta.shape != (IDENTIFIED_COUNT,):
            raise ValueError(f"theta must have {IDENTIFIED_COUNT} entries")
        if covariance.shape != (IDENTIFIED_COUNT, IDENTIFIED_COUNT):
            raise ValueError(
                f"covariance must be {IDENTIFIED_COUNT}x{IDENTIFIED_COUNT}"
            )
        if not is_plausible(theta, self._config.coefficient_bounds):
            offenders = implausible_coefficients(
                theta, self._config.coefficient_bounds
            )
            raise ValueError(
                f"persisted estimate is implausible: {', '.join(offenders)}"
            )
        self._theta = theta.astype(float)
        self._covariance = covariance.astype(float)
        self._symmetrise()
        self._bound_trace()
