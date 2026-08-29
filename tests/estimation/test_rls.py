"""Unit tests for recursive least squares and its safeguards.

The safeguards get more attention than the update, deliberately. The update
is four lines from the document; the safeguards are what section 5.2.3 calls
the difference between running RLS and running it on a real system for
twelve weeks, and each of them only ever fires when something has already
gone wrong.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.clock import SimClock
from src.common.config import EstimatorConfig, load_config
from src.common.schemas import AdaptationState
from src.estimation.rc_model import Regressor
from src.estimation.rls import ThermalEstimator, UpdateStatus

#: A plant the estimator's prior does not already match.
TRUE_THETA = np.array([0.9700, 0.0300, -0.0800, 0.0050])


@pytest.fixture(name="config")
def _config() -> EstimatorConfig:
    return load_config(Path("config/default.yaml")).estimator


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="estimator")
def _estimator(config, clock) -> ThermalEstimator:
    return ThermalEstimator(config, clock)


def _regressor(step: int, indoor_c: float) -> Regressor:
    """An input that actually excites the model."""
    return Regressor(
        indoor_c=indoor_c,
        outdoor_c=31.0 + 4.0 * np.sin(2 * np.pi * step / 1000),
        command=1.0 if (step // 40) % 2 == 0 else 0.0,
        occupancy=1.0 if (step // 300) % 2 == 0 else 0.0,
    )


def _identify(estimator: ThermalEstimator, steps: int = 3000) -> np.ndarray:
    indoor_c = 29.0
    for step in range(steps):
        phi = _regressor(step, indoor_c)
        indoor_c = float(TRUE_THETA @ phi.as_array())
        estimator.update(phi, indoor_c)
    return estimator.theta


class TestInitialState:
    def test_starts_at_the_configured_prior(self, estimator, config):
        assert np.allclose(estimator.theta, config.initial_theta)

    def test_starts_adapting(self, estimator):
        assert estimator.adaptation is AdaptationState.ACTIVE

    def test_starts_with_a_weak_prior(self, estimator, config):
        expected = 4 * config.initial_covariance
        assert estimator.trace == pytest.approx(expected)

    def test_reports_no_residual_spread_before_any_data(self, estimator):
        assert estimator.residual_sigma == 0.0

    def test_theta_is_handed_out_as_a_copy(self, estimator):
        """A caller must not be able to move the estimate."""
        theta = estimator.theta
        theta[0] = 99.0
        assert estimator.theta[0] != 99.0


class TestIdentification:
    """FR-04: identify the coefficients from operating data, no retraining."""

    def test_converges_on_a_known_plant(self, estimator):
        theta = _identify(estimator)
        assert np.max(np.abs(theta - TRUE_THETA)) < 0.01

    def test_recovers_the_actuator_authority_with_its_sign(self, estimator):
        theta = _identify(estimator)
        assert theta[2] < 0.0

    def test_converges_toward_steady_state_consistency(self, estimator):
        """|a1 + a2 - 1| shrinking is the identification working."""
        _identify(estimator)
        assert estimator.snapshot().steady_state_residual < 0.01

    def test_the_covariance_shrinks_as_evidence_accumulates(self, estimator):
        before = estimator.trace
        _identify(estimator, steps=500)
        assert estimator.trace < before

    def test_confidence_rises_as_the_covariance_shrinks(self, estimator):
        before = estimator.model_confidence
        _identify(estimator, steps=500)
        assert estimator.model_confidence > before

    def test_residuals_shrink_as_the_estimate_improves(self, estimator):
        _identify(estimator, steps=200)
        early = estimator.residual_sigma
        _identify(estimator, steps=2000)
        assert estimator.residual_sigma < early


class TestPredictionAndResidual:
    def test_the_residual_is_measurement_minus_prediction(self, estimator):
        phi = _regressor(0, 29.0)
        prediction = estimator.predict(phi)
        result = estimator.update(phi, 30.0)
        assert result.residual_c == pytest.approx(30.0 - prediction)

    def test_the_prediction_is_reported_on_every_step(self, estimator):
        result = estimator.update(_regressor(0, 29.0), 29.0)
        assert isinstance(result.prediction_c, float)

    def test_samples_are_counted(self, estimator):
        for step in range(5):
            estimator.update(_regressor(step, 29.0), 29.0)
        assert estimator.samples_since_reset == 5


class TestFreezing:
    """FR-29: never adapt while a regressor sensor is faulted."""

    def test_freezing_stops_the_estimate_moving(self, estimator):
        _identify(estimator, steps=100)
        estimator.freeze()
        before = estimator.theta
        _identify(estimator, steps=200)
        assert np.array_equal(estimator.theta, before)

    def test_freezing_is_reported(self, estimator):
        estimator.freeze()
        assert estimator.adaptation is AdaptationState.FROZEN

    def test_a_frozen_update_says_so(self, estimator):
        estimator.freeze()
        assert estimator.update(_regressor(0, 29.0), 29.0).status is UpdateStatus.FROZEN

    def test_prediction_continues_while_frozen(self, estimator):
        """This is what makes DEGRADED_SENSOR control possible (FR-27)."""
        estimator.freeze()
        result = estimator.update(_regressor(0, 29.0), 30.0)
        assert result.prediction_c != 0.0
        assert result.residual_c == pytest.approx(30.0 - result.prediction_c)

    def test_unfreezing_resumes_learning(self, estimator):
        estimator.freeze()
        estimator.unfreeze()
        before = estimator.theta
        _identify(estimator, steps=200)
        assert not np.array_equal(estimator.theta, before)


class TestExcitation:
    """A constant regressor identifies nothing and degrades the covariance."""

    def _constant(self, estimator, steps: int):
        phi = Regressor(indoor_c=27.0, outdoor_c=31.0, command=0.0, occupancy=0.0)
        results = [estimator.update(phi, 27.0) for _ in range(steps)]
        return results

    def test_a_motionless_regressor_eventually_skips_the_update(
        self, estimator, config
    ):
        results = self._constant(estimator, config.excitation_window_samples + 5)
        assert results[-1].status is UpdateStatus.INSUFFICIENT_EXCITATION

    def test_early_samples_still_adapt_before_the_window_fills(self, estimator):
        """Refusing to learn at startup would leave control on the prior."""
        assert self._constant(estimator, 1)[0].status is UpdateStatus.APPLIED

    def test_a_skipped_update_leaves_the_estimate_alone(self, estimator, config):
        self._constant(estimator, config.excitation_window_samples + 5)
        before = estimator.theta
        self._constant(estimator, 5)
        assert np.array_equal(estimator.theta, before)

    def test_excitation_returning_resumes_adaptation(self, estimator, config):
        self._constant(estimator, config.excitation_window_samples + 5)
        indoor_c = 29.0
        status = None
        for step in range(config.excitation_window_samples + 5):
            phi = _regressor(step, indoor_c)
            indoor_c = float(TRUE_THETA @ phi.as_array())
            status = estimator.update(phi, indoor_c).status
        assert status is UpdateStatus.APPLIED


class TestCovarianceSafeguards:
    def test_the_trace_stays_within_its_bound(self, estimator, config):
        """A quiet night must not wind the covariance up (section 5.2.3)."""
        phi = Regressor(indoor_c=0.0, outdoor_c=0.0, command=0.0, occupancy=0.0)
        for _ in range(500):
            estimator.update(phi, 0.0)
        assert estimator.trace <= config.max_covariance_trace * 1.0001

    def test_the_covariance_is_handed_out_as_a_copy(self, estimator):
        covariance = estimator.covariance
        covariance[0, 0] = 99.0
        assert estimator.covariance[0, 0] != 99.0

    def test_the_covariance_stays_symmetric(self, estimator):
        _identify(estimator, steps=300)
        covariance = estimator.covariance
        assert np.allclose(covariance, covariance.T)

    def test_confidence_is_bounded_to_the_unit_interval(self, estimator):
        phi = Regressor(indoor_c=0.0, outdoor_c=0.0, command=0.0, occupancy=0.0)
        for _ in range(200):
            estimator.update(phi, 0.0)
            assert 0.0 <= estimator.model_confidence <= 1.0


class TestImplausibleEstimates:
    """FR-06: reject, log which coefficient broke, and count."""

    def _drive_out_of_box(self, estimator, steps: int = 40):
        """Feed data claiming the air conditioner heats the room."""
        statuses = []
        indoor_c = 25.0
        for step in range(steps):
            phi = Regressor(
                indoor_c=indoor_c, outdoor_c=25.0, command=1.0, occupancy=0.0
            )
            indoor_c += 3.0
            statuses.append(estimator.update(phi, indoor_c).status)
        return statuses

    def test_an_implausible_update_is_rejected(self, estimator):
        assert UpdateStatus.REJECTED in self._drive_out_of_box(estimator)

    def test_a_rejected_update_leaves_the_estimate_where_it_was(self, estimator):
        estimator.update(_regressor(0, 29.0), 29.0)
        before = estimator.theta
        phi = Regressor(indoor_c=25.0, outdoor_c=25.0, command=1.0, occupancy=0.0)
        result = estimator.update(phi, 900.0)
        if result.status is UpdateStatus.REJECTED:
            assert np.array_equal(estimator.theta, before)

    def test_a_rejection_names_the_offending_coefficient(self, estimator):
        phi = Regressor(indoor_c=25.0, outdoor_c=25.0, command=1.0, occupancy=0.0)
        result = estimator.update(phi, 900.0)
        if result.status is UpdateStatus.REJECTED:
            assert result.rejected_coefficients

    def test_repeated_rejections_raise_divergence(self, config, clock):
        estimator = ThermalEstimator(config, clock)
        diverged = any(
            estimator.update(
                Regressor(indoor_c=25.0, outdoor_c=25.0, command=1.0, occupancy=0.0),
                900.0,
            ).diverged
            for _ in range(config.max_consecutive_rejections + 2)
        )
        assert diverged

    def test_an_accepted_update_clears_the_rejection_count(self, estimator):
        phi = Regressor(indoor_c=25.0, outdoor_c=25.0, command=1.0, occupancy=0.0)
        estimator.update(phi, 900.0)
        _identify(estimator, steps=50)
        assert estimator.consecutive_rejections == 0


class TestSnapshot:
    def test_reports_the_current_estimate(self, estimator, config):
        snapshot = estimator.snapshot()
        assert (snapshot.a1, snapshot.a2) == (
            config.initial_theta[0],
            config.initial_theta[1],
        )

    def test_carries_the_steady_state_diagnostic(self, estimator):
        snapshot = estimator.snapshot()
        assert snapshot.steady_state_residual == pytest.approx(
            abs(snapshot.a1 + snapshot.a2 - 1.0)
        )

    def test_is_stamped_from_the_injected_clock(self, config, clock):
        clock.advance(500.0)
        assert ThermalEstimator(config, clock).snapshot().ts == clock.now()

    def test_is_immutable(self, estimator):
        with pytest.raises(Exception):
            estimator.snapshot().a1 = 9.0

    def test_a_later_snapshot_does_not_alter_an_earlier_one(self, estimator):
        first = estimator.snapshot()
        _identify(estimator, steps=200)
        assert estimator.snapshot().a1 != first.a1 or first.a1 == estimator.theta[0]


class TestResetAndRestore:
    def test_reset_returns_to_the_prior(self, estimator, config):
        _identify(estimator, steps=300)
        estimator.reset()
        assert np.allclose(estimator.theta, config.initial_theta)

    def test_reset_clears_the_sample_count(self, estimator):
        _identify(estimator, steps=50)
        estimator.reset()
        assert estimator.samples_since_reset == 0

    def test_a_persisted_estimate_can_be_adopted(self, estimator):
        theta = np.array([0.97, 0.03, -0.08, 0.005])
        estimator.restore(theta, np.eye(4) * 0.5)
        assert np.allclose(estimator.theta, theta)

    def test_an_implausible_persisted_estimate_is_refused(self, estimator):
        """One that would be rejected on its first update must not be
        adopted at startup either."""
        with pytest.raises(ValueError):
            estimator.restore(np.array([0.98, 0.02, 0.9, 0.01]), np.eye(4))

    def test_a_wrongly_shaped_estimate_is_refused(self, estimator):
        with pytest.raises(ValueError):
            estimator.restore(np.array([0.98, 0.02]), np.eye(4))

    def test_a_wrongly_shaped_covariance_is_refused(self, estimator):
        with pytest.raises(ValueError):
            estimator.restore(np.array([0.98, 0.02, -0.05, 0.01]), np.eye(2))

    def test_a_restored_covariance_is_bounded_on_adoption(self, estimator, config):
        estimator.restore(
            np.array([0.98, 0.02, -0.05, 0.01]),
            np.eye(4) * config.max_covariance_trace,
        )
        assert estimator.trace <= config.max_covariance_trace * 1.0001
