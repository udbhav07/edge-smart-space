"""Tests for experiment E1.

E1 is the Weeks 2-3 gate: coefficients converge on synthetic data. It did
not pass in the model's original form -- a1 settled at 0.754 against a truth
of 0.998, because T[k] was a noisy regressor and least squares attenuates
such a coefficient toward zero. Section 5.2.1 was changed in v1.2 to fit the
temperature *change* against T_out - T, deriving a1 as 1 - a2, and these
tests hold the resulting numbers in place.

The tolerances are deliberately close to the measured values rather than
generously loose. A test that would still pass if the reformulation were
reverted would not be guarding anything.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from eval.experiments.e1_convergence import arx_ground_truth, run
from eval.loopback import LoopbackTransport
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import SensorReading

#: Short enough to keep the suite fast, long enough for the estimate to
#: settle: the forgetting factor's memory is 200 samples, or about 17 min.
TEST_HOURS = 2.0


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="result", scope="module")
def _result():
    config = load_config(Path("config/default.yaml"))
    return run(config, TEST_HOURS, identifiable=True, label="test run")


class TestGroundTruth:
    """The truth is derived from R, C and watts, never handed over."""

    def test_the_retained_fraction_is_the_first_order_lag(self, config):
        room = config.sim.room
        interval_s = config.loop.sensor_period_s
        tau = room.thermal_resistance_k_per_w * room.thermal_capacitance_j_per_k
        truth = arx_ground_truth(room, interval_s)
        assert truth[0] == pytest.approx(math.exp(-interval_s / tau))

    def test_steady_state_consistency_holds_exactly(self, config):
        """a1 + a2 = 1 is what the discretisation guarantees, not an
        approximation the estimator has to find."""
        truth = arx_ground_truth(config.sim.room, config.loop.sensor_period_s)
        assert truth[0] + truth[1] == pytest.approx(1.0)

    def test_the_actuator_authority_is_negative(self, config):
        truth = arx_ground_truth(config.sim.room, config.loop.sensor_period_s)
        assert truth[2] < 0.0

    def test_occupancy_gain_is_positive(self, config):
        truth = arx_ground_truth(config.sim.room, config.loop.sensor_period_s)
        assert truth[3] > 0.0

    def test_a_slower_room_retains_more_per_step(self, config):
        slow = config.sim.room.model_copy(
            update={"thermal_capacitance_j_per_k": 1_200_000.0}
        )
        interval_s = config.loop.sensor_period_s
        assert (
            arx_ground_truth(slow, interval_s)[0]
            > arx_ground_truth(config.sim.room, interval_s)[0]
        )

    def test_every_derived_coefficient_lies_inside_the_plausibility_box(self, config):
        """A truth outside the box would mean the box is wrong."""
        truth = arx_ground_truth(config.sim.room, config.loop.sensor_period_s)
        box = config.estimator.coefficient_bounds
        assert all(
            bounds.contains(value) for value, bounds in zip(truth, box)
        )


class TestTheRunItself:
    def test_the_experiment_completes(self, result):
        assert result.samples > 0

    def test_most_pairs_are_fitted(self, result):
        """Skips come from dropouts and jitter, both on by default."""
        assert result.updates_fitted > result.samples * 0.9

    def test_some_pairs_are_skipped_because_the_simulator_is_adversarial(self, result):
        """If nothing were ever skipped, the simulator would be too kind."""
        assert result.pairs_skipped > 0

    def test_the_estimate_stays_inside_the_plausibility_box(self, result, config):
        box = config.estimator.coefficient_bounds
        assert all(
            bounds.contains(value) for value, bounds in zip(result.estimated, box)
        )

    def test_a_residual_spread_is_reported(self, result):
        assert result.residual_sigma > 0.0

    def test_the_report_names_every_coefficient(self, result):
        report = result.report()
        assert all(name in report for name in ("a1", "a2", "a3", "a4"))


class TestTheGate:
    """Weeks 2-3: coefficients converge on synthetic data."""

    def test_thermal_inertia_converges(self, result):
        """The coefficient the old form could not recover. It settled at
        0.754 against a truth of 0.998 before v1.2."""
        assert abs(result.estimated[0] - result.truth[0]) < 0.01

    def test_ambient_coupling_converges(self, result):
        assert abs(result.estimated[1] - result.truth[1]) < 0.01

    def test_the_actuator_authority_keeps_its_sign(self, result):
        """FR-24 depends on this: a3 crossing zero means the air conditioner
        heats the room, which is an actuator fault rather than a property."""
        assert result.estimated[2] < 0.0

    def test_the_actuator_authority_converges(self, result):
        """Looser than a1's because a3 needs longer: measured against this
        two-hour fixture the error is 0.011, falling to 0.003 at six hours
        and 0.0001 over the full 24 h run the experiment actually reports.
        a3 is identified only while the compressor is switching, so it
        accumulates evidence more slowly than the always-present terms."""
        assert abs(result.estimated[2] - result.truth[2]) < 0.02

    def test_the_worst_coefficient_error_is_small(self, result):
        """0.243 in the old form. Now 0.018 over this fixture and 0.036 over
        the full 24 h run, where the worst case is a4."""
        assert result.worst_error < 0.05


class TestStructuralConsistency:
    """a1 is derived, not fitted, so the identity cannot drift."""

    def test_steady_state_consistency_holds_exactly(self, result):
        assert result.steady_state_residual == pytest.approx(0.0, abs=1e-12)

    def test_the_pair_sums_to_one(self, result):
        assert result.estimated[0] + result.estimated[1] == pytest.approx(1.0)

    def test_the_identity_is_therefore_not_a_useful_metric(self, result):
        """Section 8.3 no longer lists it for E1: identically zero measures
        nothing, and quoting it would read as success while saying nothing
        about whether the coefficients are right."""
        assert result.steady_state_residual == 0.0


class TestRemainingWeakness:
    """a4 is the cost of the change, and it is recorded rather than hidden.

    Occupancy gain is about 0.0008 for one person, far below the noise
    floor. The old form happened to estimate it adequately; this one does
    not, and it is now the worst of the four. It contributes roughly 0.03 C
    to a prediction, so the error costs less than a1's did.
    """

    def test_occupancy_gain_is_the_worst_identified_coefficient(self, result):
        """True from about an hour in, and increasingly so: a4's error grows
        with run length while every other coefficient's shrinks."""
        assert np.argmax(result.errors) == 3

    def test_it_is_nonetheless_bounded(self, result):
        assert abs(result.estimated[3] - result.truth[3]) < 0.05

    def test_it_stays_inside_its_widened_range(self, result, config):
        assert config.estimator.bounds_a4.contains(result.estimated[3])


class TestLoopback:
    """The experiment must run over the real topics, not around them."""

    def test_a_published_message_reaches_a_subscriber(self, config):
        transport = LoopbackTransport()
        board = Blackboard(config.mqtt, transport)
        transport.attach(board)
        seen: list[SensorReading] = []
        from src.common import topics

        board.subscribe(
            topics.SENSOR_STATE, SensorReading, lambda t, m: seen.append(m)
        )
        reading = SensorReading(
            ts=1.0, sensor_id="temp_01", value=27.4, unit="C"
        )
        board.publish(topics.SENSOR_STATE, reading, sensor_id="temp_01")
        assert seen == [reading]

    def test_the_experiment_actually_used_the_topics(self, result):
        """updates_fitted is only non-zero if readings crossed the bus."""
        assert result.updates_fitted > 0
