"""Tests for experiment E1.

These assert what E1 *measured*, not what it was hoped to measure. The
identification recovers the actuator authority and its sign well and does
not recover a1, and that gap is a finding about the estimator's structure
rather than a defect in this harness -- see the module docstring of
``eval.experiments.e1_convergence`` and the note at the bottom of this file.

Writing a test that asserts a1 converges would make the suite red for a
reason the code cannot fix, and writing one that asserts nothing would hide
the result. So the bias is pinned: if it ever improves, this test fails and
somebody has to come and read why.
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


class TestWhatIdentifiesWell:
    def test_the_actuator_authority_keeps_its_sign(self, result):
        """FR-24 depends on this: a3 crossing zero means the air conditioner
        heats the room, which is an actuator fault rather than a property."""
        assert result.estimated[2] < 0.0

    def test_the_actuator_authority_is_close_to_the_truth(self, result):
        assert abs(result.estimated[2] - result.truth[2]) < 0.02

    def test_steady_state_consistency_is_broadly_maintained(self, result):
        assert result.steady_state_residual < 0.1


class TestKnownBias:
    """a1 does not converge, and the reason is structural.

    T[k] is both a regressor and a noisy measurement. Least squares with a
    noisy predictor attenuates that predictor's coefficient toward zero and
    a2 absorbs the difference; at zero sensor noise the same estimator
    recovers a1 exactly, so the algorithm is right and the *problem* is
    mis-specified for a +/-0.15 C instrument.

    This is pinned rather than asserted away. It bears directly on section
    8.4's second success criterion, which asks for coefficients within a
    stated tolerance of ground truth, and that tolerance now has to be
    stated in the knowledge that a1 will not meet a tight one at this
    sampling cadence and noise level.
    """

    def test_a1_is_attenuated_toward_zero(self, result):
        assert result.estimated[0] < result.truth[0]

    def test_a2_absorbs_what_a1_loses(self, result):
        assert result.estimated[1] > result.truth[1]

    def test_the_pair_still_roughly_sums_to_one(self, result):
        """The attenuation moves weight between them rather than out of them,
        which is why the steady-state diagnostic stays small even though the
        individual coefficients are wrong. Reporting only |a1 + a2 - 1| would
        therefore look like success and hide this entirely."""
        assert result.estimated[0] + result.estimated[1] == pytest.approx(1.0, abs=0.1)


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
