"""Unit tests for the experiment scripts and what they measure.

The experiments themselves take minutes, so what is tested here is the part
that can be wrong without anyone noticing: the metrics, the scenario
definitions, and the verdicts drawn from a result. A comparison that quietly
scored the sensor instead of the room, or called criterion 4 met on a
technicality, would still print a confident table.

The runs are exercised at short durations in the experiments' own test for
each module where that is affordable, and not at all where it is not.
"""

from pathlib import Path

import pytest

from eval import metrics
from eval.experiments import e3_detection, e5_baseline
from src.common.config import load_config
from src.common.injection import InjectedFault
from src.common.schemas import DetectorId


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


class TestTrackingMetric:
    def test_a_room_on_setpoint_has_no_error(self):
        result = metrics.tracking([24.0] * 10, [24.0] * 10)
        assert result.rms_error_c == 0.0

    def test_error_is_measured_against_the_setpoint_in_force(self):
        """A setpoint that moves during a run must not make it meaningless."""
        result = metrics.tracking([24.0, 26.0], [24.0, 26.0])
        assert result.rms_error_c == 0.0

    def test_overshoot_counts_only_cooling_past_the_target(self):
        result = metrics.tracking([23.0, 25.0], [24.0, 24.0])
        assert result.overshoot_c == pytest.approx(1.0)

    def test_a_room_that_never_undershoots_has_no_overshoot(self):
        result = metrics.tracking([25.0, 26.0], [24.0, 24.0])
        assert result.overshoot_c == 0.0

    def test_mismatched_series_are_refused(self):
        """Pairing each temperature with the wrong setpoint would produce a
        number rather than an error."""
        with pytest.raises(ValueError):
            metrics.tracking([24.0, 25.0], [24.0])

    def test_an_empty_run_is_not_a_division(self):
        assert metrics.tracking([], []).samples == 0


class TestComfortMetric:
    def test_a_room_inside_the_band_holds_it(self):
        result = metrics.comfort([24.5, 23.5], [24.0, 24.0], band_c=1.0)
        assert result.held_the_bound

    def test_the_band_edge_is_inside(self):
        result = metrics.comfort([25.0], [24.0], band_c=1.0)
        assert result.held_the_bound

    def test_a_sample_past_the_edge_is_a_violation(self):
        result = metrics.comfort([25.01], [24.0], band_c=1.0)
        assert result.violations == 1

    def test_the_worst_excursion_is_measured_past_the_band(self):
        result = metrics.comfort([27.0], [24.0], band_c=1.0)
        assert result.worst_excursion_c == pytest.approx(2.0)

    def test_a_band_of_zero_is_refused(self):
        """Every sample would be a violation and the number would look like a
        result."""
        with pytest.raises(ValueError):
            metrics.comfort([24.0], [24.0], band_c=0.0)

    def test_an_empty_run_reports_no_violations_rather_than_dividing(self):
        assert metrics.comfort([], [], band_c=1.0).violation_fraction == 0.0


class TestLatencyMetric:
    def test_latency_is_the_gap(self):
        assert metrics.detection_latency_s(100.0, 115.0) == 15.0

    def test_a_miss_has_no_latency(self):
        """A miss and a slow detection are different outcomes, and averaging a
        sentinel would report a blind detector as a slow one."""
        assert metrics.detection_latency_s(100.0, None) is None

    def test_detecting_before_injecting_is_a_bookkeeping_error(self):
        with pytest.raises(ValueError):
            metrics.detection_latency_s(100.0, 90.0)

    def test_nothing_to_average_is_none_not_zero(self):
        assert metrics.mean([]) is None

    def test_more_successes_than_trials_is_refused(self):
        with pytest.raises(ValueError):
            metrics.rate(3, 2)


class TestE3Cases:
    def test_every_detector_has_a_case(self):
        """A detector with no trial is a detector nobody measured."""
        covered = {case.detector for case in e3_detection.CASES}
        expected = set(DetectorId) - {DetectorId.MODEL_DIVERGENCE}
        assert covered == expected

    def test_the_actuator_case_targets_the_actuator(self):
        case = [c for c in e3_detection.CASES if c.detector.value.startswith("D5")][0]
        assert case.subject == "actuator"
        assert case.fault is InjectedFault.STUCK_OFF

    def test_trials_differ_only_by_seed(self, config):
        first = e3_detection._seeded(config, 0)
        second = e3_detection._seeded(config, 1)
        assert first.sim.random_seed != second.sim.random_seed
        assert first.sim.room == second.sim.room

    def test_a_detection_rate_counts_only_detections(self):
        result = e3_detection.CaseResult(case=e3_detection.CASES[0])
        result.trials, result.detections = 4, 3
        assert result.detection_rate == 0.75


class TestE5Verdict:
    def _outcome(self, name, violations, samples=100):
        return e5_baseline.Outcome(
            system=name,
            scenario="dropout",
            comfort=metrics.ComfortMetrics(
                violations=violations,
                samples=samples,
                worst_excursion_c=0.0,
                band_c=1.0,
            ),
            tracking=metrics.tracking([24.0], [24.0]),
            final_room_c=24.0,
        )

    def test_criterion_four_needs_ours_to_hold(self):
        results = [(self._outcome("ours", 0), self._outcome("baseline", 50))]
        assert e5_baseline.decisive(results) == ["dropout"]

    def test_being_merely_better_is_not_criterion_four(self):
        """Less bad than a thermostat is not the claim the report makes."""
        results = [(self._outcome("ours", 10), self._outcome("baseline", 50))]
        assert e5_baseline.decisive(results) == []

    def test_both_holding_is_not_decisive_either(self):
        results = [(self._outcome("ours", 0), self._outcome("baseline", 0))]
        assert e5_baseline.decisive(results) == []

    def test_every_scenario_names_a_sensor_fault(self):
        for scenario in e5_baseline.SCENARIOS:
            assert scenario.fault is not InjectedFault.NONE

    def test_the_stuck_scenarios_fail_in_opposite_directions(self):
        """One makes a controller over-cool and the other makes it stop, and
        a comparison that only ran one would miss half the story."""
        stuck = [
            s.magnitude
            for s in e5_baseline.SCENARIOS
            if s.fault is InjectedFault.STUCK_AT
        ]
        assert min(stuck) < 24.0 < max(stuck)
