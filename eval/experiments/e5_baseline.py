"""E5: does the baseline fail where this system does not?

DESIGN.md section 8.3 sets the method -- identical fault injections on both
systems -- and section 8.4 criterion 4 sets the bar: the comfort bound is held
under an injected sensor fault in at least one scenario where the baseline does
not.

Both systems are assembled by the same harness from the same configuration, so
everything they share is provably shared: the same plant, the same seed, the
same sensors with the same noise, the same control law, the same setpoint. The
only differences are the two contributions -- the identified model and the
fault layer -- which is what makes a difference in the result attributable to
them.

Comfort is measured against the *room*, never against the sensor. A stuck
sensor reports a pleasant room while the real one bakes, and a metric read off
the sensor would score the broken system perfectly.

Every fault class is reported, including the ones this system loses. A run that
only quoted the scenario it wins would be a demonstration rather than an
experiment.

Run it: ``python -m eval.experiments.e5_baseline``
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from eval import harness, metrics
from src.common.config import Config, ConfigError, load_config
from src.common.injection import InjectedFault

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: How long the room runs before anything is broken. Long enough for the
#: estimator to have identified the plant and for the room to have settled,
#: because a comparison made during the initial transient measures the
#: transient.
SETTLE_S = 4 * 3600.0


@dataclass(frozen=True)
class Scenario:
    """One fault, applied identically to both systems."""

    label: str
    fault: InjectedFault
    magnitude: float | None

    def describe(self) -> str:
        if self.magnitude is None:
            return self.label
        return f"{self.label} ({self.magnitude:g} C)"


#: The sensor faults a thermostat could plausibly meet. Stuck-high and
#: stuck-low are separate scenarios because they fail in opposite directions:
#: one makes a controller over-cool and the other makes it stop.
SCENARIOS = (
    Scenario("dropout", InjectedFault.DROPOUT, None),
    Scenario("stuck low", InjectedFault.STUCK_AT, 22.0),
    Scenario("stuck high", InjectedFault.STUCK_AT, 26.0),
)


@dataclass(frozen=True)
class Outcome:
    """One system's behaviour under one fault."""

    system: str
    scenario: str
    comfort: metrics.ComfortMetrics
    tracking: metrics.TrackingMetrics
    final_room_c: float

    def report(self) -> str:
        return (
            f"    {self.system:9s} room ends {self.final_room_c:5.2f} C  "
            f"{self.comfort.report()}"
        )


def run_one(
    config: Config, scenario: Scenario, build, system_name: str
) -> Outcome:
    """Settle one system, break the sensor, and measure what the room did."""
    with tempfile.TemporaryDirectory() as state_dir:
        system = build(harness.for_experiment(config, Path(state_dir)))
        system.run_for(SETTLE_S)
        injected_at = system.inject(scenario.fault, scenario.magnitude)
        system.run_for(config.mode.degraded_sensor_budget_s)

        after = system.log.after(injected_at)
        return Outcome(
            system=system_name,
            scenario=scenario.label,
            comfort=metrics.comfort(
                after.temperatures_c,
                after.setpoints_c,
                config.evaluation.comfort_band_c,
            ),
            tracking=metrics.tracking(after.temperatures_c, after.setpoints_c),
            final_room_c=system.simulator.room_temperature_c,
        )


def run(config: Config) -> list[tuple[Outcome, Outcome]]:
    """Every scenario against both systems."""
    results = []
    for scenario in SCENARIOS:
        ours = run_one(config, scenario, harness.full_system, "ours")
        baseline = run_one(config, scenario, harness.baseline_system, "baseline")
        results.append((ours, baseline))
    return results


def decisive(results: list[tuple[Outcome, Outcome]]) -> list[str]:
    """Scenarios where this system held the bound and the baseline did not.

    Criterion 4 asks for at least one. It asks for *held*, not for *better*:
    being less bad than a thermostat is not the claim the report makes.
    """
    return [
        ours.scenario
        for ours, baseline in results
        if ours.comfort.held_the_bound and not baseline.comfort.held_the_bound
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E5.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    print("E5: the same faults, injected into both systems\n")
    print(
        f"  comfort band +/-{config.evaluation.comfort_band_c:g} C, "
        f"measured against the room for "
        f"{config.mode.degraded_sensor_budget_s:.0f} s after injection\n"
    )

    results = run(config)
    for ours, baseline in results:
        print(f"  {ours.scenario}")
        print(ours.report())
        print(baseline.report())

    won = decisive(results)
    print()
    if won:
        print(
            f"  Criterion 4 met in: {', '.join(won)} -- this system held the "
            f"comfort bound and the baseline did not."
        )
    else:
        print("  Criterion 4 NOT met: no scenario where only this system held.")

    lost = [
        ours.scenario
        for ours, baseline in results
        if ours.comfort.violation_fraction > baseline.comfort.violation_fraction
    ]
    if lost:
        print(f"  Worse than the baseline in: {', '.join(lost)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
