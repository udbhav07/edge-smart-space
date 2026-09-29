"""E4: how long is prediction-based control actually viable?

DESIGN.md section 8.3 sets the method -- inject a sensor fault and run to
failure -- and section 7.2 asserts a 1800 s budget for controlling on the
model's prediction. E4 is the experiment that says whether that number is
right, and it is the only place the budget is treated as a question rather
than a setting.

What is measured is the room, not the model's opinion of it. Once the sensor
is faulted the system is flying on dead reckoning, and the error that matters
is between the prediction it is steering by and the temperature the room
actually reached. That gap can only grow: the model was identified from data
the faulted sensor provided, and nothing corrects it.

The budget is reported against two thresholds, and neither is the budget
itself. The comfort band is what an occupant notices. The prediction error is
what the system could in principle know about. Reporting when each is crossed
says whether 1800 s is generous, tight, or beside the point.

Run it: ``python -m eval.experiments.e4_degradation``
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
from src.common.mqtt_client import Blackboard
from src.common.schemas import ThermalEstimate

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

SETTLE_S = 4 * 3600.0

#: How far past the budget to keep running, so the question "was 1800 s the
#: right number" has evidence on both sides of it.
OVERRUN_FACTOR = 2.0

#: Prediction error at which the model has stopped being a usable stand-in for
#: a measurement. Half the comfort band: a prediction wrong by more than that
#: can put the room outside the band while reporting that it is inside.
_USABLE_PREDICTION_ERROR_FRACTION = 0.5


@dataclass(frozen=True)
class Crossing:
    """When a run first went past a threshold."""

    label: str
    at_s: float | None

    def report(self, budget_s: float) -> str:
        if self.at_s is None:
            return f"  {self.label:28s} never crossed"
        verdict = "inside" if self.at_s >= budget_s else "BEFORE"
        return (
            f"  {self.label:28s} {self.at_s:6.0f} s "
            f"({verdict} the {budget_s:.0f} s budget)"
        )


def run(config: Config) -> tuple[list[Crossing], float, float]:
    """Break the sensor and watch the prediction drift away from the room."""
    with tempfile.TemporaryDirectory() as state_dir:
        run_config = harness.for_experiment(config, Path(state_dir))
        system = harness.full_system(run_config)

        predictions: list[ThermalEstimate] = []
        watcher = Blackboard(run_config.mqtt, system.transport)
        watcher.subscribe(
            harness.topics.ESTIMATE_THERMAL,
            ThermalEstimate,
            lambda _topic, estimate: predictions.append(estimate),
        )
        system.transport.attach(watcher)

        system.run_for(SETTLE_S)
        injected_at = system.inject(InjectedFault.STUCK_AT, 25.0)
        predictions.clear()

        band = config.evaluation.comfort_band_c
        usable = band * _USABLE_PREDICTION_ERROR_FRACTION
        left_band: float | None = None
        prediction_stale: float | None = None
        worst_prediction_error = 0.0

        budget_s = config.mode.degraded_sensor_budget_s
        period_s = config.loop.sensor_period_s
        elapsed = 0.0
        while elapsed < budget_s * OVERRUN_FACTOR:
            system.run_for(period_s)
            elapsed += period_s

            sample = system.log.samples[-1]
            if left_band is None and abs(sample.true_c - sample.setpoint_c) > band:
                left_band = elapsed
            if predictions:
                error = abs(predictions[-1].t_pred - sample.true_c)
                worst_prediction_error = max(worst_prediction_error, error)
                if prediction_stale is None and error > usable:
                    prediction_stale = elapsed

        return (
            [
                Crossing("room left the comfort band", left_band),
                Crossing("prediction error exceeded half the band", prediction_stale),
            ],
            worst_prediction_error,
            budget_s,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E4.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    print("E4: running a sensor fault past the degradation budget\n")
    crossings, worst_error, budget_s = run(config)
    for crossing in crossings:
        print(crossing.report(budget_s))
    print()
    print(f"  worst prediction error observed: {worst_error:.2f} C")

    early = [c.label for c in crossings if c.at_s is not None and c.at_s < budget_s]
    if early:
        print(
            f"  The budget of {budget_s:.0f} s is too generous for this plant: "
            f"{', '.join(early)} before it expired."
        )
    else:
        print(
            f"  Nothing crossed before the budget expired, so {budget_s:.0f} s "
            f"is not obviously too long for this plant."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
