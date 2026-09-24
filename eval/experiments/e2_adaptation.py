"""E2: does self-calibration improve tracking over a fixed model?

DESIGN.md section 8.3 sets the method -- the same scenario with adaptation on
and with it frozen at the configured prior -- and the metrics: RMS setpoint
error and overshoot.

The comparison is narrower than it looks, and saying so is part of the result.
The regulatory law is a deadband on the *measurement*, so while the sensor is
healthy the model barely touches the command: it contributes a prediction the
controller only uses when the sensor cannot be trusted (FR-27). What E2
therefore measures is whether adapting costs anything during normal operation,
and whether the prior is good enough to leave alone. The case *for* adaptation
is made by E4 and E5, where the prediction is doing the controlling.

Both runs share a plant, a seed and a setpoint. The frozen run keeps the
configured theta from the first sample to the last.

Run it: ``python -m eval.experiments.e2_adaptation``
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from eval import harness, metrics
from src.common.config import Config, ConfigError, load_config

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
DEFAULT_HOURS = 12.0

#: Ignored when scoring. The room starts well away from setpoint and no
#: controller can be blamed for the trip it has to make; scoring it would
#: report the initial transient twice and bury the difference under it.
WARMUP_S = 3600.0


@dataclass(frozen=True)
class E2Result:
    """One run's tracking performance."""

    label: str
    tracking: metrics.TrackingMetrics
    comfort: metrics.ComfortMetrics

    def report(self) -> str:
        return f"  {self.label:12s} {self.tracking.report()}"


def run_one(config: Config, hours: float, adapting: bool, label: str) -> E2Result:
    """Run the system with adaptation on or frozen at the prior."""
    with tempfile.TemporaryDirectory() as state_dir:
        system = harness.full_system(harness.for_experiment(config, Path(state_dir)))
        if not adapting:
            # Frozen from the first sample: the prior is the whole model, and
            # freezing it here rather than through the health topic keeps the
            # fault layer out of a comparison that is not about faults.
            system.estimator._estimator.freeze()
        system.run_for(hours * 3600.0)

        scored = system.log.after(system.log.samples[0].ts + WARMUP_S)
        return E2Result(
            label=label,
            tracking=metrics.tracking(scored.temperatures_c, scored.setpoints_c),
            comfort=metrics.comfort(
                scored.temperatures_c,
                scored.setpoints_c,
                config.evaluation.comfort_band_c,
            ),
        )


def run(config: Config, hours: float) -> tuple[E2Result, E2Result]:
    return (
        run_one(config, hours, adapting=True, label="adapting"),
        run_one(config, hours, adapting=False, label="frozen"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E2.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    print(
        f"E2: adaptation against a model frozen at the prior, "
        f"{arguments.hours:g} h\n"
    )
    adapting, frozen = run(config, arguments.hours)
    print(adapting.report())
    print(frozen.report())
    print()
    print(f"  adapting  {adapting.comfort.report()}")
    print(f"  frozen    {frozen.comfort.report()}")
    print()
    difference = frozen.tracking.rms_error_c - adapting.tracking.rms_error_c
    print(
        f"  Adapting changes RMS setpoint error by {difference:+.3f} C while "
        f"the sensor is healthy."
    )
    print(
        "  The deadband law tracks the measurement, so this is expected to be "
        "small; what adaptation buys is the prediction FR-27 controls on when "
        "the sensor fails, which E4 and E5 measure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
