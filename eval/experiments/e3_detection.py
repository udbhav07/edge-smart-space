"""E3: does the system detect every fault class it claims to?

DESIGN.md section 8.3 sets the method -- each fault class over repeated trials
-- and the metrics: detection rate, false-positive rate, and latency. Section
8.4 criterion 3 asks for detection in a clear majority of trials and a stated
false-positive bound.

Two things make this an experiment rather than a demonstration.

**The false-positive rate is measured on its own runs.** A detector that fires
on everything scores a perfect detection rate, so the healthy runs are not
decoration: they are the control, and a fault raised in one is the number that
would sink the design.

**Latency is reported, never averaged with misses.** A trial that never
detected has no latency, and folding a sentinel into a mean would report a
blind detector as a slow one. Misses are counted separately and said out loud.

Trials differ only in their seed, so the spread is the plant's noise, dropouts
and jitter rather than a different scenario each time.

Run it: ``python -m eval.experiments.e3_detection``
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from eval import harness, metrics
from src.common import topics
from src.common.config import Config, ConfigError, load_config
from src.common.injection import InjectedFault
from src.common.schemas import DetectorId

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: Time for the estimator to identify the plant before anything is broken.
#: D4 and D5 test against the model, so a trial that injected before the model
#: existed would be measuring the warm-up.
SETTLE_S = 3 * 3600.0

#: How long a trial waits for a detector to speak. Generous: what is being
#: measured is whether it detects and how fast, not whether it beats a
#: deadline, and truncating the window would turn slow detections into misses.
OBSERVE_S = 3600.0

DEFAULT_TRIALS = 5


@dataclass(frozen=True)
class FaultCase:
    """One injectable fault and the detector expected to find it."""

    label: str
    subject: str
    fault: InjectedFault
    magnitude: float | None
    detector: DetectorId


CASES = (
    FaultCase("dropout", "sensor", InjectedFault.DROPOUT, None, DetectorId.D1_DROPOUT),
    FaultCase("stuck at", "sensor", InjectedFault.STUCK_AT, 25.0, DetectorId.D2_STUCK_AT),
    FaultCase(
        "out of range", "sensor", InjectedFault.OUT_OF_RANGE, 999.0,
        DetectorId.D3_OUT_OF_RANGE,
    ),
    FaultCase("drift", "sensor", InjectedFault.DRIFT, 0.004, DetectorId.D4_DRIFT),
    FaultCase(
        "actuator dead", "actuator", InjectedFault.STUCK_OFF, None,
        DetectorId.D5_ACTUATOR_NO_RESPONSE,
    ),
)


@dataclass
class CaseResult:
    """Every trial of one fault class."""

    case: FaultCase
    detections: int = 0
    trials: int = 0
    latencies_s: list[float] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return metrics.rate(self.detections, self.trials)

    @property
    def mean_latency_s(self) -> float | None:
        return metrics.mean(self.latencies_s)

    def report(self) -> str:
        latency = self.mean_latency_s
        shown = "never detected" if latency is None else f"mean {latency:.0f} s"
        worst = f", worst {max(self.latencies_s):.0f} s" if self.latencies_s else ""
        return (
            f"  {self.case.label:14s} {self.case.detector.value:24s} "
            f"{self.detections}/{self.trials} detected, {shown}{worst}"
        )


def _seeded(config: Config, trial: int) -> Config:
    """One trial's plant. Only the seed differs between trials."""
    sim = config.sim.model_copy(update={"random_seed": config.sim.random_seed + trial})
    return config.model_copy(update={"sim": sim})


def run_trial(config: Config, case: FaultCase) -> float | None:
    """Inject one fault and time the detector that should find it.

    :returns: the detection latency, or None if it was never detected.
    """
    with tempfile.TemporaryDirectory() as state_dir:
        system = harness.full_system(harness.for_experiment(config, Path(state_dir)))
        system.run_for(SETTLE_S)

        if case.subject == "actuator":
            system.injector.inject(
                topics.AIR_CONDITIONER_ID, case.fault, case.magnitude
            )
            injected_at = system.clock.now()
        else:
            injected_at = system.inject(case.fault, case.magnitude)

        system.run_for(OBSERVE_S)
        found = system.log.first_fault(case.detector)
        if found is None:
            return None
        return metrics.detection_latency_s(injected_at, found.detected_ts)


def run_control(config: Config) -> int:
    """A healthy run. Any fault raised here is a false positive."""
    with tempfile.TemporaryDirectory() as state_dir:
        system = harness.full_system(harness.for_experiment(config, Path(state_dir)))
        system.run_for(SETTLE_S + OBSERVE_S)
        return len(system.log.faults)


def run(config: Config, trials: int) -> tuple[list[CaseResult], int, int]:
    """Every fault class over repeated trials, plus the healthy controls."""
    results = []
    for case in CASES:
        result = CaseResult(case=case)
        for trial in range(trials):
            latency = run_trial(_seeded(config, trial), case)
            result.trials += 1
            if latency is not None:
                result.detections += 1
                result.latencies_s.append(latency)
        results.append(result)

    false_positives = sum(
        run_control(_seeded(config, trial)) for trial in range(trials)
    )
    return results, false_positives, trials


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E3.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    print(f"E3: every fault class, {arguments.trials} trials each\n")
    results, false_positives, control_runs = run(config, arguments.trials)
    for result in results:
        print(result.report())

    print()
    print(
        f"  false positives: {false_positives} fault(s) raised across "
        f"{control_runs} healthy run(s) of "
        f"{(SETTLE_S + OBSERVE_S) / 3600.0:.0f} h each"
    )
    missed = [r.case.label for r in results if r.detections < r.trials]
    if missed:
        print(f"  not detected in every trial: {', '.join(missed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
