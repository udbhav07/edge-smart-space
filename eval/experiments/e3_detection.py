"""E3: does the system detect every injected fault class? (DESIGN.md section 8.3)

Method: five fault classes, ten trials each, every trial a fresh system with
its own seed, warmed up before the fault is injected at the sensor or the
plant exactly as ``tools.inject`` would. Metrics: detection rate by the
*right* detector, latency from injection to that detector's fault, and the
false-positive rate over fault-free runs of the same length.

"Right detector" is the point. A drifting sensor blamed on the air
conditioner is not a detection; it is a misdiagnosis, and it is counted as a
miss with its cause named.

Run it: ``python -m eval.experiments.e3_detection``
"""

from __future__ import annotations

import argparse
import logging
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

from eval.system import System
from src.common.clock import SimClock
from src.common.config import Config, load_config
from src.common.injection import InjectedFault
from src.common.schemas import DetectorId
from src.common.topics import AIR_CONDITIONER_ID
from sim.run_sim import INDOOR_TEMPERATURE_ID

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
WARMUP_S = 1800.0
STEP_S = 5.0


@dataclass(frozen=True)
class FaultClass:
    name: str
    expected: DetectorId
    kind: InjectedFault
    magnitude: float | None
    horizon_s: float


FAULT_CLASSES = (
    FaultClass("dropout", DetectorId.D1_DROPOUT, InjectedFault.DROPOUT, None, 120.0),
    FaultClass("stuck-at", DetectorId.D2_STUCK_AT, InjectedFault.STUCK_AT, 27.0, 900.0),
    FaultClass("out-of-range", DetectorId.D3_OUT_OF_RANGE, InjectedFault.OUT_OF_RANGE, 999.0, 120.0),
    FaultClass("drift 0.6 C/min", DetectorId.D4_DRIFT, InjectedFault.DRIFT, 0.01, 1800.0),
    FaultClass("actuator", DetectorId.D5_ACTUATOR_NO_RESPONSE, InjectedFault.NO_RESPONSE, None, 1500.0),
)


@dataclass(frozen=True)
class Trial:
    fault: str
    seed: int
    detected: bool
    latency_s: float | None
    first_detector: str


def _configured(config: Config, seed: int, directory: Path) -> Config:
    persistence = config.persistence.model_copy(update={"path": str(directory / f"c{seed}.json")})
    sim = config.sim.model_copy(update={"random_seed": seed})
    return config.model_copy(update={"persistence": persistence, "sim": sim})


def run_trial(config: Config, fault: FaultClass, seed: int, directory: Path) -> Trial:
    system = System(_configured(config, seed, directory), SimClock())
    system.run_for(WARMUP_S)
    before = {event.fault_id for event in system.faults()}
    injected_at = system.clock.now()
    subject = (
        AIR_CONDITIONER_ID if fault.kind is InjectedFault.NO_RESPONSE else INDOOR_TEMPERATURE_ID
    )
    # Every class goes through the injection channel tools.inject uses (FR-31).
    if fault.magnitude is None:
        system.injector.inject(subject, fault.kind)
    else:
        system.injector.inject(INDOOR_TEMPERATURE_ID, fault.kind, fault.magnitude)
    elapsed = 0.0
    while elapsed < fault.horizon_s:
        system.run_for(STEP_S)
        elapsed += STEP_S
        new = [event for event in system.faults() if event.fault_id not in before]
        hit = next((event for event in new if event.detector is fault.expected), None)
        if hit is not None:
            first = new[0].detector.value
            return Trial(fault.name, seed, True, hit.detected_ts - injected_at, first)
    new = [event for event in system.faults() if event.fault_id not in before]
    return Trial(fault.name, seed, False, None, new[0].detector.value if new else "")


def false_positives(config: Config, seeds: range, duration_s: float, directory: Path) -> list[int]:
    counts = []
    for seed in seeds:
        system = System(_configured(config, seed, directory), SimClock())
        system.run_for(duration_s)
        counts.append(len(system.faults()))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E3.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--trials", type=int, default=10)
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.CRITICAL)
    config = load_config(arguments.config)

    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        print("--- E3: detection by the right detector ---")
        print(f"{'fault':18s} {'detected':>9s} {'median s':>9s} {'max s':>7s}  misses")
        for fault in FAULT_CLASSES:
            trials = [
                run_trial(config, fault, seed, directory)
                for seed in range(1, arguments.trials + 1)
            ]
            hits = [trial for trial in trials if trial.detected]
            latencies = [trial.latency_s for trial in hits]
            misses = [
                f"seed {trial.seed}: {trial.first_detector or 'nothing'}"
                for trial in trials
                if not trial.detected
            ]
            print(
                f"{fault.name:18s} {len(hits):>4d}/{len(trials):<4d}"
                f" {statistics.median(latencies) if latencies else float('nan'):>9.0f}"
                f" {max(latencies) if latencies else float('nan'):>7.0f}  {'; '.join(misses)}"
            )
        duration_s = WARMUP_S + max(fault.horizon_s for fault in FAULT_CLASSES)
        counts = false_positives(config, range(101, 101 + arguments.trials), duration_s, directory)
        print(
            f"--- fault-free: {sum(1 for count in counts if count)} of {len(counts)} runs "
            f"of {duration_s / 3600:.1f} h raised any fault ({sum(counts)} faults in all) ---"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
