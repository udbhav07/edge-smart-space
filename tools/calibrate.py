"""Re-derive the detector thresholds from a recorded healthy run (R-04, Week 7).

    python -m tools.record runs/healthy.jsonl --seconds 7200
    python -m tools.calibrate runs/healthy.jsonl --out config/site.yaml
    python start.py --config config/site.yaml

Every detector threshold in ``config/default.yaml`` was set against the
simulator's noise. R-04 requires them re-derived from the room's own: what a
real sensor's spread is, how long a real WiFi node goes quiet, how far a
healthy residual wanders. This reads a recording -- the real room's, or the
simulator's -- measures those things, and writes an overlay that extends the
base configuration with the settings derived from them. Nothing else changes.

What is derived, and from what:

* **D1** ``timeout_periods`` -- the longest gap between arrivals, per sensor and
  against that sensor's own period, times ``calibration.timeout_margin``. A
  timeout shorter than a gap the healthy room produced is a false alarm
  already measured.
* **D2** ``variance_epsilon`` -- the quietest window of ``window_samples``
  readings any continuous sensor produced, times
  ``calibration.variance_fraction``. A settled room is quiet; a stuck sensor
  is silent, and the margin sits between them.
* **D4** ``threshold_sigma`` -- the D4 detector itself, replayed over the
  recorded estimates, times ``calibration.cusum_margin``. The healthy peak is
  what the threshold must clear.

Each derived value is reported beside the one it replaces and the measurement
behind it, so the overlay can be argued with in a review. D3's bounds are the
instruments' limits, not their noise, and D5's window is the room's time
constant; neither is derived here, and the report says so.

A recording that contains a fault is refused: calibrating "healthy" from a run
that was not would set the thresholds to accept the fault.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from sim.replay import Recorded, load
from src.common import topics
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import topic_matches
from src.common.schemas import FaultEvent, SensorReading, ThermalEstimate, Unit
from src.faults.detectors.drift import DriftDetector

LOGGER = logging.getLogger("tools.calibrate")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

SECONDS_PER_HOUR = 3600.0

#: Units whose variance D2 judges (as in the detector bank).
_VARIANCE_UNITS = frozenset({Unit.CELSIUS, Unit.PERCENT_RH})

#: Resolution derived values are rounded to, so the overlay reads as a
#: setting rather than as a measurement to nine places.
_TIMEOUT_STEP = 0.5
_SIGNIFICANT = 3


class CalibrationError(ValueError):
    """The recording cannot be calibrated from, and why."""


@dataclass
class SensorMeasurement:
    """What one sensor did over the recording."""

    sensor_id: str
    unit: Unit
    period_s: float
    arrivals: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    @property
    def samples(self) -> int:
        return len(self.values)

    @property
    def longest_gap_periods(self) -> float:
        """The longest silence between arrivals, in this sensor's periods."""
        if len(self.arrivals) < 2:
            return 0.0
        gaps = (later - earlier for earlier, later in zip(self.arrivals, self.arrivals[1:]))
        return max(gaps) / self.period_s

    @property
    def noise_sigma(self) -> float:
        """Sample noise, from first differences: a slow trend cancels, noise
        does not, and differencing doubles its variance."""
        if len(self.values) < 3:
            return 0.0
        steps = [later - earlier for earlier, later in zip(self.values, self.values[1:])]
        return statistics.pstdev(steps) / math.sqrt(2.0)

    def quietest_variance(self, window: int) -> float | None:
        """The smallest variance over any ``window`` consecutive readings."""
        if len(self.values) < window:
            return None
        return min(
            statistics.pvariance(self.values[start:start + window])
            for start in range(len(self.values) - window + 1)
        )


@dataclass(frozen=True)
class Derived:
    """One setting: what it was, what it becomes, and why."""

    path: str
    current: float
    derived: float
    basis: str


@dataclass(frozen=True)
class Calibration:
    """The result: what was measured and what follows from it."""

    hours: float
    sensors: tuple[SensorMeasurement, ...]
    settings: tuple[Derived, ...]
    not_derived: tuple[str, ...]
    warnings: tuple[str, ...] = ()


def measure(recorded: list[Recorded], config: Config) -> tuple[dict[str, SensorMeasurement], list[ThermalEstimate], float]:
    """Decode the recording into per-sensor series and the model's estimates.

    :raises CalibrationError: if the run contains a fault, or is too short.
    """
    if not recorded:
        raise CalibrationError("the recording is empty")
    hours = (recorded[-1].t - recorded[0].t) / SECONDS_PER_HOUR
    if hours < config.calibration.min_hours:
        raise CalibrationError(
            f"the recording covers {hours:.2f} h; at least "
            f"{config.calibration.min_hours:g} h of a healthy room is needed"
        )

    sensors = {
        sensor.sensor_id: SensorMeasurement(
            sensor.sensor_id, sensor.unit, _period_s(config, sensor.sensor_id)
        )
        for sensor in config.sensors.adapters
    }
    estimates: list[ThermalEstimate] = []
    for entry in recorded:
        if topic_matches(topics.FAULT.wildcard(), entry.topic) and entry.payload:
            fault = _decoded(FaultEvent, entry)
            if fault is not None:
                raise CalibrationError(
                    f"the recording contains a fault ({fault.detector.value} on "
                    f"{fault.subject} at t={entry.t:.0f} s); calibrate from a "
                    f"healthy run, or the thresholds will be set to accept it"
                )
        elif topic_matches(topics.SENSOR_STATE.wildcard(), entry.topic):
            reading = _decoded(SensorReading, entry)
            if reading is not None and reading.sensor_id in sensors:
                series = sensors[reading.sensor_id]
                series.arrivals.append(entry.t)
                series.values.append(reading.value)
        elif entry.topic == topics.ESTIMATE_THERMAL.pattern:
            estimate = _decoded(ThermalEstimate, entry)
            if estimate is not None:
                estimates.append(estimate)
    return sensors, estimates, hours


def derive(recorded: list[Recorded], config: Config) -> Calibration:
    """Measure the recording and derive every threshold that noise decides.

    :raises CalibrationError: if the recording cannot be calibrated from.
    """
    sensors, estimates, hours = measure(recorded, config)
    margins = config.calibration
    detectors = config.detectors
    settings: list[Derived] = []

    reporting = [series for series in sensors.values() if series.samples >= 2]
    if not reporting:
        raise CalibrationError("no configured sensor reported during the recording")
    worst = max(reporting, key=lambda series: series.longest_gap_periods)
    timeout = _step_up(max(worst.longest_gap_periods, 1.0) * margins.timeout_margin, _TIMEOUT_STEP)
    settings.append(
        Derived(
            "detectors.dropout.timeout_periods",
            detectors.dropout.timeout_periods,
            timeout,
            f"longest healthy gap {worst.longest_gap_periods:.2f} periods on "
            f"{worst.sensor_id}, x{margins.timeout_margin:g}",
        )
    )
    warnings: list[str] = []
    indoor_latency_s = (timeout + 1.0) * config.loop.sensor_period_s
    if indoor_latency_s > margins.dropout_latency_budget_s:
        warnings.append(
            f"NFR-02: a {timeout:g}-period timeout finds an indoor dropout in "
            f"about {indoor_latency_s:.0f} s, past the "
            f"{margins.dropout_latency_budget_s:g} s budget. The link is "
            f"dropping more than the requirement allows; fix the link, not "
            f"the threshold"
        )

    window = detectors.stuck_at.window_samples
    quiet = [
        (series.quietest_variance(window), series.sensor_id)
        for series in sensors.values()
        if series.unit in _VARIANCE_UNITS
    ]
    quiet = [(variance, sensor_id) for variance, sensor_id in quiet if variance is not None]
    if quiet:
        variance, sensor_id = min(quiet)
        settings.append(
            Derived(
                "detectors.stuck_at.variance_epsilon",
                detectors.stuck_at.variance_epsilon,
                _significant(variance * margins.variance_fraction),
                f"quietest healthy {window}-sample window {variance:.5f} on "
                f"{sensor_id}, x{margins.variance_fraction:g}",
            )
        )

    peak = _healthy_cusum_peak(estimates, config)
    if peak is not None:
        drift = detectors.drift
        # The most one sample can add is the cap less the slack; a threshold
        # below a few of those is tripped by magnitude, not by persistence.
        floor = (drift.max_sample_sigma - drift.slack_sigma) * margins.cusum_min_samples
        measured = peak * margins.cusum_margin
        basis = (
            f"healthy CUSUM peak {peak:.2f} sigma over {len(estimates)} "
            f"estimates, x{margins.cusum_margin:g}"
        )
        if measured < floor:
            basis += (
                f" = {measured:.2f}, below the persistence floor of "
                f"{margins.cusum_min_samples} capped samples ({floor:g}); the "
                f"floor applies"
            )
        settings.append(
            Derived(
                "detectors.drift.threshold_sigma",
                drift.threshold_sigma,
                _significant(max(measured, floor)),
                basis,
            )
        )

    return Calibration(
        hours=hours,
        sensors=tuple(sensors.values()),
        settings=tuple(settings),
        not_derived=(
            "detectors.out_of_range: the instruments' limits, not their noise",
            "detectors.actuator: set by the room's time constant and the unit's "
            "capacity, which a healthy recording alone cannot bound",
        ),
        warnings=tuple(warnings),
    )


def render(calibration: Calibration, base: Path, out: Path) -> str:
    """The overlay, as YAML that extends ``base`` and states its reasons."""
    lines = [
        "# Detector thresholds re-derived from a recorded healthy run (R-04).",
        f"# Written by tools/calibrate.py from {calibration.hours:.1f} h of recording.",
        "# Each value states what it was measured from; everything not named",
        "# here comes unchanged from the file this extends.",
        "#",
        "# Measured noise, per sensor:",
    ]
    for series in calibration.sensors:
        spread = (
            "binary"
            if series.unit is Unit.BOOLEAN
            else f"sigma {series.noise_sigma:.4f} {series.unit.value}"
        )
        lines.append(
            f"#   {series.sensor_id:11s} {series.samples:6d} samples  {spread}  "
            f"longest gap {series.longest_gap_periods:.2f} periods"
        )
    lines.extend(f"# WARNING {warning}" for warning in calibration.warnings)
    lines.append("#")
    lines.append("# Not derived:")
    lines.extend(f"#   {reason}" for reason in calibration.not_derived)
    lines.append(f"extends: {_relative(base, out)}")

    tree: dict[str, dict[str, list[Derived]]] = defaultdict(lambda: defaultdict(list))
    for setting in calibration.settings:
        section, group, _key = setting.path.split(".")
        tree[section][group].append(setting)
    for section, groups in tree.items():
        lines.append(f"{section}:")
        for group, settings in groups.items():
            lines.append(f"  {group}:")
            for setting in settings:
                key = setting.path.rsplit(".", 1)[1]
                lines.append(f"    # {setting.basis}; was {setting.current:g}")
                lines.append(f"    {key}: {setting.derived:g}")
    return "\n".join(lines) + "\n"


def report(calibration: Calibration) -> str:
    """What changed, for the terminal."""
    lines = [f"calibrated from {calibration.hours:.1f} h of recording", ""]
    for setting in calibration.settings:
        lines.append(
            f"  {setting.path:38s} {setting.current:>9g} -> {setting.derived:<9g} "
            f"({setting.basis})"
        )
    lines.append("")
    lines.extend(f"  not derived: {reason}" for reason in calibration.not_derived)
    lines.extend(f"  WARNING {warning}" for warning in calibration.warnings)
    return "\n".join(lines)


def _healthy_cusum_peak(estimates: list[ThermalEstimate], config: Config) -> float | None:
    """Replay D4 over the recorded estimates and keep its highest sum.

    The detector itself, not a reimplementation, so the peak is measured on
    exactly the test it calibrates -- warm-up, slack and the once-per-horizon
    accumulation included.
    """
    if not estimates:
        return None
    detector = DriftDetector(
        subject=config.estimator.indoor_sensor_id,
        config=config.detectors.drift,
        accumulate_every=config.estimator.drift_horizon_samples,
    )
    peak = None
    for estimate in estimates:
        detector.observe(estimate)
        evidence = detector.evaluate().evidence
        if "cusum_high" in evidence:
            worst = max(evidence["cusum_high"], evidence["cusum_low"])
            peak = worst if peak is None else max(peak, worst)
    return peak


def _period_s(config: Config, sensor_id: str) -> float:
    if sensor_id == config.estimator.outdoor_sensor_id:
        return config.loop.outdoor_period_s
    return config.loop.sensor_period_s


def _decoded(schema, entry: Recorded):
    try:
        return schema.model_validate_json(entry.payload)
    except ValidationError:
        LOGGER.debug("skipping a malformed %s at t=%.0f", schema.__name__, entry.t)
        return None


def _step_up(value: float, step: float) -> float:
    return math.ceil(value / step) * step


def _significant(value: float) -> float:
    if value == 0.0:
        return 0.0
    digits = _SIGNIFICANT - 1 - math.floor(math.log10(abs(value)))
    return round(value, digits)


def _relative(base: Path, out: Path) -> str:
    """``base`` as the overlay should name it: relative to the overlay."""
    try:
        return Path(os.path.relpath(base.resolve(), out.resolve().parent)).as_posix()
    except ValueError:
        # Different drives on Windows: no relative path exists.
        return base.resolve().as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.calibrate", description=__doc__.splitlines()[0])
    parser.add_argument("recording", type=Path, help="A tools.record recording of a healthy room")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="The configuration to extend")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the overlay; omit to only report")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        config = load_config(arguments.config)
        calibration = derive(load(arguments.recording), config)
    except (ConfigError, CalibrationError, OSError) as exc:
        LOGGER.error("cannot calibrate: %s", exc)
        return 2

    print(report(calibration))
    if arguments.out is not None:
        text = render(calibration, arguments.config, arguments.out)
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(text, encoding="utf-8")
        try:
            load_config(arguments.out)
        except ConfigError as exc:
            LOGGER.error("the overlay written does not load: %s", exc)
            return 1
        print(f"\nwrote {arguments.out}; run with --config {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
