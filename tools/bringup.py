"""Is the hardware actually there? The Week 5 bring-up check.

    python -m tools.bringup                     # listen, then report every sensor
    python -m tools.bringup --seconds 180       # a longer look, for sigma
    python -m tools.bringup --actuate           # and make the air conditioner run

Listens to the blackboard and reports, for every sensor the configuration
expects: whether it is publishing, at what interval against the one it should
have, with how much jitter, whether its values sit inside the instrument's
limits, and the mean and standard deviation of what it read. That last number
is R-04's input: Week 7 re-derives every detector threshold from measured
sigma, and this is where the sigma comes from.

``--actuate`` checks that the air conditioner takes real commands without
sending one. It proposes an operator goal two degrees below the room to the
gate on ``space/goal/proposed`` -- the validator still decides, the controller
still commands, nothing here writes an actuator topic (FR-13, FR-45) -- and
then watches the power meter for the compressor's draw. An IR path cannot
acknowledge (R-02); a watt-meter can, and this is where that is shown. The
goal expires on its own at the end of the window, so a check left running
does not leave the room held two degrees down.

It is equally a check of the simulator: the same topics, the same report.
"""

from __future__ import annotations

import argparse
import logging
import statistics
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.schemas import (
    ActuatorState,
    Goal,
    GoalSource,
    Mode,
    ModeState,
    SensorReading,
    ValidationVerdict,
)

LOGGER = logging.getLogger("bringup")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "bringup"

#: Readings kept per sensor. An hour at 5 s; bounded whatever runs this.
MAX_READINGS_KEPT = 720

#: The power meter, if one is configured. The actuation check needs it.
POWER_SENSOR_ID = "pwr_01"


@dataclass(frozen=True)
class SensorVerdict:
    """What the bring-up saw of one sensor."""

    sensor_id: str
    count: int
    expected_period_s: float
    median_interval_s: float | None
    p95_jitter_s: float | None
    mean: float | None
    sigma: float | None
    outside_limits: int
    problem: str

    @property
    def ok(self) -> bool:
        return not self.problem


@dataclass(frozen=True)
class ActuationVerdict:
    """Whether commanding cooling made the compressor draw power."""

    baseline_w: float | None
    peak_w: float | None
    required_rise_w: float
    verdict_seen: bool
    problem: str

    @property
    def ok(self) -> bool:
        return not self.problem


class BringupMonitor:
    """Watches the blackboard and judges what the hardware is doing."""

    def __init__(self, config: Config, clock: Clock, blackboard: Blackboard) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._readings: dict[str, deque[SensorReading]] = {
            sensor.sensor_id: deque(maxlen=MAX_READINGS_KEPT)
            for sensor in config.sensors.adapters
        }
        self._actuator: ActuatorState | None = None
        self._mode = Mode.INIT
        self._actuation_started_ts: float | None = None
        self._actuation_verdict: ValidationVerdict | None = None

    def subscribe(self) -> None:
        board = self._blackboard
        board.subscribe(topics.SENSOR_STATE, SensorReading, self._on_reading)
        board.subscribe(topics.ACTUATOR_STATE, ActuatorState, self._on_actuator)
        board.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)
        board.subscribe(topics.AUDIT_VALIDATION, ValidationVerdict, self._on_verdict)

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        kept = self._readings.get(reading.sensor_id)
        if kept is not None:
            kept.append(reading)

    def _on_actuator(self, _topic: str, state: ActuatorState) -> None:
        self._actuator = state

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        self._mode = state.mode

    def _on_verdict(self, _topic: str, verdict: ValidationVerdict) -> None:
        if self._actuation_started_ts is not None and verdict.ts >= self._actuation_started_ts:
            if "setpoint_c" in verdict.proposed:
                self._actuation_verdict = verdict

    # --- sensors ------------------------------------------------------

    def _expected_period_s(self, sensor_id: str) -> float:
        if sensor_id == self._config.estimator.outdoor_sensor_id:
            return self._config.loop.outdoor_period_s
        return self._config.loop.sensor_period_s

    def sensors(self) -> list[SensorVerdict]:
        """One verdict per configured sensor, in configuration order."""
        return [self._judge(sensor.sensor_id) for sensor in self._config.sensors.adapters]

    def _judge(self, sensor_id: str) -> SensorVerdict:
        readings = list(self._readings[sensor_id])
        period_s = self._expected_period_s(sensor_id)
        if not readings:
            return SensorVerdict(
                sensor_id, 0, period_s, None, None, None, None, 0,
                "SILENT: nothing published. Is the node powered, on WiFi, and "
                "its topic the one io.devices names?",
            )

        stamps = sorted(reading.ts for reading in readings)
        intervals = [later - earlier for earlier, later in zip(stamps, stamps[1:])]
        values = [reading.value for reading in readings]
        limits = self._config.sensors.by_id(sensor_id).limits
        outside = sum(1 for value in values if not limits.contains(value))

        median = statistics.median(intervals) if intervals else None
        jitter = None
        if intervals:
            deviations = sorted(abs(interval - period_s) for interval in intervals)
            jitter = deviations[min(len(deviations) - 1, int(0.95 * len(deviations)))]
        sigma = statistics.pstdev(values) if len(values) > 1 else None

        problem = ""
        tolerance = self._config.bringup.max_interval_factor
        if median is None:
            problem = "ONE READING: listen longer to judge its rate"
        elif median > period_s * tolerance:
            problem = (
                f"SLOW: median interval {median:.1f} s against {period_s:.0f} s "
                f"expected; D1 will call this a dropout"
            )
        elif outside:
            problem = (
                f"OUT OF LIMITS: {outside} reading(s) outside "
                f"[{limits.low}, {limits.high}]; wiring or units"
            )
        return SensorVerdict(
            sensor_id, len(readings), period_s, median, jitter,
            statistics.fmean(values), sigma, outside, problem,
        )

    # --- actuation ----------------------------------------------------

    def _power_since(self, since_ts: float | None, until_ts: float | None = None) -> list[float]:
        readings = self._readings.get(POWER_SENSOR_ID, ())
        return [
            r.value
            for r in readings
            if (since_ts is None or r.ts >= since_ts) and (until_ts is None or r.ts < until_ts)
        ]

    def begin_actuation(self) -> Goal:
        """Ask the gate for cooling, as an operator, for one window."""
        indoor = self._readings.get(self._config.estimator.indoor_sensor_id)
        now = self._clock.now()
        room_c = indoor[-1].value if indoor else self._config.controller.default_setpoint_c
        goal = Goal(
            ts=now,
            source=GoalSource.OPERATOR,
            setpoint_c=room_c - self._config.bringup.actuation_drop_c,
            mode=self._mode,
            rationale="bring-up: does the air conditioner take real commands?",
            expires_ts=now + self._config.bringup.actuation_window_s,
        )
        self._actuation_started_ts = now
        self._actuation_verdict = None
        self._blackboard.publish(topics.GOAL_PROPOSED, goal)
        return goal

    def actuation(self) -> ActuationVerdict:
        """Judge the window since :meth:`begin_actuation`."""
        required = self._config.bringup.min_power_rise_w
        if POWER_SENSOR_ID not in self._readings:
            return ActuationVerdict(None, None, required, False, "no power meter configured")
        started = self._actuation_started_ts
        before = self._power_since(None, started)
        after = self._power_since(started)
        # The reference is the lowest draw seen before asking: the unit's
        # off-draw. A median would mix running and standby whenever the
        # controller was already cooling, and a unit that happened to be
        # running throughout would then fail a check it should pass.
        baseline = min(before) if before else None
        peak = max(after) if after else None
        seen = self._actuation_verdict is not None
        if not seen:
            problem = "the gate never answered: is the control process running?"
        elif baseline is None or peak is None:
            problem = "no power readings either side of the command"
        elif baseline >= required:
            problem = (
                f"INCONCLUSIVE: the unit never dropped below {baseline:.0f} W "
                f"before the check, so it was already running. Run the check "
                f"with the room near its setpoint"
            )
        elif peak - baseline < required:
            problem = (
                f"NO RESPONSE: draw rose {peak - baseline:.0f} W, needed "
                f"{required:.0f} W. Check the IR protocol and the blaster's aim "
                f"(R-02); D5 would call this an actuator fault"
            )
        else:
            problem = ""
        return ActuationVerdict(baseline, peak, required, seen, problem)

    # --- the report ---------------------------------------------------

    def report(self) -> str:
        lines = [
            f"{'sensor':<11} {'count':>5} {'period':>7} {'median':>7} "
            f"{'p95 jit':>7} {'mean':>9} {'sigma':>8}  verdict"
        ]
        for verdict in self.sensors():
            def fmt(value, spec):
                return "--" if value is None else format(value, spec)

            lines.append(
                f"{verdict.sensor_id:<11} {verdict.count:>5} "
                f"{verdict.expected_period_s:>6.0f}s "
                f"{fmt(verdict.median_interval_s, '>6.1f')}s "
                f"{fmt(verdict.p95_jitter_s, '>6.2f')}s "
                f"{fmt(verdict.mean, '>9.3f')} {fmt(verdict.sigma, '>8.4f')}  "
                f"{'ok' if verdict.ok else verdict.problem}"
            )
        if self._actuator is None:
            lines.append("actuator    nothing on space/actuator/+/state")
        else:
            label = "simulated" if self._actuator.simulated else "real"
            lines.append(
                f"actuator    {self._actuator.actuator_id} ({label}) last "
                f"{self._actuator.kind.value}, ack {self._actuator.ack.value}"
            )
        lines.append(f"mode        {self._mode.value}")
        lines.append(
            "sigma is R-04's input: Week 7 sets the detector thresholds from it, "
            "not from the placeholders in config."
        )
        return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the hardware is really there.")
    parser.add_argument("--seconds", type=float, default=None, help="How long to listen")
    parser.add_argument(
        "--actuate", action="store_true", help="Also check the AC runs when asked"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    monitor = BringupMonitor(config, clock, blackboard)
    monitor.subscribe()
    blackboard.start()
    listen_s = arguments.seconds or config.bringup.listen_s
    actuation = None
    try:
        print(f"listening for {listen_s:.0f} s on {config.mqtt.host}:{config.mqtt.port} ...")
        clock.sleep(listen_s)
        if arguments.actuate:
            goal = monitor.begin_actuation()
            print(
                f"asked the gate for {goal.setpoint_c:.1f} C as an operator; "
                f"watching {POWER_SENSOR_ID} for {config.bringup.actuation_window_s:.0f} s ..."
            )
            clock.sleep(config.bringup.actuation_window_s)
            actuation = monitor.actuation()
    finally:
        blackboard.stop()

    print(monitor.report())
    healthy = all(verdict.ok for verdict in monitor.sensors())
    if actuation is not None:
        if actuation.ok:
            print(
                f"actuation   ok: draw {actuation.baseline_w:.0f} W -> "
                f"{actuation.peak_w:.0f} W when cooling was asked for"
            )
        else:
            print(f"actuation   {actuation.problem}")
        healthy = healthy and actuation.ok
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
