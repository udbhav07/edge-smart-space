"""Wires the detector bank to the blackboard.

Subscribes to every sensor topic, runs D1 to D3 at regulatory cadence, folds
their judgments through the aggregator, and publishes what comes out: a
``FaultEvent`` per raised fault (FR-20 to FR-22) and a ``SensorHealth`` per
sensor whose trustworthiness changed.

**Health is the part the rest of the system acts on.** A FaultEvent is for a
human and for the mode manager; ``space/sensor/{id}/health`` is what makes the
estimator freeze adaptation on a faulted regressor input (FR-29). It was
already subscribed there before this service existed and nothing was ever
publishing it, so the freeze was unreachable. That is why health is published
here rather than left to the mode manager: the fact "this sensor is not to be
trusted" is a detection result, not a mode decision.

**Which detectors watch which sensor is decided from config, not assumed.**
A binary occupancy sensor gets D1 only: variance says nothing about a PIR in an
empty room, and a two-valued signal has no range to leave. Getting this wrong
does not fail loudly -- it produces a fault every quiet night.

**The dropout timeout is per sensor, because the periods differ.** Ambient
updates every 60 s (FR-03, A-04) while indoor updates every 5 s. One timeout
derived from the indoor period would report the outdoor sensor as dropped
almost continuously, and the resulting alert fatigue would be indistinguishable
from the detector not working.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from src.common import topics
from src.common.clock import Clock
from src.common.config import Bounds, Config, SensorConfig
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Command,
    DetectorId,
    FaultEvent,
    Quality,
    SensorHealth,
    SensorReading,
    ThermalEstimate,
    Unit,
)
from src.faults.aggregator import AggregateOutcome, FaultAggregator
from src.faults.detectors.actuator import ActuatorResponseDetector
from src.faults.detectors.base import Finding, Judgment
from src.faults.detectors.drift import DriftDetector
from src.faults.detectors.dropout import DropoutDetector
from src.faults.detectors.out_of_range import OutOfRangeDetector
from src.faults.detectors.stuck_at import StuckAtDetector

LOGGER = logging.getLogger(__name__)

#: Units for which variance and range tests are meaningful. Everything else
#: gets dropout detection only.
_CONTINUOUS_UNITS = frozenset({Unit.CELSIUS, Unit.PERCENT_RH})


class SubjectDetectors:
    """The detectors watching one sensor.

    Grouped per subject so a reading is routed by sensor id without asking
    every detector in the bank whether it cares.
    """

    def __init__(
        self,
        dropout: DropoutDetector,
        stuck_at: StuckAtDetector | None,
        out_of_range: OutOfRangeDetector | None,
        drift: DriftDetector | None = None,
    ) -> None:
        self._dropout = dropout
        self._stuck_at = stuck_at
        self._out_of_range = out_of_range
        self._drift = drift

    def observe(self, reading: SensorReading) -> None:
        self._dropout.observe(reading)
        if self._stuck_at is not None:
            self._stuck_at.observe(reading)
        if self._out_of_range is not None:
            self._out_of_range.observe(reading)

    def observe_estimate(self, estimate: ThermalEstimate) -> None:
        """Feed the model's prediction error to D4, if this subject has one."""
        if self._drift is not None:
            self._drift.observe(estimate)

    def evaluate(self) -> tuple[Finding, ...]:
        findings = [self._dropout.evaluate()]
        if self._stuck_at is not None:
            findings.append(self._stuck_at.evaluate())
        if self._out_of_range is not None:
            findings.append(self._out_of_range.evaluate())
        if self._drift is not None:
            findings.append(self._drift.evaluate())
        return tuple(findings)

    def reset(self, detector: DetectorId) -> None:
        """Clear one detector's accumulated evidence after its fault retires.

        Only D4 accumulates anything across a fault: D1 to D3 recompute from
        their own inputs every tick, so there is nothing in them to stale.
        Without this the CUSUM stays above its threshold and re-raises on the
        next sample, and a recalibrated sensor would be faulted forever.
        """
        if detector is DetectorId.D4_DRIFT and self._drift is not None:
            self._drift.reset()


class DetectorBankService:
    """One detector bank, publishing faults and sensor health."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        aggregator: FaultAggregator,
        detectors: dict[str, SubjectDetectors],
        actuator: ActuatorResponseDetector,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._aggregator = aggregator
        self._detectors = detectors
        self._actuator = actuator
        self._last_reading_ts: dict[str, float] = {}
        self._published_quality: dict[str, Quality] = {}

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        """Listen to everything the bank tests against.

        Three sources, one per kind of question. Readings answer whether a
        sensor is behaving (D1 to D3). The model's estimate answers whether it
        is telling the truth (D4). Commands answer whether the plant responds
        to them (D5) -- commands rather than actuator state, because FR-24 asks
        what happened after a *sustained command*, and a detector that needed
        the plant to report its own state could not detect a plant that had
        stopped reporting.
        """
        self._blackboard.subscribe(
            topics.SENSOR_STATE, SensorReading, self._on_reading
        )
        self._blackboard.subscribe(
            topics.ESTIMATE_THERMAL, ThermalEstimate, self._on_estimate
        )
        self._blackboard.subscribe(
            topics.ACTUATOR_COMMAND, Command, self._on_command
        )

    @property
    def watched_subjects(self) -> frozenset[str]:
        """Sensors this bank has detectors for."""
        return frozenset(self._detectors)

    @property
    def active_faults(self) -> tuple[FaultEvent, ...]:
        """Live faults, most severe first. For the console and for tests."""
        return self._aggregator.active

    # --- observation --------------------------------------------------

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        """Route one reading to the detectors watching its sensor.

        A reading from a sensor no detector watches is logged and dropped
        rather than raising: an unconfigured sensor appearing on the bus is a
        configuration gap, and taking the service down over it would lose
        detection on every sensor that *is* configured.
        """
        subject = self._detectors.get(reading.sensor_id)
        if subject is None:
            LOGGER.debug("no detectors configured for %s", reading.sensor_id)
            return
        subject.observe(reading)
        self._last_reading_ts[reading.sensor_id] = reading.ts
        if reading.sensor_id == self._config.estimator.indoor_sensor_id:
            # D5 judges the actuator by what the room did, so it needs the
            # room's temperature and no other sensor's.
            self._actuator.observe_reading(reading)

    def _on_estimate(self, _topic: str, estimate: ThermalEstimate) -> None:
        """Route the model's prediction error to D4.

        The residual exists only for the sensor the model predicts, so only
        that subject has a drift detector to receive it.
        """
        subject = self._detectors.get(self._config.estimator.indoor_sensor_id)
        if subject is None:
            LOGGER.debug("no detectors for the indoor sensor; estimate ignored")
            return
        subject.observe_estimate(estimate)

    def _on_command(self, _topic: str, command: Command) -> None:
        """Route an actuator command to D5."""
        if command.actuator_id != self._actuator.subject:
            LOGGER.debug("no detector for actuator %s", command.actuator_id)
            return
        self._actuator.observe_command(command.kind)

    # --- the tick -----------------------------------------------------

    def tick(self) -> AggregateOutcome:
        """Evaluate every detector once and publish what changed.

        :returns: the aggregate outcome, so a caller -- the mode manager, a
            test, an experiment -- can see the transitions without
            re-subscribing to the topics they were published on.
        """
        findings: list[Finding] = []
        for subject in self._detectors.values():
            findings.extend(subject.evaluate())
        findings.append(self._actuator.evaluate())

        outcome = self._aggregator.ingest(findings)
        for event in outcome.raised:
            self._publish_fault(event)
        for event in outcome.cleared:
            self._withdraw_fault(event)
            self._reset_detector(event)
        self._publish_health(findings)
        return outcome

    def _reset_detector(self, event: FaultEvent) -> None:
        """Let a detector that accumulates evidence start afresh (FR-30)."""
        subject = self._detectors.get(event.subject)
        if subject is not None:
            subject.reset(event.detector)
        elif event.subject == self._actuator.subject:
            self._actuator.reset()

    def _publish_fault(self, event: FaultEvent) -> None:
        topic = self._blackboard.publish(
            topics.FAULT, event, fault_id=event.fault_id
        )
        LOGGER.warning("published %s on %s", event.detector.value, topic)

    def _withdraw_fault(self, event: FaultEvent) -> None:
        """A retired fault stops being current state (FR-30).

        The event is withdrawn rather than republished with a cleared flag,
        because the retained topic answers "what is wrong now" and a resolved
        fault is not an answer to that. The transition is still visible: it is
        in the log, in the health message, and in the mode change that follows.
        """
        self._blackboard.clear_retained(topics.FAULT, fault_id=event.fault_id)
        LOGGER.info("withdrew %s", event.fault_id)

    def _publish_health(self, findings: Iterable[Finding]) -> None:
        """Publish health for every sensor whose quality changed.

        Only on change: the topic is retained, so a late subscriber gets the
        current value without this service restating it every tick.

        A sensor whose every detector still answers UNKNOWN is not published at
        all. There is no Quality for "not yet observed", and OK would assert
        health nobody has measured -- which is the state the system sits in for
        the first five minutes, while D2's window fills.
        """
        faulted = self._faulted_subjects()
        for sensor_id, quality in self._observed_quality(findings).items():
            if sensor_id not in self._detectors:
                # The actuator is a subject but not a sensor; it has no
                # SensorHealth topic and publishing one would invent a sensor.
                continue
            if self._published_quality.get(sensor_id) is quality:
                continue
            self._published_quality[sensor_id] = quality
            self._blackboard.publish(
                topics.SENSOR_HEALTH,
                SensorHealth(
                    ts=self._clock.now(),
                    sensor_id=sensor_id,
                    quality=quality,
                    last_reading_ts=self._last_reading_ts.get(sensor_id),
                    active_fault_id=faulted.get(sensor_id),
                ),
                sensor_id=sensor_id,
            )
            LOGGER.info("%s is %s", sensor_id, quality.value)

    def _observed_quality(
        self, findings: Iterable[Finding]
    ) -> dict[str, Quality]:
        """Quality per sensor, for sensors something has actually been observed
        about. A subject absent from the result is a subject nothing is claimed
        about yet."""
        observed: dict[str, Quality] = {}
        for finding in findings:
            if finding.faulted:
                observed[finding.subject] = Quality.FAULTED
            elif finding.judgment is Judgment.CLEAR:
                observed.setdefault(finding.subject, Quality.OK)
        return observed

    def _faulted_subjects(self) -> dict[str, str]:
        """Which sensors have a live fault, and which fault to name.

        A sensor can be faulted by more than one detector at once -- a stuck
        sensor that then goes silent is the ordinary case -- and health carries
        one id, so it names the most severe, which is the aggregator's order.
        """
        faulted: dict[str, str] = {}
        for event in self._aggregator.active:
            faulted.setdefault(event.subject, event.fault_id)
        return faulted


def _dropout_period_s(config: Config, sensor: SensorConfig) -> float:
    """The cadence this sensor actually reports at.

    Ambient is published every 60 s (FR-03, A-04); everything else at the
    sensor period. A single timeout derived from the fast period would report
    the outdoor sensor as dropped almost continuously.
    """
    if sensor.sensor_id == config.estimator.outdoor_sensor_id:
        return config.loop.outdoor_period_s
    return config.loop.sensor_period_s


def _bounds_for(config: Config, unit: Unit) -> Bounds | None:
    """D3's bounds for a unit, or None when the unit has no range to leave."""
    out_of_range = config.detectors.out_of_range
    if unit is Unit.CELSIUS:
        return out_of_range.temperature_c
    if unit is Unit.PERCENT_RH:
        return out_of_range.humidity_pct
    return None


def build_detectors(config: Config, clock: Clock) -> dict[str, SubjectDetectors]:
    """Assemble the bank from the configured sensors.

    Every sensor gets D1. Only continuous ones get D2 and D3, because variance
    and range say nothing about a binary signal. Only the indoor sensor gets
    D4: the model predicts that one temperature, so it is the only subject
    there is a residual for (section 5.5).
    """
    bank: dict[str, SubjectDetectors] = {}
    for sensor in config.sensors.adapters:
        continuous = sensor.unit in _CONTINUOUS_UNITS
        bounds = _bounds_for(config, sensor.unit)
        bank[sensor.sensor_id] = SubjectDetectors(
            dropout=DropoutDetector(
                subject=sensor.sensor_id,
                config=config.detectors.dropout,
                sensor_period_s=_dropout_period_s(config, sensor),
                clock=clock,
            ),
            stuck_at=(
                StuckAtDetector(
                    subject=sensor.sensor_id,
                    unit=sensor.unit,
                    config=config.detectors.stuck_at,
                    clock=clock,
                )
                if continuous
                else None
            ),
            out_of_range=(
                OutOfRangeDetector(
                    subject=sensor.sensor_id,
                    unit=sensor.unit,
                    bounds=bounds,
                    debounce_samples=config.detectors.out_of_range.debounce_samples,
                )
                if continuous and bounds is not None
                else None
            ),
            drift=(
                DriftDetector(
                    subject=sensor.sensor_id, config=config.detectors.drift
                )
                if sensor.sensor_id == config.estimator.indoor_sensor_id
                else None
            ),
        )
    return bank


def build_service(
    config: Config, clock: Clock, blackboard: Blackboard
) -> DetectorBankService:
    """Assemble the bank and its aggregator from configuration."""
    return DetectorBankService(
        config=config,
        clock=clock,
        blackboard=blackboard,
        aggregator=FaultAggregator(
            clock=clock, clear_confirm_s=config.mode.fault_clear_confirm_s
        ),
        detectors=build_detectors(config, clock),
        actuator=ActuatorResponseDetector(
            subject=topics.AIR_CONDITIONER_ID,
            config=config.detectors.actuator,
            clock=clock,
        ),
    )
