"""Wires the estimator to the blackboard.

Subscribes to the sensor and actuator topics, forms the regressor, folds
each new indoor reading into the estimate, and publishes the prediction, the
residual and the coefficients (FR-04, FR-05).

The interesting part is what it refuses to do. The ARX form in section 5.2.1
is a *fixed-step* discretisation: ``a1`` means "fraction retained per step",
and a step is only meaningful if every step is the same length. A-02 assumes
uniform 5 s sampling, and A-02 will not survive WiFi -- reconnect bursts,
retries and gaps are ordinary. So a pair whose interval is wrong is skipped
rather than fitted, and so is one whose timestamps go backwards or repeat.
Fitting them would quietly rewrite what the coefficients mean, and nothing
downstream would be able to tell.

Freezing follows the same principle one level up: while a sensor feeding the
regressor is faulted, adaptation stops entirely (FR-29). A reading its own
sensor flags as suspect is never fitted either. Never adapt to bad data.

**While the indoor sensor is faulted, the model runs on itself (FR-27).** The
one-step prediction is built from the previous *reading*, so once that reading
is the fault, a one-step prediction is only the fault restated: a stuck sensor
yields the stuck value, an out-of-range one yields something derived from 999
C, and a silent one yields nothing new at all. None of that is a model standing
in for a sensor. So from the last trusted reading the service steps the model
forward on its own output, once per :meth:`tick`, and publishes that. Ticks
rather than readings drive it because a dropped sensor delivers no readings.
"""

from __future__ import annotations

import logging

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    ActuatorState,
    Coefficients,
    CommandKind,
    DetectorId,
    FaultClass,
    FaultEvent,
    Mode,
    ModeState,
    Quality,
    SensorHealth,
    SensorReading,
    ThermalEstimate,
)
from src.estimation.persistence import CoefficientStore
from src.estimation.rc_model import Regressor
from src.estimation.rls import ThermalEstimator, UpdateResult

LOGGER = logging.getLogger(__name__)

#: Normalised drive the model sees. Section 5.3's law is bang-bang, so u[k]
#: is binary; a3 is therefore identified against a two-valued input, which
#: is exactly the weak-excitation situation R-01 warns about.
COOLING_ON = 1.0
COOLING_OFF = 0.0

OCCUPIED = 1.0
VACANT = 0.0

#: Modes in which a sensor feeding the regressor cannot be trusted (FR-29).
_FREEZING_MODES = frozenset(
    {Mode.DEGRADED_SENSOR, Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD}
)

#: Confidence attached to a divergence finding. It is not a measurement:
#: three consecutive rejections is a rule, not a probability.
_DIVERGENCE_CONFIDENCE = 1.0


class ThermalEstimatorService:
    """One regulatory-cadence estimator, speaking the blackboard's language."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        estimator: ThermalEstimator,
        store: CoefficientStore,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._estimator = estimator
        self._store = store

        self._indoor_c: float | None = None
        self._indoor_ts: float | None = None
        self._outdoor_c: float | None = None
        self._command = COOLING_OFF
        self._occupancy = VACANT
        self._faulted_sensors: set[str] = set()
        self._mode = Mode.INIT
        self._skipped_pairs = 0
        self._suspect_readings = 0
        #: The model's own state while the indoor sensor is faulted; None
        #: while the sensor is trusted.
        self._open_loop_c: float | None = None
        #: The latest indoor value heard, trusted or not. Only published as
        #: t_in while running open loop, never fitted.
        self._last_heard_c: float | None = None

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        """Listen to everything the regressor is built from."""
        self._blackboard.subscribe(
            topics.SENSOR_STATE, SensorReading, self._on_reading
        )
        self._blackboard.subscribe(
            topics.SENSOR_HEALTH, SensorHealth, self._on_health
        )
        self._blackboard.subscribe(
            topics.ACTUATOR_STATE, ActuatorState, self._on_actuator
        )
        self._blackboard.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)

    def restore(self) -> bool:
        """Adopt a persisted estimate if there is a usable one (FR-07).

        :returns: whether anything was adopted. Starting from the prior is a
            normal outcome, not a failure.
        """
        persisted = self._store.load()
        if persisted is None:
            return False
        try:
            self._estimator.restore(persisted.theta, persisted.covariance)
        except ValueError as exc:
            LOGGER.warning("persisted estimate rejected on adoption: %s", exc)
            return False
        LOGGER.info("restored coefficients written %.0f s ago",
                    self._clock.now() - persisted.ts)
        return True

    # --- observed state -----------------------------------------------

    @property
    def skipped_pairs(self) -> int:
        """Pairs discarded for a bad interval. Rising means A-02 is failing."""
        return self._skipped_pairs

    @property
    def suspect_readings(self) -> int:
        """Indoor readings refused because their sensor flagged them."""
        return self._suspect_readings

    @property
    def open_loop(self) -> bool:
        """Whether the prediction is running on the model alone (FR-27)."""
        return self._open_loop_c is not None

    @property
    def adaptation_frozen(self) -> bool:
        return bool(self._faulted_sensors) or self._mode in _FREEZING_MODES

    def _on_health(self, _topic: str, health: SensorHealth) -> None:
        """Track which regressor inputs are untrustworthy (FR-29)."""
        if health.sensor_id not in self._regressor_sensor_ids():
            return
        if health.quality is Quality.FAULTED:
            self._faulted_sensors.add(health.sensor_id)
        else:
            self._faulted_sensors.discard(health.sensor_id)
        self._apply_freeze()
        if health.sensor_id == self._config.estimator.indoor_sensor_id:
            self._track_indoor_trust(health.quality is Quality.FAULTED)

    def _track_indoor_trust(self, faulted: bool) -> None:
        """Enter or leave open-loop prediction as the indoor sensor's trust changes."""
        if faulted and self._open_loop_c is None and self._indoor_c is not None:
            self._open_loop_c = self._indoor_c
            LOGGER.warning(
                "indoor sensor faulted; predicting on the model alone from the "
                "last trusted reading, %.2f C", self._indoor_c
            )
        elif not faulted and self._open_loop_c is not None:
            self._open_loop_c = None
            # The last trusted reading predates the fault. Pairing it with the
            # first reading after would fit an interval the model never saw.
            self._indoor_c = None
            self._indoor_ts = None
            LOGGER.info("indoor sensor trusted again; predicting from readings")

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        self._mode = state.mode
        self._apply_freeze()

    def _apply_freeze(self) -> None:
        if self.adaptation_frozen:
            self._estimator.freeze()
        else:
            self._estimator.unfreeze()

    def _regressor_sensor_ids(self) -> frozenset[str]:
        estimator = self._config.estimator
        return frozenset(
            {
                estimator.indoor_sensor_id,
                estimator.outdoor_sensor_id,
                estimator.occupancy_sensor_id,
            }
        )

    def _on_actuator(self, _topic: str, state: ActuatorState) -> None:
        """u[k] is what the plant is being driven with, not what was asked."""
        if state.kind is CommandKind.COOL:
            self._command = COOLING_ON
        elif state.kind is CommandKind.OFF:
            self._command = COOLING_OFF

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        estimator = self._config.estimator
        if reading.sensor_id == estimator.outdoor_sensor_id:
            self._outdoor_c = reading.value
        elif reading.sensor_id == estimator.occupancy_sensor_id:
            self._occupancy = OCCUPIED if reading.value >= 0.5 else VACANT
        elif reading.sensor_id == estimator.indoor_sensor_id:
            self._on_indoor(reading)

    # --- the step -----------------------------------------------------

    def _on_indoor(self, reading: SensorReading) -> None:
        """Form (phi[k], T[k+1]) from the previous reading and this one."""
        self._last_heard_c = reading.value
        if self._open_loop_c is not None:
            # The sensor is faulted; tick() is predicting without it.
            return
        if reading.quality is not Quality.OK:
            # Flagged by its own sensor. Kept out of the fit, the residual
            # window and the anchor alike; the interval check then refuses
            # the next pair, since it spans the gap this leaves.
            self._suspect_readings += 1
            LOGGER.debug("refusing a %s indoor reading", reading.quality.value)
            return

        previous_c, previous_ts = self._indoor_c, self._indoor_ts
        self._indoor_c, self._indoor_ts = reading.value, reading.ts

        if previous_c is None or previous_ts is None or self._outdoor_c is None:
            return

        if not self._interval_is_usable(previous_ts, reading.ts):
            self._skipped_pairs += 1
            return

        regressor = Regressor(
            indoor_c=previous_c,
            outdoor_c=self._outdoor_c,
            command=self._command,
            occupancy=self._occupancy,
        )
        result = self._estimator.update(regressor, reading.value)
        self._publish(result, reading.value)
        self._maybe_persist()
        if result.diverged:
            self._publish_divergence(result)

    def tick(self) -> ThermalEstimate | None:
        """Advance the open-loop prediction one model step (FR-27).

        Called once per sensor period. A no-op while the indoor sensor is
        trusted, since readings drive the estimate then.

        :returns: the estimate published, or None when not running open loop.
        """
        if self._open_loop_c is None or self._outdoor_c is None:
            return None
        regressor = Regressor(
            indoor_c=self._open_loop_c,
            outdoor_c=self._outdoor_c,
            command=self._command,
            occupancy=self._occupancy,
        )
        self._open_loop_c = self._estimator.predict(regressor)
        heard_c = (
            self._last_heard_c if self._last_heard_c is not None else self._open_loop_c
        )
        estimate = ThermalEstimate(
            ts=self._clock.now(),
            t_in=heard_c,
            t_pred=self._open_loop_c,
            residual=heard_c - self._open_loop_c,
            residual_sigma=self._estimator.residual_sigma,
            model_confidence=self._estimator.model_confidence,
            adaptation=self._estimator.adaptation,
        )
        self._blackboard.publish(topics.ESTIMATE_THERMAL, estimate)
        return estimate

    def _interval_is_usable(self, previous_ts: float, current_ts: float) -> bool:
        """Whether these two samples are one model step apart.

        A-02's uniform sampling will not survive WiFi. Timestamps that repeat
        or go backwards are rejected outright; an interval outside the
        configured tolerance is rejected because the ARX coefficients are
        defined against a fixed step and fitting a longer one would silently
        change what a1 means.
        """
        interval_s = current_ts - previous_ts
        if interval_s <= 0.0:
            LOGGER.debug("non-monotonic sample interval %.3f s", interval_s)
            return False
        nominal_s = self._config.loop.sensor_period_s
        tolerance = self._config.estimator.sample_interval_tolerance
        return abs(interval_s - nominal_s) <= nominal_s * tolerance

    def _publish(self, result: UpdateResult, measured_c: float) -> None:
        estimate = ThermalEstimate(
            ts=self._clock.now(),
            t_in=measured_c,
            t_pred=result.prediction_c,
            residual=result.residual_c,
            residual_sigma=self._estimator.residual_sigma,
            model_confidence=self._estimator.model_confidence,
            adaptation=self._estimator.adaptation,
        )
        self._blackboard.publish(topics.ESTIMATE_THERMAL, estimate)
        self._blackboard.publish(
            topics.ESTIMATE_COEFFICIENTS, self._estimator.snapshot()
        )

    def _maybe_persist(self) -> None:
        if not self._store.is_due():
            return
        self._store.save(
            self._estimator.theta,
            self._estimator.covariance,
            self._estimator.samples_since_reset,
        )

    def _publish_divergence(self, result: UpdateResult) -> None:
        """Raise MODEL_DIVERGENCE (FR-06).

        The estimator detects this, so the estimator reports it. Section 5.6
        routes it to SAFE_HOLD; deciding that is the mode manager's job, and
        this only states what it found and how sure it is.
        """
        detected_ts = self._clock.now()
        fault_id = f"f_model_divergence_{int(detected_ts)}"
        event = FaultEvent.model_validate(
            {
                "fault_id": fault_id,
                "detector": DetectorId.MODEL_DIVERGENCE,
                "subject": self._config.estimator.indoor_sensor_id,
                "class": FaultClass.MODEL,
                "confidence": _DIVERGENCE_CONFIDENCE,
                "detected_ts": detected_ts,
                "evidence": {
                    "consecutive_rejections": float(result.consecutive_rejections),
                    "residual": result.residual_c,
                },
                "mode_impact": Mode.SAFE_HOLD,
            }
        )
        LOGGER.error(
            "model divergence after %d consecutive rejections",
            result.consecutive_rejections,
        )
        self._blackboard.publish(topics.FAULT, event, fault_id=fault_id)
