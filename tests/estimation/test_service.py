"""Unit tests for the estimator service.

No broker. What is asserted hardest is what the service *refuses*: A-02's
uniform sampling will not survive WiFi, and an ARX model fitted across a
wrong-length step silently redefines what its coefficients mean.
"""

from pathlib import Path

import pytest

from src.common import topics
from src.common.clock import SimClock
from src.common.config import PersistenceConfig, load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    ActuatorState,
    AckStatus,
    AdaptationState,
    Coefficients,
    CommandKind,
    FaultEvent,
    Mode,
    ModeState,
    Quality,
    SensorHealth,
    SensorReading,
    ThermalEstimate,
    Unit,
)
from src.estimation.__main__ import run
from src.estimation.persistence import CoefficientStore
from src.estimation.rc_model import Regressor
from src.estimation.rls import ThermalEstimator
from src.estimation.service import COOLING_OFF, COOLING_ON, ThermalEstimatorService

INDOOR = "temp_01"
OUTDOOR = "outdoor_01"
OCCUPANCY = "pir_01"


@pytest.fixture(name="config")
def _config(tmp_path: Path):
    config = load_config(Path("config/default.yaml"))
    persistence = PersistenceConfig(
        path=str(tmp_path / "coefficients.json"),
        interval_s=config.persistence.interval_s,
        max_age_s=config.persistence.max_age_s,
    )
    return config.model_copy(update={"persistence": persistence})


class RecordingTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []

    def connect(self, host, port, keepalive): ...
    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
    def subscribe(self, topic, qos): ...
    def loop_start(self): ...
    def loop_stop(self): ...
    def disconnect(self): ...

    def on(self, topic: str) -> list[bytes]:
        return [payload for name, payload, _, _ in self.published if name == topic]

    def topics_seen(self) -> set[str]:
        return {name for name, _, _, _ in self.published}


def _service(config, clock=None):
    clock = clock or SimClock()
    transport = RecordingTransport()
    board = Blackboard(config.mqtt, transport)
    estimator = ThermalEstimator(config.estimator, clock)
    store = CoefficientStore(config.persistence, clock)
    service = ThermalEstimatorService(config, clock, board, estimator, store)
    service.subscribe()
    return service, transport, clock, board, estimator, store


def _reading(clock, sensor_id: str, value: float, unit: Unit = Unit.CELSIUS):
    return SensorReading(
        ts=clock.now(), sensor_id=sensor_id, value=value, unit=unit
    )


#: Ambient for the fed readings. The room drifts toward it, which is what a
#: room with nothing commanded actually does; feeding the opposite would be
#: physically impossible data and the estimator would rightly reject it.
FED_OUTDOOR_C = 31.0


def _feed(service, board, clock, samples: int, period_s: float, start_c: float = 29.0):
    """Deliver a run of well-spaced, physically coherent readings."""
    board.dispatch(
        topics.SENSOR_STATE.format(sensor_id=OUTDOOR),
        _reading(clock, OUTDOOR, FED_OUTDOOR_C).model_dump_json().encode(),
    )
    temperature = start_c
    for index in range(samples):
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            _reading(clock, INDOOR, temperature).model_dump_json().encode(),
        )
        temperature += 0.002 * (FED_OUTDOOR_C - temperature)
        clock.advance(period_s)


class TestRegressorAssembly:
    def test_the_first_reading_alone_produces_no_estimate(self, config):
        service, transport, clock, board, _, _ = _service(config)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=OUTDOOR),
            _reading(clock, OUTDOOR, 31.0).model_dump_json().encode(),
        )
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            _reading(clock, INDOOR, 29.0).model_dump_json().encode(),
        )
        assert transport.on("space/estimate/thermal") == []

    def test_a_pair_produces_an_estimate(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 2, config.loop.sensor_period_s)
        assert transport.on("space/estimate/thermal")

    def test_nothing_is_estimated_before_ambient_is_known(self, config):
        service, transport, clock, board, _, _ = _service(config)
        for _ in range(3):
            board.dispatch(
                topics.SENSOR_STATE.format(sensor_id=INDOOR),
                _reading(clock, INDOOR, 29.0).model_dump_json().encode(),
            )
            clock.advance(config.loop.sensor_period_s)
        assert transport.on("space/estimate/thermal") == []

    def test_cooling_state_becomes_the_drive_the_model_sees(self, config):
        service, _, clock, board, _, _ = _service(config)
        state = ActuatorState(
            ts=clock.now(),
            actuator_id="ac",
            simulated=True,
            kind=CommandKind.COOL,
            setpoint_c=24.0,
            ack=AckStatus.UNKNOWN,
        )
        board.dispatch("space/actuator/ac/state", state.model_dump_json().encode())
        assert service._command == COOLING_ON

    def test_switching_off_clears_the_drive(self, config):
        service, _, clock, board, _, _ = _service(config)
        for kind in (CommandKind.COOL, CommandKind.OFF):
            state = ActuatorState(
                ts=clock.now(),
                actuator_id="ac",
                simulated=True,
                kind=kind,
                setpoint_c=24.0 if kind is CommandKind.COOL else None,
                ack=AckStatus.UNKNOWN,
            )
            board.dispatch("space/actuator/ac/state", state.model_dump_json().encode())
        assert service._command == COOLING_OFF

    def test_occupancy_is_read_as_binary(self, config):
        service, _, clock, board, _, _ = _service(config)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=OCCUPANCY),
            _reading(clock, OCCUPANCY, 1.0, Unit.BOOLEAN).model_dump_json().encode(),
        )
        assert service._occupancy == 1.0


class TestNonUniformSampling:
    """A-02 will not survive WiFi, and the ARX form assumes a fixed step."""

    def _pair(self, config, gap_s: float):
        service, transport, clock, board, _, _ = _service(config)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=OUTDOOR),
            _reading(clock, OUTDOOR, 31.0).model_dump_json().encode(),
        )
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            _reading(clock, INDOOR, 29.0).model_dump_json().encode(),
        )
        clock.advance(gap_s)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            _reading(clock, INDOOR, 28.9).model_dump_json().encode(),
        )
        return service, transport

    def test_a_nominal_interval_is_fitted(self, config):
        _, transport = self._pair(config, config.loop.sensor_period_s)
        assert transport.on("space/estimate/thermal")

    def test_jitter_within_tolerance_is_still_fitted(self, config):
        nominal = config.loop.sensor_period_s
        within = nominal * (1.0 + config.estimator.sample_interval_tolerance / 2.0)
        _, transport = self._pair(config, within)
        assert transport.on("space/estimate/thermal")

    def test_a_gap_beyond_tolerance_is_skipped(self, config):
        """A reconnect burst must not be fitted as one model step."""
        service, transport = self._pair(config, config.loop.sensor_period_s * 5.0)
        assert transport.on("space/estimate/thermal") == []
        assert service.skipped_pairs == 1

    def test_a_burst_arriving_too_fast_is_skipped(self, config):
        service, transport = self._pair(config, config.loop.sensor_period_s / 10.0)
        assert transport.on("space/estimate/thermal") == []
        assert service.skipped_pairs == 1

    def test_a_repeated_timestamp_is_skipped(self, config):
        service, transport = self._pair(config, 0.0)
        assert service.skipped_pairs == 1

    def test_a_backwards_timestamp_is_skipped(self, config):
        service, transport, clock, board, _, _ = _service(config)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=OUTDOOR),
            _reading(clock, OUTDOOR, 31.0).model_dump_json().encode(),
        )
        clock.advance(100.0)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            _reading(clock, INDOOR, 29.0).model_dump_json().encode(),
        )
        stale = SensorReading(
            ts=clock.now() - 50.0, sensor_id=INDOOR, value=28.0, unit=Unit.CELSIUS
        )
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            stale.model_dump_json().encode(),
        )
        assert service.skipped_pairs == 1

    def test_a_skipped_pair_still_advances_the_reference_sample(self, config):
        """Skipping must not wedge the service on one stale sample."""
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 1, config.loop.sensor_period_s)
        clock.advance(config.loop.sensor_period_s * 10.0)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            _reading(clock, INDOOR, 28.0).model_dump_json().encode(),
        )
        clock.advance(config.loop.sensor_period_s)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=INDOOR),
            _reading(clock, INDOOR, 27.9).model_dump_json().encode(),
        )
        assert transport.on("space/estimate/thermal")


class TestPublishing:
    def test_the_prediction_and_residual_are_published(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        estimate = ThermalEstimate.model_validate_json(
            transport.on("space/estimate/thermal")[0]
        )
        assert estimate.residual == pytest.approx(estimate.t_in - estimate.t_pred)

    def test_the_coefficients_are_published(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        assert transport.on("space/estimate/coefficients")

    def test_the_coefficients_carry_the_steady_state_diagnostic(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        coefficients = Coefficients.model_validate_json(
            transport.on("space/estimate/coefficients")[0]
        )
        assert coefficients.steady_state_residual == pytest.approx(
            abs(coefficients.a1 + coefficients.a2 - 1.0)
        )

    def test_estimates_are_retained_for_late_subscribers(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        assert all(
            retain
            for name, _, _, retain in transport.published
            if name.startswith("space/estimate/")
        )


class TestFreezing:
    """FR-29: never adapt while a regressor sensor is faulted."""

    def _fault(self, board, clock, sensor_id: str, quality: Quality):
        health = SensorHealth(
            ts=clock.now(),
            sensor_id=sensor_id,
            quality=quality,
            active_fault_id="f_x" if quality is Quality.FAULTED else None,
        )
        board.dispatch(
            topics.SENSOR_HEALTH.format(sensor_id=sensor_id),
            health.model_dump_json().encode(),
        )

    def test_a_faulted_regressor_sensor_freezes_adaptation(self, config):
        service, _, clock, board, estimator, _ = _service(config)
        self._fault(board, clock, INDOOR, Quality.FAULTED)
        assert estimator.adaptation is AdaptationState.FROZEN

    def test_recovery_resumes_adaptation(self, config):
        service, _, clock, board, estimator, _ = _service(config)
        self._fault(board, clock, INDOOR, Quality.FAULTED)
        self._fault(board, clock, INDOOR, Quality.OK)
        assert estimator.adaptation is AdaptationState.ACTIVE

    def test_a_sensor_outside_the_regressor_does_not_freeze_anything(self, config):
        service, _, clock, board, estimator, _ = _service(config)
        self._fault(board, clock, "hum_01", Quality.FAULTED)
        assert estimator.adaptation is AdaptationState.ACTIVE

    @pytest.mark.parametrize(
        "mode", [Mode.DEGRADED_SENSOR, Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD]
    )
    def test_a_degraded_mode_freezes_adaptation(self, config, mode):
        service, _, clock, board, estimator, _ = _service(config)
        state = ModeState(ts=clock.now(), mode=mode, since_ts=clock.now())
        board.dispatch("space/system/mode", state.model_dump_json().encode())
        assert estimator.adaptation is AdaptationState.FROZEN

    def test_normal_mode_leaves_adaptation_running(self, config):
        service, _, clock, board, estimator, _ = _service(config)
        state = ModeState(ts=clock.now(), mode=Mode.NORMAL, since_ts=clock.now())
        board.dispatch("space/system/mode", state.model_dump_json().encode())
        assert estimator.adaptation is AdaptationState.ACTIVE

    def test_the_published_estimate_says_adaptation_is_frozen(self, config):
        service, transport, clock, board, _, _ = _service(config)
        self._fault(board, clock, INDOOR, Quality.FAULTED)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        estimate = ThermalEstimate.model_validate_json(
            transport.on("space/estimate/thermal")[-1]
        )
        assert estimate.adaptation is AdaptationState.FROZEN

    def test_prediction_continues_while_frozen(self, config):
        """FR-27 depends on it: DEGRADED_SENSOR control runs on this."""
        service, transport, clock, board, _, _ = _service(config)
        self._fault(board, clock, INDOOR, Quality.FAULTED)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        assert transport.on("space/estimate/thermal")


class TestDivergence:
    def test_repeated_rejections_raise_a_model_divergence_fault(self, config):
        """FR-06: the estimator detects it, so the estimator reports it."""
        service, transport, clock, board, _, _ = _service(config)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=OUTDOOR),
            _reading(clock, OUTDOOR, 25.0).model_dump_json().encode(),
        )
        state = ActuatorState(
            ts=clock.now(),
            actuator_id="ac",
            simulated=True,
            kind=CommandKind.COOL,
            setpoint_c=24.0,
            ack=AckStatus.UNKNOWN,
        )
        board.dispatch("space/actuator/ac/state", state.model_dump_json().encode())

        temperature = 25.0
        for _ in range(config.estimator.divergence_window_samples + 10):
            board.dispatch(
                topics.SENSOR_STATE.format(sensor_id=INDOOR),
                _reading(clock, INDOOR, temperature).model_dump_json().encode(),
            )
            temperature += 40.0
            clock.advance(config.loop.sensor_period_s)

        faults = [
            name for name in transport.topics_seen() if name.startswith("space/fault/")
        ]
        assert faults

    def test_the_fault_names_the_model_as_its_class(self, config):
        service, transport, clock, board, _, _ = _service(config)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=OUTDOOR),
            _reading(clock, OUTDOOR, 25.0).model_dump_json().encode(),
        )
        temperature = 25.0
        for _ in range(config.estimator.divergence_window_samples + 10):
            board.dispatch(
                topics.SENSOR_STATE.format(sensor_id=INDOOR),
                _reading(clock, INDOOR, temperature).model_dump_json().encode(),
            )
            temperature += 40.0
            clock.advance(config.loop.sensor_period_s)

        payloads = [
            payload
            for name, payload, _, _ in transport.published
            if name.startswith("space/fault/")
        ]
        if payloads:
            event = FaultEvent.model_validate_json(payloads[0])
            assert event.mode_impact is Mode.SAFE_HOLD
            assert event.fault_class.value == "model"


class TestDivergenceRecovery:
    """Section 7.1: divergence resets the coefficients to theta_0."""

    def _diverge(self, config, extra_samples: int = 10):
        service, transport, clock, board, estimator, _ = _service(config)
        board.dispatch(
            topics.SENSOR_STATE.format(sensor_id=OUTDOOR),
            _reading(clock, OUTDOOR, 25.0).model_dump_json().encode(),
        )
        temperature = 25.0
        for _ in range(config.estimator.divergence_window_samples + extra_samples):
            board.dispatch(
                topics.SENSOR_STATE.format(sensor_id=INDOOR),
                _reading(clock, INDOOR, temperature).model_dump_json().encode(),
            )
            temperature += 40.0
            clock.advance(config.loop.sensor_period_s)
        return service, transport, estimator

    def _divergence_faults(self, transport) -> list[str]:
        return [
            name
            for name, payload, _, _ in transport.published
            if name.startswith("space/fault/f_model_divergence") and payload
        ]

    def test_the_coefficients_return_to_the_prior(self, config):
        _, _, estimator = self._diverge(config, extra_samples=1)
        assert list(estimator.theta) == list(config.estimator.initial_theta)

    def test_a_divergence_is_announced_once_not_every_update(self, config):
        """Regression: the full window kept diverged true, so the same event
        was republished as a new fault every 5 s."""
        _, transport, _ = self._diverge(config, extra_samples=10)
        assert len(self._divergence_faults(transport)) == 1

    def test_a_second_divergence_needs_a_fresh_window(self, config):
        window = config.estimator.divergence_window_samples
        _, transport, _ = self._diverge(config, extra_samples=window + 10)
        assert len(self._divergence_faults(transport)) == 2


class TestPersistence:
    def test_the_estimate_is_written_as_it_runs(self, config):
        service, _, clock, board, _, store = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        assert store.path.exists()

    def test_a_persisted_estimate_is_adopted_on_startup(self, config):
        first, _, clock, board, estimator, store = _service(config)
        _feed(first, board, clock, 40, config.loop.sensor_period_s)
        store.save(estimator.theta, estimator.covariance, estimator.samples_since_reset)
        learned = estimator.theta

        second, _, _, _, fresh, _ = _service(config, clock=clock)
        assert second.restore() is True
        assert fresh.theta == pytest.approx(learned)

    def test_starting_from_the_prior_is_a_normal_outcome(self, config):
        service, _, _, _, _, _ = _service(config)
        assert service.restore() is False


def _health(board, clock, sensor_id: str, quality: Quality) -> None:
    health = SensorHealth(
        ts=clock.now(),
        sensor_id=sensor_id,
        quality=quality,
        active_fault_id="f_x" if quality is Quality.FAULTED else None,
    )
    board.dispatch(
        topics.SENSOR_HEALTH.format(sensor_id=sensor_id),
        health.model_dump_json().encode(),
    )


def _indoor(board, clock, value: float, quality: Quality = Quality.OK) -> None:
    reading = SensorReading(
        ts=clock.now(),
        sensor_id=INDOOR,
        value=value,
        unit=Unit.CELSIUS,
        quality=quality,
    )
    board.dispatch(
        topics.SENSOR_STATE.format(sensor_id=INDOOR),
        reading.model_dump_json().encode(),
    )


class TestSuspectReadings:
    """A reading its own sensor flags is never fitted (FR-29, section 5.1)."""

    def test_a_suspect_reading_produces_no_estimate(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        before = len(transport.on("space/estimate/thermal"))
        _indoor(board, clock, 999.0, Quality.SUSPECT)
        assert len(transport.on("space/estimate/thermal")) == before

    def test_a_suspect_reading_is_counted(self, config):
        service, _, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        _indoor(board, clock, 999.0, Quality.SUSPECT)
        assert service.suspect_readings == 1

    def test_a_suspect_reading_leaves_the_residual_spread_alone(self, config):
        """The live run saw sigma reach 480 C after two 999 C samples, which
        blinded D4 for minutes after the fault had cleared."""
        service, _, clock, board, estimator, _ = _service(config)
        _feed(service, board, clock, 10, config.loop.sensor_period_s)
        sigma = estimator.residual_sigma
        _indoor(board, clock, 999.0, Quality.SUSPECT)
        assert estimator.residual_sigma == sigma

    def test_a_suspect_reading_leaves_the_coefficients_alone(self, config):
        service, _, clock, board, estimator, _ = _service(config)
        _feed(service, board, clock, 30, config.loop.sensor_period_s)
        theta = estimator.theta
        _indoor(board, clock, 999.0, Quality.SUSPECT)
        assert (estimator.theta == theta).all()

    def test_the_pair_spanning_a_suspect_reading_is_not_fitted(self, config):
        """The gap it leaves is two periods, which the interval check refuses."""
        service, _, clock, board, _, _ = _service(config)
        period_s = config.loop.sensor_period_s
        _feed(service, board, clock, 3, period_s)
        skipped = service.skipped_pairs
        _indoor(board, clock, 999.0, Quality.SUSPECT)
        clock.advance(period_s)
        _indoor(board, clock, 29.0)
        assert service.skipped_pairs == skipped + 1


class TestOpenLoopPrediction:
    """FR-27: while the indoor sensor is faulted, the model runs on itself."""

    def _faulted_after(self, config, samples: int = 10, start_c: float = 29.0):
        service, transport, clock, board, estimator, _ = _service(config)
        _feed(service, board, clock, samples, config.loop.sensor_period_s, start_c)
        _health(board, clock, INDOOR, Quality.FAULTED)
        return service, transport, clock, board, estimator

    def test_a_trusted_sensor_means_tick_does_nothing(self, config):
        service, _, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        assert service.tick() is None
        assert not service.open_loop

    def test_a_faulted_indoor_sensor_starts_open_loop(self, config):
        service, *_ = self._faulted_after(config)
        assert service.open_loop

    def test_a_fault_before_any_reading_has_nothing_to_start_from(self, config):
        service, _, clock, board, _, _ = _service(config)
        _health(board, clock, INDOOR, Quality.FAULTED)
        assert not service.open_loop
        assert service.tick() is None

    def test_another_sensor_faulting_does_not_start_open_loop(self, config):
        service, _, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 3, config.loop.sensor_period_s)
        _health(board, clock, OUTDOOR, Quality.FAULTED)
        assert not service.open_loop

    def test_each_tick_publishes_a_prediction(self, config):
        service, transport, *_ = self._faulted_after(config)
        before = len(transport.on("space/estimate/thermal"))
        service.tick()
        service.tick()
        assert len(transport.on("space/estimate/thermal")) == before + 2

    def test_the_first_step_starts_from_the_last_trusted_reading(self, config):
        service, _, clock, board, estimator = self._faulted_after(config)
        last_trusted_c = _last_fed_c(10)
        expected = estimator.predict(
            Regressor(
                indoor_c=last_trusted_c,
                outdoor_c=FED_OUTDOOR_C,
                command=COOLING_OFF,
                occupancy=0.0,
            )
        )
        assert service.tick().t_pred == pytest.approx(expected)

    def test_the_prediction_evolves_rather_than_holding(self, config):
        """A dropout delivers no readings; the prediction must still move."""
        service, *_ = self._faulted_after(config)
        first = service.tick().t_pred
        for _ in range(50):
            latest = service.tick().t_pred
        assert latest != first

    def test_a_faulted_reading_does_not_become_the_prediction(self, config):
        """The defect the live run found: the prediction restated the fault."""
        service, _, clock, board, _ = self._faulted_after(config)
        for _ in range(20):
            clock.advance(config.loop.sensor_period_s)
            _indoor(board, clock, 999.0)
            estimate = service.tick()
        assert abs(estimate.t_pred - _last_fed_c(10)) < 1.0

    def test_the_faulted_reading_is_published_as_heard(self, config):
        service, _, clock, board, _ = self._faulted_after(config)
        _indoor(board, clock, 27.0)
        estimate = service.tick()
        assert estimate.t_in == 27.0
        assert estimate.residual == pytest.approx(27.0 - estimate.t_pred)

    def test_open_loop_estimates_are_marked_frozen(self, config):
        service, *_ = self._faulted_after(config)
        assert service.tick().adaptation is AdaptationState.FROZEN

    def test_open_loop_leaves_the_residual_spread_alone(self, config):
        service, _, clock, board, estimator = self._faulted_after(config)
        sigma = estimator.residual_sigma
        for _ in range(5):
            _indoor(board, clock, 999.0)
            service.tick()
        assert estimator.residual_sigma == sigma

    def test_the_fault_clearing_returns_to_readings(self, config):
        service, _, clock, board, _ = self._faulted_after(config)
        service.tick()
        _health(board, clock, INDOOR, Quality.OK)
        assert not service.open_loop
        assert service.tick() is None

    def test_the_first_reading_after_recovery_is_not_paired_across_the_fault(
        self, config
    ):
        """The last trusted reading predates the fault by an unknown time."""
        service, transport, clock, board, _ = self._faulted_after(config)
        _health(board, clock, INDOOR, Quality.OK)
        before = len(transport.on("space/estimate/thermal"))
        _indoor(board, clock, 29.0)
        assert len(transport.on("space/estimate/thermal")) == before

    def test_nothing_is_predicted_before_ambient_is_known(self, config):
        service, _, clock, board, _, _ = _service(config)
        _indoor(board, clock, 29.0)
        _health(board, clock, INDOOR, Quality.FAULTED)
        assert service.open_loop
        assert service.tick() is None


def _last_fed_c(samples: int, start_c: float = 29.0) -> float:
    """The last indoor value :func:`_feed` delivers for a run of ``samples``."""
    temperature = start_c
    for _ in range(samples - 1):
        temperature += 0.002 * (FED_OUTDOOR_C - temperature)
    return temperature


class TestRunLoop:
    def test_the_loop_ticks_once_per_period(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 10, config.loop.sensor_period_s)
        _health(board, clock, INDOOR, Quality.FAULTED)
        before = len(transport.on("space/estimate/thermal"))
        period_s = config.loop.sensor_period_s
        run(service, clock, period_s, wakes=int(period_s) * 3)
        assert len(transport.on("space/estimate/thermal")) == before + 3

    def test_a_trusted_sensor_publishes_nothing_from_the_loop(self, config):
        service, transport, clock, board, _, _ = _service(config)
        _feed(service, board, clock, 10, config.loop.sensor_period_s)
        before = len(transport.on("space/estimate/thermal"))
        run(service, clock, config.loop.sensor_period_s, wakes=20)
        assert len(transport.on("space/estimate/thermal")) == before
