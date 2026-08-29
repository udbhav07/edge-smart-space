"""Unit tests for the simulated sensors and their injection path."""

import random
from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import SensorNoiseConfig, load_config
from src.common.schemas import Unit
from src.common.injection import FaultInjection, InjectedFault
from sim.sensors import BinarySensor, SimulatedSensor

SENSOR_ID = "temp_01"
TRUE_VALUE_C = 27.4
NOMINAL_PERIOD_S = 5.0
SAMPLE_COUNT = 400
SEED = 20260826


@pytest.fixture(name="noise")
def _noise() -> SensorNoiseConfig:
    return load_config(Path("config/default.yaml")).sim.sensor_noise


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


def _perfect(noise: SensorNoiseConfig) -> SensorNoiseConfig:
    """Imperfection switched off, for tests that need an exact value."""
    return noise.model_copy(
        update={
            "sigma_c": 0.0,
            "quantisation_c": 0.0,
            "jitter_s": 0.0,
            "dropout_probability": 0.0,
            "bias_c": 0.0,
        }
    )


def _sensor(noise: SensorNoiseConfig, clock: SimClock) -> SimulatedSensor:
    return SimulatedSensor(
        SENSOR_ID, Unit.CELSIUS, noise, random.Random(SEED), clock
    )


class TestConstruction:
    def test_a_boolean_unit_is_refused(self, noise, clock):
        """Noise and quantisation cannot be applied to a two-valued signal."""
        with pytest.raises(ValueError):
            SimulatedSensor("pir_01", Unit.BOOLEAN, noise, random.Random(SEED), clock)

    def test_reports_its_own_id(self, noise, clock):
        assert _sensor(noise, clock).sensor_id == SENSOR_ID

    def test_starts_with_no_fault_injected(self, noise, clock):
        assert _sensor(noise, clock).injected_fault is InjectedFault.NONE


class TestNominalObservation:
    def test_a_perfect_sensor_reports_the_true_value(self, noise, clock):
        reading = _sensor(_perfect(noise), clock).sample(TRUE_VALUE_C)
        assert reading is not None
        assert reading.value == pytest.approx(TRUE_VALUE_C)

    def test_bias_offsets_every_reading(self, noise, clock):
        biased = _perfect(noise).model_copy(update={"bias_c": 1.5})
        reading = _sensor(biased, clock).sample(TRUE_VALUE_C)
        assert reading.value == pytest.approx(TRUE_VALUE_C + 1.5)

    def test_quantisation_snaps_to_the_configured_step(self, noise, clock):
        quantised = _perfect(noise).model_copy(update={"quantisation_c": 0.5})
        reading = _sensor(quantised, clock).sample(27.4)
        assert reading.value == pytest.approx(27.5)

    def test_noise_perturbs_readings_without_moving_the_mean_far(self, noise, clock):
        sensor = _sensor(noise, clock)
        values = [
            reading.value
            for reading in (sensor.sample(TRUE_VALUE_C) for _ in range(SAMPLE_COUNT))
            if reading is not None
        ]
        assert len(set(values)) > 1
        assert sum(values) / len(values) == pytest.approx(TRUE_VALUE_C, abs=0.1)

    def test_the_reading_carries_the_clock_timestamp(self, noise, clock):
        clock.advance(NOMINAL_PERIOD_S)
        reading = _sensor(_perfect(noise), clock).sample(TRUE_VALUE_C)
        assert reading.ts == clock.now()


class TestSamplingJitter:
    def test_jitter_moves_the_interval_off_nominal(self, noise, clock):
        sensor = _sensor(noise, clock)
        delays = {sensor.sampling_delay_s(NOMINAL_PERIOD_S) for _ in range(50)}
        assert len(delays) > 1

    def test_jitter_stays_within_the_configured_band(self, noise, clock):
        sensor = _sensor(noise, clock)
        for _ in range(200):
            delay = sensor.sampling_delay_s(NOMINAL_PERIOD_S)
            assert abs(delay - NOMINAL_PERIOD_S) <= noise.jitter_s

    def test_a_delay_is_never_zero_or_negative(self, noise, clock):
        """Two readings at the same instant is a bug, not a jitter model."""
        wild = noise.model_copy(update={"jitter_s": 100.0})
        sensor = _sensor(wild, clock)
        for _ in range(200):
            assert sensor.sampling_delay_s(NOMINAL_PERIOD_S) > 0.0

    def test_zero_jitter_gives_the_nominal_period(self, noise, clock):
        sensor = _sensor(_perfect(noise), clock)
        assert sensor.sampling_delay_s(NOMINAL_PERIOD_S) == NOMINAL_PERIOD_S

    def test_a_non_positive_period_is_rejected(self, noise, clock):
        with pytest.raises(ValueError):
            _sensor(noise, clock).sampling_delay_s(0.0)


class TestInjectedFaults:
    def test_dropout_produces_no_reading_at_all(self, noise, clock):
        """D1 detects absence, so absence is what the sensor must produce."""
        sensor = _sensor(_perfect(noise), clock)
        sensor.inject(FaultInjection(kind=InjectedFault.DROPOUT))
        assert all(sensor.sample(TRUE_VALUE_C) is None for _ in range(20))

    def test_stuck_at_freezes_the_reported_value(self, noise, clock):
        sensor = _sensor(noise, clock)
        sensor.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        values = {sensor.sample(TRUE_VALUE_C).value for _ in range(50)}
        assert values == {27.0}

    def test_stuck_at_ignores_the_true_value_changing(self, noise, clock):
        sensor = _sensor(noise, clock)
        sensor.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        assert sensor.sample(10.0).value == 27.0
        assert sensor.sample(40.0).value == 27.0

    def test_out_of_range_reports_the_requested_value(self, noise, clock):
        sensor = _sensor(noise, clock)
        sensor.inject(FaultInjection(kind=InjectedFault.OUT_OF_RANGE, magnitude=95.0))
        assert sensor.sample(TRUE_VALUE_C).value == 95.0

    def test_drift_accumulates_with_elapsed_time(self, noise, clock):
        sensor = _sensor(_perfect(noise), clock)
        sensor.inject(FaultInjection(kind=InjectedFault.DRIFT, magnitude=0.01))
        clock.advance(100.0)
        assert sensor.sample(TRUE_VALUE_C).value == pytest.approx(TRUE_VALUE_C + 1.0)

    def test_drift_is_zero_at_the_moment_of_injection(self, noise, clock):
        sensor = _sensor(_perfect(noise), clock)
        sensor.inject(FaultInjection(kind=InjectedFault.DRIFT, magnitude=0.01))
        assert sensor.sample(TRUE_VALUE_C).value == pytest.approx(TRUE_VALUE_C)

    def test_clearing_restores_nominal_behaviour(self, noise, clock):
        sensor = _sensor(_perfect(noise), clock)
        sensor.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        sensor.clear()
        assert sensor.sample(TRUE_VALUE_C).value == pytest.approx(TRUE_VALUE_C)

    def test_the_active_injection_is_reportable_for_audit(self, noise, clock):
        sensor = _sensor(noise, clock)
        sensor.inject(FaultInjection(kind=InjectedFault.DRIFT, magnitude=0.01))
        assert sensor.injected_fault is InjectedFault.DRIFT

    def test_a_faulted_reading_is_still_labelled_ok(self, noise, clock):
        """The sensor does not know it is broken; that is the detectors' job."""
        sensor = _sensor(noise, clock)
        sensor.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        assert sensor.sample(TRUE_VALUE_C).quality.value == "ok"


class TestRandomDropout:
    def test_some_samples_are_lost_at_the_configured_rate(self, noise, clock):
        certain = noise.model_copy(update={"dropout_probability": 1.0})
        sensor = _sensor(certain, clock)
        assert all(sensor.sample(TRUE_VALUE_C) is None for _ in range(20))

    def test_no_samples_are_lost_when_dropout_is_disabled(self, noise, clock):
        sensor = _sensor(_perfect(noise), clock)
        assert all(sensor.sample(TRUE_VALUE_C) is not None for _ in range(50))


class TestDeterminism:
    def test_the_same_seed_reproduces_the_same_readings(self, noise):
        def run() -> list[float | None]:
            clock = SimClock()
            sensor = SimulatedSensor(
                SENSOR_ID, Unit.CELSIUS, noise, random.Random(SEED), clock
            )
            return [
                None if reading is None else reading.value
                for reading in (sensor.sample(TRUE_VALUE_C) for _ in range(100))
            ]

        assert run() == run()


class TestBinarySensor:
    def _binary(self, noise: SensorNoiseConfig, clock: SimClock) -> BinarySensor:
        return BinarySensor("pir_01", noise, random.Random(SEED), clock)

    def test_reports_occupancy_as_one(self, noise, clock):
        assert self._binary(_perfect(noise), clock).sample(True).value == 1.0

    def test_reports_vacancy_as_zero(self, noise, clock):
        assert self._binary(_perfect(noise), clock).sample(False).value == 0.0

    def test_the_reading_is_labelled_boolean(self, noise, clock):
        reading = self._binary(_perfect(noise), clock).sample(True)
        assert reading.unit is Unit.BOOLEAN

    def test_dropout_produces_no_reading(self, noise, clock):
        sensor = self._binary(_perfect(noise), clock)
        sensor.inject(FaultInjection(kind=InjectedFault.DROPOUT))
        assert sensor.sample(True) is None

    def test_stuck_at_freezes_the_reported_state(self, noise, clock):
        sensor = self._binary(_perfect(noise), clock)
        sensor.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=1.0))
        assert sensor.sample(False).value == 1.0

    @pytest.mark.parametrize(
        "kind", [InjectedFault.DRIFT, InjectedFault.OUT_OF_RANGE]
    )
    def test_a_meaningless_fault_is_refused_rather_than_ignored(
        self, noise, clock, kind
    ):
        """Silently ignoring it would look like a missed detection later."""
        sensor = self._binary(noise, clock)
        with pytest.raises(ValueError):
            sensor.inject(FaultInjection(kind=kind, magnitude=1.0))

    def test_refusing_an_injection_leaves_the_sensor_nominal(self, noise, clock):
        sensor = self._binary(_perfect(noise), clock)
        with pytest.raises(ValueError):
            sensor.inject(FaultInjection(kind=InjectedFault.DRIFT, magnitude=1.0))
        assert sensor.injected_fault is InjectedFault.NONE
        assert sensor.sample(True).value == 1.0

    def test_clearing_restores_nominal_behaviour(self, noise, clock):
        sensor = self._binary(_perfect(noise), clock)
        sensor.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=1.0))
        sensor.clear()
        assert sensor.sample(False).value == 0.0

    def test_reports_its_own_id(self, noise, clock):
        assert self._binary(noise, clock).sensor_id == "pir_01"
