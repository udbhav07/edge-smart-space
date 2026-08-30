"""Unit tests for the Layer 1 sensor adapters.

No hardware and no broker: the source and the transport are both injected.
What is asserted is the contract everything above Layer 1 depends on --
that a lost sample is absence, that an implausible reading still reaches the
bus, and that injection behaves identically to a real fault.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import SensorConfig, load_config
from src.common.injection import FaultInjection, InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import Quality, SensorHealth, SensorReading, Unit
from src.io.sensor_adapters import EsphomeSource, SensorAdapter

SENSOR_ID = "temp_01"
NOMINAL_C = 27.4
STALE_AFTER_S = 15.0


@pytest.fixture(name="config")
def _config() -> SensorConfig:
    return load_config(Path("config/default.yaml")).sensors.by_id(SENSOR_ID)


@pytest.fixture(name="mqtt_config")
def _mqtt_config():
    return load_config(Path("config/default.yaml")).mqtt


class FakeSource:
    def __init__(self, value: float | None = NOMINAL_C) -> None:
        self.value = value
        self.reads = 0

    def read(self) -> float | None:
        self.reads += 1
        return self.value


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

    def retained(self, topic: str) -> bool:
        return all(retain for name, _, _, retain in self.published if name == topic)


def _adapter(config, mqtt_config, source=None, clock=None):
    clock = clock or SimClock()
    transport = RecordingTransport()
    board = Blackboard(mqtt_config, transport)
    adapter = SensorAdapter(config, source or FakeSource(), clock, board)
    return adapter, transport, clock


class TestIdentity:
    def test_reports_its_configured_id(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        assert adapter.sensor_id == SENSOR_ID

    def test_reports_its_configured_unit(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        assert adapter.unit is Unit.CELSIUS

    def test_reports_its_physical_limits(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        assert adapter.limits.low == -10.0 and adapter.limits.high == 60.0

    def test_has_no_reading_timestamp_before_the_first_poll(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        assert adapter.last_reading_ts is None


class TestReading:
    def test_wraps_the_source_value_in_the_shared_schema(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        reading = adapter.read()
        assert isinstance(reading, SensorReading)
        assert reading.value == NOMINAL_C

    def test_stamps_the_reading_from_the_injected_clock(self, config, mqtt_config):
        clock = SimClock()
        clock.advance(500.0)
        adapter, _, _ = _adapter(config, mqtt_config, clock=clock)
        assert adapter.read().ts == clock.now()

    def test_a_silent_source_produces_no_reading(self, config, mqtt_config):
        """Absence is what D1 detects, so absence is what must come out."""
        adapter, _, _ = _adapter(config, mqtt_config, source=FakeSource(None))
        assert adapter.read() is None

    def test_a_plausible_reading_is_marked_ok(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        assert adapter.read().quality is Quality.OK


class TestOutOfRangeIsNotSuppressed:
    """FR-22: D3 detects out-of-range readings, so they must reach it."""

    @pytest.mark.parametrize("value", [-40.0, 150.0])
    def test_an_implausible_reading_is_still_produced(
        self, config, mqtt_config, value
    ):
        adapter, _, _ = _adapter(config, mqtt_config, source=FakeSource(value))
        assert adapter.read().value == value

    def test_an_implausible_reading_is_flagged_suspect(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config, source=FakeSource(150.0))
        assert adapter.read().quality is Quality.SUSPECT

    def test_an_implausible_reading_is_still_published(self, config, mqtt_config):
        adapter, transport, _ = _adapter(
            config, mqtt_config, source=FakeSource(150.0)
        )
        adapter.poll()
        assert transport.on(f"space/sensor/{SENSOR_ID}/state")

    @pytest.mark.parametrize("value", [-10.0, 60.0])
    def test_the_limits_themselves_are_plausible(self, config, mqtt_config, value):
        adapter, _, _ = _adapter(config, mqtt_config, source=FakeSource(value))
        assert adapter.read().quality is Quality.OK


class TestInjection:
    def test_dropout_produces_no_reading(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        adapter.inject(FaultInjection(kind=InjectedFault.DROPOUT))
        assert adapter.read() is None

    def test_dropout_publishes_nothing_on_the_state_topic(self, config, mqtt_config):
        adapter, transport, _ = _adapter(config, mqtt_config)
        adapter.inject(FaultInjection(kind=InjectedFault.DROPOUT))
        adapter.poll()
        assert transport.on(f"space/sensor/{SENSOR_ID}/state") == []

    def test_stuck_at_freezes_the_value(self, config, mqtt_config):
        source = FakeSource(NOMINAL_C)
        adapter, _, _ = _adapter(config, mqtt_config, source=source)
        adapter.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        source.value = 10.0
        assert adapter.read().value == 27.0

    def test_stuck_at_does_not_even_consult_the_source(self, config, mqtt_config):
        source = FakeSource()
        adapter, _, _ = _adapter(config, mqtt_config, source=source)
        adapter.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        adapter.read()
        assert source.reads == 0

    def test_out_of_range_reports_the_requested_value(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        adapter.inject(FaultInjection(kind=InjectedFault.OUT_OF_RANGE, magnitude=95.0))
        assert adapter.read().value == 95.0

    def test_drift_accumulates_with_elapsed_time(self, config, mqtt_config):
        clock = SimClock()
        adapter, _, _ = _adapter(config, mqtt_config, clock=clock)
        adapter.inject(FaultInjection(kind=InjectedFault.DRIFT, magnitude=0.01))
        clock.advance(100.0)
        assert adapter.read().value == pytest.approx(NOMINAL_C + 1.0)

    def test_drift_is_zero_at_the_moment_of_injection(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        adapter.inject(FaultInjection(kind=InjectedFault.DRIFT, magnitude=0.01))
        assert adapter.read().value == pytest.approx(NOMINAL_C)

    def test_clearing_restores_the_instrument(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        adapter.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        adapter.clear()
        assert adapter.read().value == NOMINAL_C

    def test_the_active_injection_is_reportable(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        adapter.inject(FaultInjection(kind=InjectedFault.DRIFT, magnitude=0.01))
        assert adapter.injected_fault is InjectedFault.DRIFT

    def test_an_injected_reading_still_looks_ordinary_downstream(
        self, config, mqtt_config
    ):
        """Nothing above Layer 1 can tell injection from a real fault."""
        adapter, transport, _ = _adapter(config, mqtt_config)
        adapter.inject(FaultInjection(kind=InjectedFault.STUCK_AT, magnitude=27.0))
        adapter.poll()
        payload = transport.on(f"space/sensor/{SENSOR_ID}/state")[0]
        restored = SensorReading.model_validate_json(payload)
        assert restored.sensor_id == SENSOR_ID and restored.value == 27.0


class TestPublishing:
    def test_readings_go_to_the_sensor_state_topic(self, config, mqtt_config):
        adapter, transport, _ = _adapter(config, mqtt_config)
        adapter.poll()
        assert transport.on(f"space/sensor/{SENSOR_ID}/state")

    def test_readings_are_not_retained(self, config, mqtt_config):
        adapter, transport, _ = _adapter(config, mqtt_config)
        adapter.poll()
        assert not any(
            retain
            for name, _, _, retain in transport.published
            if name.endswith("/state")
        )

    def test_health_is_retained_for_late_subscribers(self, config, mqtt_config):
        """FR-61: what is trusted must be readable without waiting."""
        adapter, transport, _ = _adapter(config, mqtt_config)
        adapter.poll()
        assert transport.retained(f"space/sensor/{SENSOR_ID}/health")

    def test_health_is_published_even_when_the_reading_was_lost(
        self, config, mqtt_config
    ):
        adapter, transport, _ = _adapter(config, mqtt_config, source=FakeSource(None))
        adapter.poll()
        assert transport.on(f"space/sensor/{SENSOR_ID}/health")

    def test_health_reports_the_last_reading_time(self, config, mqtt_config):
        clock = SimClock()
        adapter, transport, _ = _adapter(config, mqtt_config, clock=clock)
        adapter.poll()
        payload = transport.on(f"space/sensor/{SENSOR_ID}/health")[0]
        assert SensorHealth.model_validate_json(payload).last_reading_ts == clock.now()

    def test_health_is_suspect_before_anything_has_arrived(self, config, mqtt_config):
        adapter, transport, _ = _adapter(config, mqtt_config, source=FakeSource(None))
        adapter.poll()
        payload = transport.on(f"space/sensor/{SENSOR_ID}/health")[0]
        assert SensorHealth.model_validate_json(payload).quality is Quality.SUSPECT

    def test_health_reports_a_fault_the_detectors_raised(self, config, mqtt_config):
        adapter, transport, _ = _adapter(config, mqtt_config)
        adapter.attach_fault("f_temp01_stuck_1756032")
        adapter.poll()
        payload = transport.on(f"space/sensor/{SENSOR_ID}/health")[-1]
        health = SensorHealth.model_validate_json(payload)
        assert health.quality is Quality.FAULTED
        assert health.active_fault_id == "f_temp01_stuck_1756032"

    def test_poll_returns_what_it_published(self, config, mqtt_config):
        adapter, _, _ = _adapter(config, mqtt_config)
        assert adapter.poll().value == NOMINAL_C


class TestEsphomeSource:
    def _source(self, clock=None) -> tuple[EsphomeSource, SimClock]:
        clock = clock or SimClock()
        return EsphomeSource(clock, STALE_AFTER_S), clock

    def test_reports_nothing_before_the_device_has_spoken(self):
        source, _ = self._source()
        assert source.read() is None

    def test_is_stale_before_the_device_has_spoken(self):
        source, _ = self._source()
        assert source.is_stale is True

    def test_reports_the_value_the_device_sent(self):
        source, _ = self._source()
        source.accept(NOMINAL_C)
        assert source.read() == NOMINAL_C

    def test_a_fresh_value_is_not_stale(self):
        source, clock = self._source()
        source.accept(NOMINAL_C)
        clock.advance(STALE_AFTER_S / 2.0)
        assert source.is_stale is False

    def test_a_value_goes_silent_rather_than_repeating_forever(self):
        """A node that stopped publishing must look like silence: repeating
        would turn a dropped node into a stuck sensor and send D1 and D2
        looking for different faults in the same place."""
        source, clock = self._source()
        source.accept(NOMINAL_C)
        clock.advance(STALE_AFTER_S + 1.0)
        assert source.read() is None

    def test_a_new_value_refreshes_the_horizon(self):
        source, clock = self._source()
        source.accept(NOMINAL_C)
        clock.advance(STALE_AFTER_S + 1.0)
        source.accept(28.0)
        assert source.read() == 28.0

    def test_a_non_positive_horizon_is_refused(self):
        with pytest.raises(ValueError):
            EsphomeSource(SimClock(), 0.0)

    def test_it_satisfies_the_adapter_source_contract(self, config, mqtt_config):
        source, _ = self._source()
        source.accept(NOMINAL_C)
        adapter, _, _ = _adapter(config, mqtt_config, source=source)
        assert adapter.read().value == NOMINAL_C
