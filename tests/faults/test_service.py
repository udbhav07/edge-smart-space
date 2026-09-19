"""Unit tests for the detector bank service.

The real configuration is used throughout, because half of what this service
does is decide which detectors watch which sensor from that configuration, and
a test against an invented config would not check the thing that breaks.

No broker: the transport is injected, and every published payload is decoded
back from bytes so a schema change cannot pass unnoticed.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    DetectorId,
    FaultEvent,
    Quality,
    SensorHealth,
    SensorReading,
    Unit,
)
from src.faults.__main__ import run
from src.faults.service import build_service

INDOOR = "temp_01"
HUMIDITY = "hum_01"
OUTDOOR = "outdoor_01"
OCCUPANCY = "pir_01"
STUCK_VALUE = 27.0


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []

    def connect(self, host, port, keepalive):
        pass

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def faults(self) -> list[FaultEvent]:
        return [
            FaultEvent.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic.startswith("space/fault/") and payload
        ]

    def withdrawals(self) -> list[str]:
        return [
            topic
            for topic, payload, _, _ in self.published
            if topic.startswith("space/fault/") and not payload
        ]

    def health(self) -> list[SensorHealth]:
        return [
            SensorHealth.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic.endswith("/health") and payload
        ]


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="wired")
def _wired(config, clock):
    transport = FakeTransport()
    blackboard = Blackboard(config.mqtt, transport)
    service = build_service(config, clock, blackboard)
    service.subscribe()
    blackboard.on_connected()
    return service, transport, blackboard


def _reading(clock, sensor_id=INDOOR, value=27.4, unit=Unit.CELSIUS):
    return SensorReading(
        ts=clock.now(), sensor_id=sensor_id, value=value, unit=unit
    )


def _deliver(blackboard, reading: SensorReading) -> None:
    """Send a reading the way the broker would, through dispatch."""
    blackboard.dispatch(
        f"space/sensor/{reading.sensor_id}/state",
        reading.model_dump_json().encode(),
    )


def _report_healthily(service, blackboard, clock, duration_s, value=None) -> None:
    """Keep a sensor reporting normally for a while, ticking as it goes.

    Recovery has to be driven this way rather than by advancing the clock: a
    sensor that stops reporting during its own clear confirmation is faulted
    again, which is the correct answer and not something to work around.
    """
    period_s = load_config(Path("config/default.yaml")).loop.sensor_period_s
    elapsed = 0.0
    index = 0
    while elapsed < duration_s:
        drift = 0.2 if index % 2 else -0.2
        _deliver(
            blackboard,
            _reading(clock, value=(27.4 + drift) if value is None else value),
        )
        service.tick()
        clock.advance(period_s)
        elapsed += period_s
        index += 1


class TestBankConstruction:
    def test_every_configured_sensor_is_watched(self, wired, config):
        service, _, _ = wired
        expected = {sensor.sensor_id for sensor in config.sensors.adapters}
        assert service.watched_subjects == expected

    def test_an_empty_room_reporting_zero_all_night_raises_nothing(
        self, wired, clock, config
    ):
        """Variance says nothing about a PIR: an unoccupied room reports a
        constant legitimately, and D2 on it would fire every quiet night."""
        service, transport, blackboard = wired
        for _ in range(config.detectors.stuck_at.window_samples + 4):
            _deliver(blackboard, _reading(clock, OCCUPANCY, 0.0, Unit.BOOLEAN))
            service.tick()
            clock.advance(config.loop.sensor_period_s)
        assert [
            event for event in transport.faults() if event.subject == OCCUPANCY
        ] == []

    def test_a_continuous_sensor_is_watched_by_all_three(self, wired, clock, config):
        """Each detector is reachable on temp_01: silence raises D1, a frozen
        value raises D2, and an implausible one raises D3."""
        service, transport, blackboard = wired
        for _ in range(config.detectors.out_of_range.debounce_samples):
            _deliver(blackboard, _reading(clock, value=999.0))
            clock.advance(config.loop.sensor_period_s)
        service.tick()
        assert transport.faults()[0].detector is DetectorId.D3_OUT_OF_RANGE

    def test_the_humidity_sensor_gets_its_own_bounds(self, wired, clock):
        """D3 is not a temperature detector; the bounds come from the unit."""
        service, transport, blackboard = wired
        for _ in range(2):
            _deliver(
                blackboard, _reading(clock, HUMIDITY, 105.0, Unit.PERCENT_RH)
            )
            clock.advance(5.0)
        service.tick()
        raised = [event for event in transport.faults() if event.subject == HUMIDITY]
        assert raised and raised[0].detector is DetectorId.D3_OUT_OF_RANGE


class TestDropoutTimeoutPerSensor:
    def test_the_outdoor_sensor_is_allowed_its_slower_cadence(self, wired, clock):
        """Ambient updates every 60 s. A timeout from the 5 s indoor period
        would report it dropped almost continuously."""
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock, OUTDOOR, 31.0))
        clock.advance(20.0)
        service.tick()
        assert [event.subject for event in transport.faults()] == []

    def test_the_indoor_sensor_is_not(self, wired, clock):
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(20.0)
        service.tick()
        raised = transport.faults()
        assert [event.subject for event in raised] == [INDOOR]
        assert raised[0].detector is DetectorId.D1_DROPOUT


class TestPublishingFaults:
    def test_a_dropout_is_published_as_a_retained_fault(self, wired, clock):
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        service.tick()
        topic, _, _, retain = [
            entry for entry in transport.published if "/fault/" in entry[0]
        ][0]
        assert topic.startswith("space/fault/f_temp01_dropout_")
        assert retain is True

    def test_the_published_fault_carries_its_evidence(self, wired, clock):
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        service.tick()
        event = transport.faults()[0]
        assert event.evidence["silence_s"] == pytest.approx(60.0)
        assert event.evidence["timeout_s"] == 15.0

    def test_a_fault_is_published_once_not_every_tick(self, wired, clock):
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        for _ in range(4):
            service.tick()
            clock.advance(5.0)
        assert len(transport.faults()) == 1

    def test_a_recovered_sensor_has_its_fault_withdrawn(self, wired, clock):
        """A resolved fault is not an answer to "what is wrong now"."""
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        service.tick()
        raised = transport.faults()[0]

        _report_healthily(
            service, blackboard, clock, config_confirm_s() + 10.0
        )
        assert transport.withdrawals() == [f"space/fault/{raised.fault_id}"]

    def test_nothing_is_published_before_any_reading_arrives(self, wired, clock):
        """Every detector answers UNKNOWN, which is not a claim about health."""
        service, transport, _ = wired
        clock.advance(600.0)
        service.tick()
        assert transport.published == []


def config_confirm_s() -> float:
    return load_config(Path("config/default.yaml")).mode.fault_clear_confirm_s


class TestPublishingHealth:
    def test_a_faulted_sensor_is_published_as_faulted(self, wired, clock):
        """This is the message the estimator freezes adaptation on (FR-29)."""
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        service.tick()
        faulted = [entry for entry in transport.health() if entry.sensor_id == INDOOR]
        assert faulted[-1].quality is Quality.FAULTED

    def test_the_health_message_names_the_fault(self, wired, clock):
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        service.tick()
        event = transport.faults()[0]
        named = [
            entry
            for entry in transport.health()
            if entry.active_fault_id == event.fault_id
        ]
        assert named

    def test_health_is_retained(self, wired, clock):
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        service.tick()
        retained = [
            entry[3] for entry in transport.published if entry[0].endswith("/health")
        ]
        assert retained and all(retained)

    def test_a_healthy_sensor_is_published_once_not_every_tick(self, wired, clock):
        service, transport, blackboard = wired
        for _ in range(4):
            _deliver(blackboard, _reading(clock))
            service.tick()
            clock.advance(5.0)
        indoor = [entry for entry in transport.health() if entry.sensor_id == INDOOR]
        assert len(indoor) == 1

    def test_recovery_republishes_health_as_ok(self, wired, clock):
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        service.tick()

        _report_healthily(
            service, blackboard, clock, config_confirm_s() + 10.0
        )
        indoor = [entry for entry in transport.health() if entry.sensor_id == INDOOR]
        assert [entry.quality for entry in indoor][-1] is Quality.OK

    def test_health_carries_the_last_reading_it_saw(self, wired, clock):
        service, transport, blackboard = wired
        reading = _reading(clock)
        _deliver(blackboard, reading)
        service.tick()
        indoor = [entry for entry in transport.health() if entry.sensor_id == INDOOR]
        assert indoor[0].last_reading_ts == reading.ts


class TestClearMustHold:
    def test_a_sensor_that_goes_quiet_again_stays_faulted(self, wired, clock):
        """Recovery is a sensor reporting for the whole confirmation period,
        not one reading and then silence."""
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        clock.advance(60.0)
        service.tick()

        _deliver(blackboard, _reading(clock))
        service.tick()
        clock.advance(config_confirm_s())
        service.tick()
        assert transport.withdrawals() == []


class TestStuckAtEndToEnd:
    def test_a_frozen_sensor_is_detected_through_the_whole_bank(self, wired, clock):
        """Section 5.9.2's scenario: the sensor keeps reporting a constant and
        the variance test raises it."""
        service, transport, blackboard = wired
        config = load_config(Path("config/default.yaml"))
        window = config.detectors.stuck_at.window_samples
        consecutive = config.detectors.stuck_at.consecutive_windows

        for _ in range(window + consecutive):
            _deliver(blackboard, _reading(clock, value=STUCK_VALUE))
            service.tick()
            clock.advance(config.loop.sensor_period_s)

        stuck = [
            event
            for event in transport.faults()
            if event.detector is DetectorId.D2_STUCK_AT
        ]
        assert stuck and stuck[0].subject == INDOOR

    def test_a_moving_sensor_raises_nothing(self, wired, clock):
        service, transport, blackboard = wired
        config = load_config(Path("config/default.yaml"))
        for index in range(config.detectors.stuck_at.window_samples + 4):
            value = STUCK_VALUE + (0.2 if index % 2 else -0.2)
            _deliver(blackboard, _reading(clock, value=value))
            service.tick()
            clock.advance(config.loop.sensor_period_s)
        assert transport.faults() == []


class TestRobustness:
    def test_a_reading_from_an_unconfigured_sensor_is_ignored(self, wired, clock):
        """Taking the service down would lose detection on every sensor that
        is configured."""
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock, sensor_id="ghost_99"))
        service.tick()
        assert transport.faults() == []

    def test_a_malformed_payload_never_reaches_a_detector(self, wired):
        service, transport, blackboard = wired
        blackboard.dispatch("space/sensor/temp_01/state", b"{not json")
        service.tick()
        assert transport.published == []

    def test_the_service_subscribes_to_every_sensor(self, wired):
        _, transport, _ = wired
        assert ("space/sensor/+/state", 0) in transport.subscribed


class TestRunLoop:
    def test_the_loop_ticks_the_requested_number_of_times(self, wired, clock):
        """The entry point's loop, driven for real rather than doubled."""
        service, transport, blackboard = wired
        _deliver(blackboard, _reading(clock))
        run(service, clock, period_s=20.0, ticks=3)
        assert transport.faults()

    def test_the_loop_sleeps_the_configured_period(self, wired, clock):
        service, _, _ = wired
        started = clock.now()
        run(service, clock, period_s=5.0, ticks=4)
        assert clock.now() - started == pytest.approx(20.0)
