"""Injection to detection, over the real topics (FR-31, FR-20 to FR-22).

Every other test in this directory exercises one component. This one wires the
simulator and the detector bank through the loopback transport and asks the
question the Week 3 gate actually asks: if somebody breaks a sensor while the
system is running, does the system say so, with evidence?

Nothing here calls a detector. The injector publishes, the plant obeys at the
sensor, readings go out on the same topics an ESP32 will use, and the bank
reads them back. If this passes and the unit tests pass, the wiring between
them is what was tested -- which is the part that was missing when the
estimator sat subscribed to a health topic nobody published to.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    DetectorId,
    FaultEvent,
    Quality,
    SensorHealth,
)
from src.faults.injector import FaultInjector
from src.faults.service import build_service
from eval.loopback import LoopbackTransport
from sim.run_sim import INDOOR_TEMPERATURE_ID, build_simulator

STUCK_VALUE = 27.0
IMPLAUSIBLE_VALUE = 999.0


@pytest.fixture(name="config")
def _config():
    """Sampling imperfection off, so a detection is attributable.

    Dropout probability is switched off deliberately here and nowhere else:
    with it on, a fault raised by D1 could be the injection or could be the
    ordinary 1% loss, and the test would not be measuring what it claims.
    """
    config = load_config(Path("config/default.yaml"))
    noise = config.sim.sensor_noise.model_copy(
        update={"dropout_probability": 0.0, "jitter_s": 0.0}
    )
    return config.model_copy(
        update={"sim": config.sim.model_copy(update={"sensor_noise": noise})}
    )


class Wiring:
    """A plant and a detector bank on one loopback bus."""

    def __init__(self, config) -> None:
        self.clock = SimClock()
        self.transport = LoopbackTransport()

        plant_board = Blackboard(config.mqtt, self.transport)
        bank_board = Blackboard(config.mqtt, self.transport)
        operator_board = Blackboard(config.mqtt, self.transport)

        self.simulator = build_simulator(config, self.clock, plant_board)
        self.simulator.subscribe()
        self.bank = build_service(config, self.clock, bank_board)
        self.bank.subscribe()
        self.injector = FaultInjector(operator_board, self.clock)

        for board in (plant_board, bank_board, operator_board):
            self.transport.attach(board)

        self._period_s = config.loop.sensor_period_s

    def run_for(self, seconds: float) -> None:
        """Step the plant and tick the bank, in lockstep with the clock."""
        elapsed = 0.0
        while elapsed < seconds:
            self.simulator.step()
            self.bank.tick()
            self.clock.advance(self._period_s)
            elapsed += self._period_s

    def faults(self) -> list[FaultEvent]:
        return [
            FaultEvent.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic.startswith("space/fault/") and payload
        ]

    def health(self, sensor_id: str) -> list[SensorHealth]:
        return [
            SensorHealth.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic == f"space/sensor/{sensor_id}/health" and payload
        ]


@pytest.fixture(name="wiring")
def _wiring(config) -> Wiring:
    return Wiring(config)


class TestNothingWrong:
    def test_a_healthy_room_raises_no_faults(self, wiring):
        """The precondition for every other test here: if a working plant
        raises faults, a detection proves nothing."""
        wiring.run_for(600.0)
        assert wiring.faults() == []

    def test_a_healthy_sensor_is_reported_as_trusted(self, wiring):
        wiring.run_for(60.0)
        assert wiring.health(INDOOR_TEMPERATURE_ID)[-1].quality is Quality.OK


class TestDropoutEndToEnd:
    def test_a_sensor_told_to_go_quiet_is_detected(self, wiring):
        wiring.run_for(60.0)
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        wiring.run_for(60.0)
        detectors = {event.detector for event in wiring.faults()}
        assert DetectorId.D1_DROPOUT in detectors

    def test_the_fault_names_the_sensor_that_went_quiet(self, wiring):
        wiring.run_for(60.0)
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        wiring.run_for(60.0)
        assert wiring.faults()[0].subject == INDOOR_TEMPERATURE_ID

    def test_the_fault_carries_the_silence_that_produced_it(self, wiring):
        """Evidence, not just an outcome: the number a human checks."""
        wiring.run_for(60.0)
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        wiring.run_for(60.0)
        evidence = wiring.faults()[0].evidence
        assert evidence["silence_s"] > evidence["timeout_s"]

    def test_detection_lands_inside_the_documented_latency(self, wiring, config):
        """Section 5.5 targets under 20 s for D1."""
        wiring.run_for(60.0)
        injected_at = wiring.clock.now()
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        wiring.run_for(20.0)
        raised = wiring.faults()
        assert raised and raised[0].detected_ts - injected_at <= 20.0

    def test_the_sensor_is_marked_untrusted_so_adaptation_freezes(self, wiring):
        """This is the message the estimator acts on (FR-29)."""
        wiring.run_for(60.0)
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        wiring.run_for(60.0)
        assert (
            wiring.health(INDOOR_TEMPERATURE_ID)[-1].quality is Quality.FAULTED
        )

    def test_clearing_the_injection_returns_the_sensor_to_trusted(self, wiring, config):
        wiring.run_for(60.0)
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        wiring.run_for(60.0)

        wiring.injector.clear(INDOOR_TEMPERATURE_ID)
        wiring.run_for(config.mode.fault_clear_confirm_s + 30.0)
        assert wiring.health(INDOOR_TEMPERATURE_ID)[-1].quality is Quality.OK

    def test_the_retired_fault_is_withdrawn_from_the_blackboard(self, wiring, config):
        wiring.run_for(60.0)
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        wiring.run_for(60.0)
        raised = wiring.faults()[0]

        wiring.injector.clear(INDOOR_TEMPERATURE_ID)
        wiring.run_for(config.mode.fault_clear_confirm_s + 30.0)
        withdrawals = [
            topic
            for topic, payload, _, _ in wiring.transport.published
            if topic == f"space/fault/{raised.fault_id}" and not payload
        ]
        assert withdrawals


class TestStuckAtEndToEnd:
    def test_a_frozen_sensor_is_detected(self, wiring, config):
        """Section 5.9.2's scenario, run for real: the sensor keeps reporting
        a plausible constant and the variance test catches it."""
        wiring.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE
        )
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        wiring.run_for(window_s + 60.0)
        detectors = {event.detector for event in wiring.faults()}
        assert DetectorId.D2_STUCK_AT in detectors

    def test_the_frozen_sensor_never_goes_quiet(self, wiring, config):
        """The point of the fault: readings keep arriving and look fine, which
        is why a threshold controller would act on them."""
        wiring.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE
        )
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        wiring.run_for(window_s + 60.0)
        assert DetectorId.D1_DROPOUT not in {
            event.detector for event in wiring.faults()
        }

    def test_the_fault_carries_the_variance_that_produced_it(self, wiring, config):
        wiring.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE
        )
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        wiring.run_for(window_s + 60.0)
        stuck = [
            event
            for event in wiring.faults()
            if event.detector is DetectorId.D2_STUCK_AT
        ][0]
        assert stuck.evidence["variance"] < stuck.evidence["variance_epsilon"]


class TestOutOfRangeEndToEnd:
    def test_an_impossible_reading_is_detected(self, wiring):
        wiring.run_for(30.0)
        wiring.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.OUT_OF_RANGE, IMPLAUSIBLE_VALUE
        )
        wiring.run_for(30.0)
        detectors = {event.detector for event in wiring.faults()}
        assert DetectorId.D3_OUT_OF_RANGE in detectors

    def test_the_fault_carries_the_value_and_the_bounds_it_left(self, wiring):
        wiring.run_for(30.0)
        wiring.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.OUT_OF_RANGE, IMPLAUSIBLE_VALUE
        )
        wiring.run_for(30.0)
        event = [
            fault
            for fault in wiring.faults()
            if fault.detector is DetectorId.D3_OUT_OF_RANGE
        ][0]
        assert event.evidence["value"] == IMPLAUSIBLE_VALUE
        assert event.evidence["high"] < IMPLAUSIBLE_VALUE

    def test_detection_lands_inside_the_documented_latency(self, wiring):
        """Section 5.5 targets under 10 s for D3."""
        wiring.run_for(30.0)
        injected_at = wiring.clock.now()
        wiring.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.OUT_OF_RANGE, IMPLAUSIBLE_VALUE
        )
        wiring.run_for(10.0)
        raised = [
            fault
            for fault in wiring.faults()
            if fault.detector is DetectorId.D3_OUT_OF_RANGE
        ]
        assert raised and raised[0].detected_ts - injected_at <= 10.0


class TestTheFaultIsIndistinguishableFromAReadOne:
    def test_no_published_reading_says_it_was_injected(self, wiring):
        """If a detector could tell, the trial would measure nothing."""
        wiring.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE
        )
        wiring.run_for(60.0)
        readings = [
            payload
            for topic, payload, _, _ in wiring.transport.published
            if topic.startswith("space/sensor/") and topic.endswith("/state")
        ]
        assert readings
        assert not any(b"inject" in payload for payload in readings)

    def test_the_injection_is_visible_on_its_own_topic(self, wiring):
        """Visible to a human reading the tree, and retained, so it answers
        what is being injected right now."""
        wiring.injector.inject(INDOOR_TEMPERATURE_ID, InjectedFault.DROPOUT)
        injections = [
            (topic, retain)
            for topic, _, _, retain in wiring.transport.published
            if topic.startswith("space/inject/")
        ]
        assert injections == [(f"space/inject/{INDOOR_TEMPERATURE_ID}", True)]


class TestBeforeAnythingHasBeenObserved:
    """The UNKNOWN case, live.

    Until this was fixed, hum_01 was configured and nothing ever published it,
    which made this class's point for free. FR-01 requires humidity to be
    measured, so the simulator now publishes it and the point has to be made
    honestly: at the start of a run, nothing has been observed about any
    sensor, and nothing is claimed about any of them.
    """

    def test_nothing_is_claimed_before_the_first_tick(self, wiring):
        assert wiring.faults() == []
        assert wiring.health(INDOOR_TEMPERATURE_ID) == []

    def test_a_sensor_is_not_reported_healthy_until_it_has_reported(
        self, wiring
    ):
        """There is no Quality for "not yet observed", so nothing is published
        rather than OK being asserted about a sensor nobody has heard."""
        assert wiring.health("hum_01") == []

    def test_every_configured_sensor_does_eventually_report(self, wiring, config):
        """A configured sensor nothing publishes is a hole: its detectors sit
        UNKNOWN forever and no fault on it can ever be raised."""
        wiring.run_for(300.0)
        for sensor in config.sensors.adapters:
            assert wiring.health(sensor.sensor_id), (
                f"{sensor.sensor_id} is configured but never reported"
            )
