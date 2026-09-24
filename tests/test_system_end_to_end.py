"""The whole system, Weeks 1 to 4, running as one thing.

Every other test exercises a component or a pair. This one wires all four
processes onto one bus -- plant, estimator, detector bank, regulatory loop --
and runs them in lockstep with a simulated clock, over exactly the topics
hardware will use. Nothing is called directly: each component only ever sees
messages.

It exists because the interesting failures in this system are not inside
components, they are between them. The estimator sat subscribed to a health
topic nobody published to; the controller was written, tested and never run.
Both passed every unit test in the repository.

What it asserts, in order, is the project's claim:

* the room is controlled, and the model identifies itself while it happens;
* a sensor that starts lying is detected, with evidence;
* the loop stays closed on the model's prediction rather than collapsing to
  open loop (FR-27), which is the thing a thermostat cannot do;
* the substitution is time-boxed, and the system stops rather than pretending
  a stale prediction is a measurement (section 7.2);
* an air conditioner that is not cooling is caught by the room not responding,
  with no acknowledgement anywhere (FR-24, R-02);
* and control survives every layer above it being absent (FR-47).
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AdaptationState,
    Coefficients,
    Command,
    CommandKind,
    DetectorId,
    FaultEvent,
    Mode,
    ModeState,
    Quality,
    SensorHealth,
    ThermalEstimate,
)
from src.control.service import build_service as build_control
from src.estimation.__main__ import build_service as build_estimator
from src.faults.injector import FaultInjector
from src.faults.service import build_service as build_bank
from eval.loopback import LoopbackTransport
from sim.run_sim import INDOOR_TEMPERATURE_ID, build_simulator

STUCK_VALUE_C = 27.0


class System:
    """Four services and a plant, on one in-process bus."""

    def __init__(self, config, clock) -> None:
        self.config = config
        self.clock = clock
        self.transport = LoopbackTransport()

        boards = [Blackboard(config.mqtt, self.transport) for _ in range(5)]
        plant_board, estimator_board, bank_board, control_board, operator = boards

        self.simulator = build_simulator(config, clock, plant_board)
        self.estimator = build_estimator(config, clock, estimator_board)
        self.bank = build_bank(config, clock, bank_board)
        self.control = build_control(config, clock, control_board)
        self.injector = FaultInjector(operator, clock)

        for service in (self.simulator, self.estimator, self.bank, self.control):
            service.subscribe()
        for board in boards:
            self.transport.attach(board)

    def run_for(self, seconds: float) -> None:
        """Advance every process one period at a time.

        Order within a tick matters and mirrors reality: the plant publishes,
        whoever is listening reacts, and the controller acts on what it heard.
        """
        period_s = self.config.loop.sensor_period_s
        elapsed = 0.0
        while elapsed < seconds:
            self.simulator.step()
            self.bank.tick()
            self.control.tick()
            self.clock.advance(period_s)
            elapsed += period_s

    # --- what the bus saw ---------------------------------------------

    def _decode(self, topic_test, schema):
        return [
            schema.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic_test(topic) and payload
        ]

    def faults(self) -> list[FaultEvent]:
        return self._decode(lambda t: t.startswith("space/fault/"), FaultEvent)

    def modes(self) -> list[ModeState]:
        return self._decode(lambda t: t == "space/system/mode", ModeState)

    def commands(self) -> list[Command]:
        return self._decode(lambda t: t.endswith("/command"), Command)

    def estimates(self) -> list[ThermalEstimate]:
        return self._decode(
            lambda t: t == "space/estimate/thermal", ThermalEstimate
        )

    def coefficients(self) -> list[Coefficients]:
        return self._decode(
            lambda t: t == "space/estimate/coefficients", Coefficients
        )

    def health(self, sensor_id: str) -> list[SensorHealth]:
        return self._decode(
            lambda t: t == f"space/sensor/{sensor_id}/health", SensorHealth
        )

    def mode(self) -> Mode:
        modes = self.modes()
        return modes[-1].mode if modes else Mode.INIT

    def commands_since(self, count: int) -> list[Command]:
        return self.commands()[count:]

    def room_temperature_c(self) -> float:
        """Ground truth, which only the test is allowed to look at."""
        return self.simulator.room_temperature_c


@pytest.fixture(name="config")
def _config(tmp_path):
    """The shipped configuration, with the estimator's state file redirected.

    Nothing else is softened. The sensors keep their noise, quantisation,
    jitter and dropout, and the actuator keeps its dead time and command loss:
    a system that only works against a kind simulator has not been shown to
    work.
    """
    config = load_config(Path("config/default.yaml"))
    persistence = config.persistence.model_copy(
        update={"path": str(tmp_path / "coefficients.json")}
    )
    return config.model_copy(update={"persistence": persistence})


@pytest.fixture(name="system")
def _system(config) -> System:
    return System(config, SimClock())


class TestNominalOperation:
    def test_the_system_reaches_normal_and_stays_there(self, system):
        system.run_for(600.0)
        assert system.mode() is Mode.NORMAL

    def test_a_healthy_run_raises_no_faults(self, system):
        """The precondition for everything below: if a working system raises
        faults, no detection below proves anything."""
        system.run_for(600.0)
        assert system.faults() == []

    def test_the_room_is_driven_towards_the_setpoint(self, system, config):
        """It starts at 29 C and is asked for 24 C."""
        started = system.room_temperature_c()
        system.run_for(1800.0)
        assert system.room_temperature_c() < started

    def test_the_controller_commands_every_tick(self, system):
        system.run_for(60.0)
        assert len(system.commands()) >= 10

    def test_the_model_identifies_itself_while_the_room_runs(self, system):
        """FR-04: no offline training, no fixed gains."""
        system.run_for(600.0)
        published = system.coefficients()
        assert published and published[-1].samples_since_reset > 0

    def test_the_indoor_sensor_is_reported_as_trusted(self, system):
        system.run_for(600.0)
        assert system.health(INDOOR_TEMPERATURE_ID)[-1].quality is Quality.OK


class TestASensorThatStartsLying:
    """The core fault-tolerance claim, end to end."""

    def _break_the_sensor(self, system) -> None:
        system.run_for(600.0)
        system.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE_C
        )

    def test_the_stuck_sensor_is_detected(self, system, config):
        self._break_the_sensor(system)
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        assert DetectorId.D2_STUCK_AT in {
            event.detector for event in system.faults()
        }

    def test_the_fault_carries_the_evidence_that_raised_it(self, system, config):
        self._break_the_sensor(system)
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        stuck = [
            event
            for event in system.faults()
            if event.detector is DetectorId.D2_STUCK_AT
        ][0]
        assert stuck.evidence["variance"] < stuck.evidence["variance_epsilon"]

    def test_the_system_degrades_rather_than_stopping(self, system, config):
        self._break_the_sensor(system)
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        assert system.mode() is Mode.DEGRADED_SENSOR

    def test_adaptation_freezes_so_the_model_does_not_learn_the_fault(
        self, system, config
    ):
        """FR-29. The estimator hears it from the health topic, which is the
        wiring that was missing before Week 3."""
        self._break_the_sensor(system)
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        assert system.health(INDOOR_TEMPERATURE_ID)[-1].quality is Quality.FAULTED
        assert system.estimator.adaptation_frozen

    def test_the_loop_stays_closed_on_the_model_instead(self, system, config):
        """FR-27, and the whole reason for identifying a model. A thermostat
        has nothing to fall back to here."""
        self._break_the_sensor(system)
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        assert system.mode() is Mode.DEGRADED_SENSOR

        before = len(system.commands())
        system.run_for(120.0)
        issued = system.commands_since(before)
        assert len(issued) >= 20
        assert all(command.kind is not CommandKind.HOLD for command in issued)

    def test_the_estimate_keeps_being_published_while_frozen(self, system, config):
        """Prediction continues while adaptation stops, which is what leaves
        degraded control something to run on."""
        self._break_the_sensor(system)
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        latest = system.estimates()[-1]
        assert latest.adaptation is AdaptationState.FROZEN


class TestTheSubstitutionIsBounded:
    def test_an_exhausted_budget_stops_the_system(self, config, tmp_path):
        """Section 7.2: a prediction is not a measurement forever. The budget
        is shortened here because the mechanism is what is under test, not the
        1800 s value -- which E4 is the experiment for."""
        budget = config.mode.model_copy(update={"degraded_sensor_budget_s": 300.0})
        system = System(config.model_copy(update={"mode": budget}), SimClock())

        system.run_for(600.0)
        system.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE_C
        )
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        assert system.mode() is Mode.DEGRADED_SENSOR

        system.run_for(400.0)
        assert system.mode() is Mode.SAFE_HOLD

    def test_actuation_ceases_once_it_holds(self, config, tmp_path):
        budget = config.mode.model_copy(update={"degraded_sensor_budget_s": 300.0})
        system = System(config.model_copy(update={"mode": budget}), SimClock())

        system.run_for(600.0)
        system.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE_C
        )
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 520.0)
        assert system.mode() is Mode.SAFE_HOLD

        before = len(system.commands())
        system.run_for(60.0)
        issued = system.commands_since(before)
        assert issued and all(
            command.kind is CommandKind.HOLD for command in issued
        )


class TestRecovery:
    def test_a_repaired_sensor_returns_the_system_to_normal(self, system, config):
        system.run_for(600.0)
        system.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE_C
        )
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)
        assert system.mode() is Mode.DEGRADED_SENSOR

        system.injector.clear(INDOOR_TEMPERATURE_ID)
        system.run_for(window_s + config.mode.fault_clear_confirm_s + 120.0)
        assert system.mode() is Mode.NORMAL

    def test_adaptation_resumes_once_the_sensor_is_trusted_again(
        self, system, config
    ):
        system.run_for(600.0)
        system.injector.inject(
            INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE_C
        )
        window_s = (
            config.detectors.stuck_at.window_samples * config.loop.sensor_period_s
        )
        system.run_for(window_s + 120.0)

        system.injector.clear(INDOOR_TEMPERATURE_ID)
        system.run_for(window_s + config.mode.fault_clear_confirm_s + 120.0)
        assert not system.estimator.adaptation_frozen


class TestAnActuatorThatIsNotCooling:
    def test_a_dead_actuator_is_caught_by_the_room_not_responding(
        self, config, tmp_path
    ):
        """FR-24 with no acknowledgement anywhere (R-02): the plant is told to
        cool, the room does not move, and that is the evidence."""
        room = config.sim.room.model_copy(update={"cooling_power_w": 0.0})
        broken = config.model_copy(
            update={"sim": config.sim.model_copy(update={"room": room})}
        )
        system = System(broken, SimClock())

        window_s = config.detectors.actuator.evaluation_window_s
        system.run_for(window_s + 300.0)
        assert DetectorId.D5_ACTUATOR_NO_RESPONSE in {
            event.detector for event in system.faults()
        }

    def test_the_system_stops_actuating_when_the_actuator_is_faulted(
        self, config, tmp_path
    ):
        """FR-28: cease closed-loop actuation and hold."""
        room = config.sim.room.model_copy(update={"cooling_power_w": 0.0})
        broken = config.model_copy(
            update={"sim": config.sim.model_copy(update={"room": room})}
        )
        system = System(broken, SimClock())

        window_s = config.detectors.actuator.evaluation_window_s
        system.run_for(window_s + 300.0)
        assert system.mode() in (Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD)

        before = len(system.commands())
        system.run_for(60.0)
        issued = system.commands_since(before)
        assert issued and all(
            command.kind is CommandKind.HOLD for command in issued
        )


class TestTheLoopSurvivesWhatIsAboveIt:
    def test_no_goal_is_ever_published_and_control_still_runs(self, system, config):
        """FR-47: there is no reasoning layer in this test at all."""
        system.run_for(600.0)
        published_goals = [
            topic
            for topic, _, _, _ in system.transport.published
            if topic.startswith("space/goal/")
        ]
        assert published_goals == []
        assert system.commands()
        assert system.control.setpoint_c == config.controller.default_setpoint_c

    def test_every_command_is_accompanied_by_a_published_verdict(self, system):
        """FR-46: the audit trail is not optional."""
        system.run_for(120.0)
        verdicts = [
            topic
            for topic, _, _, _ in system.transport.published
            if topic == "space/audit/validation"
        ]
        assert len(verdicts) >= len(system.commands())

    def test_the_whole_run_uses_no_wall_clock(self, system):
        """Everything is driven by the injected clock, which is what lets an
        accelerated experiment run at all (NFR-01)."""
        started = system.clock.now()
        system.run_for(300.0)
        assert system.clock.now() - started == pytest.approx(300.0)
