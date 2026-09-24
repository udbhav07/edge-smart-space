"""The Week 5 bring-up check, run against the simulator.

The simulator publishes on exactly the topics hardware will, so a check that
passes here and fails on the Orin has found something about the hardware --
which is its whole job.
"""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.control.service import build_service as build_control
from src.faults.injector import FaultInjector
from sim.run_sim import build_simulator
from tools.bringup import POWER_SENSOR_ID, BringupMonitor


@pytest.fixture(name="config")
def _config():
    """Sampling loss off, so a verdict is about the injection, not the 1%."""
    config = load_config(Path("config/default.yaml"))
    noise = config.sim.sensor_noise.model_copy(update={"dropout_probability": 0.0})
    meter = config.sim.power_meter.model_copy(update={"dropout_probability": 0.0})
    actuator = config.sim.actuator.model_copy(update={"command_loss_probability": 0.0})
    sim = config.sim.model_copy(
        update={"sensor_noise": noise, "power_meter": meter, "actuator": actuator}
    )
    return config.model_copy(update={"sim": sim})


class Room:
    def __init__(self, config, with_control=False) -> None:
        self.config = config
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        plant = Blackboard(config.mqtt, self.transport)
        self.simulator = build_simulator(config, self.clock, plant)
        self.simulator.subscribe()
        watch = Blackboard(config.mqtt, self.transport)
        self.monitor = BringupMonitor(config, self.clock, watch)
        self.monitor.subscribe()
        operator = Blackboard(config.mqtt, self.transport)
        self.injector = FaultInjector(operator, self.clock)
        boards = [plant, watch, operator]
        self.control = None
        if with_control:
            control_board = Blackboard(config.mqtt, self.transport)
            self.control = build_control(config, self.clock, control_board)
            self.control.subscribe()
            boards.append(control_board)
        for board in boards:
            self.transport.attach(board)

    def run_for(self, seconds: float) -> None:
        elapsed = 0.0
        period = self.config.loop.sensor_period_s
        while elapsed < seconds:
            self.simulator.step()
            if self.control is not None:
                self.control.tick()
            self.clock.advance(period)
            elapsed += period


class TestSensors:
    def test_a_healthy_room_passes(self, config):
        room = Room(config)
        room.run_for(180.0)
        assert all(verdict.ok for verdict in room.monitor.sensors())

    def test_every_configured_sensor_is_judged(self, config):
        room = Room(config)
        room.run_for(30.0)
        judged = {verdict.sensor_id for verdict in room.monitor.sensors()}
        assert judged == {sensor.sensor_id for sensor in config.sensors.adapters}

    def test_a_silent_sensor_is_named_and_explained(self, config):
        room = Room(config)
        room.injector.inject("hum_01", InjectedFault.DROPOUT)
        room.run_for(60.0)
        humidity = next(v for v in room.monitor.sensors() if v.sensor_id == "hum_01")
        assert humidity.problem.startswith("SILENT")

    def test_the_outdoor_sensor_is_judged_against_its_own_period(self, config):
        """FR-03: 60 s is right for ambient, and must not read as slow."""
        room = Room(config)
        room.run_for(300.0)
        outdoor = next(v for v in room.monitor.sensors() if v.sensor_id == "outdoor_01")
        assert outdoor.ok and outdoor.expected_period_s == config.loop.outdoor_period_s

    def test_a_reading_outside_the_instruments_limits_is_flagged(self, config):
        room = Room(config)
        room.injector.inject("temp_01", InjectedFault.OUT_OF_RANGE, 999.0)
        room.run_for(30.0)
        indoor = next(v for v in room.monitor.sensors() if v.sensor_id == "temp_01")
        assert indoor.problem.startswith("OUT OF LIMITS")

    def test_sigma_is_measured_for_r04(self, config):
        """The number Week 7 sets detector thresholds from."""
        room = Room(config)
        room.run_for(180.0)
        indoor = next(v for v in room.monitor.sensors() if v.sensor_id == "temp_01")
        assert indoor.sigma is not None and indoor.sigma > 0.0

    def test_the_report_names_every_sensor(self, config):
        room = Room(config)
        room.run_for(30.0)
        report = room.monitor.report()
        assert all(sensor.sensor_id in report for sensor in config.sensors.adapters)

    def test_the_report_says_whether_the_actuator_is_real(self, config):
        room = Room(config)
        room.run_for(10.0)
        assert "(simulated)" in room.monitor.report()


class TestActuation:
    def test_asking_for_cooling_makes_the_meter_see_the_compressor(self, config):
        """The air conditioner takes real commands: the gate decided, the
        controller commanded, and the draw rose. Nothing here wrote to an
        actuator topic."""
        room = Room(config, with_control=True)
        room.run_for(60.0)
        room.monitor.begin_actuation()
        room.run_for(config.bringup.actuation_window_s)
        verdict = room.monitor.actuation()
        assert verdict.ok, verdict.problem

    def test_a_dead_unit_fails_the_check(self, config):
        room = Room(config, with_control=True)
        room.injector.inject("ac", InjectedFault.STUCK_OFF)
        room.run_for(60.0)
        room.monitor.begin_actuation()
        room.run_for(config.bringup.actuation_window_s)
        verdict = room.monitor.actuation()
        assert verdict.problem.startswith("NO RESPONSE")

    def test_with_no_gate_running_it_says_so(self, config):
        room = Room(config, with_control=False)
        room.run_for(30.0)
        room.monitor.begin_actuation()
        room.run_for(30.0)
        assert "gate never answered" in room.monitor.actuation().problem

    def test_the_check_asks_the_gate_it_never_commands(self, config):
        """FR-13: every command passes the validator, bring-up included."""
        room = Room(config, with_control=True)
        room.run_for(10.0)
        before = [n for n, _, _, _ in room.transport.published]
        room.monitor.begin_actuation()
        after = [n for n, _, _, _ in room.transport.published][len(before):]
        assert after[0] == "space/goal/proposed"

    def test_the_operator_goal_expires_by_itself(self, config):
        room = Room(config, with_control=True)
        goal = room.monitor.begin_actuation()
        assert goal.expires_ts - goal.ts == config.bringup.actuation_window_s

    def test_a_unit_already_running_is_inconclusive_not_failed(self, config):
        """The off-draw is the reference; a unit that never switched off
        before the check gives none, and a working unit must not fail."""
        room = Room(config, with_control=True)
        for watts in (1290.0, 1300.0, 1310.0):
            room.monitor._on_reading("", _power(room.clock, watts))
            room.clock.advance(5.0)
        room.monitor.begin_actuation()
        room.run_for(config.bringup.actuation_window_s)
        assert room.monitor.actuation().problem.startswith("INCONCLUSIVE")

    def test_the_reference_is_the_off_draw_not_an_average(self, config):
        room = Room(config, with_control=True)
        for watts in (4.0, 1300.0, 1300.0, 1300.0):
            room.monitor._on_reading("", _power(room.clock, watts))
            room.clock.advance(5.0)
        room.monitor.begin_actuation()
        room.run_for(config.bringup.actuation_window_s)
        assert room.monitor.actuation().baseline_w == 4.0

    def test_a_room_with_no_meter_cannot_be_checked(self, config):
        adapters = tuple(
            a for a in config.sensors.adapters if a.sensor_id != POWER_SENSOR_ID
        )
        unmetered = config.model_copy(
            update={"sensors": config.sensors.model_copy(update={"adapters": adapters})}
        )
        room = Room(unmetered, with_control=True)
        room.monitor.begin_actuation()
        assert "no power meter" in room.monitor.actuation().problem


def _power(clock, watts):
    from src.common.schemas import SensorReading, Unit

    return SensorReading(
        ts=clock.now(), sensor_id=POWER_SENSOR_ID, value=watts, unit=Unit.WATT
    )
