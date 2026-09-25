"""The whole system on one in-process bus, for tests and experiments.

Four services and a plant -- simulator, estimator, detector bank, regulatory
loop -- wired onto a loopback blackboard and driven by a simulated clock, so
an hour of room time runs in about a second. Every message crosses the same
topics and schemas it would against mosquitto; no component is called
directly.

Lives in ``eval`` because experiments need it as much as tests do, and
``eval`` may import anything.
"""

from __future__ import annotations

from src.common.schemas import (
    Coefficients,
    Command,
    FaultEvent,
    Mode,
    ModeState,
    SensorHealth,
    ThermalEstimate,
)
from src.control.service import build_service as build_control
from src.estimation.__main__ import build_service as build_estimator
from src.faults.injector import FaultInjector
from src.faults.service import build_service as build_bank
from eval.loopback import LoopbackTransport
from src.common.mqtt_client import Blackboard
from sim.run_sim import build_simulator


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
            self.estimator.tick()
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
