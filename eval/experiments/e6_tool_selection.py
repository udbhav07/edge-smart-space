"""E6: is the supervisor's tool selection reliable? (DESIGN.md sections 5.7.4, 8.3)

Labelled scenarios, each a room state put on a blackboard, then one
supervisory cycle against the real model. Three things are scored and
reported separately, because section 5.7.4 is emphatic that they are not the
same thing:

* **Schema validity** -- the proposal parsed and passed the argument check.
  Near 100% by construction, and not a result.
* **Tool selection** -- all four read tools consulted, then propose_setpoint
  called exactly once.
* **Argument correctness** -- the proposed setpoint is the one the policy
  gives for that scenario.

Run it (needs the inference server): ``python -m eval.experiments.e6_tool_selection``
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import Config, load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AdaptationState,
    FaultEvent,
    Goal,
    GoalSource,
    Mode,
    ModeState,
    SensorReading,
    TariffBand,
    TariffState,
    ThermalEstimate,
    Unit,
)
from src.control.service import build_service as build_control
from src.reasoning.chat import ChatClient
from src.reasoning.supervisor_agent import SupervisorAgent
from src.reasoning.supervisor_tools import SUPERVISOR_TOOLS, SupervisorState

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
READ_TOOLS = {spec.name for spec in SUPERVISOR_TOOLS} - {"propose_setpoint"}
SETPOINT_IN_FORCE_C = 25.0

OCCUPANCIES = ("occupied", "vacant 2 min", "vacant 30 min", "unknown")
FAULTS = (False, True)
TARIFFS = (TariffBand.NORMAL, TariffBand.PEAK)
TEMPERATURES_C = (23.0, 26.0, 29.0)


@dataclass(frozen=True)
class Scenario:
    occupancy: str
    faulted: bool
    tariff: TariffBand
    indoor_c: float

    def expected_c(self, config: Config) -> float:
        if self.faulted:
            return SETPOINT_IN_FORCE_C
        if self.occupancy == "vacant 30 min":
            return config.supervisor.vacant_setpoint_c
        return config.supervisor.occupied_setpoint_c

    def label(self) -> str:
        fault = "fault" if self.faulted else "no fault"
        return f"{self.occupancy}, {fault}, {self.tariff.value}, {self.indoor_c:g} C"


@dataclass(frozen=True)
class Outcome:
    scenario: Scenario
    schema_valid: bool
    tools_right: bool
    argument_right: bool
    proposed: float | None
    wall_s: float


def scenarios() -> list[Scenario]:
    return [
        Scenario(*values)
        for values in itertools.product(OCCUPANCIES, FAULTS, TARIFFS, TEMPERATURES_C)
    ]


def _stage(world: Blackboard, clock: SimClock, config: Config, scenario: Scenario) -> None:
    """Put the scenario's room on the blackboard, as the services would."""
    now = clock.now()
    world.publish(topics.GOAL_ACTIVE, Goal(
        ts=now, source=GoalSource.DEFAULT, setpoint_c=SETPOINT_IN_FORCE_C,
        mode=Mode.NORMAL, expires_ts=now + 3600.0,
    ))
    world.publish(topics.SENSOR_STATE, SensorReading(
        ts=now, sensor_id=config.estimator.outdoor_sensor_id, value=31.0, unit=Unit.CELSIUS,
    ), sensor_id=config.estimator.outdoor_sensor_id)
    world.publish(topics.TARIFF_STATE, TariffState(
        ts=now, band=scenario.tariff, next_transition_ts=now + 3600.0,
    ))
    pir = config.estimator.occupancy_sensor_id
    if scenario.occupancy != "unknown":
        world.publish(topics.SENSOR_STATE, SensorReading(
            ts=clock.now(), sensor_id=pir, value=1.0, unit=Unit.BOOLEAN), sensor_id=pir)
    if scenario.occupancy.startswith("vacant"):
        world.publish(topics.SENSOR_STATE, SensorReading(
            ts=clock.now(), sensor_id=pir, value=0.0, unit=Unit.BOOLEAN), sensor_id=pir)
        clock.advance(120.0 if scenario.occupancy == "vacant 2 min" else 1800.0)
    mode = Mode.DEGRADED_SENSOR if scenario.faulted else Mode.NORMAL
    fault_ids: tuple[str, ...] = ()
    if scenario.faulted:
        event = FaultEvent.model_validate({
            "fault_id": "f_temp01_stuck_1", "detector": "D2_STUCK_AT", "subject": "temp_01",
            "class": "sensor", "confidence": 1.0, "detected_ts": clock.now(),
            "evidence": {"variance": 0.0}, "mode_impact": "DEGRADED_SENSOR",
        })
        fault_ids = (event.fault_id,)
        world.publish(topics.SYSTEM_MODE, ModeState(
            ts=clock.now(), mode=mode, since_ts=clock.now(), active_fault_ids=fault_ids))
        world.publish(topics.FAULT, event, fault_id=event.fault_id)
    else:
        world.publish(topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=mode, since_ts=clock.now()))
    world.publish(topics.ESTIMATE_THERMAL, ThermalEstimate(
        ts=clock.now(), t_in=scenario.indoor_c, t_pred=scenario.indoor_c, residual=0.0,
        residual_sigma=0.2, model_confidence=0.9, adaptation=AdaptationState.ACTIVE,
    ))


def run_scenario(config: Config, chat: ChatClient, scenario: Scenario) -> Outcome:
    clock = SimClock()
    transport = LoopbackTransport()
    boards = [Blackboard(config.mqtt, transport) for _ in range(3)]
    for board in boards:
        transport.attach(board)
    control = build_control(config, clock, boards[0])
    control.subscribe()
    supervisor = SupervisorAgent(config.supervisor, clock, boards[1], chat, SupervisorState(config, clock))
    supervisor.subscribe()
    _stage(boards[2], clock, config, scenario)

    started = time.perf_counter()
    record = supervisor.run_cycle("periodic")
    wall_s = time.perf_counter() - started

    calls = list(record.tool_calls)
    proposals = [
        Goal.model_validate_json(payload)
        for topic, payload, _, _ in transport.published
        if topic == topics.GOAL_PROPOSED.pattern and payload
    ]
    schema_valid = bool(proposals) or "discarded" not in record.verdict
    tools_right = READ_TOOLS <= set(calls[:-1]) and calls.count("propose_setpoint") == 1 and calls[-1] == "propose_setpoint"
    proposed = proposals[-1].setpoint_c if proposals else None
    argument_right = proposed is not None and abs(proposed - scenario.expected_c(config)) < 0.01
    return Outcome(scenario, schema_valid, tools_right, argument_right, proposed, wall_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run experiment E6.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N scenarios")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.CRITICAL)
    config = load_config(arguments.config)
    chat = ChatClient(config.reasoning, SimClock())

    outcomes = []
    for scenario in scenarios()[: arguments.limit]:
        outcome = run_scenario(config, chat, scenario)
        outcomes.append(outcome)
        mark = "ok " if outcome.tools_right and outcome.argument_right else "BAD"
        print(
            f"{mark} {outcome.wall_s:5.1f}s {scenario.label():36s} "
            f"expected {scenario.expected_c(config):g}, proposed {outcome.proposed}",
            flush=True,
        )
    total = len(outcomes)
    rate = lambda attribute: sum(getattr(o, attribute) for o in outcomes)
    print(f"--- E6 over {total} scenarios, {config.reasoning.model} ---")
    print(f"schema validity      {rate('schema_valid')}/{total}  (by construction; not a result)")
    print(f"tool selection       {rate('tools_right')}/{total}  (all four reads, then one proposal)")
    print(f"argument correctness {rate('argument_right')}/{total}  (the policy's setpoint)")
    print(f"median wall time     {sorted(o.wall_s for o in outcomes)[total // 2]:.1f} s per cycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
