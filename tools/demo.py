"""Run the whole system against a scripted fault, and narrate what it does.

    python -m tools.demo                 # the full story, about 80 lines
    python -m tools.demo --quiet         # just the transitions
    python -m tools.demo --scenario actuator

No broker, no hardware, no wall-clock waiting: the four services are wired onto
an in-process bus and driven by a simulated clock, so an hour of room time runs
in a couple of seconds. Every message crosses the same topics and the same
schemas it would against mosquitto, and no component is called directly.

Two jobs. It is the fastest way to see the system work end to end without
standing up a broker, and it is the fallback if a live demonstration fails --
the same sequence, the same decisions, with nothing that can go wrong on the
night (section 3 of the skill: a recorded run must replay).

What it shows is the claim the project is making: a sensor starts lying, the
system notices with evidence, and the control loop keeps running on the model's
prediction instead of collapsing to open loop -- then stops when that has gone
on long enough to stop being trustworthy.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, SimClock
from src.common.config import Config, ConfigError, load_config
from src.common.injection import InjectedFault
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Command,
    FaultEvent,
    Mode,
    ModeState,
)
from src.control.service import build_service as build_control
from src.estimation.__main__ import build_service as build_estimator
from src.faults.injector import FaultInjector
from src.faults.service import build_service as build_bank
from eval.loopback import LoopbackTransport
from sim.run_sim import INDOOR_TEMPERATURE_ID, build_simulator

LOGGER = logging.getLogger("demo")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: The value a stuck sensor freezes at. Plausible on purpose: the fault is
#: only findable because the number stops *moving*, not because it looks wrong.
STUCK_VALUE_C = 27.0

#: Shortened for the demonstration so the whole story fits in one run. The
#: shipped value is 1800 s (section 7.2); what is being shown is that the
#: budget exists and is enforced, not what it should be.
DEMO_BUDGET_S = 420.0


class Narrator:
    """Watches the bus and says what changed, in English."""

    def __init__(self, blackboard: Blackboard, clock: Clock, quiet: bool) -> None:
        self._clock = clock
        self._quiet = quiet
        self._started = clock.now()
        self._commands = 0
        blackboard.subscribe(topics.FAULT, FaultEvent, self._on_fault)
        blackboard.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)
        blackboard.subscribe(topics.ACTUATOR_COMMAND, Command, self._on_command)

    @property
    def commands(self) -> int:
        return self._commands

    def _at(self) -> str:
        elapsed = self._clock.now() - self._started
        return f"[{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}]"

    def say(self, message: str) -> None:
        print(f"{self._at()} {message}")

    def detail(self, message: str) -> None:
        if not self._quiet:
            print(f"{self._at()}     {message}")

    def _on_fault(self, _topic: str, event: FaultEvent) -> None:
        self.say(f"FAULT   {event.detector.value} on {event.subject}")
        for name, value in sorted(event.evidence.items()):
            self.detail(f"{name} = {value:.4g}")

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        self.say(f"MODE    {state.mode.value} -- {state.reason}")

    def _on_command(self, _topic: str, command: Command) -> None:
        self._commands += 1


class Demo:
    """The four services, a plant, and someone watching."""

    def __init__(self, config: Config, quiet: bool) -> None:
        self.clock = SimClock()
        self.config = config
        transport = LoopbackTransport()

        boards = [Blackboard(config.mqtt, transport) for _ in range(6)]
        plant, estimator, bank, control, operator, watcher = boards

        self.simulator = build_simulator(config, self.clock, plant)
        self.estimator = build_estimator(config, self.clock, estimator)
        self.bank = build_bank(config, self.clock, bank)
        self.control = build_control(config, self.clock, control)
        self.injector = FaultInjector(operator, self.clock)
        self.narrator = Narrator(watcher, self.clock, quiet)

        for service in (self.simulator, self.estimator, self.bank, self.control):
            service.subscribe()
        for board in boards:
            transport.attach(board)

    def run_for(self, seconds: float) -> None:
        period_s = self.config.loop.sensor_period_s
        elapsed = 0.0
        while elapsed < seconds:
            self.simulator.step()
            self.bank.tick()
            self.control.tick()
            self.clock.advance(period_s)
            elapsed += period_s

    def report_room(self, label: str) -> None:
        self.narrator.say(
            f"ROOM    {label}: {self.simulator.room_temperature_c:.2f} C, "
            f"setpoint {self.control.setpoint_c:.1f} C, "
            f"{self.narrator.commands} commands issued"
        )


def _stuck_sensor(demo: Demo) -> None:
    """The headline scenario: a sensor that lies, and a loop that survives it."""
    window_s = (
        demo.config.detectors.stuck_at.window_samples
        * demo.config.loop.sensor_period_s
    )

    demo.narrator.say("START   healthy room, identifying its own thermal model")
    demo.run_for(600.0)
    demo.report_room("after ten minutes")

    demo.narrator.say(
        f"INJECT  freezing {INDOOR_TEMPERATURE_ID} at {STUCK_VALUE_C} C "
        f"-- it keeps reporting, and the number is plausible"
    )
    demo.injector.inject(
        INDOOR_TEMPERATURE_ID, InjectedFault.STUCK_AT, STUCK_VALUE_C
    )

    before = demo.narrator.commands
    demo.run_for(window_s + 120.0)
    demo.narrator.say(
        f"STILL   controlling: {demo.narrator.commands - before} commands since "
        f"the sensor broke, on the model's prediction (FR-27)"
    )
    demo.report_room("while degraded")

    demo.narrator.say("WAIT    letting the prediction budget run out")
    demo.run_for(DEMO_BUDGET_S)
    demo.report_room("after the budget")

    demo.narrator.say("REPAIR  clearing the injection; the sensor is honest again")
    demo.injector.clear(INDOOR_TEMPERATURE_ID)
    demo.run_for(window_s + demo.config.mode.fault_clear_confirm_s + 120.0)
    demo.narrator.say(
        "NOTE    the hold is not self-clearing: "
        "python -m tools.reset --reason '...'"
    )


def _dead_actuator(demo: Demo) -> None:
    """The other detector a thermostat cannot have."""
    window_s = demo.config.detectors.actuator.evaluation_window_s
    demo.narrator.say(
        "START   the air conditioner is commanded but cools nothing. "
        "No acknowledgement exists to tell us (R-02)."
    )
    demo.run_for(window_s + 300.0)
    demo.report_room("after the evaluation window")
    demo.narrator.say(
        "NOTE    the fault came from the room not responding, "
        "not from a missing acknowledgement"
    )


SCENARIOS = {"stuck": _stuck_sensor, "actuator": _dead_actuator}


def _configure(config: Config, scenario: str) -> Config:
    """Shorten the budget, and for the actuator scenario break the plant.

    The budget is the only policy number changed, so the whole story fits in
    one run. Nothing about the sensors is softened: the noise, quantisation,
    jitter and dropout are the shipped ones.
    """
    mode = config.mode.model_copy(
        update={"degraded_sensor_budget_s": DEMO_BUDGET_S}
    )
    config = config.model_copy(update={"mode": mode})
    if scenario != "actuator":
        return config
    room = config.sim.room.model_copy(update={"cooling_power_w": 0.0})
    return config.model_copy(
        update={"sim": config.sim.model_copy(update={"room": room})}
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.demo",
        description="Run the whole system against a scripted fault, with no broker.",
    )
    parser.add_argument(
        "--scenario", choices=sorted(SCENARIOS), default="stuck"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--quiet", action="store_true", help="Transitions only, no evidence"
    )
    arguments = parser.parse_args(argv)

    # The services log at INFO; here their output would drown the narration,
    # which is the whole point of this script.
    logging.basicConfig(level=logging.ERROR, format="%(message)s")

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    demo = Demo(_configure(config, arguments.scenario), arguments.quiet)
    SCENARIOS[arguments.scenario](demo)
    print(
        f"\nDone. {demo.narrator.commands} commands issued, "
        f"final mode {demo.control.mode.value}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
