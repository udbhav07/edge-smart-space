"""Run the reasoning layer as a service.

``python -m src.reasoning``

Two call sites in one process (section 5.7.1): the Environmental Supervisor,
on its cadence and on events (FR-40, FR-41), and the assistant, which acts on
what the speech pipeline heard and says what happened (FR-42, FR-53). Each has
its own blackboard client -- they share nothing but the model server, which a
single instance serves for every call site (section 5.7.5).

This is the process the demonstration kills. Nothing here can stop the room
being controlled: the regulatory loop holds the last validated setpoint and
keeps its cadence without it (FR-11, FR-47).
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.reasoning.assistant import Assistant
from src.reasoning.chat import ChatClient
from src.reasoning.diagnosis import FaultDiagnoser
from src.reasoning.supervisor_agent import SupervisorAgent
from src.reasoning.supervisor_tools import SupervisorState

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
ASSISTANT_CLIENT_ID = "reasoning-assistant"
SUPERVISOR_CLIENT_ID = "reasoning-supervisor"
DIAGNOSIS_CLIENT_ID = "reasoning-diagnosis"

#: How often the main loop looks for work. Short, so a spoken request is
#: answered promptly; the supervisor's own cadence is far longer.
_IDLE_INTERVAL_S = 0.2


def build_assistant(
    config: Config, clock: Clock, blackboard: Blackboard, chat: ChatClient
) -> Assistant:
    return Assistant(
        config.assistance,
        clock,
        blackboard,
        chat,
        safe_range_c=config.validator.setpoint_bounds_c,
    )


def build_supervisor(
    config: Config, clock: Clock, blackboard: Blackboard, chat: ChatClient
) -> SupervisorAgent:
    return SupervisorAgent(
        config.supervisor, clock, blackboard, chat, SupervisorState(config, clock)
    )


def run(assistant: Assistant, clock: Clock, iterations: int | None = None) -> None:
    """Answer utterances as they arrive."""
    completed = 0
    while iterations is None or completed < iterations:
        assistant.process_pending()
        clock.sleep(_IDLE_INTERVAL_S)
        completed += 1


def supervise(
    supervisor: SupervisorAgent,
    clock: Clock,
    stop: threading.Event,
    iterations: int | None = None,
    diagnoser: FaultDiagnoser | None = None,
) -> None:
    """Run diagnoses and supervisory cycles, on a thread of their own.

    Separate from the assistant because a cycle is several completions long,
    and a person asking for something must not wait behind one. Diagnosis
    runs here too: it explains a fault after the mode has already changed,
    so it is never on anyone's critical path (FR-26).
    """
    completed = 0
    while not stop.is_set() and (iterations is None or completed < iterations):
        try:
            if diagnoser is not None:
                diagnoser.process_pending()
            supervisor.maybe_run()
        except Exception:
            # A supervisor bug costs supervisory cycles, never the assistant.
            LOGGER.exception("supervisory cycle failed; previous goal retained")
        clock.sleep(_IDLE_INTERVAL_S)
        completed += 1


def _board(config: Config, client_id: str) -> Blackboard:
    holder: list = []
    transport = build_transport(config.mqtt, client_id, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)
    return blackboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reasoning layer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--no-supervisor",
        action="store_true",
        help="Answer utterances only; run no supervisory cycles.",
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    clock = RealClock()
    chat = ChatClient(config.reasoning, clock)
    assistant_board = _board(config, ASSISTANT_CLIENT_ID)
    supervisor_board = _board(config, SUPERVISOR_CLIENT_ID)
    diagnosis_board = _board(config, DIAGNOSIS_CLIENT_ID)
    diagnoser = FaultDiagnoser(clock, diagnosis_board, chat)
    diagnoser.subscribe()
    assistant = build_assistant(config, clock, assistant_board, chat)
    supervisor = build_supervisor(config, clock, supervisor_board, chat)
    assistant.subscribe()
    if not arguments.no_supervisor:
        supervisor.subscribe()
    assistant_board.start()
    supervisor_board.start()
    diagnosis_board.start()
    LOGGER.info(
        "reasoning on %s with %s; supervisor every %.0f s%s",
        config.reasoning.base_url,
        config.reasoning.model,
        config.supervisor.period_s,
        " (disabled)" if arguments.no_supervisor else "",
    )
    stop = threading.Event()
    supervising = threading.Thread(
        target=supervise,
        args=(supervisor, clock, stop),
        kwargs={"diagnoser": diagnoser},
        name="supervisor",
        daemon=True,
    )
    supervising.start()
    try:
        run(assistant, clock)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        stop.set()
        assistant_board.stop()
        supervisor_board.stop()
        diagnosis_board.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
