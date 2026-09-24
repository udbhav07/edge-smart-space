"""Run the assistance executor as a service.

``python -m src.assistance``

Binds the providers to the declared tool surface, announces the catalogue, and
runs whatever the reasoning layer proposes -- subject to FR-74's confirmation
gate for anything that commits the occupant to an outside party.

It is a separate process for the usual reason: killing it costs assistance and
nothing else. The regulatory loop does not know it exists.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.tools import (
    BOOK_TRAVEL,
    GET_EVENTS,
    SCHEDULE_EVENT,
    ToolRegistry,
)
from src.assistance.executor import AssistanceExecutor
from src.assistance.providers.local_calendar import LocalCalendar
from src.assistance.providers.mock_travel import MockTravel

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "assistance-executor"

#: The service is callback-driven; the main thread only stays interruptible.
_IDLE_INTERVAL_S = 1.0


def build_registry(config: Config, clock: Clock) -> ToolRegistry:
    """Bind an implementation behind each declared tool.

    The surface is declared by the registry itself, with the contract, and
    this only says which implementation happens to be installed. They answer
    different questions: the declaration is what the reasoning layer is shown
    and what the catalogue publishes, the binding is what runs. Moving from
    this calendar to a hosted one changes the second and leaves the first
    untouched, which is the whole point of section 5.7.6.
    """
    registry = ToolRegistry(clock=clock)
    calendar = LocalCalendar(
        path=Path(config.assistance.calendar_path), clock=clock
    )
    registry.bind(SCHEDULE_EVENT.name, calendar)
    registry.bind(GET_EVENTS.name, calendar)
    registry.bind(BOOK_TRAVEL.name, MockTravel(clock=clock))
    return registry


def build_service(
    config: Config, clock: Clock, blackboard: Blackboard
) -> AssistanceExecutor:
    return AssistanceExecutor(
        config=config,
        clock=clock,
        blackboard=blackboard,
        registry=build_registry(config, clock),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the assistance executor.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    clock = RealClock()
    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    holder.append(blackboard)

    service = build_service(config, clock, blackboard)
    service.subscribe()
    blackboard.start()
    service.publish_catalogue()
    LOGGER.info(
        "assisting; watch outcomes with: mosquitto_sub -t '%s' -v",
        topics.ASSIST_RESULT.pattern,
    )
    try:
        while True:
            clock.sleep(_IDLE_INTERVAL_S)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
