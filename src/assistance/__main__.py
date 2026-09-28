"""Run the assistance executor as a service.

``python -m src.assistance``

Binds the providers to the declared tools and executes invocations published
on the blackboard (DESIGN.md section 5.7.6). This is the only process that
holds a provider, so it is the only place a provider swap happens (FR-72): to
move the calendar somewhere else, change the binding in :func:`build_registry`
and nothing else.

Killing it costs the occupant their calendar and bookings until it is back.
It costs the room nothing: no tool reaches the plant (FR-45).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.assistance.executor import AssistanceExecutor
from src.assistance.providers.local_calendar import LocalCalendar
from src.assistance.providers.mock_travel import MockTravel
from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.tools import BOOK_TRAVEL, GET_EVENTS, SCHEDULE_EVENT, ToolRegistry

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "assistance-executor"

#: Callback-driven; the main thread only has to stay alive and interruptible.
_IDLE_INTERVAL_S = 1.0


def build_registry(config: Config, clock: Clock) -> ToolRegistry:
    """Declare the surface and bind what fulfils it (FR-71, FR-72)."""
    registry = ToolRegistry(clock)
    calendar = LocalCalendar(
        Path(config.assistance.calendar_path), config.assistance.calendar_max_events
    )
    registry.bind(SCHEDULE_EVENT.name, calendar)
    registry.bind(GET_EVENTS.name, calendar)
    registry.bind(BOOK_TRAVEL.name, MockTravel())
    return registry


def build_service(
    config: Config, clock: Clock, blackboard: Blackboard
) -> AssistanceExecutor:
    return AssistanceExecutor(clock, blackboard, build_registry(config, clock))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the assistance executor.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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
        "executing tool invocations from %s; results on %s",
        topics.ASSIST_PROPOSED.pattern,
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
