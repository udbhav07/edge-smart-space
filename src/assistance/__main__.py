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
from dataclasses import dataclass
from pathlib import Path

from src.assistance.executor import AssistanceExecutor
from src.assistance.google.account import GoogleAccount
from src.assistance.google.api import RealGoogleApi
from src.assistance.google.calendar import SyncedCalendar
from src.assistance.google.mail import GmailOrMock
from src.assistance.google.outbox import Outbox
from src.assistance.google.sync import GoogleSync
from src.assistance.google.travel import TravelWithItinerary
from src.assistance.memory import ProfileStore
from src.assistance.profiles import ProfileKeeper
from src.assistance.providers.local_calendar import LocalCalendar
from src.assistance.providers.mock_mail import MockMail
from src.assistance.providers.mock_travel import MockTravel
from src.assistance.providers.profile_memory import ProfileMemory
from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import Config, ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.tools import (
    BOOK_TRAVEL,
    SEND_MAIL,
    CANCEL_EVENT,
    FORGET,
    GET_EVENTS,
    KEEP_EVENT,
    MOVE_EVENT,
    PENCIL_EVENT,
    REMEMBER,
    SCHEDULE_EVENT,
    SETTLE_FACT,
    SUGGEST_FACT,
    ToolRegistry,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "assistance-executor"

#: Callback-driven; the main thread only has to stay alive and interruptible.
_IDLE_INTERVAL_S = 1.0


@dataclass
class GoogleParts:
    """What reaches Google, when the service runs for real (FR-84 to FR-86)."""

    account: GoogleAccount
    outbox: Outbox
    #: The one calendar both the tools and the sync work on: two instances of
    #: one file would each overwrite what the other wrote.
    calendar: LocalCalendar
    sync: GoogleSync | None = None


def build_registry(config: Config, clock: Clock, google: GoogleParts | None = None) -> ToolRegistry:
    """Declare the surface and bind what fulfils it (FR-71, FR-72).

    With Google parts, the calendar is mirrored to Google Calendar, mail goes
    through Gmail once an account is connected, and a booking adds its entry
    and itinerary. Without them -- every test, and a node nobody signed in
    on -- it is the local calendar and the mocks, exactly as before.
    """
    registry = ToolRegistry(clock)
    local = google.calendar if google is not None else LocalCalendar(
        Path(config.assistance.calendar_path), config.assistance.calendar_max_events
    )
    calendar = local
    travel = MockTravel()
    mail = MockMail()
    if google is not None:
        poke = google.sync.poke if google.sync else None
        flush = google.sync.flush if google.sync else None
        lock = google.sync.lock if google.sync else None
        calendar = SyncedCalendar(local, google.outbox, google.account, clock, config.site.utc_offset_h, on_queued=poke, lock=lock)
        travel = TravelWithItinerary(calendar, google.account, google.outbox, clock, flush=flush)
        mail = GmailOrMock(google.account, google.outbox, clock, flush=flush)
    registry.bind(SCHEDULE_EVENT.name, calendar)
    registry.bind(GET_EVENTS.name, calendar)
    registry.bind(CANCEL_EVENT.name, calendar)
    registry.bind(MOVE_EVENT.name, calendar)
    registry.bind(PENCIL_EVENT.name, calendar)
    registry.bind(KEEP_EVENT.name, calendar)
    registry.bind(BOOK_TRAVEL.name, travel)
    registry.bind(SEND_MAIL.name, mail)
    memory = ProfileMemory(build_store(config, clock))
    registry.bind(REMEMBER.name, memory)
    registry.bind(FORGET.name, memory)
    registry.bind(SUGGEST_FACT.name, memory)
    registry.bind(SETTLE_FACT.name, memory)
    return registry


def build_store(config: Config, clock: Clock) -> ProfileStore:
    """The files that hold what the room remembers about each person (FR-52)."""
    return ProfileStore(config.profiles, clock, config.site.utc_offset_h)


def build_keeper(config: Config, clock: Clock, blackboard: Blackboard) -> ProfileKeeper:
    """What publishes each person's context, beside the executor that writes it."""
    return ProfileKeeper(config.profiles, clock, blackboard, build_store(config, clock))


def build_service(
    config: Config, clock: Clock, blackboard: Blackboard, google: GoogleParts | None = None
) -> AssistanceExecutor:
    return AssistanceExecutor(clock, blackboard, build_registry(config, clock, google))


def build_google(config: Config, clock: Clock, blackboard: Blackboard) -> GoogleParts:
    """The account, the outbox and the sync, publishing its status (FR-84)."""
    account = GoogleAccount(config.google)
    outbox = Outbox(Path(config.google.outbox_path), config.google.outbox_max, config.google.retry_min_s, config.google.retry_max_s)
    calendar = LocalCalendar(Path(config.assistance.calendar_path), config.assistance.calendar_max_events)
    api = RealGoogleApi(account.credentials)
    sync = GoogleSync(config.google, clock, account, api, outbox, calendar, config.site.utc_offset_h, publish=blackboard.publish)
    return GoogleParts(account=account, outbox=outbox, calendar=calendar, sync=sync)


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

    google = build_google(config, clock, blackboard)
    service = build_service(config, clock, blackboard, google)
    service.subscribe()
    keeper = build_keeper(config, clock, blackboard)
    keeper.subscribe()
    blackboard.start()
    google.sync.start()
    service.publish_catalogue()
    LOGGER.info(
        "executing tool invocations from %s; results on %s",
        topics.ASSIST_PROPOSED.pattern,
        topics.ASSIST_RESULT.pattern,
    )
    try:
        while True:
            keeper.tick()
            clock.sleep(_IDLE_INTERVAL_S)
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
