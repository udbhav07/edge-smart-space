"""Answer the room's questions about bookings, from a terminal (FR-54, FR-74).

    python -m tools.confirm             # watch, and ask about each one
    python -m tools.confirm --list      # watch a few seconds, list, exit

A ``commit`` tool -- a flight, a hotel -- is refused by the executor with
``CONFIRMATION_REQUIRED`` until an occupant agrees. This is where they agree,
until the local console of Week 8 exists: it keeps the invocations it has
seen on ``space/assist/proposed``, notices which came back needing
confirmation, asks, and on a yes republishes the invocation *verbatim* on
``space/assist/confirmed``.

Verbatim is the whole design. Confirmation is carried by the topic, never by
a field (section 5.7.6), and agreeing to a booking is agreeing to the one that
was described -- so nothing here may edit the arguments, and an invocation
past its ``expires_ts`` is still republished only to be refused by the
executor as EXPIRED, which is the gate that owns that decision.

A no publishes nothing. The booking was never made, and it expires.
"""

from __future__ import annotations

import argparse
import logging
import threading
from collections import OrderedDict
from pathlib import Path

from src.common import topics
from src.common.clock import Clock, RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.common.tools import ToolInvocation, ToolResult, ToolStatus

LOGGER = logging.getLogger("confirm")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "confirm-desk"

#: Invocations remembered while their results are awaited. Bounded: this
#: watches a busy topic and must not grow with it.
MAX_REMEMBERED = 32

#: How often the interactive loop looks for something new to ask about.
_POLL_S = 0.5

#: How long ``--list`` watches before printing.
_LIST_WINDOW_S = 5.0


class ConfirmationDesk:
    """Holds the bookings waiting for a person, and republishes the agreed ones."""

    def __init__(self, clock: Clock, blackboard: Blackboard) -> None:
        self._clock = clock
        self._blackboard = blackboard
        self._lock = threading.Lock()
        self._seen: OrderedDict[str, ToolInvocation] = OrderedDict()
        self._waiting: OrderedDict[str, ToolInvocation] = OrderedDict()
        self._unmatched: OrderedDict[str, ToolResult] = OrderedDict()

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.ASSIST_PROPOSED, ToolInvocation, self._on_proposed)
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)

    def _on_proposed(self, _topic: str, invocation: ToolInvocation) -> None:
        with self._lock:
            self._seen[invocation.invocation_id] = invocation
            _bound(self._seen)
            # The refusal can arrive first: two publishers, no ordering
            # promised between them. Match them up whichever way round.
            if self._unmatched.pop(invocation.invocation_id, None) is not None:
                self._await_answer(invocation)

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        if result.status is not ToolStatus.CONFIRMATION_REQUIRED:
            return
        with self._lock:
            invocation = self._seen.get(result.invocation_id)
            if invocation is None:
                self._unmatched[result.invocation_id] = result
                _bound(self._unmatched)
                return
            self._await_answer(invocation)

    def _await_answer(self, invocation: ToolInvocation) -> None:
        self._waiting[invocation.invocation_id] = invocation
        _bound(self._waiting)

    def waiting(self) -> list[ToolInvocation]:
        """Bookings an occupant has not yet answered, oldest first."""
        with self._lock:
            return list(self._waiting.values())

    def confirm(self, invocation_id: str) -> ToolInvocation:
        """Agree to one, exactly as it was described.

        :raises KeyError: if nothing by that id is waiting.
        """
        with self._lock:
            invocation = self._waiting.pop(invocation_id)
        self._blackboard.publish(topics.ASSIST_CONFIRMED, invocation)
        LOGGER.info("confirmed %s (%s)", invocation_id, invocation.tool)
        return invocation

    def decline(self, invocation_id: str) -> ToolInvocation:
        """Refuse one. Nothing is published; it was never made.

        :raises KeyError: if nothing by that id is waiting.
        """
        with self._lock:
            invocation = self._waiting.pop(invocation_id)
        LOGGER.info("declined %s (%s)", invocation_id, invocation.tool)
        return invocation


def _bound(table: OrderedDict) -> None:
    """Drop the oldest entries past the limit. Every table here is bounded."""
    while len(table) > MAX_REMEMBERED:
        table.popitem(last=False)


def describe(invocation: ToolInvocation) -> str:
    """One line a person can say yes or no to."""
    arguments = ", ".join(f"{name}={value}" for name, value in invocation.arguments.items())
    return f"{invocation.tool}({arguments}) -- asked as: {invocation.rationale!r}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Confirm or decline pending bookings.")
    parser.add_argument("--list", action="store_true", help="List what is waiting and exit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
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
    desk = ConfirmationDesk(clock, blackboard)
    desk.subscribe()
    blackboard.start()
    try:
        if arguments.list:
            # Nothing is retained on the assistance topics, so this lists what
            # is asked while it watches -- a short window, then out.
            clock.sleep(_LIST_WINDOW_S)
            for invocation in desk.waiting():
                print(describe(invocation))
            return 0
        print("watching for bookings that need your confirmation; Ctrl-C to stop")
        while True:
            for invocation in desk.waiting():
                print(f"\n{describe(invocation)}")
                answer = input("Confirm? This reaches a MOCK endpoint. [y/N] ")
                if answer.strip().lower() in ("y", "yes"):
                    desk.confirm(invocation.invocation_id)
                else:
                    desk.decline(invocation.invocation_id)
            clock.sleep(_POLL_S)
    except KeyboardInterrupt:
        return 0
    finally:
        blackboard.stop()


if __name__ == "__main__":
    raise SystemExit(main())
