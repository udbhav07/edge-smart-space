"""A confirmed booking, also in the calendar and in the occupant's inbox (FR-85).

Google has no booking API, so the reservation itself stays the mock it was,
labelled as one (FR-55). What the occupant gains is the rest of what a real
booking brings: an entry in their calendar for the day they travel, and,
with an account connected, an itinerary emailed to themselves -- queued if
the node is offline (FR-86).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

from src.assistance.google.account import GoogleAccount
from src.assistance.google.mail import send_or_queue
from src.assistance.google.outbox import Outbox
from src.assistance.providers.mock_travel import MockTravel
from src.common.tools import BOOK_TRAVEL, SCHEDULE_EVENT, ArgumentValue, ProviderOutcome, ToolProvider

LOGGER = logging.getLogger(__name__)

#: When a booking named only a day: a flight is put in the morning, a hotel's
#: check-in in the afternoon. Either is only where the entry sits.
_FLIGHT_AT = "09:00:00"
_CHECK_IN_AT = "14:00:00"


class TravelWithItinerary:
    """The mock booking, plus a calendar entry and an itinerary email."""

    def __init__(self, calendar: ToolProvider, account: GoogleAccount, outbox: Outbox, clock, flush=None, travel: MockTravel | None = None) -> None:
        self._travel = travel or MockTravel()
        self._calendar = calendar
        self._account = account
        self._outbox = outbox
        self._clock = clock
        self._flush = flush

    @property
    def name(self) -> str:
        return self._travel.name

    @property
    def simulated(self) -> bool:
        return True

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if tool != BOOK_TRAVEL.name:
            raise ValueError(f"{self.name} does not serve {tool!r}")
        booked = self._travel.invoke(tool, arguments)
        notes = [booked.message]
        what = _what(arguments)
        try:
            self._calendar.invoke(SCHEDULE_EVENT.name, {"subject": what, "starts_at": _when(arguments), "duration_min": 120})
            notes.append("It is in your calendar.")
        except ValueError as exc:
            LOGGER.warning("booked, but not put in the calendar: %s", exc)
        if self._account.connected and self._account.email:
            emailed = send_or_queue(self._outbox, self._clock, self._flush, self._account.email, f"Itinerary: {what}", _itinerary(booked, arguments, what))
            notes.append("The itinerary is queued for your inbox." if emailed.detail.get("queued") else "The itinerary is in your inbox.")
        return ProviderOutcome(message=" ".join(notes), detail=dict(booked.detail))


def _what(arguments: Mapping[str, ArgumentValue]) -> str:
    destination = str(arguments.get("destination", "")).strip()
    if arguments.get("kind") == "hotel":
        return f"Hotel in {destination}"
    origin = str(arguments.get("origin") or "").strip()
    return f"Flight {'from ' + origin + ' ' if origin else ''}to {destination}"


def _when(arguments: Mapping[str, ArgumentValue]) -> str:
    value = str(arguments.get("depart_on", ""))
    if "T" in value:
        return value
    return f"{value}T{_CHECK_IN_AT if arguments.get('kind') == 'hotel' else _FLIGHT_AT}"


def _itinerary(booked: ProviderOutcome, arguments: Mapping[str, ArgumentValue], what: str) -> str:
    day = str(arguments.get("depart_on", ""))[:10]
    try:
        day = datetime.fromisoformat(day).strftime("%A %d %B %Y")
    except ValueError:
        pass
    lines = [
        what,
        f"Date: {day}",
        f"Reference: {booked.detail.get('reference', '')}",
    ]
    if arguments.get("nights"):
        lines.append(f"Nights: {arguments['nights']}")
    lines += ["", booked.message, "", "Sent by your smart space."]
    return "\n".join(lines)
