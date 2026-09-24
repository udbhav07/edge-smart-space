"""Flights and hotels, which this system does not actually book (FR-55).

A mock, and it says so in every result it produces. That is not a disclaimer
bolted on -- ``simulated`` is read off the provider by the registry, so a
result cannot be published without carrying it, and identifying a mock never
depends on anyone remembering to mention it.

**Why a mock at all.** Booking travel is the one action in the declared
surface whose consequences reach outside the room and cannot be undone by the
system that took them: a calendar entry can be deleted, a flight is somebody
else's money. Section 5.7.6 draws the line exactly there, and ``book_travel``
is the only tool marked COMMIT. Wiring a real booking endpoint behind this
would change nothing above the provider, which is the point of the provider
boundary -- and is precisely why the confirmation gate has to be in front of
it rather than inside it.

**It still refuses impossible requests.** A mock that accepted anything would
make the confirmation flow look like it worked while proving nothing about
whether the arguments a model produced were usable. E6 measures argument
plausibility separately from schema validity for the same reason (section
5.7.4).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

from src.common.clock import Clock
from src.common.localtime import local_time
from src.common.tools import ArgumentValue, ProviderOutcome

LOGGER = logging.getLogger(__name__)

#: What the mock claims to be able to book.
FLIGHT = "flight"
HOTEL = "hotel"

#: Longest stay it will pretend to book. Not a rule about hotels; a bound on
#: a number a model produced, so an obvious mistake is refused rather than
#: confirmed by somebody skim-reading.
MAX_NIGHTS = 30

#: Reference this mock returns. Deliberately unmistakable: a reference that
#: looked like an airline's would end up in somebody's email.
_REFERENCE_PREFIX = "MOCK"


class TravelRequestError(ValueError):
    """The request could not be made sense of."""


class MockTravel:
    """Pretends to book travel, and never pretends it did."""

    def __init__(self, clock: Clock, utc_offset_h: float | None = None) -> None:
        self._clock = clock
        self._utc_offset_h = utc_offset_h
        self._bookings = 0

    @property
    def name(self) -> str:
        return "mock_travel"

    @property
    def simulated(self) -> bool:
        """True, always. FR-55 exists so that this can never be forgotten."""
        return True

    @property
    def bookings(self) -> int:
        """How many times it has been asked to book, for the audit trail."""
        return self._bookings

    def invoke(
        self, tool: str, arguments: Mapping[str, ArgumentValue]
    ) -> ProviderOutcome:
        if tool != "book_travel":
            raise TravelRequestError(f"{self.name} does not serve {tool!r}")

        kind = str(arguments["kind"])
        destination = str(arguments["destination"])
        depart_on = self._as_datetime(arguments["depart_on"])

        if depart_on < self._as_now():
            raise TravelRequestError(
                f"{depart_on.date()} is in the past"
            )

        if kind == HOTEL:
            return self._hotel(destination, depart_on, arguments)
        return self._flight(destination, depart_on, arguments)

    def _flight(
        self,
        destination: str,
        depart_on: datetime,
        arguments: Mapping[str, ArgumentValue],
    ) -> ProviderOutcome:
        origin = str(arguments.get("origin") or "").strip()
        if not origin:
            # Refused rather than guessed. A flight from the wrong airport is
            # a worse outcome than a question, and inventing one here would
            # hide a gap in what the model actually extracted.
            raise TravelRequestError(
                "a flight needs somewhere to leave from, and none was given"
            )
        reference = self._reference()
        return ProviderOutcome(
            message=(
                f"Simulated only: no flight was booked. A flight from {origin} "
                f"to {destination} on {depart_on.date()} would be "
                f"reference {reference}."
            ),
            detail={
                "reference": reference,
                "kind": FLIGHT,
                "origin": origin,
                "destination": destination,
                "depart_on": depart_on.isoformat(),
            },
        )

    def _hotel(
        self,
        destination: str,
        depart_on: datetime,
        arguments: Mapping[str, ArgumentValue],
    ) -> ProviderOutcome:
        # Not ``or 1``: that idiom reads zero as absent, so a request for no
        # nights would be silently turned into a booking for one.
        supplied = arguments.get("nights")
        nights = 1 if supplied is None else int(supplied)
        if nights < 1:
            raise TravelRequestError(f"{nights} nights is not a stay")
        if nights > MAX_NIGHTS:
            raise TravelRequestError(
                f"{nights} nights is longer than this will pretend to book"
            )
        reference = self._reference()
        return ProviderOutcome(
            message=(
                f"Simulated only: nothing was booked. {nights} night"
                f"{'s' if nights > 1 else ''} in {destination} from "
                f"{depart_on.date()} would be reference {reference}."
            ),
            detail={
                "reference": reference,
                "kind": HOTEL,
                "destination": destination,
                "depart_on": depart_on.isoformat(),
                "nights": nights,
            },
        )

    def _reference(self) -> str:
        self._bookings += 1
        return f"{_REFERENCE_PREFIX}-{int(self._clock.now())}-{self._bookings:03d}"

    def _as_now(self) -> datetime:
        """Now, in the frame the model wrote ``depart_on`` in.

        The site offset when one is given, so a node provisioned in UTC does
        not call a flight tomorrow morning "in the past" (localtime.py).
        """
        if self._utc_offset_h is None:
            return datetime.fromtimestamp(self._clock.now())
        return local_time(self._clock.now(), self._utc_offset_h)

    @staticmethod
    def _as_datetime(value: object) -> datetime:
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError) as exc:
            raise TravelRequestError(f"{value!r} is not a date") from exc
