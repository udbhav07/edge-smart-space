"""Flights and hotels, against a mock endpoint (FR-55).

No real reservation is ever made, and every sentence this produces says so.
``simulated`` is True, which the registry copies onto every result, so the
mock is identifiable on the blackboard without relying on anyone reading the
message text (FR-55). A booking reaches this only after the occupant has
confirmed it (FR-54, FR-74); that gate is the registry's, not this class's.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import datetime

from src.assistance.providers.local_calendar import describe_day
from src.common.tools import BOOK_TRAVEL, ArgumentValue, ProviderOutcome

LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "mock_travel"


class MockTravel:
    """Records a booking and makes none."""

    def __init__(self) -> None:
        self._next_reference = 1

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def simulated(self) -> bool:
        return True

    def invoke(
        self, tool: str, arguments: Mapping[str, ArgumentValue]
    ) -> ProviderOutcome:
        """Pretend to book, and say plainly that it was pretend.

        :raises ValueError: for a tool this provider does not serve.
        """
        if tool != BOOK_TRAVEL.name:
            raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")
        reference = f"MOCK-{self._next_reference:04d}"
        self._next_reference += 1
        departs = datetime.fromisoformat(str(arguments["depart_on"])).replace(
            tzinfo=None
        )
        kind = str(arguments["kind"])
        destination = str(arguments["destination"])
        origin = str(arguments.get("origin") or "").strip()
        if kind == "flight":
            what = f"flight {'from ' + origin + ' ' if origin else ''}to {destination}"
        else:
            what = f"hotel in {destination}"
            nights = arguments.get("nights")
            if nights:
                what += f" for {nights} night{'s' if int(nights) != 1 else ''}"
        LOGGER.info("MOCK booking %s: %s on %s", reference, what, departs.date())
        return ProviderOutcome(
            message=(
                f"Mock booking {reference} recorded for a {what} on "
                f"{describe_day(departs)}. This is a mock: no real "
                f"reservation was made."
            ),
            detail={"reference": reference, "kind": kind, "destination": destination},
        )
