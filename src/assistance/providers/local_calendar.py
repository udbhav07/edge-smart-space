"""The occupant's calendar, kept on this node (FR-58).

A :class:`~src.common.tools.ToolProvider` for ``schedule_event`` and
``get_events``. It is the provider that ships because a hosted calendar needs
the outbound connection NFR-06 forbids (DESIGN.md section 2.2); the tool
surface admits one, and nothing above this module would change if one were
bound instead (FR-71, FR-72).

The calendar is a JSON file, written atomically the way the estimator's
coefficients are: a crash mid-write must leave the previous calendar, not
half of a new one. It holds no clock. The provider contract gives it none,
and it needs none -- every time it handles was written by the model as an
ISO-8601 local date and time, and is stored and compared as one.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from src.common.tools import (
    DEFAULT_EVENT_DURATION_MIN,
    GET_EVENTS,
    SCHEDULE_EVENT,
    ArgumentValue,
    ProviderOutcome,
)

LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "local_calendar"

SCHEMA_VERSION = 1
_ENCODING = "utf-8"
_TEMPORARY_SUFFIX = ".tmp"

#: How many entries a reply lists by name before summarising the rest. A
#: spoken answer that reads out twenty meetings is not an answer.
_ENTRIES_NAMED_IN_A_REPLY = 5


class CalendarFullError(RuntimeError):
    """The configured bound on entries has been reached."""


def describe_time(moment: datetime) -> str:
    """A time the way a person says it: ``15:00 on Thursday 1 October``.

    Built by hand rather than with ``%-d``, which Windows' strftime rejects.
    """
    return f"{moment:%H:%M} on {moment:%A} {moment.day} {moment:%B}"


class LocalCalendar:
    """Calendar entries in a local file."""

    def __init__(self, path: Path, max_events: int) -> None:
        if max_events <= 0:
            raise ValueError(f"max_events must be positive, got {max_events!r}")
        self._path = path
        self._max_events = max_events
        self._events, self._next_id = self._load()

    # --- ToolProvider -------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def simulated(self) -> bool:
        """A real calendar: the entry exists afterwards (FR-55 is for mocks)."""
        return False

    def invoke(
        self, tool: str, arguments: Mapping[str, ArgumentValue]
    ) -> ProviderOutcome:
        """Perform one calendar tool.

        :raises ValueError: for a tool this provider does not serve, or a
            window that ends before it starts.
        :raises CalendarFullError: when the entry bound is reached.
        :raises OSError: when the calendar could not be written.
        """
        if tool == SCHEDULE_EVENT.name:
            return self._schedule(arguments)
        if tool == GET_EVENTS.name:
            return self._list(arguments)
        raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")

    # --- entries ------------------------------------------------------

    @property
    def events(self) -> tuple[dict[str, str | int], ...]:
        """Every entry, earliest first. A copy; the file is the record."""
        return tuple(dict(event) for event in self._sorted())

    def remove(self, event_id: str) -> bool:
        """Delete one entry (FR-58: an unwanted entry is deleted, not undone).

        :returns: whether anything was removed.
        """
        kept = [event for event in self._events if event["event_id"] != event_id]
        if len(kept) == len(self._events):
            return False
        self._events = kept
        self._save()
        return True

    def _schedule(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if len(self._events) >= self._max_events:
            raise CalendarFullError(
                f"the calendar already holds {self._max_events} entries"
            )
        starts = _parse(arguments["starts_at"])
        duration_min = int(arguments.get("duration_min") or DEFAULT_EVENT_DURATION_MIN)
        subject = str(arguments["subject"]).strip()
        event = {
            "event_id": f"ev_{self._next_id:04d}",
            "subject": subject,
            "starts_at": starts.isoformat(timespec="minutes"),
            "ends_at": (starts + timedelta(minutes=duration_min)).isoformat(
                timespec="minutes"
            ),
            "duration_min": duration_min,
        }
        self._events.append(event)
        self._next_id += 1
        self._save()
        LOGGER.info("added %s: %s at %s", event["event_id"], subject, event["starts_at"])
        return ProviderOutcome(
            message=f"Added {subject} at {describe_time(starts)}.",
            detail={
                "event_id": event["event_id"],
                "starts_at": event["starts_at"],
                "duration_min": duration_min,
            },
        )

    def _list(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        window_start = _parse(arguments["from_time"])
        window_end = _parse(arguments["to_time"])
        if window_end < window_start:
            raise ValueError("the window ends before it starts")
        found = [
            event
            for event in self._sorted()
            if _parse(event["starts_at"]) < window_end
            and _parse(event["ends_at"]) > window_start
        ]
        return ProviderOutcome(
            message=_summarise(found),
            detail={"count": len(found), "entries": _listing(found)},
        )

    def _sorted(self) -> list[dict[str, str | int]]:
        return sorted(self._events, key=lambda event: str(event["starts_at"]))

    # --- storage ------------------------------------------------------

    def _load(self) -> tuple[list[dict[str, str | int]], int]:
        """Read the calendar back. Missing is empty; unreadable is refused.

        An unreadable calendar is not silently replaced with an empty one:
        the next write would destroy every entry the occupant had.
        """
        try:
            raw = json.loads(self._path.read_text(encoding=_ENCODING))
        except FileNotFoundError:
            LOGGER.info("no calendar at %s yet; starting empty", self._path)
            return [], 1
        if not isinstance(raw, dict) or raw.get("version") != SCHEMA_VERSION:
            raise ValueError(f"calendar at {self._path} is not version {SCHEMA_VERSION}")
        events = [dict(event) for event in raw.get("events", [])]
        return events, int(raw.get("next_id", len(events) + 1))

    def _save(self) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "next_id": self._next_id,
            "events": self._events,
        }
        temporary = self._path.with_suffix(self._path.suffix + _TEMPORARY_SUFFIX)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, indent=2), encoding=_ENCODING)
        os.replace(temporary, self._path)


def _parse(value: ArgumentValue | str | int) -> datetime:
    """An ISO-8601 local date and time, as the tool declaration asks for.

    A trailing zone is dropped rather than converted: every time here is the
    occupant's local wall-clock time, and mixing aware and naive values would
    make comparisons raise.
    """
    moment = datetime.fromisoformat(str(value))
    return moment.replace(tzinfo=None)


def _summarise(found: list[dict[str, str | int]]) -> str:
    if not found:
        return "Nothing is in the calendar then."
    named = [
        f"{event['subject']} at {describe_time(_parse(event['starts_at']))}"
        for event in found[:_ENTRIES_NAMED_IN_A_REPLY]
    ]
    remainder = len(found) - len(named)
    listing = "; ".join(named)
    if remainder:
        listing += f"; and {remainder} more"
    count = "1 entry" if len(found) == 1 else f"{len(found)} entries"
    return f"{count}: {listing}."


def _listing(found: list[dict[str, str | int]]) -> str:
    """Flat text for the result detail, which carries no nested values."""
    return "; ".join(
        f"{event['event_id']} {event['starts_at']} {event['subject']}" for event in found
    )
