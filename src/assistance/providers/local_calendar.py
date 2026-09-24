"""The calendar that ships with the system (FR-58).

First-party on purpose. A hosted calendar would put an OAuth credential on a
device sitting in a room, and NFR-06 keeps the node self-contained: nothing
here reaches the network, so there is nothing to leak and nothing to expire at
the worst moment. Section 5.7.6 makes moving to a hosted calendar later a
matter of writing another provider, because the reasoning layer never learns
which one is bound.

**It is a real calendar, not a mock.** Entries persist, they are read back,
and ``simulated`` is False -- which matters because FR-55 requires a mock to
say so, and a provider that lied either way would make that flag worthless.
The travel provider is the mock, and it says so.

**Storage is one JSON file, written whole.** A calendar for one room holds
tens of entries, so an index would be complexity nobody pays for, and writing
the file whole means a crash leaves either the old file or the new one rather
than half of each. The file is read on every call rather than cached, because
another process -- the console, an examiner with an editor -- may have changed
it, and a cache would quietly serve an entry somebody had deleted.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path

from src.common.clock import Clock
from src.common.tools import (
    DEFAULT_EVENT_DURATION_MIN,
    ArgumentValue,
    ProviderOutcome,
)

LOGGER = logging.getLogger(__name__)

#: Fields of a stored entry.
_ID = "event_id"
_STARTS_AT = "starts_at"
_ENDS_AT = "ends_at"
_SUBJECT = "subject"

#: How an entry is described back to an occupant. Day and time without the
#: year: somebody asking about Thursday is not helped by "2026".
_SPOKEN_FORMAT = "%H:%M on %-d %B"
_SPOKEN_FORMAT_WINDOWS = "%H:%M on %#d %B"

#: Most entries anyone will ask to see at once. A read that returned four
#: hundred events would be answered by a model that then said "several".
MAX_EVENTS_RETURNED = 50


class CalendarError(RuntimeError):
    """The calendar could not be read or written."""


def _spoken(moment: datetime) -> str:
    """Format a time the way it would be said aloud."""
    try:
        return moment.strftime(_SPOKEN_FORMAT)
    except ValueError:
        # Windows rejects the dash modifier rather than ignoring it.
        return moment.strftime(_SPOKEN_FORMAT_WINDOWS)


class LocalCalendar:
    """A calendar kept in a file on this machine."""

    def __init__(self, path: Path, clock: Clock) -> None:
        self._path = path
        self._clock = clock

    @property
    def name(self) -> str:
        return "local_calendar"

    @property
    def simulated(self) -> bool:
        """False. The entries are real and they are still there tomorrow."""
        return False

    @property
    def path(self) -> Path:
        return self._path

    # --- storage ------------------------------------------------------

    def _load(self) -> list[dict[str, object]]:
        """Read the file, treating absence as an empty calendar.

        A missing file is the ordinary state before the first entry, not an
        error. A *corrupt* file is an error and is raised: silently starting
        again from empty would discard somebody's appointments and report
        success.
        """
        if not self._path.is_file():
            return []
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise CalendarError(f"{self._path} is not readable: {exc}") from exc
        if not isinstance(parsed, list):
            raise CalendarError(f"{self._path} does not hold a list of entries")
        return parsed

    def _save(self, entries: list[dict[str, object]]) -> None:
        """Write the file whole, atomically.

        Through a temporary file and a rename: a crash mid-write then leaves
        the old calendar rather than half of a new one, and a half-written
        JSON file is a calendar nobody can read again.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(self._path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(entries, stream, indent=2, sort_keys=True)
            os.replace(temporary, self._path)
        except OSError as exc:
            Path(temporary).unlink(missing_ok=True)
            raise CalendarError(f"could not write {self._path}: {exc}") from exc

    # --- the tools ----------------------------------------------------

    def invoke(
        self, tool: str, arguments: Mapping[str, ArgumentValue]
    ) -> ProviderOutcome:
        """Perform one of the calendar's tools.

        The registry has already validated the arguments against the declared
        specification, so what arrives here is the right shape; what it cannot
        check is whether the calendar can be written, which is why this may
        still fail.
        """
        if tool == "schedule_event":
            return self._schedule(arguments)
        if tool == "get_events":
            return self._read(arguments)
        raise CalendarError(f"{self.name} does not serve {tool!r}")

    def _schedule(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        starts_at = self._as_datetime(arguments["starts_at"])
        # Not ``or DEFAULT``: that reads a zero-minute entry as an absent
        # one, so an explicit request for a marker with no duration would
        # silently become an hour.
        supplied = arguments.get("duration_min")
        duration = (
            DEFAULT_EVENT_DURATION_MIN if supplied is None else int(supplied)
        )
        subject = str(arguments["subject"])
        ends_at = starts_at + timedelta(minutes=duration)

        entries = self._load()
        event_id = f"ev_{len(entries) + 1:04d}"
        entries.append(
            {
                _ID: event_id,
                _STARTS_AT: starts_at.isoformat(),
                _ENDS_AT: ends_at.isoformat(),
                _SUBJECT: subject,
            }
        )
        self._save(entries)

        clash = self._overlapping(entries, starts_at, ends_at, event_id)
        message = f"Added {subject} at {_spoken(starts_at)}."
        if clash:
            # Reported rather than refused. It is the occupant's calendar and
            # double-booking it is their business; saying nothing would be the
            # system quietly deciding it knew better.
            message += f" It overlaps {clash}."
        return ProviderOutcome(
            message=message,
            detail={_ID: event_id, _STARTS_AT: starts_at.isoformat()},
        )

    def _read(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        from_time = self._as_datetime(arguments["from_time"])
        to_time = self._as_datetime(arguments["to_time"])
        if to_time < from_time:
            raise CalendarError("the window ends before it starts")

        found = [
            entry
            for entry in self._load()
            if from_time <= self._as_datetime(entry[_STARTS_AT]) <= to_time
        ]
        found.sort(key=lambda entry: entry[_STARTS_AT])
        shown = found[:MAX_EVENTS_RETURNED]

        if not shown:
            return ProviderOutcome(
                message="Nothing in the calendar for that.", detail={"count": 0}
            )

        described = "; ".join(
            f"{entry[_SUBJECT]} at {_spoken(self._as_datetime(entry[_STARTS_AT]))}"
            for entry in shown
        )
        message = f"{len(found)} in that window: {described}."
        if len(found) > len(shown):
            message = (
                f"{len(found)} in that window, the first {len(shown)}: "
                f"{described}."
            )
        return ProviderOutcome(
            message=message,
            detail={"count": len(found), "returned": len(shown)},
        )

    def _overlapping(
        self,
        entries: list[dict[str, object]],
        starts_at: datetime,
        ends_at: datetime,
        ignore_id: str,
    ) -> str | None:
        """The first existing entry this one runs into, if any."""
        for entry in entries:
            if entry[_ID] == ignore_id:
                continue
            existing_start = self._as_datetime(entry[_STARTS_AT])
            existing_end = self._as_datetime(entry[_ENDS_AT])
            if starts_at < existing_end and existing_start < ends_at:
                return str(entry[_SUBJECT])
        return None

    @staticmethod
    def _as_datetime(value: object) -> datetime:
        """Parse a stored or supplied timestamp.

        :raises CalendarError: if it is not one. A stored entry that cannot be
            parsed is corruption, and a supplied one that cannot be parsed
            would otherwise be written and become corruption.
        """
        try:
            return datetime.fromisoformat(str(value))
        except (TypeError, ValueError) as exc:
            raise CalendarError(f"{value!r} is not a date and time") from exc
