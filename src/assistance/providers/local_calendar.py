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
    CANCEL_EVENT,
    DEFAULT_EVENT_DURATION_MIN,
    GET_EVENTS,
    KEEP_EVENT,
    MOVE_EVENT,
    PENCIL_EVENT,
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

#: Where a pencilled entry came from (FR-77). Kept on the entry so the
#: console can say why it is there.
ORIGIN_OVERHEARD = "overheard"

#: The longest overheard sentence kept with an entry: enough to recognise it.
_HEARD_KEPT_CHARS = 200

#: Words too common to tell two subjects apart when matching a repeat mention.
_FILLER = frozenset({"the", "and", "with", "for", "our", "my", "a", "an", "to", "at", "on", "in"})


class CalendarFullError(RuntimeError):
    """The configured bound on entries has been reached."""


class NoSuchEntryError(ValueError):
    """Nothing, or more than one thing, matches the entry asked about.

    The message is said to the occupant, so it is written for them.
    """


def describe_time(moment: datetime) -> str:
    """A time the way a person says it: ``15:00 on Thursday 1 October``.

    Built by hand rather than with ``%-d``, which Windows' strftime rejects.
    """
    return f"{moment:%H:%M} on {moment:%A} {moment.day} {moment:%B}"


def describe_day(moment: datetime) -> str:
    """A date the way a person says it: ``Thursday 1 October``, and the time
    too only if one was given -- a flight on a day is not a flight at midnight."""
    day = f"{moment:%A} {moment.day} {moment:%B}"
    if moment.hour or moment.minute:
        return f"{day} at {moment:%H:%M}"
    return day


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
        if tool == CANCEL_EVENT.name:
            return self._cancel(arguments)
        if tool == MOVE_EVENT.name:
            return self._move(arguments)
        if tool == PENCIL_EVENT.name:
            return self._pencil(arguments)
        if tool == KEEP_EVENT.name:
            return self._keep(arguments)
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
        clashes = self._overlapping(starts, starts + timedelta(minutes=duration_min))
        self._events.append(event)
        self._next_id += 1
        self._save()
        LOGGER.info("added %s: %s at %s", event["event_id"], subject, event["starts_at"])
        clash = (
            f" Note that you already have {clashes[0]['subject']} then."
            if clashes
            else ""
        )
        return ProviderOutcome(
            message=f"Added {subject} at {describe_time(starts)}.{clash}",
            detail={
                "event_id": event["event_id"],
                "starts_at": event["starts_at"],
                "duration_min": duration_min,
            },
        )

    def _pencil(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """A plan overheard, written in as tentative (FR-77).

        A plan is mentioned more than once in a conversation, so an entry
        already on that day with the same subject -- tentative or firm --
        takes it, and nothing is added.
        """
        starts = _parse(arguments["starts_at"])
        subject = str(arguments["subject"]).strip()
        same = self._same_plan(subject, starts)
        if same is not None:
            return ProviderOutcome(
                message=f"{same['subject']} is already in the calendar then.",
                detail={"event_id": same["event_id"], "added": False},
            )
        if len(self._events) >= self._max_events:
            raise CalendarFullError(
                f"the calendar already holds {self._max_events} entries"
            )
        duration_min = int(arguments.get("duration_min") or DEFAULT_EVENT_DURATION_MIN)
        event = {
            "event_id": f"ev_{self._next_id:04d}",
            "subject": subject,
            "starts_at": starts.isoformat(timespec="minutes"),
            "ends_at": (starts + timedelta(minutes=duration_min)).isoformat(timespec="minutes"),
            "duration_min": duration_min,
            "tentative": True,
            "origin": ORIGIN_OVERHEARD,
            "heard": " ".join(str(arguments.get("heard") or "").split())[:_HEARD_KEPT_CHARS],
        }
        self._events.append(event)
        self._next_id += 1
        self._save()
        LOGGER.info("pencilled %s: %s at %s", event["event_id"], subject, event["starts_at"])
        return ProviderOutcome(
            message=f"Pencilled in {subject} at {describe_time(starts)}, tentatively.",
            detail={"event_id": event["event_id"], "starts_at": event["starts_at"], "added": True},
        )

    def _keep(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """A tentative entry the occupant kept becomes a firm one."""
        event_id = str(arguments["event_id"])
        event = next((event for event in self._events if event["event_id"] == event_id), None)
        if event is None:
            raise NoSuchEntryError(f"There is no entry {event_id}.")
        event.pop("tentative", None)
        self._save()
        LOGGER.info("kept %s: %s", event_id, event["subject"])
        return ProviderOutcome(
            message=f"Kept {event['subject']} at {describe_time(_parse(event['starts_at']))}.",
            detail={"event_id": event_id},
        )

    def _same_plan(self, subject: str, starts: datetime) -> dict[str, str | int] | None:
        """An entry that day sharing the subject's telling words, if any."""
        words = _telling(subject)
        for event in self._sorted():
            if _parse(event["starts_at"]).date() != starts.date():
                continue
            theirs = _telling(str(event["subject"]))
            if words and theirs and (words <= theirs or theirs <= words):
                return event
        return None

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
        tentative = [event for event in found if event.get("tentative")]
        # The detail carries no nested values, so what was heard for each
        # tentative entry travels under a key of its own.
        heard = {f"heard_{event['event_id']}": str(event.get("heard", "")) for event in tentative}
        return ProviderOutcome(
            message=_summarise(found),
            detail={
                "count": len(found),
                "entries": _listing(found),
                "tentative_ids": ",".join(str(event["event_id"]) for event in tentative),
                **heard,
            },
        )

    def _cancel(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        event = self._one_matching(str(arguments["subject"]), arguments.get("on"))
        self.remove(str(event["event_id"]))
        LOGGER.info("cancelled %s: %s", event["event_id"], event["subject"])
        return ProviderOutcome(
            message=f"Cancelled {event['subject']} at {describe_time(_parse(event['starts_at']))}.",
            detail={"event_id": event["event_id"]},
        )

    def _move(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        event = self._one_matching(str(arguments["subject"]), None)
        starts = _parse(arguments["starts_at"])
        if arguments.get("keep_day"):
            # "Move it to four": the entry's own day, at the new time.
            starts = datetime.combine(_parse(event["starts_at"]).date(), starts.time())
        duration_min = int(event.get("duration_min") or DEFAULT_EVENT_DURATION_MIN)
        event["starts_at"] = starts.isoformat(timespec="minutes")
        event["ends_at"] = (starts + timedelta(minutes=duration_min)).isoformat(timespec="minutes")
        self._save()
        LOGGER.info("moved %s to %s", event["event_id"], event["starts_at"])
        return ProviderOutcome(
            message=f"Moved {event['subject']} to {describe_time(starts)}.",
            detail={"event_id": event["event_id"], "starts_at": event["starts_at"]},
        )

    def _one_matching(self, words: str, on: ArgumentValue | None) -> dict[str, str | int]:
        """The single entry the occupant meant, or a question back.

        :raises NoSuchEntryError: when nothing matches, or several do and
            nothing said tells them apart.
        """
        wanted = [word for word in words.lower().split() if len(word) > 2]
        found = [
            event
            for event in self._sorted()
            if wanted and all(word in str(event["subject"]).lower() for word in wanted)
        ]
        if on is not None and len(found) > 1:
            day = _parse(on).date()
            found = [event for event in found if _parse(event["starts_at"]).date() == day] or found
        if not found:
            raise NoSuchEntryError(f"Nothing in your calendar matches {words!r}.")
        if len(found) > 1:
            listed = "; ".join(
                f"{event['subject']} at {describe_time(_parse(event['starts_at']))}" for event in found[:3]
            )
            raise NoSuchEntryError(f"More than one entry matches: {listed}. Which one?")
        return found[0]

    def _overlapping(self, starts: datetime, ends: datetime) -> list[dict[str, str | int]]:
        return [
            event
            for event in self._sorted()
            if _parse(event["starts_at"]) < ends and _parse(event["ends_at"]) > starts
        ]

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
        + (" (tentative)" if event.get("tentative") else "")
        for event in found[:_ENTRIES_NAMED_IN_A_REPLY]
    ]
    remainder = len(found) - len(named)
    listing = "; ".join(named)
    if remainder:
        listing += f"; and {remainder} more"
    count = "1 entry" if len(found) == 1 else f"{len(found)} entries"
    return f"{count}: {listing}."


def _telling(subject: str) -> frozenset[str]:
    """The words of a subject that identify it, lower-cased."""
    return frozenset(
        word for word in "".join(c if c.isalnum() else " " for c in subject.lower()).split()
        if word not in _FILLER and len(word) > 2
    )


def _listing(found: list[dict[str, str | int]]) -> str:
    """Flat text for the result detail, which carries no nested values."""
    return "; ".join(
        f"{event['event_id']} {event['starts_at']} {event['subject']}" for event in found
    )
