"""The local calendar, mirrored to Google Calendar (FR-85, FR-86).

:class:`SyncedCalendar` is a tool provider that wraps :class:`LocalCalendar`:
every calendar tool runs locally first -- so it works offline and every
existing behaviour stays, tentative entries included -- and then, if an
account is connected, the matching Google operation is queued in the
outbox. The reply never waits on Google.

A tentative entry pencilled from overheard speech (FR-77) is a Google event
with ``status: tentative``; keeping it makes it confirmed. Room-made events
carry ``extendedProperties.private.smartspace = "1"`` so the pull in
:mod:`src.assistance.google.sync` never mirrors the room's own entries back.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from src.assistance.google.account import GoogleAccount
from src.assistance.google.outbox import DELETE, INSERT, PATCH, Outbox, OutboxFull
from src.assistance.providers.local_calendar import ORIGIN_GOOGLE, LocalCalendar
from src.common.tools import (
    CANCEL_EVENT,
    KEEP_EVENT,
    MOVE_EVENT,
    PENCIL_EVENT,
    SCHEDULE_EVENT,
    ArgumentValue,
    ProviderOutcome,
)

LOGGER = logging.getLogger(__name__)

#: Marks an event in Google as the room's own (FR-85).
ROOM_MARK = "smartspace"


def google_id_for(entry: Mapping) -> str:
    """A Google event id for a room entry, the same on every retry (FR-86).

    Google accepts ids of 5 to 1024 characters from base32hex (0-9, a-v);
    lowercase hex is a subset. Derived from the entry's id and what it was
    created as, so a calendar file started afresh cannot reuse an old id.
    """
    seed = f"{entry['event_id']}|{entry['subject']}|{entry['starts_at']}"
    return "ss" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:30]


def to_google(entry: Mapping, utc_offset_h: float) -> dict:
    """A room entry as a Google Calendar event body."""
    zone = timezone(timedelta(hours=utc_offset_h))
    starts = datetime.fromisoformat(str(entry["starts_at"])).replace(tzinfo=zone)
    ends = datetime.fromisoformat(str(entry["ends_at"])).replace(tzinfo=zone)
    body = {
        "summary": str(entry["subject"]),
        "start": {"dateTime": starts.isoformat()},
        "end": {"dateTime": ends.isoformat()},
        "status": "tentative" if entry.get("tentative") else "confirmed",
        "extendedProperties": {"private": {ROOM_MARK: "1", "local_id": str(entry["event_id"]), "origin": str(entry.get("origin", "room"))}},
    }
    if entry.get("heard"):
        body["description"] = f"Pencilled in by the smart space from what it overheard: “{entry['heard']}”"
    return body


def from_google(event: Mapping, utc_offset_h: float) -> dict | None:
    """A Google event as a local mirror entry, or None for one the room made
    or one it cannot place (cancelled, or without a start)."""
    if event.get("status") == "cancelled":
        return None
    if event.get("extendedProperties", {}).get("private", {}).get(ROOM_MARK) == "1":
        return None
    start, end = event.get("start", {}), event.get("end", {})
    zone = timezone(timedelta(hours=utc_offset_h))
    if "dateTime" in start:
        starts = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00")).astimezone(zone).replace(tzinfo=None)
        ends = datetime.fromisoformat(end.get("dateTime", start["dateTime"]).replace("Z", "+00:00")).astimezone(zone).replace(tzinfo=None)
    elif "date" in start:
        starts = datetime.fromisoformat(start["date"])
        ends = datetime.fromisoformat(end.get("date", start["date"]))
    else:
        return None
    duration_min = max(5, int((ends - starts).total_seconds() // 60) or 60)
    return {
        "google_id": str(event["id"]),
        "subject": str(event.get("summary") or "(no title)"),
        "starts_at": starts.isoformat(timespec="minutes"),
        "duration_min": duration_min,
        "tentative": event.get("status") == "tentative",
    }


class SyncedCalendar:
    """The local calendar, with each change queued for Google when connected."""

    def __init__(self, local: LocalCalendar, outbox: Outbox, account: GoogleAccount, clock, utc_offset_h: float, on_queued=None, lock=None) -> None:
        self._local = local
        self._outbox = outbox
        self._account = account
        self._clock = clock
        self._utc_offset_h = utc_offset_h
        #: Told when something was queued, so the sync can try at once.
        self._on_queued = on_queued
        #: Shared with the sync, which mirrors into the same calendar from
        #: its own thread.
        self._lock = lock or threading.RLock()

    @property
    def name(self) -> str:
        return self._local.name

    @property
    def simulated(self) -> bool:
        return False

    @property
    def local(self) -> LocalCalendar:
        return self._local

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            before = {str(event["event_id"]): event for event in self._local.events}
            outcome = self._local.invoke(tool, arguments)
            if self._account.connected:
                try:
                    self._queue(tool, outcome, before)
                except OutboxFull as exc:
                    # The entry is made here; only its copy in Google waits.
                    LOGGER.warning("calendar change not queued for Google: %s", exc)
        return outcome

    def _queue(self, tool: str, outcome: ProviderOutcome, before: dict) -> None:
        event_id = str(outcome.detail.get("event_id", ""))
        now = self._clock.now()
        if tool in (SCHEDULE_EVENT.name, PENCIL_EVENT.name):
            if outcome.detail.get("added") is False:
                return  # a repeat mention: nothing new was made
            entry = self._local.entry(event_id)
            if entry is None:
                return
            google_id = google_id_for(entry)
            self._local.set_google_id(event_id, google_id)
            self._outbox.add(INSERT, {"google_id": google_id, "body": {"id": google_id, **to_google(entry, self._utc_offset_h)}}, now)
        elif tool in (MOVE_EVENT.name, KEEP_EVENT.name):
            entry = self._local.entry(event_id)
            if entry is None or not entry.get("google_id"):
                return
            body = to_google(entry, self._utc_offset_h)
            if entry.get("origin") == ORIGIN_GOOGLE:
                body.pop("extendedProperties")  # theirs, not the room's
            self._outbox.add(PATCH, {"google_id": entry["google_id"], "body": body}, now)
        elif tool == CANCEL_EVENT.name:
            gone = before.get(event_id)
            if gone is None or not gone.get("google_id"):
                return
            self._outbox.add(DELETE, {"google_id": gone["google_id"]}, now)
        else:
            return
        if self._on_queued is not None:
            self._on_queued()
