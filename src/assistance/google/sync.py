"""Draining the outbox to Google, mirroring Google in, and saying how it is going.

:class:`GoogleSync` runs on a thread of its own in the assistance service.
Each pass:

1. if an account is connected, sends every operation that is due, in order,
   stopping at the first sign Google cannot be reached -- the rest wait, with
   their back-off (FR-86);
2. every ``pull_interval_s``, mirrors entries made in Google itself into the
   local calendar for the coming ``pull_days`` (FR-85);
3. publishes :class:`GoogleStatus` when anything the occupant would see has
   changed -- never a token (FR-84).

:meth:`flush` runs one pass at once: a confirmed mail tries to go
immediately, and is only queued if it cannot.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone

from src.assistance.google.account import GoogleAccount
from src.assistance.google.api import (
    GoogleAlreadyExists,
    GoogleApi,
    GoogleAuthError,
    GoogleNotFound,
    GoogleOffline,
    GoogleRejected,
)
from src.assistance.google.calendar import from_google
from src.assistance.google.outbox import DELETE, INSERT, PATCH, SEND, Outbox
from src.assistance.providers.local_calendar import LocalCalendar
from src.common import topics
from src.common.clock import Clock
from src.common.config import GoogleConfig
from src.common.schemas import GoogleStatus

LOGGER = logging.getLogger(__name__)

#: Seconds between passes of the worker thread.
_PASS_S = 5.0

#: Restate the status at least this often, so a console sees it is alive.
_STATUS_EVERY_S = 60.0


class GoogleSync:
    """Sends what is waiting, mirrors Google in, and reports."""

    def __init__(
        self,
        config: GoogleConfig,
        clock: Clock,
        account: GoogleAccount,
        api: GoogleApi,
        outbox: Outbox,
        calendar: LocalCalendar,
        utc_offset_h: float,
        publish=None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._account = account
        self._api = api
        self._outbox = outbox
        self._calendar = calendar
        self._utc_offset_h = utc_offset_h
        self._publish = publish
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._online = False
        self._last_sync_ts: float | None = None
        self._last_error = ""
        self._last_pull_s: float | None = None
        self._last_status: tuple | None = None
        self._last_status_s: float | None = None
        self._was_connected = account.connected

    # --- the pass ---------------------------------------------------------

    @property
    def lock(self) -> threading.RLock:
        """Held while the calendar or the outbox is changed, from any thread."""
        return self._lock

    def flush(self) -> None:
        """Try at once, from any thread."""
        with self._lock:
            self._pass()

    def poke(self) -> None:
        """Something was queued: run a pass soon rather than on the next tick."""
        self._wake.set()

    def run_forever(self) -> None:
        while True:
            self._wake.wait(timeout=_PASS_S)
            self._wake.clear()
            try:
                self.flush()
            except Exception:
                # The sync is a courtesy: a bug in it must not take the tool
                # executor down. Logged with its traceback.
                LOGGER.exception("Google sync pass failed")

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_forever, name="google-sync", daemon=True)
        thread.start()
        return thread

    def _pass(self) -> None:
        connected = self._account.connected
        if connected and not self._was_connected:
            # Signed in (again): everything waiting is tried now.
            self._outbox.retry_all_now(self._clock.now())
            self._last_error = ""
        self._was_connected = connected
        if connected and self._drain() and self._pull_due():
            # Not when the drain just found Google unreachable: the pull
            # would only find the same.
            self._pull()
        self._report()

    def _drain(self) -> bool:
        """Send what is due. Returns False if Google turned out unreachable."""
        for item in self._outbox.due(self._clock.now()):
            try:
                self._send(item)
            except GoogleOffline as exc:
                self._online = False
                self._last_error = str(exc)
                self._outbox.retry_later(item["op_id"], self._clock.now())
                return False  # the rest wait too
            except GoogleAuthError as exc:
                self._online = False
                self._last_error = str(exc)
                self._account.refused()
                return False
            except GoogleRejected as exc:
                LOGGER.warning("Google refused %s %s: %s", item["kind"], item["op_id"], exc)
                self._last_error = str(exc)
                self._outbox.done(item["op_id"])
                continue
            self._online = True
            self._last_sync_ts = self._clock.now()
            self._outbox.done(item["op_id"])
        return True

    def _send(self, item: dict) -> None:
        kind, args, calendar_id = item["kind"], item["args"], self._config.calendar_id
        if kind == INSERT:
            try:
                self._api.insert_event(calendar_id, args["body"])
            except GoogleAlreadyExists:
                LOGGER.info("event %s already in Google: an earlier attempt landed", args["google_id"])
        elif kind == PATCH:
            self._api.patch_event(calendar_id, args["google_id"], args["body"])
        elif kind == DELETE:
            try:
                self._api.delete_event(calendar_id, args["google_id"])
            except GoogleNotFound:
                LOGGER.info("event %s already gone from Google", args["google_id"])
        elif kind == SEND:
            self._api.send_mail(args["raw"])
            LOGGER.info("mail %s sent through Gmail", item["op_id"])

    # --- the pull -----------------------------------------------------------

    def _pull_due(self) -> bool:
        now_s = self._clock.monotonic()
        return self._last_pull_s is None or now_s - self._last_pull_s >= self._config.pull_interval_s

    def _pull(self) -> None:
        zone = timezone(timedelta(hours=self._utc_offset_h))
        start = datetime.fromtimestamp(self._clock.now(), zone).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=self._config.pull_days)
        # Counted from the attempt, so an unreachable Google is asked again on
        # the pull's own schedule, not on every pass.
        self._last_pull_s = self._clock.monotonic()
        try:
            events = self._api.list_events(self._config.calendar_id, start.isoformat(), end.isoformat())
        except GoogleOffline as exc:
            self._online, self._last_error = False, str(exc)
            return
        except GoogleAuthError as exc:
            self._online, self._last_error = False, str(exc)
            self._account.refused()
            return
        except GoogleRejected as exc:
            self._last_error = str(exc)
            return
        self._online = True
        self._last_sync_ts = self._clock.now()
        mirrored = [entry for entry in (from_google(event, self._utc_offset_h) for event in events) if entry is not None]
        changed = self._calendar.mirror(mirrored, start.replace(tzinfo=None), end.replace(tzinfo=None))
        if changed:
            LOGGER.info("mirrored %d change(s) from Google Calendar", changed)

    # --- the status -----------------------------------------------------------

    def status(self) -> GoogleStatus:
        return GoogleStatus(
            ts=self._clock.now(),
            configured=self._account.configured,
            connected=self._account.connected,
            account=self._account.email if self._account.connected or self._account.needs_reconnect else "",
            outbox=len(self._outbox),
            online=self._online and self._account.connected,
            last_sync_ts=self._last_sync_ts,
            last_error=self._last_error,
            needs_reconnect=self._account.needs_reconnect,
        )

    def _report(self) -> None:
        if self._publish is None:
            return
        status = self.status()
        key = (status.configured, status.connected, status.account, status.outbox, status.online, status.last_error, status.needs_reconnect)
        now_s = self._clock.monotonic()
        if key == self._last_status and self._last_status_s is not None and now_s - self._last_status_s < _STATUS_EVERY_S:
            return
        self._last_status, self._last_status_s = key, now_s
        try:
            self._publish(topics.ASSIST_GOOGLE, status)
        except (OSError, ValueError) as exc:
            LOGGER.warning("could not publish the Google status: %s", exc)
