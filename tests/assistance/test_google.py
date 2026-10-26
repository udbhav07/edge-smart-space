"""Google Calendar and Gmail behind the tool surface (FR-84 to FR-86).

A fake Google stands in for the network: it records what it was asked, can
be taken offline, and can refuse the token. The local calendar, the outbox
and the sync are real, writing real files in a temporary directory.
"""

import json
import time
from pathlib import Path

import pytest

from src.assistance.google.account import GoogleAccount
from src.assistance.google.api import GoogleAlreadyExists, GoogleAuthError, GoogleNotFound, GoogleOffline, GoogleRejected
from src.assistance.google.calendar import SyncedCalendar, from_google, google_id_for, to_google
from src.assistance.google.mail import GmailOrMock
from src.assistance.google.outbox import DELETE, INSERT, PATCH, SEND, Outbox, OutboxFull
from src.assistance.google.sync import GoogleSync
from src.assistance.google.travel import TravelWithItinerary
from src.assistance.providers.local_calendar import LocalCalendar
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config

CONFIG = load_config(Path("config/default.yaml"))
OFFSET_H = CONFIG.site.utc_offset_h


class FakeGoogle:
    def __init__(self):
        self.calendar: dict[str, dict] = {}
        self.sent: list[str] = []
        self.offline = False
        self.refuse = False
        self.calls: list[str] = []

    def _gate(self, call):
        self.calls.append(call)
        if self.refuse:
            raise GoogleAuthError("refused")
        if self.offline:
            raise GoogleOffline("no route")

    def insert_event(self, calendar_id, body):
        self._gate("insert")
        if body["id"] in self.calendar:
            raise GoogleAlreadyExists("409")
        self.calendar[body["id"]] = dict(body)
        return body["id"]

    def patch_event(self, calendar_id, event_id, body):
        self._gate("patch")
        if event_id not in self.calendar:
            raise GoogleNotFound("404")
        self.calendar[event_id].update(body)

    def delete_event(self, calendar_id, event_id):
        self._gate("delete")
        if event_id not in self.calendar:
            raise GoogleNotFound("404")
        del self.calendar[event_id]

    def list_events(self, calendar_id, time_min, time_max):
        self._gate("list")
        return [{**event, "id": key} for key, event in self.calendar.items()]

    def send_mail(self, raw):
        self._gate("send")
        self.sent.append(raw)
        return f"m{len(self.sent)}"


class FakeAccount:
    def __init__(self, connected=True, email="me@example.com"):
        self.connected, self.email, self.configured = connected, email, True
        self.needs_reconnect = False

    def refused(self):
        self.connected, self.needs_reconnect = False, True


class Rig:
    def __init__(self, tmp_path, connected=True):
        self.clock = SimClock()
        self.google = FakeGoogle()
        self.account = FakeAccount(connected)
        self.outbox = Outbox(tmp_path / "outbox.json", 50, retry_min_s=10.0, retry_max_s=600.0)
        self.local = LocalCalendar(tmp_path / "calendar.json", 50)
        self.published = []
        self.sync = GoogleSync(CONFIG.google, self.clock, self.account, self.google, self.outbox, self.local, OFFSET_H,
                               publish=lambda topic, status: self.published.append((topic, status)))
        self.calendar = SyncedCalendar(self.local, self.outbox, self.account, self.clock, OFFSET_H, lock=self.sync.lock)
        self.mail = GmailOrMock(self.account, self.outbox, self.clock, flush=self.sync.flush)
        self.travel = TravelWithItinerary(self.calendar, self.account, self.outbox, self.clock, flush=self.sync.flush)

    def schedule(self, subject="dentist", starts_at="2026-10-01T16:00:00"):
        return self.calendar.invoke("schedule_event", {"subject": subject, "starts_at": starts_at})


class TestOutbox:
    def _outbox(self, tmp_path, size=5):
        return Outbox(tmp_path / "outbox.json", size, retry_min_s=10.0, retry_max_s=80.0)

    def test_it_survives_a_restart(self, tmp_path):
        self._outbox(tmp_path).add(SEND, {"raw": "x"}, 1.0)
        assert len(self._outbox(tmp_path)) == 1

    def test_failures_wait_longer_each_time(self, tmp_path):
        outbox = self._outbox(tmp_path)
        op = outbox.add(SEND, {"raw": "x"}, 0.0)
        waits = []
        for _ in range(5):
            outbox.retry_later(op, 0.0)
            waits.append(outbox.items[0]["next_try_ts"])
        assert waits == [10.0, 20.0, 40.0, 80.0, 80.0]

    def test_a_delete_of_an_entry_never_sent_sends_nothing(self, tmp_path):
        outbox = self._outbox(tmp_path)
        outbox.add(INSERT, {"google_id": "ssabc", "body": {"id": "ssabc"}}, 0.0)
        outbox.add(DELETE, {"google_id": "ssabc"}, 0.0)
        assert len(outbox) == 0

    def test_a_move_before_the_insert_left_is_folded_in(self, tmp_path):
        outbox = self._outbox(tmp_path)
        outbox.add(INSERT, {"google_id": "ssabc", "body": {"id": "ssabc", "summary": "a"}}, 0.0)
        outbox.add(PATCH, {"google_id": "ssabc", "body": {"summary": "b"}}, 0.0)
        (only,) = outbox.items
        assert only["kind"] == INSERT and only["args"]["body"]["summary"] == "b"

    def test_a_full_outbox_says_so(self, tmp_path):
        outbox = self._outbox(tmp_path, size=1)
        outbox.add(SEND, {"raw": "x"}, 0.0)
        with pytest.raises(OutboxFull):
            outbox.add(SEND, {"raw": "y"}, 0.0)

    def test_an_unreadable_outbox_is_refused_not_replaced(self, tmp_path):
        (tmp_path / "outbox.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ValueError):
            self._outbox(tmp_path)


class TestTheCalendarIsMirrored:
    def test_nothing_is_queued_without_an_account(self, tmp_path):
        rig = Rig(tmp_path, connected=False)
        rig.schedule()
        assert len(rig.outbox) == 0 and len(rig.local.events) == 1

    def test_an_entry_is_queued_under_a_stable_id(self, tmp_path):
        rig = Rig(tmp_path)
        rig.schedule()
        (item,) = rig.outbox.items
        entry = rig.local.events[0]
        assert item["kind"] == INSERT and item["args"]["google_id"] == entry["google_id"] == google_id_for(entry)

    def test_and_reaches_google_on_the_next_pass(self, tmp_path):
        rig = Rig(tmp_path)
        rig.schedule()
        rig.sync.flush()
        (event,) = rig.google.calendar.values()
        assert event["summary"] == "dentist" and event["start"]["dateTime"].startswith("2026-10-01T16:00:00")

    def test_a_tentative_entry_is_tentative_in_google(self, tmp_path):
        rig = Rig(tmp_path)
        rig.calendar.invoke("pencil_event", {"subject": "eye test", "starts_at": "2026-10-01T17:00:00", "heard": "eye test at five"})
        rig.sync.flush()
        (event,) = rig.google.calendar.values()
        assert event["status"] == "tentative" and "eye test at five" in event["description"]

    def test_keeping_it_confirms_it_in_google(self, tmp_path):
        rig = Rig(tmp_path)
        outcome = rig.calendar.invoke("pencil_event", {"subject": "eye test", "starts_at": "2026-10-01T17:00:00"})
        rig.sync.flush()
        rig.calendar.invoke("keep_event", {"event_id": outcome.detail["event_id"]})
        rig.sync.flush()
        assert next(iter(rig.google.calendar.values()))["status"] == "confirmed"

    def test_a_move_and_a_cancel_follow(self, tmp_path):
        rig = Rig(tmp_path)
        rig.schedule()
        rig.sync.flush()
        rig.calendar.invoke("move_event", {"subject": "dentist", "starts_at": "2026-10-01T17:00:00"})
        rig.sync.flush()
        assert next(iter(rig.google.calendar.values()))["start"]["dateTime"].startswith("2026-10-01T17:00")
        rig.calendar.invoke("cancel_event", {"subject": "dentist"})
        rig.sync.flush()
        assert rig.google.calendar == {}

    def test_a_repeat_mention_queues_nothing_more(self, tmp_path):
        rig = Rig(tmp_path)
        rig.calendar.invoke("pencil_event", {"subject": "dentist", "starts_at": "2026-10-01T16:00:00"})
        rig.calendar.invoke("pencil_event", {"subject": "the dentist", "starts_at": "2026-10-01T16:30:00"})
        assert len(rig.outbox) == 1


class TestOffline:
    """FR-86: queued while offline, sent once back, never twice."""

    def test_offline_it_waits_and_says_so(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.offline = True
        rig.schedule()
        rig.sync.flush()
        assert len(rig.outbox) == 1 and rig.published[-1][1].online is False

    def test_back_online_it_is_sent(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.offline = True
        rig.schedule()
        rig.sync.flush()
        rig.google.offline = False
        rig.clock.advance(11.0)
        rig.sync.flush()
        assert len(rig.outbox) == 0 and len(rig.google.calendar) == 1

    def test_a_retry_of_an_insert_that_had_landed_is_not_a_duplicate(self, tmp_path):
        rig = Rig(tmp_path)
        rig.schedule()
        item = rig.outbox.items[0]
        rig.google.insert_event("primary", item["args"]["body"])  # it landed; the answer was lost
        rig.sync.flush()
        assert len(rig.google.calendar) == 1 and len(rig.outbox) == 0

    def test_a_restart_mid_queue_loses_nothing(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.offline = True
        rig.schedule()
        rig.mail.invoke("send_mail", {"to": "anu@example.com", "subject": "Dinner", "body": "at eight"})
        again = Rig(tmp_path)
        again.clock.advance(11.0)  # past the first back-off
        again.sync.flush()
        assert len(again.google.calendar) == 1 and len(again.google.sent) == 1

    def test_order_is_kept_behind_a_failure(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.offline = True
        rig.schedule("first", "2026-10-01T10:00:00")
        rig.schedule("second", "2026-10-01T11:00:00")
        rig.sync.flush()
        assert rig.google.calls == ["insert"]  # stopped at the first failure

    def test_a_refused_token_asks_to_reconnect(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.refuse = True
        rig.schedule()
        rig.sync.flush()
        assert rig.account.needs_reconnect and len(rig.outbox) == 1

    def test_a_rejected_operation_is_dropped_with_its_reason(self, tmp_path):
        rig = Rig(tmp_path)
        rig.schedule()

        def rejected(calendar_id, body):
            raise GoogleRejected("Google refused it (400)")

        rig.google.insert_event = rejected
        rig.sync.flush()
        assert len(rig.outbox) == 0 and "400" in rig.sync.status().last_error


class TestThePull:
    def test_an_entry_made_in_google_appears_here(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.calendar["g1"] = {"summary": "Standup", "status": "confirmed",
                                      "start": {"dateTime": "2026-10-01T09:00:00+05:30"}, "end": {"dateTime": "2026-10-01T09:15:00+05:30"}}
        rig.clock = rig.sync._clock  # noqa: the pull window starts at today
        rig.sync.flush()
        mirrored = [e for e in rig.local.events if e.get("origin") == "google"]
        assert mirrored and mirrored[0]["subject"] == "Standup"

    def test_the_rooms_own_entries_are_not_mirrored_back(self, tmp_path):
        rig = Rig(tmp_path)
        rig.schedule()
        rig.sync.flush()
        rig.sync.flush()
        assert len(rig.local.events) == 1

    def test_from_google_reads_local_time(self):
        entry = from_google({"id": "g1", "summary": "Call", "status": "tentative",
                             "start": {"dateTime": "2026-10-01T03:30:00Z"}, "end": {"dateTime": "2026-10-01T04:00:00Z"}}, 5.5)
        assert entry["starts_at"] == "2026-10-01T09:00" and entry["duration_min"] == 30 and entry["tentative"]

    def test_to_google_is_marked_as_the_rooms(self):
        body = to_google({"event_id": "ev_0001", "subject": "x", "starts_at": "2026-10-01T09:00", "ends_at": "2026-10-01T10:00"}, 5.5)
        assert body["extendedProperties"]["private"]["smartspace"] == "1" and body["start"]["dateTime"].endswith("+05:30")


class TestMail:
    def test_without_an_account_it_is_the_mock(self, tmp_path):
        rig = Rig(tmp_path, connected=False)
        outcome = rig.mail.invoke("send_mail", {"to": "anu@example.com", "subject": "Hi", "body": "x"})
        assert "mock" in outcome.message.lower() and rig.google.sent == []

    def test_connected_and_online_it_is_sent(self, tmp_path):
        rig = Rig(tmp_path)
        outcome = rig.mail.invoke("send_mail", {"to": "anu@example.com", "subject": "Dinner", "body": "at eight"})
        assert outcome.message.startswith("Sent the email to anu@example.com") and len(rig.google.sent) == 1

    def test_offline_it_is_queued_and_says_so(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.offline = True
        outcome = rig.mail.invoke("send_mail", {"to": "anu@example.com", "subject": "Dinner", "body": "at eight"})
        assert outcome.message.startswith("Queued") and outcome.detail["queued"] is True

    def test_it_is_still_a_real_send_not_a_mock(self, tmp_path):
        assert Rig(tmp_path).mail.simulated is False


class TestBookings:
    def test_a_booking_lands_in_the_calendar_and_the_inbox(self, tmp_path):
        rig = Rig(tmp_path)
        outcome = rig.travel.invoke("book_travel", {"kind": "flight", "destination": "Delhi", "depart_on": "2026-10-02"})
        assert "mock" in outcome.message.lower() and "in your calendar" in outcome.message
        assert rig.local.events[0]["subject"] == "Flight to Delhi" and rig.local.events[0]["starts_at"] == "2026-10-02T09:00"
        assert len(rig.google.sent) == 1 and "in your inbox" in outcome.message

    def test_offline_the_itinerary_waits(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.offline = True
        outcome = rig.travel.invoke("book_travel", {"kind": "hotel", "destination": "Pune", "depart_on": "2026-10-02", "nights": 2})
        assert "queued for your inbox" in outcome.message and rig.local.events[0]["starts_at"] == "2026-10-02T14:00"

    def test_without_an_account_it_is_the_mock_and_a_local_entry(self, tmp_path):
        rig = Rig(tmp_path, connected=False)
        outcome = rig.travel.invoke("book_travel", {"kind": "flight", "destination": "Delhi", "depart_on": "2026-10-02"})
        assert "inbox" not in outcome.message and len(rig.local.events) == 1


class TestTheStatus:
    def test_it_never_carries_a_token(self, tmp_path):
        rig = Rig(tmp_path)
        rig.sync.flush()
        topic, status = rig.published[-1]
        assert topic is topics.ASSIST_GOOGLE and "token" not in status.model_dump_json()

    def test_it_says_how_much_is_waiting(self, tmp_path):
        rig = Rig(tmp_path)
        rig.google.offline = True
        rig.schedule()
        rig.sync.flush()
        assert rig.published[-1][1].outbox == 1


class TestTheAccount:
    def _account(self, tmp_path):
        config = CONFIG.google.model_copy(update={"token_path": str(tmp_path / "token.json"), "client_secret_path": str(tmp_path / "client.json")})
        return GoogleAccount(config)

    def test_no_token_is_not_connected(self, tmp_path):
        account = self._account(tmp_path)
        assert not account.connected and not account.configured

    def test_a_refused_token_needs_reconnecting_until_a_new_one_is_written(self, tmp_path):
        account = self._account(tmp_path)
        (tmp_path / "token.json").write_text("{}", encoding="utf-8")
        account.refused()
        assert account.needs_reconnect and not account.connected
        time.sleep(0.02)
        (tmp_path / "token.json").write_text('{"new": 1}', encoding="utf-8")
        import os
        os.utime(tmp_path / "token.json", (time.time() + 5, time.time() + 5))
        assert account.connected and not account.needs_reconnect

    def test_the_address_comes_from_the_account_file(self, tmp_path):
        (tmp_path / "account.json").write_text(json.dumps({"email": "me@example.com"}), encoding="utf-8")
        assert self._account(tmp_path).email == "me@example.com"
