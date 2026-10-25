"""Unit tests for the local calendar provider (FR-58)."""

import json
from datetime import datetime

import pytest

from src.assistance.providers.local_calendar import (
    CalendarFullError,
    LocalCalendar,
    describe_time,
)
from src.common.tools import DEFAULT_EVENT_DURATION_MIN

MAX_EVENTS = 5
DAY_START = "2026-10-01T00:00:00"
DAY_END = "2026-10-02T00:00:00"


@pytest.fixture(name="path")
def _path(tmp_path):
    return tmp_path / "calendar.json"


@pytest.fixture(name="calendar")
def _calendar(path) -> LocalCalendar:
    return LocalCalendar(path, MAX_EVENTS)


def _schedule(calendar, starts_at="2026-10-01T15:00:00", subject="design review", **extra):
    return calendar.invoke(
        "schedule_event", {"starts_at": starts_at, "subject": subject, **extra}
    )


def _window(calendar, start=DAY_START, end=DAY_END):
    return calendar.invoke("get_events", {"from_time": start, "to_time": end})


class TestScheduling:
    def test_an_entry_is_added(self, calendar):
        _schedule(calendar)
        assert [event["subject"] for event in calendar.events] == ["design review"]

    def test_the_reply_says_what_and_when(self, calendar):
        outcome = _schedule(calendar)
        assert outcome.message == "Added design review at 15:00 on Thursday 1 October."

    def test_the_entry_id_is_reported(self, calendar):
        assert _schedule(calendar).detail["event_id"] == "ev_0001"

    def test_ids_are_not_reused(self, calendar):
        _schedule(calendar)
        assert _schedule(calendar).detail["event_id"] == "ev_0002"

    def test_a_missing_duration_uses_the_declared_default(self, calendar):
        assert _schedule(calendar).detail["duration_min"] == DEFAULT_EVENT_DURATION_MIN

    def test_a_given_duration_sets_the_end(self, calendar):
        _schedule(calendar, duration_min=30)
        assert calendar.events[0]["ends_at"] == "2026-10-01T15:30"

    def test_a_time_zone_suffix_is_read_as_local_time(self, calendar):
        _schedule(calendar, starts_at="2026-10-01T15:00:00+05:30")
        assert calendar.events[0]["starts_at"] == "2026-10-01T15:00"

    def test_a_full_calendar_refuses_more(self, calendar):
        for _ in range(MAX_EVENTS):
            _schedule(calendar)
        with pytest.raises(CalendarFullError):
            _schedule(calendar)

    def test_an_unreadable_time_is_refused(self, calendar):
        with pytest.raises(ValueError):
            _schedule(calendar, starts_at="next thursday")


class TestReading:
    def test_an_empty_window_says_so(self, calendar):
        outcome = _window(calendar)
        assert outcome.message == "Nothing is in the calendar then."
        assert outcome.detail["count"] == 0

    def test_an_entry_in_the_window_is_listed(self, calendar):
        _schedule(calendar)
        assert _window(calendar).message == (
            "1 entry: design review at 15:00 on Thursday 1 October."
        )

    def test_an_entry_outside_the_window_is_not(self, calendar):
        _schedule(calendar, starts_at="2026-10-05T09:00:00")
        assert _window(calendar).detail["count"] == 0

    def test_an_entry_overlapping_the_start_is_listed(self, calendar):
        _schedule(calendar, starts_at="2026-09-30T23:30:00")
        assert _window(calendar).detail["count"] == 1

    def test_entries_are_listed_earliest_first(self, calendar):
        _schedule(calendar, starts_at="2026-10-01T16:00:00", subject="later")
        _schedule(calendar, starts_at="2026-10-01T09:00:00", subject="earlier")
        message = _window(calendar).message
        assert message.index("earlier") < message.index("later")

    def test_a_long_day_is_summarised_not_read_out(self, path):
        calendar = LocalCalendar(path, 50)
        for hour in range(8, 16):
            _schedule(calendar, starts_at=f"2026-10-01T{hour:02d}:00:00", subject=f"m{hour}")
        assert _window(calendar).message.endswith("and 3 more.")

    def test_a_window_ending_before_it_starts_is_refused(self, calendar):
        with pytest.raises(ValueError):
            _window(calendar, start=DAY_END, end=DAY_START)


class TestPersistence:
    def test_entries_survive_a_restart(self, path, calendar):
        _schedule(calendar)
        reopened = LocalCalendar(path, MAX_EVENTS)
        assert [event["subject"] for event in reopened.events] == ["design review"]

    def test_ids_continue_after_a_restart(self, path, calendar):
        _schedule(calendar)
        reopened = LocalCalendar(path, MAX_EVENTS)
        assert _schedule(reopened).detail["event_id"] == "ev_0002"

    def test_an_unreadable_calendar_is_refused_not_replaced(self, path):
        """Starting empty would let the next write destroy every entry."""
        path.write_text(json.dumps({"version": 99}), encoding="utf-8")
        with pytest.raises(ValueError):
            LocalCalendar(path, MAX_EVENTS)

    def test_an_entry_can_be_removed(self, path, calendar):
        event_id = _schedule(calendar).detail["event_id"]
        assert calendar.remove(event_id) is True
        assert LocalCalendar(path, MAX_EVENTS).events == ()

    def test_removing_an_unknown_entry_reports_nothing_removed(self, calendar):
        assert calendar.remove("ev_9999") is False


class TestContract:
    def test_it_is_not_a_mock(self, calendar):
        assert calendar.simulated is False

    def test_it_refuses_a_tool_it_does_not_serve(self, calendar):
        with pytest.raises(ValueError):
            calendar.invoke("book_travel", {})

    def test_a_bound_of_zero_is_refused(self, path):
        with pytest.raises(ValueError):
            LocalCalendar(path, 0)

    def test_times_are_described_as_a_person_says_them(self):
        assert describe_time(datetime(2026, 10, 1, 9, 5)) == "09:05 on Thursday 1 October"


class TestCancellingAndMoving:
    """U-02: 'cancel the meeting' once *created* an entry; FR-58 says removable."""

    def test_an_entry_is_cancelled_by_words_from_its_title(self, calendar):
        _schedule(calendar, subject="Meeting with Ravi")
        outcome = calendar.invoke("cancel_event", {"subject": "meeting with Ravi"})
        assert "Cancelled Meeting with Ravi" in outcome.message
        assert calendar.events == ()

    def test_cancelling_what_is_not_there_says_so(self, calendar):
        with pytest.raises(ValueError, match="Nothing in your calendar matches"):
            calendar.invoke("cancel_event", {"subject": "dentist"})

    def test_two_matches_ask_which_one(self, calendar):
        _schedule(calendar, starts_at="2026-10-01T15:00:00", subject="Meeting with Ravi")
        _schedule(calendar, starts_at="2026-10-02T15:00:00", subject="Meeting with Ravi")
        with pytest.raises(ValueError, match="Which one"):
            calendar.invoke("cancel_event", {"subject": "meeting with Ravi"})
        assert len(calendar.events) == 2

    def test_a_date_tells_two_matches_apart(self, calendar):
        _schedule(calendar, starts_at="2026-10-01T15:00:00", subject="Meeting with Ravi")
        _schedule(calendar, starts_at="2026-10-02T15:00:00", subject="Meeting with Ravi")
        calendar.invoke("cancel_event", {"subject": "meeting with Ravi", "on": "2026-10-02T00:00:00"})
        assert [event["starts_at"] for event in calendar.events] == ["2026-10-01T15:00"]

    def test_an_entry_is_moved_keeping_its_length(self, calendar):
        _schedule(calendar, subject="Meeting with Ravi", duration_min=30)
        outcome = calendar.invoke("move_event", {"subject": "Ravi", "starts_at": "2026-10-01T16:00:00"})
        assert "Moved Meeting with Ravi to 16:00" in outcome.message
        event = calendar.events[0]
        assert (event["starts_at"], event["ends_at"]) == ("2026-10-01T16:00", "2026-10-01T16:30")

    def test_a_clash_is_mentioned(self, calendar):
        _schedule(calendar, subject="Meeting with Ravi")
        message = _schedule(calendar, subject="Dentist").message
        assert "you already have Meeting with Ravi then" in message


def _pencil(calendar, starts_at="2026-10-01T16:00:00", subject="dentist", heard="dentist on thursday at four"):
    return calendar.invoke("pencil_event", {"starts_at": starts_at, "subject": subject, "heard": heard})


class TestPencilling:
    """FR-77: a plan overheard is written in, marked tentative."""

    def test_an_overheard_plan_is_added_as_tentative(self, calendar):
        _pencil(calendar)
        (event,) = calendar.events
        assert event["tentative"] is True and event["origin"] == "overheard"

    def test_what_was_heard_is_kept_with_it(self, calendar):
        _pencil(calendar)
        assert calendar.events[0]["heard"] == "dentist on thursday at four"

    def test_the_reply_says_it_is_tentative(self, calendar):
        assert "tentatively" in _pencil(calendar).message

    def test_a_repeat_mention_that_day_adds_nothing(self, calendar):
        _pencil(calendar)
        outcome = _pencil(calendar, starts_at="2026-10-01T16:30:00", subject="the dentist")
        assert len(calendar.events) == 1 and outcome.detail["added"] is False

    def test_a_plan_already_firmly_in_the_calendar_adds_nothing(self, calendar):
        _schedule(calendar, starts_at="2026-10-01T16:00:00", subject="dentist appointment")
        _pencil(calendar)
        assert len(calendar.events) == 1 and "tentative" not in calendar.events[0]

    def test_the_same_subject_on_another_day_is_another_plan(self, calendar):
        _pencil(calendar)
        _pencil(calendar, starts_at="2026-10-08T16:00:00")
        assert len(calendar.events) == 2

    def test_a_different_subject_the_same_day_is_another_plan(self, calendar):
        _pencil(calendar)
        _pencil(calendar, subject="flight to Goa")
        assert len(calendar.events) == 2

    def test_a_full_calendar_refuses_it(self, calendar):
        for day in range(1, MAX_EVENTS + 1):
            _schedule(calendar, starts_at=f"2026-10-0{day}T09:00:00", subject=f"entry {day}")
        with pytest.raises(CalendarFullError):
            _pencil(calendar, starts_at="2026-10-09T16:00:00")

    def test_it_survives_a_restart(self, calendar, path):
        _pencil(calendar)
        assert LocalCalendar(path, MAX_EVENTS).events[0]["tentative"] is True


class TestKeeping:
    def test_a_kept_entry_is_firm(self, calendar):
        event_id = _pencil(calendar).detail["event_id"]
        calendar.invoke("keep_event", {"event_id": event_id})
        assert "tentative" not in calendar.events[0]

    def test_it_stays_marked_as_overheard(self, calendar):
        event_id = _pencil(calendar).detail["event_id"]
        calendar.invoke("keep_event", {"event_id": event_id})
        assert calendar.events[0]["origin"] == "overheard"

    def test_an_unknown_entry_is_refused(self, calendar):
        from src.assistance.providers.local_calendar import NoSuchEntryError

        with pytest.raises(NoSuchEntryError):
            calendar.invoke("keep_event", {"event_id": "ev_0099"})


class TestTentativeInListings:
    def test_a_reply_marks_it(self, calendar):
        _pencil(calendar)
        assert "(tentative)" in _window(calendar).message

    def test_the_detail_names_the_tentative_entries(self, calendar):
        _schedule(calendar)
        _pencil(calendar)
        assert _window(calendar).detail["tentative_ids"] == "ev_0002"

    def test_the_detail_carries_what_was_heard(self, calendar):
        _pencil(calendar)
        assert _window(calendar).detail["heard_ev_0001"] == "dentist on thursday at four"

    def test_a_firm_entry_is_not_marked(self, calendar):
        _schedule(calendar)
        detail = _window(calendar).detail
        assert detail["tentative_ids"] == "" and "(tentative)" not in _window(calendar).message
