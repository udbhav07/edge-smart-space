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
