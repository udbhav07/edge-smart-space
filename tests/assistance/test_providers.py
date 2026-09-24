"""Unit tests for the two providers behind the declared tools.

The calendar is real and says so; the travel provider is a mock and says so.
That pair of claims is the thing FR-55 exists to protect, so both halves are
asserted rather than assumed.
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.assistance.providers.local_calendar import (
    MAX_EVENTS_RETURNED,
    CalendarError,
    LocalCalendar,
)
from src.assistance.providers.mock_travel import (
    MAX_NIGHTS,
    MockTravel,
    TravelRequestError,
)
from src.common.clock import SimClock

#: SimClock's epoch sits in 2025, so a "future" date has to be after it.
FUTURE = "2026-12-01T09:00:00"
PAST = "2020-01-01T09:00:00"


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="calendar")
def _calendar(tmp_path, clock) -> LocalCalendar:
    return LocalCalendar(path=tmp_path / "calendar.json", clock=clock)


@pytest.fixture(name="travel")
def _travel(clock) -> MockTravel:
    return MockTravel(clock=clock)


class TestTheCalendarIsReal:
    def test_it_does_not_claim_to_be_simulated(self, calendar):
        """FR-55 is about mocks saying so. A real provider claiming to be a
        mock would make the flag mean nothing in either direction."""
        assert calendar.simulated is False

    def test_an_entry_survives_being_written(self, calendar):
        calendar.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T15:00:00", "subject": "design review"},
        )
        assert calendar.path.is_file()

    def test_an_entry_is_read_back(self, calendar):
        calendar.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T15:00:00", "subject": "design review"},
        )
        outcome = calendar.invoke(
            "get_events",
            {
                "from_time": "2026-10-01T00:00:00",
                "to_time": "2026-10-02T00:00:00",
            },
        )
        assert "design review" in outcome.message

    def test_a_second_reader_sees_what_the_first_wrote(self, tmp_path, clock):
        """The file is read on every call rather than cached, because another
        process may have changed it."""
        first = LocalCalendar(path=tmp_path / "c.json", clock=clock)
        first.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T15:00:00", "subject": "standup"},
        )
        second = LocalCalendar(path=tmp_path / "c.json", clock=clock)
        outcome = second.invoke(
            "get_events",
            {"from_time": "2026-10-01T00:00:00", "to_time": "2026-10-02T00:00:00"},
        )
        assert "standup" in outcome.message

    def test_the_outcome_names_the_entry_it_made(self, calendar):
        outcome = calendar.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T15:00:00", "subject": "design review"},
        )
        assert outcome.detail["event_id"]


class TestReadingBack:
    def _add(self, calendar, when: str, subject: str) -> None:
        calendar.invoke(
            "schedule_event", {"starts_at": when, "subject": subject}
        )

    def test_an_empty_window_says_so_rather_than_failing(self, calendar):
        outcome = calendar.invoke(
            "get_events",
            {"from_time": "2026-10-01T00:00:00", "to_time": "2026-10-02T00:00:00"},
        )
        assert outcome.detail["count"] == 0

    def test_only_entries_inside_the_window_are_returned(self, calendar):
        self._add(calendar, "2026-10-01T15:00:00", "inside")
        self._add(calendar, "2026-11-01T15:00:00", "outside")
        outcome = calendar.invoke(
            "get_events",
            {"from_time": "2026-10-01T00:00:00", "to_time": "2026-10-02T00:00:00"},
        )
        assert outcome.detail["count"] == 1
        assert "outside" not in outcome.message

    def test_entries_come_back_in_order(self, calendar):
        self._add(calendar, "2026-10-01T16:00:00", "later")
        self._add(calendar, "2026-10-01T09:00:00", "earlier")
        outcome = calendar.invoke(
            "get_events",
            {"from_time": "2026-10-01T00:00:00", "to_time": "2026-10-02T00:00:00"},
        )
        assert outcome.message.index("earlier") < outcome.message.index("later")

    def test_a_backwards_window_is_refused(self, calendar):
        with pytest.raises(CalendarError):
            calendar.invoke(
                "get_events",
                {
                    "from_time": "2026-10-02T00:00:00",
                    "to_time": "2026-10-01T00:00:00",
                },
            )

    def test_a_huge_window_is_bounded(self, calendar):
        """A read returning four hundred entries is answered by a model that
        then says 'several'."""
        for hour in range(MAX_EVENTS_RETURNED + 5):
            self._add(calendar, f"2026-10-01T{hour % 24:02d}:00:00", f"e{hour}")
        outcome = calendar.invoke(
            "get_events",
            {"from_time": "2026-10-01T00:00:00", "to_time": "2026-10-02T00:00:00"},
        )
        assert outcome.detail["returned"] == MAX_EVENTS_RETURNED
        assert outcome.detail["count"] > MAX_EVENTS_RETURNED


class TestClashes:
    def test_an_overlap_is_reported_not_refused(self, calendar):
        """It is the occupant's calendar. Refusing would be the system
        quietly deciding it knew better."""
        calendar.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T15:00:00", "subject": "first"},
        )
        outcome = calendar.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T15:30:00", "subject": "second"},
        )
        assert "overlaps" in outcome.message
        assert "first" in outcome.message

    def test_a_non_overlapping_entry_says_nothing_about_clashes(self, calendar):
        calendar.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T09:00:00", "subject": "first"},
        )
        outcome = calendar.invoke(
            "schedule_event",
            {"starts_at": "2026-10-01T15:00:00", "subject": "second"},
        )
        assert "overlaps" not in outcome.message


class TestCalendarFailures:
    def test_a_corrupt_file_is_raised_rather_than_discarded(self, tmp_path, clock):
        """Starting again from empty would throw away somebody's appointments
        and report success."""
        path = tmp_path / "calendar.json"
        path.write_text("{not json", encoding="utf-8")
        calendar = LocalCalendar(path=path, clock=clock)
        with pytest.raises(CalendarError):
            calendar.invoke(
                "get_events",
                {
                    "from_time": "2026-10-01T00:00:00",
                    "to_time": "2026-10-02T00:00:00",
                },
            )

    def test_a_file_holding_the_wrong_shape_is_refused(self, tmp_path, clock):
        path = tmp_path / "calendar.json"
        path.write_text('{"events": []}', encoding="utf-8")
        with pytest.raises(CalendarError):
            LocalCalendar(path=path, clock=clock).invoke(
                "get_events",
                {
                    "from_time": "2026-10-01T00:00:00",
                    "to_time": "2026-10-02T00:00:00",
                },
            )

    def test_a_missing_file_is_an_empty_calendar_not_an_error(self, calendar):
        outcome = calendar.invoke(
            "get_events",
            {"from_time": "2026-10-01T00:00:00", "to_time": "2026-10-02T00:00:00"},
        )
        assert outcome.detail["count"] == 0

    def test_an_unserved_tool_is_refused(self, calendar):
        with pytest.raises(CalendarError):
            calendar.invoke("book_travel", {})


class TestTheTravelProviderIsAMock:
    def test_it_says_it_is_simulated(self, travel):
        """Read off the provider by the registry, so a result cannot be
        published without carrying it."""
        assert travel.simulated is True

    def test_every_message_says_nothing_was_booked(self, travel):
        outcome = travel.invoke(
            "book_travel",
            {
                "kind": "flight",
                "destination": "Delhi",
                "depart_on": FUTURE,
                "origin": "Hyderabad",
            },
        )
        assert "Simulated" in outcome.message

    def test_the_reference_is_unmistakably_fake(self, travel):
        """A reference that looked like an airline's would end up in
        somebody's email."""
        outcome = travel.invoke(
            "book_travel",
            {
                "kind": "flight",
                "destination": "Delhi",
                "depart_on": FUTURE,
                "origin": "Hyderabad",
            },
        )
        assert str(outcome.detail["reference"]).startswith("MOCK")

    def test_a_hotel_is_described_in_nights(self, travel):
        outcome = travel.invoke(
            "book_travel",
            {
                "kind": "hotel",
                "destination": "Delhi",
                "depart_on": FUTURE,
                "nights": 3,
            },
        )
        assert outcome.detail["nights"] == 3


class TestTheMockStillRefusesNonsense:
    """A mock that accepted anything would make the confirmation flow look
    like it worked while proving nothing about the arguments."""

    def test_a_flight_with_no_origin_is_refused(self, travel):
        """A flight from the wrong airport is worse than a question, and
        inventing one would hide a gap in what was extracted."""
        with pytest.raises(TravelRequestError):
            travel.invoke(
                "book_travel",
                {"kind": "flight", "destination": "Delhi", "depart_on": FUTURE},
            )

    def test_a_departure_in_the_past_is_refused(self, travel):
        with pytest.raises(TravelRequestError):
            travel.invoke(
                "book_travel",
                {
                    "kind": "flight",
                    "destination": "Delhi",
                    "depart_on": PAST,
                    "origin": "Hyderabad",
                },
            )

    def test_the_site_offset_decides_what_is_in_the_past(self):
        """A node provisioned in UTC must judge "now" in the room's frame.

        Simulated now is 2025-08-24 10:40 UTC, which is 16:10 in India. A
        flight at 14:00 local has already left in India and has not in UTC.
        """
        clock = SimClock()
        flight = {
            "kind": "flight",
            "destination": "Delhi",
            "depart_on": "2025-08-24T14:00:00",
            "origin": "Hyderabad",
        }
        MockTravel(clock=clock, utc_offset_h=0.0).invoke("book_travel", flight)
        with pytest.raises(TravelRequestError):
            MockTravel(clock=clock, utc_offset_h=5.5).invoke("book_travel", flight)

    def test_an_absurd_stay_is_refused(self, travel):
        with pytest.raises(TravelRequestError):
            travel.invoke(
                "book_travel",
                {
                    "kind": "hotel",
                    "destination": "Delhi",
                    "depart_on": FUTURE,
                    "nights": MAX_NIGHTS + 1,
                },
            )

    def test_a_stay_of_no_nights_is_refused(self, travel):
        with pytest.raises(TravelRequestError):
            travel.invoke(
                "book_travel",
                {
                    "kind": "hotel",
                    "destination": "Delhi",
                    "depart_on": FUTURE,
                    "nights": 0,
                },
            )

    def test_an_unparseable_date_is_refused(self, travel):
        with pytest.raises(TravelRequestError):
            travel.invoke(
                "book_travel",
                {
                    "kind": "flight",
                    "destination": "Delhi",
                    "depart_on": "next Tuesday",
                    "origin": "Hyderabad",
                },
            )

    def test_an_unserved_tool_is_refused(self, travel):
        with pytest.raises(TravelRequestError):
            travel.invoke("schedule_event", {})
