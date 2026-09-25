"""Unit tests for the mock travel provider (FR-55)."""

import pytest

from src.assistance.providers.mock_travel import MockTravel

FLIGHT = {"kind": "flight", "destination": "Delhi", "depart_on": "2026-10-01T09:00:00"}


def test_it_declares_itself_a_mock():
    assert MockTravel().simulated is True


def test_every_reply_says_it_was_a_mock():
    message = MockTravel().invoke("book_travel", FLIGHT).message
    assert "no real reservation was made" in message


def test_the_reference_is_marked_as_a_mock():
    assert MockTravel().invoke("book_travel", FLIGHT).detail["reference"] == "MOCK-0001"


def test_references_are_not_reused():
    travel = MockTravel()
    travel.invoke("book_travel", FLIGHT)
    assert travel.invoke("book_travel", FLIGHT).detail["reference"] == "MOCK-0002"


def test_a_hotel_is_described_as_a_hotel():
    hotel = {"kind": "hotel", "destination": "Goa", "depart_on": "2026-10-01T14:00:00"}
    assert "hotel in Goa" in MockTravel().invoke("book_travel", hotel).message


def test_it_refuses_a_tool_it_does_not_serve():
    with pytest.raises(ValueError):
        MockTravel().invoke("schedule_event", {})
