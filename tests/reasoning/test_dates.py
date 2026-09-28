"""Unit tests for resolving the day an occupant named (FR-44)."""

from datetime import date

import pytest

from src.reasoning.dates import annotation, named_days, on_day, the_one_day

SATURDAY = date(2026, 9, 26)


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("remind me tomorrow at 10", date(2026, 9, 27)),
        ("what is on today", date(2026, 9, 26)),
        ("dinner this evening at 8", date(2026, 9, 26)),
        ("gym the day after tomorrow", date(2026, 9, 28)),
        ("dentist on Tuesday at 11", date(2026, 9, 29)),
        ("flight next Monday", date(2026, 9, 28)),
        ("meeting Thursday at 3", date(2026, 10, 1)),
        ("anything on Saturday", date(2026, 9, 26)),
    ],
)
def test_the_named_day_is_resolved(utterance, expected):
    assert the_one_day(utterance, SATURDAY) == expected


def test_the_day_after_tomorrow_is_not_also_read_as_tomorrow():
    assert list(named_days("gym the day after tomorrow", SATURDAY)) == ["day after tomorrow"]


def test_no_day_named_resolves_to_nothing():
    assert the_one_day("call mom", SATURDAY) is None


def test_two_days_named_are_left_to_the_model():
    assert the_one_day("meetings Thursday and Friday", SATURDAY) is None


def test_the_annotation_lists_what_was_resolved():
    assert annotation("dentist on Tuesday", SATURDAY) == "(Dates named: Tuesday = 2026-09-29.)"


def test_no_annotation_when_no_day_is_named():
    assert annotation("call mom", SATURDAY) == ""


class TestMovingOntoTheDay:
    def test_the_date_is_replaced_and_the_time_kept(self):
        assert on_day("2026-09-28T11:00:00", date(2026, 9, 29)) == "2026-09-29T11:00:00"

    def test_a_value_already_on_that_day_is_left_alone(self):
        assert on_day("2026-09-29T11:00:00", date(2026, 9, 29)) is None

    def test_something_that_is_not_a_time_is_left_for_the_argument_check(self):
        assert on_day("next week", date(2026, 9, 29)) is None
