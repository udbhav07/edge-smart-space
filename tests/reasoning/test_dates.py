"""Unit tests for resolving the day an occupant named (FR-44)."""

from datetime import date, datetime, timedelta

import pytest

from src.reasoning.dates import annotation, from_now, named_days, on_day, the_one_day

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


class TestFromNow:
    """ "In 10 minutes" is arithmetic, not a judgement."""

    @pytest.mark.parametrize(
        "said, ahead",
        [
            ("remind me to call mom in 10 minutes", timedelta(minutes=10)),
            ("in an hour", timedelta(hours=1)),
            ("ping me in half an hour", timedelta(minutes=30)),
            ("in 2 hrs", timedelta(hours=2)),
        ],
    )
    def test_the_delay_is_read(self, said, ahead):
        assert from_now(said) == ahead

    def test_no_delay_is_none(self):
        assert from_now("put a meeting in at three") is None

    def test_the_moment_is_given_to_the_model(self):
        now = datetime(2026, 10, 13, 10, 53)
        assert "in 10 minutes = 2026-10-13T11:03" in annotation("remind me in 10 minutes", now.date(), now)


class TestTimeSaid:
    """S-03: the time of day in the words, read as arithmetic."""

    @pytest.mark.parametrize("words, nearest, expected", [
        ("actually make it half past two", 14, (14, 30)),
        ("make it half past eleven", 11, (11, 30)),
        ("move it to four", 15, (16, 0)),
        ("move it to four", 9, (4, 0)),
        ("change it to 10:30", 9, (10, 30)),
        ("make it quarter to five", 16, (16, 45)),
        ("push it to 6 pm", 9, (18, 0)),
        ("make it 7 am", 19, (7, 0)),
        ("at noon please", 9, (12, 0)),
        ("make it 18:15", 9, (18, 15)),
    ])
    def test_it_is_read(self, words, nearest, expected):
        from src.reasoning.dates import time_said

        assert time_said(words).nearest_to(nearest) == expected

    @pytest.mark.parametrize("words", ["make it warmer", "cancel it", "what is on Thursday", "make it 22 degrees"])
    def test_no_time_is_none(self, words):
        from src.reasoning.dates import time_said

        assert time_said(words) is None
