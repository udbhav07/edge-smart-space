"""Unit tests for the room's local wall time."""

from datetime import datetime, timezone

from src.common.localtime import local_time

#: 2026-09-24 18:30 UTC, which is midnight on the 25th in India.
EPOCH = datetime(2026, 9, 24, 18, 30, tzinfo=timezone.utc).timestamp()


class TestLocalTime:
    def test_the_offset_is_applied(self):
        assert local_time(EPOCH, 5.5) == datetime(2026, 9, 25, 0, 0)

    def test_a_zero_offset_is_utc(self):
        assert local_time(EPOCH, 0.0) == datetime(2026, 9, 24, 18, 30)

    def test_a_negative_offset_goes_back(self):
        assert local_time(EPOCH, -4.0) == datetime(2026, 9, 24, 14, 30)

    def test_the_result_is_naive(self):
        """Tool arguments are naive local times; an aware value would refuse
        to compare with them rather than answer."""
        assert local_time(EPOCH, 5.5).tzinfo is None

    def test_the_machine_zone_plays_no_part(self):
        """Same epoch, same offset, same answer, wherever this runs."""
        assert local_time(EPOCH, 5.5) == local_time(EPOCH, 5.5)
