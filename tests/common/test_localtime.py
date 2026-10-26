"""Unit tests for the room's one conversion from epoch to wall time."""

from datetime import datetime, timezone

import pytest

from src.common.localtime import epoch_of, local_time

EPOCH_MIDNIGHT_UTC = datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()


def test_zero_offset_is_utc():
    assert local_time(EPOCH_MIDNIGHT_UTC, 0.0) == datetime(2026, 10, 1, 0, 0)


def test_a_fractional_offset_is_honoured():
    assert local_time(EPOCH_MIDNIGHT_UTC, 5.5) == datetime(2026, 10, 1, 5, 30)


def test_a_negative_offset_crosses_back_a_day():
    assert local_time(EPOCH_MIDNIGHT_UTC, -3.0) == datetime(2026, 9, 30, 21, 0)


def test_the_result_is_naive():
    assert local_time(EPOCH_MIDNIGHT_UTC, 5.5).tzinfo is None


@pytest.mark.parametrize("offset_h", [-12.0, 0.0, 5.5, 14.0])
def test_epoch_of_inverts_local_time(offset_h):
    assert epoch_of(local_time(EPOCH_MIDNIGHT_UTC, offset_h), offset_h) == EPOCH_MIDNIGHT_UTC


def test_epoch_of_refuses_an_aware_value():
    with pytest.raises(ValueError):
        epoch_of(datetime(2026, 10, 1, tzinfo=timezone.utc), 5.5)
