"""Unit tests for the tariff schedule (FR-16).

Timestamps are built from local wall-clock times, because the schedule is
defined in local hours; a fixed epoch would pass in one time zone and fail in
another.
"""

from datetime import datetime

import pytest

from src.common.config import TariffConfig
from src.common.schemas import TariffBand
from src.common.tariff import TariffSchedule


def _at(hour: int, minute: int = 0, day: int = 1) -> float:
    return datetime(2026, 10, day, hour, minute).timestamp()


@pytest.fixture(name="schedule")
def _schedule() -> TariffSchedule:
    return TariffSchedule(TariffConfig(peak_start_hour=18, peak_end_hour=22, peak_offset_c=1.0))


class TestBands:
    def test_the_afternoon_is_normal(self, schedule):
        assert schedule.band_at(_at(15)) is TariffBand.NORMAL

    def test_the_start_hour_is_peak(self, schedule):
        assert schedule.band_at(_at(18)) is TariffBand.PEAK

    def test_the_last_minute_of_peak_is_peak(self, schedule):
        assert schedule.band_at(_at(21, 59)) is TariffBand.PEAK

    def test_the_end_hour_is_normal_again(self, schedule):
        assert schedule.band_at(_at(22)) is TariffBand.NORMAL


class TestTransitions:
    def test_before_peak_the_next_change_is_its_start(self, schedule):
        assert schedule.next_transition_ts(_at(15)) == _at(18)

    def test_during_peak_the_next_change_is_its_end(self, schedule):
        assert schedule.next_transition_ts(_at(19)) == _at(22)

    def test_after_peak_the_next_change_is_tomorrow(self, schedule):
        assert schedule.next_transition_ts(_at(23)) == _at(18, day=2)

    def test_the_state_carries_band_and_transition(self, schedule):
        state = schedule.state(_at(19))
        assert state.band is TariffBand.PEAK
        assert state.next_transition_ts == _at(22)


class TestConfiguration:
    def test_a_peak_ending_before_it_starts_is_refused(self):
        with pytest.raises(ValueError):
            TariffConfig(peak_start_hour=20, peak_end_hour=18, peak_offset_c=1.0)
