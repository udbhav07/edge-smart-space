"""Unit tests for the tariff schedule and its publisher (FR-16).

Times are built from a known local midnight so every assertion reads as a
clock time rather than an epoch.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.common.clock import SimClock
from src.common.config import PeakWindow, TariffConfig, load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import TariffBand, TariffState
from src.control.__main__ import run
from src.control.service import build_service
from src.control.tariff import SECONDS_PER_HOUR, TariffPublisher, TariffSchedule

#: 2026-09-25 00:00 in UTC+5:30.
LOCAL_MIDNIGHT = datetime(2026, 9, 24, 18, 30, tzinfo=timezone.utc).timestamp()
IST = 5.5


def _at(hour: float) -> float:
    return LOCAL_MIDNIGHT + hour * SECONDS_PER_HOUR


def _schedule(*windows, offset_h=IST, offset_c=1.0) -> TariffSchedule:
    return TariffSchedule(
        TariffConfig(
            peak_windows=tuple(PeakWindow(start_h=a, end_h=b) for a, b in windows),
            utc_offset_h=offset_h,
            peak_offset_c=offset_c,
        )
    )


class RecordingTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []

    def connect(self, host, port, keepalive): ...
    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))
    def subscribe(self, topic, qos): ...
    def loop_start(self): ...
    def loop_stop(self): ...
    def disconnect(self): ...

    def tariffs(self) -> list[TariffState]:
        return [
            TariffState.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/context/tariff"
        ]


class TestSchedule:
    def test_inside_a_window_is_peak(self):
        assert _schedule((18, 22)).state_at(_at(19)).band is TariffBand.PEAK

    def test_outside_every_window_is_normal(self):
        assert _schedule((18, 22)).state_at(_at(12)).band is TariffBand.NORMAL

    def test_the_start_of_a_window_is_peak(self):
        """Boundaries belong to the band they begin."""
        assert _schedule((18, 22)).state_at(_at(18)).band is TariffBand.PEAK

    def test_the_end_of_a_window_is_normal(self):
        assert _schedule((18, 22)).state_at(_at(22)).band is TariffBand.NORMAL

    def test_peak_says_when_it_ends(self):
        state = _schedule((18, 22)).state_at(_at(19))
        assert state.next_transition_ts == pytest.approx(_at(22))

    def test_peak_says_when_it_began(self):
        state = _schedule((18, 22)).state_at(_at(19))
        assert state.since_ts == pytest.approx(_at(18))

    def test_after_midnight_normal_began_yesterday_evening(self):
        """The band at 00:30 was set by a window that closed the day before."""
        state = _schedule((18, 22)).state_at(_at(0.5))
        assert state.since_ts == pytest.approx(_at(22 - 24))

    def test_late_evening_looks_ahead_to_tomorrow(self):
        state = _schedule((18, 22)).state_at(_at(23))
        assert state.next_transition_ts == pytest.approx(_at(18 + 24))

    def test_two_windows_are_both_peak(self):
        schedule = _schedule((7, 9), (18, 22))
        assert schedule.state_at(_at(8)).band is TariffBand.PEAK
        assert schedule.state_at(_at(20)).band is TariffBand.PEAK

    def test_between_two_windows_is_normal_until_the_second(self):
        state = _schedule((7, 9), (18, 22)).state_at(_at(12))
        assert state.next_transition_ts == pytest.approx(_at(18))

    def test_no_windows_is_always_normal(self):
        state = _schedule().state_at(_at(19))
        assert state.band is TariffBand.NORMAL
        assert state.next_transition_ts > state.since_ts

    def test_the_utc_offset_moves_the_window(self):
        """The same instant is 19:00 in India and 13:30 in UTC."""
        assert _schedule((18, 22), offset_h=0.0).state_at(_at(19)).band is (
            TariffBand.NORMAL
        )

    def test_the_configured_offset_rides_along(self):
        assert _schedule((18, 22), offset_c=1.5).state_at(_at(19)).offset_c == 1.5


class TestConfiguration:
    def test_a_window_that_ends_before_it_starts_is_refused(self):
        with pytest.raises(ValidationError):
            PeakWindow(start_h=22.0, end_h=18.0)

    def test_overlapping_windows_are_refused(self):
        with pytest.raises(ValidationError):
            TariffConfig(
                peak_windows=(
                    PeakWindow(start_h=17.0, end_h=20.0),
                    PeakWindow(start_h=19.0, end_h=22.0),
                )
            )

    def test_touching_windows_are_refused(self):
        """Adjacent windows are one window, written as two by mistake."""
        with pytest.raises(ValidationError):
            TariffConfig(
                peak_windows=(
                    PeakWindow(start_h=17.0, end_h=19.0),
                    PeakWindow(start_h=19.0, end_h=22.0),
                )
            )

    def test_the_shipped_config_has_an_evening_peak(self):
        config = load_config(Path("config/default.yaml"))
        assert config.tariff.peak_windows


class TestPublisher:
    def _wired(self, hour: float):
        config = load_config(Path("config/default.yaml"))
        clock = SimClock(start_epoch_s=_at(hour))
        transport = RecordingTransport()
        blackboard = Blackboard(config.mqtt, transport)
        publisher = TariffPublisher(TariffSchedule(config.tariff), clock, blackboard)
        return publisher, transport, clock

    def test_the_first_tick_publishes(self):
        publisher, transport, _ = self._wired(12)
        publisher.tick()
        assert len(transport.tariffs()) == 1

    def test_the_tariff_is_retained(self):
        """A supervisor starting mid-peak must know it is mid-peak."""
        publisher, transport, _ = self._wired(12)
        publisher.tick()
        assert transport.published[-1][3] is True

    def test_an_unchanged_band_is_not_republished(self):
        publisher, transport, clock = self._wired(12)
        publisher.tick()
        clock.advance(600.0)
        assert publisher.tick() is None
        assert len(transport.tariffs()) == 1

    def test_a_change_of_band_is_published(self):
        publisher, transport, clock = self._wired(17.9)
        publisher.tick()
        clock.advance(0.2 * SECONDS_PER_HOUR)
        publisher.tick()
        assert [t.band for t in transport.tariffs()] == [
            TariffBand.NORMAL,
            TariffBand.PEAK,
        ]

    def test_the_control_loop_publishes_it(self):
        """Published from the regulatory tick, so a band change is announced
        by the one process that must stay up."""
        config = load_config(Path("config/default.yaml"))
        clock = SimClock(start_epoch_s=_at(12))
        transport = RecordingTransport()
        blackboard = Blackboard(config.mqtt, transport)
        service = build_service(config, clock, blackboard)
        publisher = TariffPublisher(TariffSchedule(config.tariff), clock, blackboard)
        run(service, clock, config.loop.regulatory_period_s, ticks=2, tariff=publisher)
        assert transport.tariffs()
