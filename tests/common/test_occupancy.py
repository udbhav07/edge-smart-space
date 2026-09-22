"""Unit tests for occupancy derivation (FR-02).

The boundary that matters is the hold-off: a person who stops moving must stay
detected, and an empty room must eventually be reported empty.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.occupancy import OccupancyTracker

HOLD_OFF_S = 600.0


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="tracker")
def _tracker(clock) -> OccupancyTracker:
    return OccupancyTracker(hold_off_s=HOLD_OFF_S, clock=clock)


class TestBeforeAnyEvidence:
    def test_a_room_nothing_has_been_seen_in_is_empty(self, tracker):
        """The conservative-to-occupied rule is about losing evidence, not
        about never having had any: otherwise the system would cool an empty
        building from the moment it booted."""
        assert not tracker.occupied

    def test_it_stays_empty_however_long_nothing_happens(self, tracker, clock):
        clock.advance(HOLD_OFF_S * 10)
        assert not tracker.occupied

    def test_nothing_is_reported_as_quiet_time_yet(self, tracker, clock):
        clock.advance(60.0)
        assert tracker.quiet_for_s == 0.0


class TestTheHoldOff:
    def test_motion_makes_the_room_occupied(self, tracker):
        tracker.motion()
        assert tracker.occupied

    def test_a_person_who_stops_moving_stays_detected(self, tracker, clock):
        """A PIR reports motion, not presence. Somebody reading quietly sets
        off nothing."""
        tracker.motion()
        clock.advance(HOLD_OFF_S - 1.0)
        assert tracker.occupied

    def test_the_room_empties_once_the_hold_off_expires(self, tracker, clock):
        tracker.motion()
        clock.advance(HOLD_OFF_S)
        assert not tracker.occupied

    def test_fresh_motion_restarts_the_hold_off(self, tracker, clock):
        tracker.motion()
        clock.advance(HOLD_OFF_S - 1.0)
        tracker.motion()
        clock.advance(HOLD_OFF_S - 1.0)
        assert tracker.occupied

    def test_the_quiet_time_is_reported(self, tracker, clock):
        tracker.motion()
        clock.advance(120.0)
        assert tracker.quiet_for_s == pytest.approx(120.0)

    def test_a_zero_hold_off_empties_the_room_at_once(self, clock):
        """Configurable includes configurably immediate, which is raw PIR."""
        tracker = OccupancyTracker(hold_off_s=0.0, clock=clock)
        tracker.motion()
        assert not tracker.occupied


class TestTheDoor:
    def test_a_door_transition_counts_as_presence(self, tracker):
        """An opening cannot distinguish arrival from departure, and the
        conservative reading of an ambiguous signal is occupied."""
        tracker.door_transition()
        assert tracker.occupied

    def test_a_door_transition_restarts_the_hold_off(self, tracker, clock):
        tracker.motion()
        clock.advance(HOLD_OFF_S - 1.0)
        tracker.door_transition()
        clock.advance(HOLD_OFF_S - 1.0)
        assert tracker.occupied

    def test_somebody_leaving_still_holds_the_room_briefly(self, tracker, clock):
        """They may come back, and the cost of being wrong is asymmetric."""
        tracker.door_transition()
        clock.advance(HOLD_OFF_S / 2)
        assert tracker.occupied


class TestConstruction:
    def test_a_negative_hold_off_is_refused(self, clock):
        with pytest.raises(ValueError):
            OccupancyTracker(hold_off_s=-1.0, clock=clock)

    def test_the_configured_hold_off_is_reported(self, tracker):
        assert tracker.hold_off_s == HOLD_OFF_S

    def test_the_shipped_default_is_the_documented_ten_minutes(self):
        """FR-02 names 10 minutes as the default."""
        config = load_config(Path("config/default.yaml"))
        assert config.sensors.vacancy_hold_off_s == 600.0
