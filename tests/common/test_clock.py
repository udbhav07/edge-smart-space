"""Unit tests for the injectable time source."""

import threading
import time

import pytest

from src.common.clock import DEFAULT_SIM_EPOCH_S, Clock, RealClock, SimClock

SECONDS_PER_DAY = 86400.0
SHORT_SLEEP_S = 0.01
WALL_CLOCK_TOLERANCE_S = 1.0
NEGATIVE_DURATION_S = -0.001

CLOCK_FACTORIES = [RealClock, SimClock]


@pytest.mark.parametrize("make_clock", CLOCK_FACTORIES)
class TestClockContract:
    """Behaviour every implementation must exhibit (Liskov substitutability).

    ``isinstance(x, Clock)`` is deliberately not the test here. Clock is a
    runtime-checkable Protocol, so isinstance only confirms that three
    correctly-named methods exist: an object whose ``now()`` returns a string
    passes it. These tests assert the contract the docstring actually
    promises, so a substituted implementation cannot satisfy the type and
    still break its callers.
    """

    def test_now_returns_epoch_seconds_as_a_float(self, make_clock):
        assert isinstance(make_clock().now(), float)

    def test_monotonic_returns_seconds_as_a_float(self, make_clock):
        assert isinstance(make_clock().monotonic(), float)

    def test_monotonic_never_decreases_across_a_sleep(self, make_clock):
        clock = make_clock()
        before = clock.monotonic()
        clock.sleep(SHORT_SLEEP_S)
        assert clock.monotonic() >= before

    def test_sleep_advances_monotonic_time(self, make_clock):
        clock = make_clock()
        before = clock.monotonic()
        clock.sleep(SHORT_SLEEP_S)
        assert clock.monotonic() > before

    def test_now_advances_across_a_sleep(self, make_clock):
        clock = make_clock()
        before = clock.now()
        clock.sleep(SHORT_SLEEP_S)
        assert clock.now() > before

    def test_sleep_rejects_a_negative_duration(self, make_clock):
        with pytest.raises(ValueError):
            make_clock().sleep(NEGATIVE_DURATION_S)

    def test_sleep_accepts_a_zero_duration(self, make_clock):
        make_clock().sleep(0.0)

    def test_satisfies_the_clock_protocol_structurally(self, make_clock):
        assert isinstance(make_clock(), Clock)


class TestRealClock:
    def test_now_tracks_wall_clock_epoch(self):
        assert abs(RealClock().now() - time.time()) < WALL_CLOCK_TOLERANCE_S

    def test_sleep_actually_blocks_for_the_requested_duration(self):
        """NFR-01 measures loop period against real elapsed time."""
        clock = RealClock()
        wall_before = time.monotonic()
        clock.sleep(SHORT_SLEEP_S)
        assert time.monotonic() - wall_before >= SHORT_SLEEP_S


class TestSimClock:
    def test_starts_at_the_documented_default_epoch(self):
        assert SimClock().now() == DEFAULT_SIM_EPOCH_S

    def test_starts_at_a_caller_supplied_epoch(self):
        assert SimClock(start_epoch_s=0.0).now() == 0.0

    def test_monotonic_starts_at_zero_regardless_of_epoch(self):
        assert SimClock(start_epoch_s=DEFAULT_SIM_EPOCH_S).monotonic() == 0.0

    def test_advance_moves_wall_clock_time_forward(self):
        clock = SimClock(start_epoch_s=0.0)
        clock.advance(SECONDS_PER_DAY)
        assert clock.now() == SECONDS_PER_DAY

    def test_advance_moves_monotonic_time_by_the_same_amount(self):
        clock = SimClock()
        before = clock.monotonic()
        clock.advance(SECONDS_PER_DAY)
        assert clock.monotonic() - before == SECONDS_PER_DAY

    def test_sleep_advances_virtual_time(self):
        clock = SimClock(start_epoch_s=0.0)
        clock.sleep(SECONDS_PER_DAY)
        assert clock.now() == SECONDS_PER_DAY

    def test_sleep_does_not_block_the_calling_thread(self):
        """A simulated day must cost negligible wall-clock time (E1)."""
        clock = SimClock()
        wall_before = time.monotonic()
        clock.sleep(SECONDS_PER_DAY)
        assert time.monotonic() - wall_before < WALL_CLOCK_TOLERANCE_S

    def test_advance_rejects_a_negative_duration(self):
        with pytest.raises(ValueError):
            SimClock().advance(NEGATIVE_DURATION_S)

    def test_advance_accepts_a_zero_duration(self):
        clock = SimClock(start_epoch_s=0.0)
        clock.advance(0.0)
        assert clock.now() == 0.0


class TestWaitFor:
    """Waiting for another thread's answer, without a direct time call."""

    def test_a_set_event_returns_at_once_on_the_real_clock(self):
        event = threading.Event()
        event.set()
        assert RealClock().wait_for(event, 5.0) is True

    def test_an_unset_event_times_out_on_the_real_clock(self):
        assert RealClock().wait_for(threading.Event(), 0.01) is False

    def test_the_sim_clock_reports_a_set_event(self):
        event = threading.Event()
        event.set()
        assert SimClock().wait_for(event, 5.0) is True

    def test_the_sim_clock_never_blocks_on_an_unset_event(self):
        """A missing answer must not cost a batch run real seconds."""
        started = RealClock().monotonic()
        assert SimClock().wait_for(threading.Event(), 30.0) is False
        assert RealClock().monotonic() - started < 1.0

    def test_waiting_does_not_move_simulated_time(self):
        clock = SimClock()
        before = clock.now()
        clock.wait_for(threading.Event(), 30.0)
        assert clock.now() == before

    @pytest.mark.parametrize("make_clock", [RealClock, SimClock])
    def test_a_negative_timeout_is_refused(self, make_clock):
        with pytest.raises(ValueError):
            make_clock().wait_for(threading.Event(), -1.0)
