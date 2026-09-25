"""Unit tests for the goal path (FR-53) and proposer arbitration."""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.schemas import (
    ReasonCode,
    ValidationVerdict,
    Verdict,
    Comfort,
    Goal,
    GoalSource,
    Intent,
    Mode,
    PreferenceHint,
)
from src.control.goal_manager import GoalManager

IN_FORCE_C = 24.0


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml")).goals


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="manager")
def _manager(config, clock) -> GoalManager:
    return GoalManager(config, clock)


def _hint(clock, **fields) -> PreferenceHint:
    values = {
        "intent": Intent.ENVIRONMENT,
        "comfort": Comfort.UNCHANGED,
        "subject": "temperature",
        "rationale": "asked",
        "transcript": "make it 22 degrees",
        **fields,
    }
    return PreferenceHint(ts=clock.now(), **values)


def _supervisor_goal(clock, setpoint_c=25.0) -> Goal:
    return Goal(
        ts=clock.now(),
        source=GoalSource.SUPERVISOR,
        setpoint_c=setpoint_c,
        mode=Mode.NORMAL,
        rationale="cycle",
        expires_ts=clock.now() + 600.0,
    )


class TestPreferencesBecomeProposals:
    def test_a_named_temperature_is_proposed_as_named(self, manager, clock):
        goal = manager.from_preference(_hint(clock, target_c=22.0), IN_FORCE_C, Mode.NORMAL)
        assert goal.setpoint_c == 22.0

    def test_an_unsafe_temperature_is_still_proposed_as_named(self, manager, clock):
        """The validator decides what 5 C becomes, not the goal path."""
        goal = manager.from_preference(_hint(clock, target_c=5.0), IN_FORCE_C, Mode.NORMAL)
        assert goal.setpoint_c == 5.0

    def test_cooler_moves_the_setpoint_in_force_down_one_step(
        self, manager, clock, config
    ):
        goal = manager.from_preference(
            _hint(clock, comfort=Comfort.COOLER), IN_FORCE_C, Mode.NORMAL
        )
        assert goal.setpoint_c == IN_FORCE_C - config.comfort_step_c

    def test_warmer_moves_it_up_one_step(self, manager, clock, config):
        goal = manager.from_preference(
            _hint(clock, comfort=Comfort.WARMER), IN_FORCE_C, Mode.NORMAL
        )
        assert goal.setpoint_c == IN_FORCE_C + config.comfort_step_c

    def test_a_named_temperature_wins_over_a_direction(self, manager, clock):
        goal = manager.from_preference(
            _hint(clock, comfort=Comfort.COOLER, target_c=22.0), IN_FORCE_C, Mode.NORMAL
        )
        assert goal.setpoint_c == 22.0

    def test_the_proposal_is_marked_as_a_preference(self, manager, clock):
        goal = manager.from_preference(_hint(clock, target_c=22.0), IN_FORCE_C, Mode.NORMAL)
        assert goal.source is GoalSource.PREFERENCE

    def test_the_rationale_is_what_was_said(self, manager, clock):
        goal = manager.from_preference(_hint(clock, target_c=22.0), IN_FORCE_C, Mode.NORMAL)
        assert goal.rationale == "make it 22 degrees"

    def test_the_proposal_expires_after_the_configured_ttl(self, manager, clock, config):
        goal = manager.from_preference(_hint(clock, target_c=22.0), IN_FORCE_C, Mode.NORMAL)
        assert goal.expires_ts == clock.now() + config.preference_ttl_s

    def test_nothing_asked_proposes_nothing(self, manager, clock):
        assert manager.from_preference(_hint(clock), IN_FORCE_C, Mode.NORMAL) is None

    def test_a_service_request_proposes_nothing(self, manager, clock):
        hint = _hint(clock, intent=Intent.SERVICE, subject="booking")
        assert manager.from_preference(hint, IN_FORCE_C, Mode.NORMAL) is None

    def test_a_request_about_the_lights_proposes_nothing(self, manager, clock):
        hint = _hint(clock, subject="lights", comfort=Comfort.WARMER)
        assert manager.from_preference(hint, IN_FORCE_C, Mode.NORMAL) is None

    def test_a_blank_subject_is_read_as_the_temperature(self, manager, clock):
        hint = _hint(clock, subject="", target_c=23.0)
        assert manager.from_preference(hint, IN_FORCE_C, Mode.NORMAL).setpoint_c == 23.0


class TestArbitration:
    def test_a_supervisor_proposal_is_admitted_with_no_preference(self, manager, clock):
        assert manager.admits(_supervisor_goal(clock)) is True

    def test_a_recent_preference_holds_the_supervisor_back(self, manager, clock):
        manager.from_preference(_hint(clock, target_c=22.0), IN_FORCE_C, Mode.NORMAL)
        assert manager.admits(_supervisor_goal(clock)) is False

    def test_the_hold_lapses_after_the_configured_time(self, manager, clock, config):
        manager.from_preference(_hint(clock, target_c=22.0), IN_FORCE_C, Mode.NORMAL)
        clock.advance(config.preference_hold_s + 1.0)
        assert manager.admits(_supervisor_goal(clock)) is True

    def test_a_request_that_proposed_nothing_does_not_hold(self, manager, clock):
        manager.from_preference(_hint(clock), IN_FORCE_C, Mode.NORMAL)
        assert manager.admits(_supervisor_goal(clock)) is True

    def test_an_operator_is_never_held_back(self, manager, clock):
        manager.from_preference(_hint(clock, target_c=22.0), IN_FORCE_C, Mode.NORMAL)
        operator = _supervisor_goal(clock).model_copy(update={"source": GoalSource.OPERATOR})
        assert manager.admits(operator) is True


def _verdict(clock, reason: ReasonCode) -> ValidationVerdict:
    verdict = Verdict.ACCEPTED if reason is ReasonCode.NONE else Verdict.CLAMPED
    applied = 22.0 if reason is ReasonCode.NONE else 24.0
    return ValidationVerdict(
        ts=clock.now(),
        proposed={"setpoint_c": 22.0, "source": "preference"},
        verdict=verdict,
        reason=reason,
        applied={"setpoint_c": applied},
    )


class TestPursuit:
    """A part-granted preference is proposed again until it is settled."""

    def _asked(self, manager, clock):
        return manager.from_preference(_hint(clock, target_c=22.0), 26.0, Mode.NORMAL)

    def test_a_rate_limited_preference_is_pursued(self, manager, clock, config):
        self._asked(manager, clock)
        manager.settle(_verdict(clock, ReasonCode.RATE_LIMIT))
        clock.advance(config.pursue_interval_s)
        assert manager.next_pursuit(Mode.NORMAL).setpoint_c == 22.0

    def test_pursuit_waits_for_its_interval(self, manager, clock, config):
        self._asked(manager, clock)
        manager.settle(_verdict(clock, ReasonCode.RATE_LIMIT))
        clock.advance(config.pursue_interval_s - 1.0)
        assert manager.next_pursuit(Mode.NORMAL) is None

    def test_each_pursuit_is_a_fresh_proposal(self, manager, clock, config):
        first = self._asked(manager, clock)
        manager.settle(_verdict(clock, ReasonCode.RATE_LIMIT))
        clock.advance(config.pursue_interval_s)
        assert manager.next_pursuit(Mode.NORMAL).ts > first.ts

    @pytest.mark.parametrize(
        "reason", [ReasonCode.NONE, ReasonCode.BOUND_CLAMP, ReasonCode.STALE_GOAL]
    )
    def test_anything_but_a_rate_limit_settles_it(self, manager, clock, config, reason):
        self._asked(manager, clock)
        manager.settle(_verdict(clock, reason))
        clock.advance(config.pursue_interval_s)
        assert manager.next_pursuit(Mode.NORMAL) is None

    def test_a_lapsed_preference_is_no_longer_pursued(self, manager, clock, config):
        self._asked(manager, clock)
        manager.settle(_verdict(clock, ReasonCode.RATE_LIMIT))
        clock.advance(config.preference_ttl_s + 1.0)
        assert manager.next_pursuit(Mode.NORMAL) is None
