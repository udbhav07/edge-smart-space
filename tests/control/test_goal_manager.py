"""Unit tests for goal arbitration.

Three things carry the weight: an occupant outranks a model, an expired
proposal is withdrawn rather than weakened, and a hint that was not about
temperature never becomes one. The arbiter holds no blackboard, so these call
it directly; what reaches the gate is tested through the service in
``test_service.py``.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.common.schemas import (
    Comfort,
    Goal,
    GoalSource,
    Intent,
    Mode,
    PreferenceHint,
)
from src.control.goal_manager import GoalManager


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="manager")
def _manager(config, clock) -> GoalManager:
    return GoalManager(config, clock)


def _goal(clock, source: GoalSource, setpoint_c: float, expires_in_s=600.0):
    return Goal(
        ts=clock.now(),
        source=source,
        setpoint_c=setpoint_c,
        mode=Mode.NORMAL,
        rationale=f"{source.value} asked",
        expires_ts=clock.now() + expires_in_s,
    )


def _hint(clock, **overrides) -> PreferenceHint:
    return PreferenceHint(
        **{
            "ts": clock.now(),
            "intent": Intent.ENVIRONMENT,
            "comfort": Comfort.COOLER,
            "subject": "temperature",
            "rationale": "it is too warm in here",
            "spoken_reply": "I have passed that on.",
            **overrides,
        }
    )


class TestWithNothingProposed:
    def test_the_configured_default_stands(self, manager, config):
        """FR-11: a valid setpoint is held when the layers above are gone."""
        assert manager.setpoint_c == config.controller.default_setpoint_c

    def test_the_default_is_named_as_the_source(self, manager):
        assert manager.winning_source is GoalSource.DEFAULT

    def test_expiring_with_nothing_proposed_hands_on_nothing(self, manager):
        assert manager.expire() is None


class TestAuthority:
    def test_an_occupant_outranks_a_supervisor(self, manager, clock):
        """The supervisor optimises on that person's behalf, so overriding
        them would answer a question nobody asked."""
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        assert manager.setpoint_c == 23.0

    def test_order_of_arrival_does_not_change_who_wins(self, manager, clock):
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        assert manager.setpoint_c == 23.0

    def test_an_operator_outranks_an_occupant(self, manager, clock):
        """An operator is running a demonstration and needs the room to do
        what they said."""
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        manager.propose(_goal(clock, GoalSource.OPERATOR, 27.0))
        assert manager.setpoint_c == 27.0

    def test_a_later_proposal_from_one_source_replaces_its_earlier_one(
        self, manager, clock
    ):
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        clock.advance(60.0)
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 25.0))
        assert manager.setpoint_c == 25.0

    def test_any_source_outranks_the_default(self, manager, clock, config):
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        assert manager.setpoint_c != config.controller.default_setpoint_c


class TestExpiry:
    def test_an_expired_proposal_stops_winning(self, manager, clock):
        """A reasoning layer that has since crashed must not keep steering
        the room."""
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=60.0))
        clock.advance(120.0)
        assert manager.winning_source is GoalSource.DEFAULT

    def test_an_expired_proposal_yields_to_a_lesser_live_one(self, manager, clock):
        manager.propose(_goal(clock, GoalSource.OPERATOR, 27.0, expires_in_s=60.0))
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=600.0))
        clock.advance(120.0)
        assert manager.setpoint_c == 26.0

    def test_expiring_hands_on_the_new_winner(self, manager, clock):
        manager.propose(_goal(clock, GoalSource.OPERATOR, 27.0, expires_in_s=60.0))
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=600.0))
        clock.advance(120.0)
        successor = manager.expire()
        assert successor is not None and successor.setpoint_c == 26.0

    def test_expiring_nothing_hands_on_nothing(self, manager, clock):
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        assert manager.expire() is None

    def test_everything_expiring_returns_to_the_default(self, manager, clock, config):
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=60.0))
        clock.advance(120.0)
        manager.expire()
        assert manager.setpoint_c == config.controller.default_setpoint_c


class TestWhatReachesTheGate:
    def test_a_winning_proposal_is_handed_on(self, manager, clock):
        winner = manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        assert winner is not None and winner.setpoint_c == 23.0

    def test_a_losing_proposal_is_not(self, manager, clock):
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        assert manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0)) is None

    def test_an_unchanged_setpoint_is_not_handed_on_again(self, manager, clock):
        """Gating it again would reset the validator's rate limit against a
        setpoint nobody moved."""
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        clock.advance(60.0)
        assert manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0)) is None

    def test_an_out_of_bounds_proposal_is_still_handed_on_for_the_gate(
        self, manager, clock
    ):
        """The validator clamps and says why. Filtering here would remove the
        evidence that the gate works."""
        winner = manager.propose(_goal(clock, GoalSource.PREFERENCE, 5.0))
        assert winner is not None and winner.setpoint_c == 5.0


class TestFromSpeech:
    """FR-53: what an occupant said, turned into something the gate can see."""

    def test_a_named_temperature_is_taken_as_given(self, manager, clock):
        winner = manager.consider(_hint(clock, target_c=22.0))
        assert winner.setpoint_c == 22.0

    def test_cooler_moves_the_setpoint_down_a_step(self, manager, clock, config):
        before = manager.setpoint_c
        winner = manager.consider(_hint(clock))
        assert winner.setpoint_c == pytest.approx(
            before - config.controller.comfort_step_c
        )

    def test_warmer_moves_it_up(self, manager, clock, config):
        before = manager.setpoint_c
        winner = manager.consider(_hint(clock, comfort=Comfort.WARMER))
        assert winner.setpoint_c == pytest.approx(
            before + config.controller.comfort_step_c
        )

    def test_asking_twice_moves_it_twice(self, manager, clock, config):
        """The step applies to what is in force, not to a constant."""
        start = manager.setpoint_c
        for _ in range(2):
            clock.advance(10.0)
            manager.consider(_hint(clock))
        assert manager.setpoint_c == pytest.approx(
            start - 2 * config.controller.comfort_step_c
        )

    def test_a_request_about_something_else_is_not_a_temperature(
        self, manager, clock
    ):
        """Reading every hint as a setpoint would turn 'put that in my
        calendar' into a goal."""
        hint = _hint(
            clock,
            intent=Intent.SERVICE,
            comfort=Comfort.UNCHANGED,
            subject="calendar",
            target_c=None,
        )
        assert manager.consider(hint) is None

    def test_a_hint_asking_for_nothing_proposes_nothing(self, manager, clock):
        hint = _hint(
            clock, intent=Intent.NONE, comfort=Comfort.UNCHANGED, target_c=None
        )
        assert manager.consider(hint) is None

    def test_a_spoken_request_is_attributed_to_the_occupant(self, manager, clock):
        assert manager.consider(_hint(clock)).source is GoalSource.PREFERENCE

    def test_the_occupants_words_survive_as_the_rationale(self, manager, clock):
        """A clamped proposal is a finding to display, and the display needs
        to say what was asked for."""
        assert "too warm" in manager.consider(_hint(clock)).rationale

    def test_a_spoken_request_carries_the_mode_in_force(self, manager, clock):
        manager.observe_mode(Mode.DEGRADED_SENSOR)
        assert manager.consider(_hint(clock)).mode is Mode.DEGRADED_SENSOR
