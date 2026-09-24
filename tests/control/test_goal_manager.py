"""Unit tests for goal arbitration.

Three things carry the weight: an occupant outranks a model, an expired
proposal is withdrawn rather than weakened, and a hint that was not about
temperature never becomes one.
"""

from pathlib import Path

import pytest

from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Comfort,
    Goal,
    GoalSource,
    Intent,
    Mode,
    PreferenceHint,
)
from src.control.goal_manager import GoalManager


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []

    def connect(self, host, port, keepalive):
        pass

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def proposals(self) -> list[Goal]:
        return [
            Goal.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/goal/proposed"
        ]


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="wired")
def _wired(config, clock):
    transport = FakeTransport()
    blackboard = Blackboard(config.mqtt, transport)
    manager = GoalManager(config, clock, blackboard)
    manager.subscribe()
    blackboard.on_connected()
    return manager, transport, blackboard


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
    def test_the_configured_default_stands(self, wired, config):
        """FR-11: a valid setpoint is held when the layers above are gone."""
        manager, _, _ = wired
        assert manager.setpoint_c == config.controller.default_setpoint_c

    def test_the_default_is_named_as_the_source(self, wired):
        manager, _, _ = wired
        assert manager.winning_source is GoalSource.DEFAULT

    def test_nothing_is_published_until_something_proposes(self, wired):
        _, transport, _ = wired
        assert transport.proposals() == []


class TestAuthority:
    def test_an_occupant_outranks_a_supervisor(self, wired, clock):
        """The supervisor optimises on that person's behalf, so overriding
        them would answer a question nobody asked."""
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        assert manager.setpoint_c == 23.0

    def test_order_of_arrival_does_not_change_who_wins(self, wired, clock):
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        assert manager.setpoint_c == 23.0

    def test_an_operator_outranks_an_occupant(self, wired, clock):
        """An operator is running a demonstration and needs the room to do
        what they said."""
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        manager.propose(_goal(clock, GoalSource.OPERATOR, 27.0))
        assert manager.setpoint_c == 27.0

    def test_a_later_proposal_from_one_source_replaces_its_earlier_one(
        self, wired, clock
    ):
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        clock.advance(60.0)
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 25.0))
        assert manager.setpoint_c == 25.0

    def test_any_source_outranks_the_default(self, wired, clock, config):
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        assert manager.setpoint_c != config.controller.default_setpoint_c


class TestExpiry:
    def test_an_expired_proposal_stops_winning(self, wired, clock):
        """A reasoning layer that has since crashed must not keep steering
        the room."""
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=60.0))
        clock.advance(120.0)
        assert manager.winning_source is GoalSource.DEFAULT

    def test_an_expired_proposal_yields_to_a_lesser_live_one(self, wired, clock):
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.OPERATOR, 27.0, expires_in_s=60.0))
        manager.propose(
            _goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=600.0)
        )
        clock.advance(120.0)
        assert manager.setpoint_c == 26.0

    def test_expiring_republishes_when_the_winner_changed(self, wired, clock):
        manager, transport, _ = wired
        manager.propose(_goal(clock, GoalSource.OPERATOR, 27.0, expires_in_s=60.0))
        manager.propose(
            _goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=600.0)
        )
        clock.advance(120.0)
        assert manager.expire() is not None
        assert transport.proposals()[-1].setpoint_c == 26.0

    def test_expiring_nothing_publishes_nothing(self, wired, clock):
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        assert manager.expire() is None

    def test_everything_expiring_returns_to_the_default(
        self, wired, clock, config
    ):
        manager, _, _ = wired
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0, expires_in_s=60.0))
        clock.advance(120.0)
        manager.expire()
        assert manager.setpoint_c == config.controller.default_setpoint_c


class TestPublishing:
    def test_a_winning_proposal_is_published(self, wired, clock):
        manager, transport, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        assert transport.proposals()[-1].setpoint_c == 23.0

    def test_a_losing_proposal_is_not(self, wired, clock):
        manager, transport, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        before = len(transport.proposals())
        manager.propose(_goal(clock, GoalSource.SUPERVISOR, 26.0))
        assert len(transport.proposals()) == before

    def test_an_unchanged_setpoint_is_not_republished(self, wired, clock):
        """Republishing would reset the validator's rate limit against a
        setpoint nobody moved."""
        manager, transport, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        before = len(transport.proposals())
        clock.advance(60.0)
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        assert len(transport.proposals()) == before

    def test_it_proposes_rather_than_activates(self, wired, clock):
        """Winning the argument is not the same as being allowed: the
        validator still gates it."""
        manager, transport, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 23.0))
        assert all(
            topic == "space/goal/proposed"
            for topic, _, _, _ in transport.published
        )

    def test_an_out_of_bounds_proposal_is_still_published_for_the_gate(
        self, wired, clock
    ):
        """The validator clamps and says why. Filtering here would remove the
        evidence that the gate works."""
        manager, transport, _ = wired
        manager.propose(_goal(clock, GoalSource.PREFERENCE, 5.0))
        assert transport.proposals()[-1].setpoint_c == 5.0


class TestFromSpeech:
    """FR-53: what an occupant said, turned into something the gate can see."""

    def test_a_named_temperature_is_taken_as_given(self, wired, clock):
        _, transport, blackboard = wired
        blackboard.dispatch(
            "space/context/preference",
            _hint(clock, target_c=22.0).model_dump_json().encode(),
        )
        assert transport.proposals()[-1].setpoint_c == 22.0

    def test_cooler_moves_the_setpoint_down_a_step(self, wired, clock, config):
        manager, transport, blackboard = wired
        before = manager.setpoint_c
        blackboard.dispatch(
            "space/context/preference", _hint(clock).model_dump_json().encode()
        )
        assert transport.proposals()[-1].setpoint_c == pytest.approx(
            before - config.controller.comfort_step_c
        )

    def test_warmer_moves_it_up(self, wired, clock, config):
        manager, transport, blackboard = wired
        before = manager.setpoint_c
        blackboard.dispatch(
            "space/context/preference",
            _hint(clock, comfort=Comfort.WARMER).model_dump_json().encode(),
        )
        assert transport.proposals()[-1].setpoint_c == pytest.approx(
            before + config.controller.comfort_step_c
        )

    def test_asking_twice_moves_it_twice(self, wired, clock, config):
        """The step applies to what is in force, not to a constant."""
        manager, _, blackboard = wired
        start = manager.setpoint_c
        for _ in range(2):
            clock.advance(10.0)
            blackboard.dispatch(
                "space/context/preference", _hint(clock).model_dump_json().encode()
            )
        assert manager.setpoint_c == pytest.approx(
            start - 2 * config.controller.comfort_step_c
        )

    def test_a_request_about_something_else_is_not_a_temperature(
        self, wired, clock
    ):
        """Reading every hint as a setpoint would turn 'put that in my
        calendar' into a goal."""
        _, transport, blackboard = wired
        blackboard.dispatch(
            "space/context/preference",
            _hint(
                clock,
                intent=Intent.SERVICE,
                comfort=Comfort.UNCHANGED,
                subject="calendar",
                target_c=None,
            ).model_dump_json().encode(),
        )
        assert transport.proposals() == []

    def test_a_hint_asking_for_nothing_proposes_nothing(self, wired, clock):
        _, transport, blackboard = wired
        blackboard.dispatch(
            "space/context/preference",
            _hint(
                clock,
                intent=Intent.NONE,
                comfort=Comfort.UNCHANGED,
                target_c=None,
            ).model_dump_json().encode(),
        )
        assert transport.proposals() == []

    def test_a_spoken_request_is_attributed_to_the_occupant(self, wired, clock):
        _, transport, blackboard = wired
        blackboard.dispatch(
            "space/context/preference", _hint(clock).model_dump_json().encode()
        )
        assert transport.proposals()[-1].source is GoalSource.PREFERENCE

    def test_the_occupants_words_survive_as_the_rationale(self, wired, clock):
        """A clamped proposal is a finding to display, and the display needs
        to say what was asked for."""
        _, transport, blackboard = wired
        blackboard.dispatch(
            "space/context/preference", _hint(clock).model_dump_json().encode()
        )
        assert "too warm" in transport.proposals()[-1].rationale

    def test_it_subscribes_to_what_speech_publishes(self, wired):
        _, transport, _ = wired
        patterns = {topic for topic, _ in transport.subscribed}
        assert topics.CONTEXT_PREFERENCE.pattern in patterns
