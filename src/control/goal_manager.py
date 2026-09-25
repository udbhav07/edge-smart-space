"""Deciding whose setpoint wins (FR-40, FR-45, FR-53).

Several things may want the room at different temperatures at once: an
occupant who said it was too warm, a supervisor optimising against a tariff, an
operator during a demonstration, and the configured default underneath all of
them. Something has to choose, and it should not be the safety validator --
that gate exists to apply hard limits without knowing intent, and teaching it
whose intent to prefer would give it exactly the knowledge it is supposed to
lack.

So the arbitration happens here, and what wins is handed to the validator.
Section 4.4 draws the two as one box, "Goal Manager + Validator", subscribed to
``space/goal/proposed``: every proposal from outside -- the supervisor's, an
operator's -- arrives there, is arbitrated, and only the winner is gated. They
are two classes because they answer two questions, and one component because
there must be no topic between them: if arbitration published its winner back
onto ``space/goal/proposed``, the gate could not tell an arbitrated goal from a
raw one, and a supervisor's proposal would override an occupant simply by
arriving. That was the wiring before Week 6, and it went unnoticed only because
no supervisor existed yet.

**An occupant outranks a model.** Somebody in the room saying it is too warm
beats a supervisor's tariff optimisation, because the supervisor is optimising
on that person's behalf and a model that overrode them would be answering a
question nobody asked. An operator outranks both, because an operator is
running a demonstration and needs the room to do what they said.

**A stale proposal loses to nothing.** Goals expire (V-6), and an expired one
is not weakened but withdrawn: leaving a supervisor's hour-old proposal in the
running would let a reasoning layer that has since crashed keep steering the
room: when a more authoritative proposal expires, the next one down takes over
and is gated. When the *last* live proposal expires there is no new winner to
gate, and the gate keeps what it last admitted -- FR-11's "holding the last
valid setpoint", literally. The arbiter reports DEFAULT as the winning source
then, and the configured default is what it would hand on only if nothing had
ever been admitted.

**Speech reaches the plant through here.** A ``PreferenceHint`` is what an
occupant said, turned into structure (FR-53). It is converted into a proposal
and arbitrated like any other, then gated like any other: FR-45 holds of spoken
requests exactly as it holds of the reasoning layer.

This class holds no blackboard. The control service subscribes, feeds it, and
gates what it returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.common.clock import Clock
from src.common.config import Config
from src.common.schemas import (
    Comfort,
    Goal,
    GoalSource,
    Intent,
    Mode,
    PreferenceHint,
)

LOGGER = logging.getLogger(__name__)

#: Who wins when two sources disagree, least authoritative first. Derived from
#: whose question is being answered rather than from who is cleverest: the
#: supervisor optimises on an occupant's behalf, so the occupant outranks it,
#: and an operator running a demonstration outranks them both.
_AUTHORITY: dict[GoalSource, int] = {
    GoalSource.DEFAULT: 0,
    GoalSource.SUPERVISOR: 1,
    GoalSource.PREFERENCE: 2,
    GoalSource.OPERATOR: 3,
}


@dataclass(frozen=True)
class _Standing:
    """A proposal still in the running."""

    goal: Goal

    @property
    def authority(self) -> int:
        return _AUTHORITY[self.goal.source]


class GoalManager:
    """Holds the live proposals and says which one wins."""

    def __init__(self, config: Config, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._standing: dict[GoalSource, _Standing] = {}
        self._mode = Mode.INIT
        self._handed_on_c: float | None = None

    @property
    def setpoint_c(self) -> float:
        """The setpoint currently winning.

        Falls back to the configured default, which is the answer when every
        source has gone quiet or expired.
        """
        winner = self._winner()
        if winner is None:
            return self._config.controller.default_setpoint_c
        return winner.goal.setpoint_c

    @property
    def winning_source(self) -> GoalSource:
        """Whose proposal is in force. DEFAULT when nobody's is."""
        winner = self._winner()
        return GoalSource.DEFAULT if winner is None else winner.goal.source

    def observe_mode(self, mode: Mode) -> None:
        """Remember the mode, so a spoken request is stamped with it."""
        self._mode = mode

    # --- proposals ----------------------------------------------------

    def consider(self, hint: PreferenceHint) -> Goal | None:
        """Turn what an occupant said into a proposal, if it asked for one.

        A hint about anything other than the environment is not a setpoint
        request and is left alone: section 6.4 makes a hint able to carry a
        service request or nothing at all, and reading every hint as a
        temperature would turn "put that in my calendar" into a goal.

        :returns: the goal to gate, when the winner changed as a result.
        """
        if hint.intent is not Intent.ENVIRONMENT:
            LOGGER.debug("hint with intent %s is not a goal", hint.intent.value)
            return None

        target_c = self._target_from(hint)
        if target_c is None:
            LOGGER.info("hint %r asked for no particular temperature", hint.rationale)
            return None

        now = self._clock.now()
        return self.propose(
            Goal(
                ts=now,
                source=GoalSource.PREFERENCE,
                setpoint_c=target_c,
                mode=self._mode,
                rationale=hint.rationale or "an occupant asked",
                expires_ts=now + self._config.validator.goal_max_age_s,
            )
        )

    def _target_from(self, hint: PreferenceHint) -> float | None:
        """The temperature a hint is asking for, if it names one.

        A stated target is taken as given. A direction is applied to what is
        currently winning rather than to the configured default, because
        "cooler" means cooler than it is now -- resolving it against a
        constant would make repeating it do nothing the second time.
        """
        if hint.target_c is not None:
            return hint.target_c

        step = self._config.controller.comfort_step_c
        if hint.comfort is Comfort.COOLER:
            return self.setpoint_c - step
        if hint.comfort is Comfort.WARMER:
            return self.setpoint_c + step
        return None

    def propose(self, goal: Goal) -> Goal | None:
        """Enter a proposal.

        :returns: the winning goal when the winning setpoint changed, for the
            validator to gate; None when the outcome is unchanged. Handing on
            an unchanged setpoint every time anything spoke would reset the
            validator's rate limit against a setpoint nobody moved.
        """
        self._standing[goal.source] = _Standing(goal=goal)
        return self._changed_winner()

    def expire(self) -> Goal | None:
        """Drop proposals that have aged out.

        Called on the regulatory tick rather than only when something arrives:
        a source going silent is exactly the case where nothing arrives, and a
        supervisor that crashed mid-proposal would otherwise keep steering the
        room from beyond the grave.

        :returns: the new winner, when losing a proposal changed who wins.
        """
        now = self._clock.now()
        expired = [
            source
            for source, standing in self._standing.items()
            if standing.goal.expires_ts <= now
        ]
        for source in expired:
            LOGGER.info("%s's proposal expired", source.value)
            del self._standing[source]
        if not expired:
            return None
        return self._changed_winner()

    # --- arbitration --------------------------------------------------

    def _winner(self) -> _Standing | None:
        """The most authoritative proposal that has not expired."""
        now = self._clock.now()
        live = [
            standing
            for standing in self._standing.values()
            if standing.goal.expires_ts > now
        ]
        if not live:
            return None
        return max(live, key=lambda standing: (standing.authority, standing.goal.ts))

    def _changed_winner(self) -> Goal | None:
        winner = self._winner()
        if winner is None:
            return None
        if self._handed_on_c == winner.goal.setpoint_c:
            return None
        self._handed_on_c = winner.goal.setpoint_c
        LOGGER.info(
            "%.2f C from %s wins: %s",
            winner.goal.setpoint_c,
            winner.goal.source.value,
            winner.goal.rationale or "no reason given",
        )
        return winner.goal
