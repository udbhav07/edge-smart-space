"""The goal path: what becomes a proposed setpoint, and whose proposal wins.

Two jobs, both deterministic and both ahead of the safety validator, which
still judges everything they pass on (FR-13):

* **A spoken preference becomes a setpoint proposal (FR-53).** A
  :class:`PreferenceHint` is a request weighed like any other, never a
  command: a named temperature is proposed as named -- the validator, not
  this module, decides what 5 C becomes -- and "cooler" or "warmer" moves
  the setpoint in force by one configured step. A hint about the lights, a
  booking, or nothing at all proposes nothing.
* **An occupant's wish outranks the supervisor for a while.** Without that,
  someone who asked for 22 C would have it undone by the next 300-second
  supervisory cycle that preferred 25 C, and the system would appear to
  ignore them. A supervisor proposal inside the hold is refused, and the
  refusal is published like any other verdict, so the override is visible
  rather than silent.

It holds no blackboard and publishes nothing: the control service does, so
the arbitration is testable without a transport.
"""

from __future__ import annotations

import logging

from src.common.clock import Clock
from src.common.config import GoalsConfig
from src.common.schemas import Comfort, Goal, GoalSource, Intent, Mode, PreferenceHint

LOGGER = logging.getLogger(__name__)

#: Subjects a spoken preference may name and still be about the temperature.
#: Empty covers the common case of the model leaving it blank.
_THERMAL_SUBJECTS = frozenset(
    {"", "temperature", "heat", "cooling", "air conditioner", "ac", "air conditioning"}
)


class GoalManager:
    """Turns preferences into goals and arbitrates between proposers."""

    def __init__(self, config: GoalsConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._preference_at_s: float | None = None

    @property
    def preference_holding(self) -> bool:
        """Whether an occupant's recent preference is still outranking the supervisor."""
        if self._preference_at_s is None:
            return False
        elapsed = self._clock.monotonic() - self._preference_at_s
        return elapsed < self._config.preference_hold_s

    def from_preference(
        self, hint: PreferenceHint, setpoint_in_force_c: float, mode: Mode
    ) -> Goal | None:
        """The goal a spoken preference asks for, or None if it asks for none.

        None is the ordinary answer to a hint that is not about the
        temperature, so the return is typed Optional rather than raising.
        """
        if hint.intent is not Intent.ENVIRONMENT:
            return None
        if hint.subject.strip().lower() not in _THERMAL_SUBJECTS:
            return None
        setpoint_c = self._requested_setpoint_c(hint, setpoint_in_force_c)
        if setpoint_c is None:
            return None
        now = self._clock.now()
        self._preference_at_s = self._clock.monotonic()
        return Goal(
            ts=now,
            source=GoalSource.PREFERENCE,
            setpoint_c=setpoint_c,
            mode=mode,
            rationale=hint.transcript or hint.rationale,
            expires_ts=now + self._config.preference_ttl_s,
        )

    def admits(self, goal: Goal) -> bool:
        """Whether a proposal from the blackboard may go on to the validator.

        Only a supervisor proposal is ever held back, and only while an
        occupant's preference is holding. An operator is never overruled.
        """
        if goal.source is GoalSource.SUPERVISOR and self.preference_holding:
            LOGGER.info(
                "supervisor proposal of %.1f C held back: the occupant asked "
                "for something else within the last %.0f s",
                goal.setpoint_c,
                self._config.preference_hold_s,
            )
            return False
        return True

    def _requested_setpoint_c(
        self, hint: PreferenceHint, setpoint_in_force_c: float
    ) -> float | None:
        if hint.target_c is not None:
            return hint.target_c
        if hint.comfort is Comfort.COOLER:
            return setpoint_in_force_c - self._config.comfort_step_c
        if hint.comfort is Comfort.WARMER:
            return setpoint_in_force_c + self._config.comfort_step_c
        return None
