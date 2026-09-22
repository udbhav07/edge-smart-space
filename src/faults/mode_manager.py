"""The degradation state machine (DESIGN.md section 5.6).

Detection that does not change behaviour is not tolerance. This is where a
fault becomes a decision: what the system is now allowed to do, published
before anything slow is consulted.

**The transition never waits for an explanation.** FR-26 gives 2 s from
confirmation to published mode, and the Fault Diagnosis call runs in parallel
to enrich the notification (section 5.9.2). Nothing here calls a model, and
nothing here can block on one. The worst case for a diagnosis that never
arrives is a mode change with a terse reason, not a mode change that never
happens.

**Escalation counts broken things, not findings.** Section 5.6 escalates to
SAFE_HOLD on "multiple faults", and the naive reading -- more than one active
FaultEvent -- is wrong in a way that would destroy the feature it exists to
protect. A stuck temperature sensor raises D2, and its frozen reading then
makes the model residual grow until D4 raises as well. That is two faults and
one broken sensor. Escalating there would mean every single sensor failure
ended in SAFE_HOLD with actuation blocked, and FR-27's whole point -- that the
loop keeps running on the model's prediction -- would never once be exercised.
So the count is over distinct *subjects*.

**SAFE_HOLD is the one mode the system will not leave on its own.** Every
other degradation is undone by the fault clearing, because the evidence that
raised it is the evidence that retires it. SAFE_HOLD means several independent
things are wrong or the model itself has diverged, and no observation means "a
person has looked at this" -- so a person says so, over ``space/system/reset``.

**A known dead end, recorded rather than hidden.** Section 5.6 returns
DEGRADED_ACTUATOR to NORMAL on "actuator ack restored and response observed".
On an open-loop IR path there is no ack to restore (R-02), and the mode blocks
actuation, so no cooling can be commanded and no response can be observed. The
transition as written is unreachable on our hardware. The operator reset is
therefore the route back from DEGRADED_ACTUATOR too, which is honest about
what the system can actually determine by itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from src.common.clock import Clock
from src.common.config import ModeConfig
from src.common.schemas import FaultEvent, Mode, ModeReset, ModeState

LOGGER = logging.getLogger(__name__)

#: Modes the system does not leave without a person saying so.
_OPERATOR_ONLY_EXITS = frozenset({Mode.SAFE_HOLD, Mode.DEGRADED_ACTUATOR})

#: How many distinct broken things it takes to stop trusting the picture.
_MULTIPLE_FAULTS = 2


class ModeManager:
    """Turns the active fault set into the mode the system runs in.

    Pure decision: it holds no blackboard and publishes nothing. The caller
    publishes what it returns, so the state machine is testable without a
    transport and the 2 s deadline is not spent inside a network call.
    """

    def __init__(self, config: ModeConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._mode = Mode.INIT
        self._since_ts = clock.now()
        self._reason = "starting up"
        self._degraded_since_s: float | None = None
        self._fault_ids: tuple[str, ...] = ()
        self._reset_pending = False

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def substitution_elapsed_s(self) -> float:
        """How long control has been running on prediction (FR-27, section 7.2).

        Zero when not substituting. This is the number the budget is spent
        against, and it is worth watching during a demonstration.
        """
        if self._degraded_since_s is None:
            return 0.0
        return self._clock.monotonic() - self._degraded_since_s

    def request_reset(self, reset: ModeReset) -> None:
        """Record an operator's acknowledgement (section 5.6).

        Applied on the next update rather than immediately, so one code path
        decides the mode. A reset that arrives while nothing is held is kept
        rather than discarded: the operator may simply have got there first,
        and dropping it silently would make the button look broken.
        """
        self._reset_pending = True
        LOGGER.warning(
            "reset requested by %s: %s",
            reset.requester,
            reset.reason or "no reason given",
        )

    def update(
        self, active_faults: Sequence[FaultEvent], sensors_reporting: bool = True
    ) -> ModeState | None:
        """Decide the mode from what is currently broken.

        :param active_faults: every live fault, most severe first, as the
            aggregator orders them.
        :param sensors_reporting: whether anything has been heard from at all.
            INIT is left only once it has, so a system whose sensors never
            arrive stays visibly in INIT rather than claiming to be NORMAL.
        :returns: the new mode to publish, or None when nothing changed.
            Optional is the honest type: most ticks change nothing, and
            republishing an unchanged retained mode every 5 s would bury the
            transitions that matter in the ones that do not.
        """
        target, reason = self._decide(active_faults, sensors_reporting)
        fault_ids = tuple(event.fault_id for event in active_faults)

        if target is self._mode and fault_ids == self._fault_ids:
            return None

        return self._transition(target, reason, fault_ids)

    def _decide(
        self, active_faults: Sequence[FaultEvent], sensors_reporting: bool
    ) -> tuple[Mode, str]:
        """Which mode the current evidence calls for, and why in one line."""
        subjects = {event.subject for event in active_faults}

        if self._reset_pending:
            self._reset_pending = False
            if not active_faults:
                return Mode.NORMAL, "operator reset; no faults active"
            LOGGER.warning(
                "reset refused: %d fault(s) still active", len(active_faults)
            )

        if self._mode in _OPERATOR_ONLY_EXITS:
            return self._mode, self._reason

        if self._mode is Mode.INIT and not sensors_reporting:
            return Mode.INIT, "waiting for the first sensor reading"

        if not active_faults:
            return Mode.NORMAL, "no active faults"

        dominant = active_faults[0]
        dominant_cause = f"{dominant.detector.value} on {dominant.subject}"

        if dominant.mode_impact is Mode.SAFE_HOLD:
            return Mode.SAFE_HOLD, dominant_cause

        if len(subjects) >= _MULTIPLE_FAULTS:
            return Mode.SAFE_HOLD, f"{len(subjects)} subjects faulted at once"

        if self._substitution_budget_spent():
            return (
                Mode.SAFE_HOLD,
                f"prediction budget of {self._config.degraded_sensor_budget_s:.0f} s "
                f"exhausted",
            )

        return dominant.mode_impact, dominant_cause

    def _substitution_budget_spent(self) -> bool:
        """FR-27 is time-boxed: a prediction is not a measurement forever.

        The model was identified from data the faulted sensor provided, and it
        drifts away from the room with nothing to correct it. Section 7.2 caps
        how long the system is allowed to pretend otherwise.
        """
        if self._degraded_since_s is None:
            return False
        elapsed = self._clock.monotonic() - self._degraded_since_s
        return elapsed >= self._config.degraded_sensor_budget_s

    def _transition(
        self, target: Mode, reason: str, fault_ids: tuple[str, ...]
    ) -> ModeState:
        """Enter a mode and describe the entry."""
        now = self._clock.now()
        if target is not self._mode:
            LOGGER.warning(
                "mode %s -> %s: %s", self._mode.value, target.value, reason
            )
            self._track_substitution(target)
            self._mode = target
            self._since_ts = now
            self._reason = reason

        self._fault_ids = fault_ids
        return ModeState(
            ts=now,
            mode=self._mode,
            since_ts=self._since_ts,
            active_fault_ids=fault_ids,
            reason=self._reason,
        )

    def _track_substitution(self, target: Mode) -> None:
        """Start or stop the clock on the prediction budget."""
        if target is Mode.DEGRADED_SENSOR:
            if self._degraded_since_s is None:
                self._degraded_since_s = self._clock.monotonic()
            return
        self._degraded_since_s = None
