"""Safety validation: the gate every proposal passes through.

This is the only component permitted to be paranoid. It has no knowledge of
intent, applies its rules unconditionally, and does not care which layer
proposed something (DESIGN.md sections 5.4, 1.3).

That indifference is the whole point. A reasoning layer cannot be prompted
into satisfying a hard bound, so it is placed where its worst failure mode is
a suboptimal goal rather than an unsafe command — and this is the component
that makes that true. A supervisor asking for 5 degrees is clamped to the
configured minimum, the verdict says so, and nothing downstream ever sees the
original number.

Rules V-1 to V-6 split by what they act on:

* :class:`GoalValidator` gates *setpoints*: V-1 bounds, V-2 rate, V-6
  staleness.
* :class:`CommandValidator` gates *actuation*: V-3 compressor dwell, V-4
  command rate, V-5 mode consistency.

They are separate classes because they hold unrelated state and callers need
one or the other, never both.

A clamped proposal is a finding, not a failure. Every verdict carries the
proposal, the reason code, and the applied value so the audit trail can show
the gate working.
"""

from __future__ import annotations

from src.common.clock import Clock
from src.common.config import ValidatorConfig
from src.common.schemas import (
    Command,
    CommandKind,
    CommandVerdict,
    Goal,
    Mode,
    ReasonCode,
    ValidationVerdict,
    Verdict,
)

#: Modes in which no actuation may be commanded at all (rule V-5).
NON_ACTUATING_MODES = frozenset({Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD})

#: Commands that change the compressor's state, and therefore count against
#: the dwell and rate limits. MAINTAIN and HOLD assert the status quo and are
#: not throttled: throttling them would leave the controller unable to say
#: "keep doing what you are doing".
_STATE_CHANGING_COMMANDS = frozenset({CommandKind.COOL, CommandKind.OFF})

#: What a blocked or throttled command is downgraded to.
_SUPPRESSED_COMMAND = CommandKind.MAINTAIN
_BLOCKED_COMMAND = CommandKind.HOLD


class GoalValidator:
    """Gates proposed setpoints. Rules V-1, V-2, V-6.

    Holds the last admitted setpoint, because V-2 limits movement relative to
    what is actually in force, not relative to what was last proposed. A
    supervisor cannot walk past the rate limit by proposing large steps
    repeatedly.
    """

    def __init__(
        self,
        config: ValidatorConfig,
        clock: Clock,
        initial_setpoint_c: float,
    ) -> None:
        if not config.setpoint_bounds_c.contains(initial_setpoint_c):
            raise ValueError(
                f"initial setpoint {initial_setpoint_c!r} is outside "
                f"[{config.setpoint_bounds_c.low}, {config.setpoint_bounds_c.high}]"
            )
        self._config = config
        self._clock = clock
        self._applied_setpoint_c = initial_setpoint_c

    @property
    def applied_setpoint_c(self) -> float:
        """The setpoint currently in force."""
        return self._applied_setpoint_c

    def validate(self, goal: Goal) -> ValidationVerdict:
        """Admit, clamp, or reject a proposed goal.

        Rules apply in order V-6, V-1, V-2. Where more than one clamps the
        value, the reported reason is the last rule that changed it, since
        that is the rule which produced the number actually in force.
        """
        now = self._clock.now()

        if self._is_stale(goal, now):
            return self._verdict(
                now, goal.setpoint_c, Verdict.BLOCKED, ReasonCode.STALE_GOAL
            )

        candidate = goal.setpoint_c
        reason = ReasonCode.NONE

        bounded = self._config.setpoint_bounds_c.clamp(candidate)
        if bounded != candidate:
            candidate, reason = bounded, ReasonCode.BOUND_CLAMP

        rate_limited = self._limit_rate(candidate)
        if rate_limited != candidate:
            candidate, reason = rate_limited, ReasonCode.RATE_LIMIT

        verdict = Verdict.ACCEPTED if reason is ReasonCode.NONE else Verdict.CLAMPED
        self._applied_setpoint_c = candidate
        return self._verdict(now, goal.setpoint_c, verdict, reason, candidate)

    def _is_stale(self, goal: Goal, now: float) -> bool:
        """V-6. Also rejects a goal that has passed its own expiry."""
        if now - goal.ts > self._config.goal_max_age_s:
            return True
        return now > goal.expires_ts

    def _limit_rate(self, candidate: float) -> float:
        """V-2. Movement per invocation, measured from the setpoint in force."""
        step = candidate - self._applied_setpoint_c
        largest = self._config.max_step_c
        if abs(step) <= largest:
            return candidate
        return self._applied_setpoint_c + (largest if step > 0.0 else -largest)

    def _verdict(
        self,
        now: float,
        proposed: float,
        verdict: Verdict,
        reason: ReasonCode,
        applied: float | None = None,
    ) -> ValidationVerdict:
        return ValidationVerdict(
            ts=now,
            proposed_setpoint_c=proposed,
            verdict=verdict,
            reason=reason,
            applied_setpoint_c=(
                self._applied_setpoint_c if applied is None else applied
            ),
        )


class CommandValidator:
    """Gates actuation. Rules V-3, V-4, V-5.

    Enforces compressor dwell independently of the regulatory controller,
    which enforces it too. The duplication is deliberate: a controller bug
    must not be able to damage the compressor.
    """

    def __init__(self, config: ValidatorConfig, clock: Clock) -> None:
        self._config = config
        self._clock = clock
        self._last_state_change_ts: float | None = None
        self._last_off_ts: float | None = None
        self._compressor_on = False

    @property
    def compressor_on(self) -> bool:
        """Whether the last admitted command left the compressor running."""
        return self._compressor_on

    def validate(self, command: Command, mode: Mode) -> CommandVerdict:
        """Admit, downgrade, or block a command.

        Rules apply in order V-5, V-3, V-4: most restrictive first, so a mode
        that forbids actuation is never overridden by a timing rule.
        """
        now = self._clock.now()

        if mode in NON_ACTUATING_MODES and command.kind in _STATE_CHANGING_COMMANDS:
            return self._verdict(
                now, command, _BLOCKED_COMMAND, Verdict.BLOCKED, ReasonCode.MODE_BLOCK
            )

        if command.kind not in _STATE_CHANGING_COMMANDS:
            return self._verdict(
                now, command, command.kind, Verdict.ACCEPTED, ReasonCode.NONE
            )

        if self._violates_dwell(command.kind, now):
            return self._verdict(
                now, command, _SUPPRESSED_COMMAND, Verdict.BLOCKED, ReasonCode.DWELL
            )

        if self._violates_command_rate(now):
            return self._verdict(
                now, command, _SUPPRESSED_COMMAND, Verdict.BLOCKED, ReasonCode.CMD_RATE
            )

        self._record_admitted(command.kind, now)
        return self._verdict(
            now, command, command.kind, Verdict.ACCEPTED, ReasonCode.NONE
        )

    def _violates_dwell(self, kind: CommandKind, now: float) -> bool:
        """V-3. No restart until the compressor has been off long enough."""
        if kind is not CommandKind.COOL or self._compressor_on:
            return False
        if self._last_off_ts is None:
            return False
        return now - self._last_off_ts < self._config.min_off_s

    def _violates_command_rate(self, now: float) -> bool:
        """V-4. Caps how often the compressor state may be changed."""
        if self._last_state_change_ts is None:
            return False
        return now - self._last_state_change_ts < self._config.min_command_interval_s

    def _record_admitted(self, kind: CommandKind, now: float) -> None:
        if kind is CommandKind.OFF and self._compressor_on:
            self._last_off_ts = now
        self._compressor_on = kind is CommandKind.COOL
        self._last_state_change_ts = now

    def _verdict(
        self,
        now: float,
        command: Command,
        applied: CommandKind,
        verdict: Verdict,
        reason: ReasonCode,
    ) -> CommandVerdict:
        return CommandVerdict(
            ts=now,
            actuator_id=command.actuator_id,
            proposed_kind=command.kind,
            verdict=verdict,
            reason=reason,
            applied_kind=applied,
        )
