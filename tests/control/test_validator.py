"""Unit tests for the safety validator, rules V-1 to V-6.

These are the tests that make the architecture's central claim checkable: a
proposal from anywhere, however unreasonable, cannot reach the plant intact.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import ValidatorConfig, load_config
from src.common.schemas import (
    Command,
    CommandKind,
    Goal,
    GoalSource,
    Mode,
    ReasonCode,
    Verdict,
)
from src.control.validator import CommandValidator, GoalValidator

ACTUATOR_ID = "ac"
START_SETPOINT_C = 24.0


@pytest.fixture(name="config")
def _config() -> ValidatorConfig:
    return load_config(Path("config/default.yaml")).validator


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="goals")
def _goals(config: ValidatorConfig, clock: SimClock) -> GoalValidator:
    return GoalValidator(config, clock, initial_setpoint_c=START_SETPOINT_C)


@pytest.fixture(name="commands")
def _commands(config: ValidatorConfig, clock: SimClock) -> CommandValidator:
    return CommandValidator(config, clock)


def _goal(clock: SimClock, setpoint_c: float, **overrides) -> Goal:
    return Goal(
        **{
            "ts": clock.now(),
            "source": GoalSource.SUPERVISOR,
            "setpoint_c": setpoint_c,
            "mode": Mode.NORMAL,
            "expires_ts": clock.now() + 600.0,
            **overrides,
        }
    )


def _command(clock: SimClock, kind: CommandKind, setpoint_c: float | None = None):
    return Command(
        ts=clock.now(), actuator_id=ACTUATOR_ID, kind=kind, setpoint_c=setpoint_c
    )


class TestConstruction:
    def test_an_inadmissible_initial_setpoint_is_refused(self, config, clock):
        with pytest.raises(ValueError):
            GoalValidator(config, clock, initial_setpoint_c=5.0)

    def test_the_shipped_default_setpoint_is_admissible(self, config, clock):
        default = load_config(Path("config/default.yaml")).controller.default_setpoint_c
        assert GoalValidator(config, clock, default).applied_setpoint_c == default


class TestRuleV1AbsoluteBounds:
    def test_a_setpoint_below_the_floor_is_clamped_up(self, goals, clock, config):
        verdict = goals.validate(_goal(clock, 5.0))
        assert verdict.verdict is Verdict.CLAMPED
        assert verdict.applied_setpoint_c == pytest.approx(
            START_SETPOINT_C - config.max_step_c
        )

    def test_a_setpoint_above_the_ceiling_is_clamped_down(self, goals, clock, config):
        verdict = goals.validate(_goal(clock, 99.0))
        assert verdict.verdict is Verdict.CLAMPED
        assert verdict.applied_setpoint_c == pytest.approx(
            START_SETPOINT_C + config.max_step_c
        )

    def test_the_bound_alone_is_reported_when_it_is_the_binding_rule(
        self, config, clock
    ):
        """From 19.0, a request for 5.0 is stopped by V-1 before V-2 bites."""
        validator = GoalValidator(config, clock, initial_setpoint_c=19.0)
        verdict = validator.validate(_goal(clock, 5.0))
        assert verdict.reason is ReasonCode.BOUND_CLAMP
        assert verdict.applied_setpoint_c == config.setpoint_bounds_c.low

    def test_the_proposal_is_preserved_in_the_audit_record(self, goals, clock):
        assert goals.validate(_goal(clock, 99.0)).proposed_setpoint_c == 99.0

    @pytest.mark.parametrize("setpoint", [18.0, 30.0])
    def test_the_bounds_themselves_are_admissible(self, config, clock, setpoint):
        validator = GoalValidator(config, clock, initial_setpoint_c=setpoint)
        assert validator.validate(_goal(clock, setpoint)).verdict is Verdict.ACCEPTED


class TestRuleV2RateLimit:
    def test_a_step_within_the_limit_is_accepted_unchanged(self, goals, clock):
        verdict = goals.validate(_goal(clock, START_SETPOINT_C + 1.0))
        assert verdict.verdict is Verdict.ACCEPTED
        assert verdict.applied_setpoint_c == START_SETPOINT_C + 1.0

    def test_an_upward_step_beyond_the_limit_is_clamped(self, goals, clock, config):
        verdict = goals.validate(_goal(clock, START_SETPOINT_C + 10.0))
        assert verdict.reason is ReasonCode.RATE_LIMIT
        assert verdict.applied_setpoint_c == pytest.approx(
            START_SETPOINT_C + config.max_step_c
        )

    def test_a_downward_step_beyond_the_limit_is_clamped(self, goals, clock, config):
        verdict = goals.validate(_goal(clock, START_SETPOINT_C - 10.0))
        assert verdict.applied_setpoint_c == pytest.approx(
            START_SETPOINT_C - config.max_step_c
        )

    def test_the_limit_is_measured_from_what_is_in_force_not_what_was_asked(
        self, goals, clock, config
    ):
        """Repeated large proposals must not walk past the rate limit."""
        goals.validate(_goal(clock, 30.0))
        assert goals.applied_setpoint_c == pytest.approx(
            START_SETPOINT_C + config.max_step_c
        )
        goals.validate(_goal(clock, 30.0))
        assert goals.applied_setpoint_c == pytest.approx(
            START_SETPOINT_C + 2 * config.max_step_c
        )

    def test_the_exact_limit_is_admissible(self, goals, clock, config):
        verdict = goals.validate(_goal(clock, START_SETPOINT_C + config.max_step_c))
        assert verdict.verdict is Verdict.ACCEPTED


class TestRuleV6Staleness:
    def test_a_goal_older_than_the_horizon_is_blocked(self, goals, clock, config):
        goal = _goal(clock, 26.0)
        clock.advance(config.goal_max_age_s + 1.0)
        verdict = goals.validate(goal)
        assert verdict.verdict is Verdict.BLOCKED
        assert verdict.reason is ReasonCode.STALE_GOAL

    def test_a_blocked_stale_goal_leaves_the_setpoint_in_force(
        self, goals, clock, config
    ):
        goal = _goal(clock, 26.0)
        clock.advance(config.goal_max_age_s + 1.0)
        verdict = goals.validate(goal)
        assert verdict.applied_setpoint_c == START_SETPOINT_C
        assert goals.applied_setpoint_c == START_SETPOINT_C

    def test_a_goal_past_its_own_expiry_is_blocked(self, goals, clock):
        goal = _goal(clock, 26.0, expires_ts=clock.now() + 10.0)
        clock.advance(20.0)
        assert goals.validate(goal).reason is ReasonCode.STALE_GOAL

    def test_a_fresh_goal_is_not_stale(self, goals, clock):
        assert goals.validate(_goal(clock, 25.0)).verdict is Verdict.ACCEPTED


class TestGoalSourceIndifference:
    @pytest.mark.parametrize("source", list(GoalSource))
    def test_every_source_is_gated_identically(self, goals, clock, source):
        """The validator has no knowledge of intent (section 5.4)."""
        verdict = goals.validate(_goal(clock, 99.0, source=source))
        assert verdict.verdict is Verdict.CLAMPED


class TestRuleV5ModeConsistency:
    @pytest.mark.parametrize("mode", [Mode.DEGRADED_ACTUATOR, Mode.SAFE_HOLD])
    @pytest.mark.parametrize("kind", [CommandKind.COOL, CommandKind.OFF])
    def test_actuation_is_blocked_in_a_non_actuating_mode(
        self, commands, clock, mode, kind
    ):
        setpoint = 25.0 if kind is CommandKind.COOL else None
        verdict = commands.validate(_command(clock, kind, setpoint), mode)
        assert verdict.verdict is Verdict.BLOCKED
        assert verdict.reason is ReasonCode.MODE_BLOCK
        assert verdict.applied_kind is CommandKind.HOLD

    @pytest.mark.parametrize("mode", [Mode.NORMAL, Mode.DEGRADED_SENSOR])
    def test_actuation_is_permitted_in_an_actuating_mode(self, commands, clock, mode):
        verdict = commands.validate(_command(clock, CommandKind.COOL, 25.0), mode)
        assert verdict.verdict is Verdict.ACCEPTED

    def test_mode_block_takes_precedence_over_timing_rules(self, commands, clock):
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        verdict = commands.validate(
            _command(clock, CommandKind.OFF), Mode.SAFE_HOLD
        )
        assert verdict.reason is ReasonCode.MODE_BLOCK


class TestRuleV3CompressorDwell:
    def _cool_then_off(self, commands, clock, config):
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        clock.advance(config.min_command_interval_s)
        commands.validate(_command(clock, CommandKind.OFF), Mode.NORMAL)

    def test_a_restart_inside_the_dwell_window_is_downgraded(
        self, commands, clock, config
    ):
        self._cool_then_off(commands, clock, config)
        clock.advance(config.min_off_s / 2.0)
        verdict = commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        assert verdict.reason is ReasonCode.DWELL
        assert verdict.applied_kind is CommandKind.MAINTAIN

    def test_a_restart_after_the_dwell_window_is_admitted(
        self, commands, clock, config
    ):
        self._cool_then_off(commands, clock, config)
        clock.advance(config.min_off_s + 1.0)
        verdict = commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        assert verdict.verdict is Verdict.ACCEPTED

    def test_the_first_ever_command_is_not_held_by_dwell(self, commands, clock):
        verdict = commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        assert verdict.verdict is Verdict.ACCEPTED

    def test_switching_off_is_never_held_by_dwell(self, commands, clock, config):
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        clock.advance(config.min_command_interval_s)
        verdict = commands.validate(_command(clock, CommandKind.OFF), Mode.NORMAL)
        assert verdict.verdict is Verdict.ACCEPTED


class TestRuleV4CommandRate:
    def test_a_second_change_too_soon_is_suppressed(self, commands, clock):
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        verdict = commands.validate(_command(clock, CommandKind.OFF), Mode.NORMAL)
        assert verdict.reason is ReasonCode.CMD_RATE
        assert verdict.applied_kind is CommandKind.MAINTAIN

    def test_a_change_after_the_interval_is_admitted(self, commands, clock, config):
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        clock.advance(config.min_command_interval_s + 1.0)
        verdict = commands.validate(_command(clock, CommandKind.OFF), Mode.NORMAL)
        assert verdict.verdict is Verdict.ACCEPTED

    @pytest.mark.parametrize("kind", [CommandKind.MAINTAIN, CommandKind.HOLD])
    def test_a_status_quo_command_is_never_throttled(self, commands, clock, kind):
        """Throttling MAINTAIN would leave the controller unable to say
        'keep doing what you are doing' on every tick."""
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        verdict = commands.validate(_command(clock, kind), Mode.NORMAL)
        assert verdict.verdict is Verdict.ACCEPTED

    def test_a_suppressed_command_does_not_reset_the_rate_window(
        self, commands, clock, config
    ):
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        clock.advance(config.min_command_interval_s / 2.0)
        commands.validate(_command(clock, CommandKind.OFF), Mode.NORMAL)
        clock.advance(config.min_command_interval_s / 2.0 + 1.0)
        verdict = commands.validate(_command(clock, CommandKind.OFF), Mode.NORMAL)
        assert verdict.verdict is Verdict.ACCEPTED


class TestCompressorState:
    def test_tracks_the_compressor_across_admitted_commands(
        self, commands, clock, config
    ):
        assert commands.compressor_on is False
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        assert commands.compressor_on is True
        clock.advance(config.min_command_interval_s + 1.0)
        commands.validate(_command(clock, CommandKind.OFF), Mode.NORMAL)
        assert commands.compressor_on is False

    def test_a_blocked_command_does_not_change_the_compressor_state(
        self, commands, clock
    ):
        commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.SAFE_HOLD)
        assert commands.compressor_on is False


class TestAuditRecord:
    def test_every_verdict_names_the_actuator(self, commands, clock):
        verdict = commands.validate(_command(clock, CommandKind.COOL, 25.0), Mode.NORMAL)
        assert verdict.actuator_id == ACTUATOR_ID

    def test_every_verdict_preserves_what_was_proposed(self, commands, clock):
        verdict = commands.validate(
            _command(clock, CommandKind.COOL, 25.0), Mode.SAFE_HOLD
        )
        assert verdict.proposed_kind is CommandKind.COOL
        assert verdict.applied_kind is CommandKind.HOLD

    def test_verdicts_carry_the_clock_timestamp(self, goals, clock):
        clock.advance(100.0)
        assert goals.validate(_goal(clock, 25.0)).ts == clock.now()
