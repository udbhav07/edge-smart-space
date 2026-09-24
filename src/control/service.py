"""Wires the regulatory loop to the blackboard.

This is the process that keeps the room at a temperature. Everything else in
the system informs it; none of it is allowed to stop it.

**It ticks on the clock, not on messages.** NFR-01 puts the loop at 5 s, and a
callback-driven controller would run at whatever rate its inputs happened to
arrive -- which on WiFi is not a rate at all. A controller that stops
commanding because a sensor stopped reporting is a controller that fails
silently, and the dwell timer would stop with it.

**Every input is optional, and the loop survives each one being absent
(FR-11, FR-47).** No goal ever arrives: the configured default setpoint stands.
The reasoning layer is down: the last validated setpoint stands, because the
validator holds it rather than the proposer. The estimator is down: there is no
prediction, which matters only in DEGRADED_SENSOR, and the mode manager will
not put the system there without a sensor fault. The detector bank is down: the
mode stays whatever was last retained, which is the conservative answer.

**What it will not do is run without a measurement.** Until the first reading
arrives there is nothing to control on, and a loop that commanded cooling from
a default would drive a room it has never observed. It publishes nothing and
says so, once.

**FR-27 lives here, and it is one line of substitution.** In DEGRADED_SENSOR
the controller acts on the model's prediction instead of the reading it cannot
trust, so the loop stays closed on a room whose sensor has failed. That single
substitution is the whole payoff of having identified a model, and it is the
thing a thermostat cannot do at all.

Control law: DESIGN.md section 5.3. Validation rules: section 5.4.
"""

from __future__ import annotations

import logging

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    COMMAND_KIND_KEY,
    SETPOINT_KEY,
    Command,
    CommandKind,
    Goal,
    Mode,
    ModeState,
    PreferenceHint,
    ReasonCode,
    SensorReading,
    ThermalEstimate,
    ValidationVerdict,
    Verdict,
)
from src.control.goal_manager import GoalManager
from src.control.regulatory import RegulatoryController
from src.control.validator import CommandValidator, GoalValidator

LOGGER = logging.getLogger(__name__)

#: Commands that ask the plant to do nothing new. They are still published --
#: silence is indistinguishable from a crashed controller -- but they are not
#: worth a log line every 5 s.
_QUIET_COMMANDS = frozenset({CommandKind.MAINTAIN, CommandKind.HOLD})


class ControlService:
    """One regulatory loop, gated by the safety validator."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        blackboard: Blackboard,
        controller: RegulatoryController,
        goal_validator: GoalValidator,
        command_validator: CommandValidator,
        goals: GoalManager,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._controller = controller
        self._goal_validator = goal_validator
        self._command_validator = command_validator
        self._goals = goals

        self._measured_c: float | None = None
        self._predicted_c: float | None = None
        self._mode = Mode.INIT
        self._warned_about_no_reading = False

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        """Listen to everything that informs the loop, none of it required."""
        self._blackboard.subscribe(
            topics.SENSOR_STATE, SensorReading, self._on_reading
        )
        self._blackboard.subscribe(
            topics.ESTIMATE_THERMAL, ThermalEstimate, self._on_estimate
        )
        self._blackboard.subscribe(topics.SYSTEM_MODE, ModeState, self._on_mode)
        self._blackboard.subscribe(topics.GOAL_PROPOSED, Goal, self._on_goal)
        self._blackboard.subscribe(
            topics.CONTEXT_PREFERENCE, PreferenceHint, self._on_preference
        )

    @property
    def setpoint_c(self) -> float:
        """The setpoint in force. Held by the validator, not by the proposer."""
        return self._goal_validator.applied_setpoint_c

    @property
    def mode(self) -> Mode:
        """The mode last heard. INIT until the detector bank says otherwise."""
        return self._mode

    # --- inputs -------------------------------------------------------

    def _on_reading(self, _topic: str, reading: SensorReading) -> None:
        """Keep the latest indoor temperature, ignoring every other sensor."""
        if reading.sensor_id != self._config.estimator.indoor_sensor_id:
            return
        self._measured_c = reading.value

    def _on_estimate(self, _topic: str, estimate: ThermalEstimate) -> None:
        """Keep the model's prediction, for the mode that controls on it."""
        self._predicted_c = estimate.t_pred

    def _on_mode(self, _topic: str, state: ModeState) -> None:
        if state.mode is not self._mode:
            LOGGER.info("control mode is now %s", state.mode.value)
        self._mode = state.mode
        self._goals.observe_mode(state.mode)

    def _on_goal(self, _topic: str, goal: Goal) -> None:
        """A proposal from outside: the supervisor, an operator (FR-40).

        Arbitrated first, then gated. A stale one goes straight to the gate,
        which refuses it with the reason that is actually true (V-6). One that
        loses arbitration is published as a verdict too: a supervisor goal
        that changed nothing because an occupant had spoken is a decision, and
        a decision nobody can see did not happen.
        """
        if self._goal_validator.is_stale(goal):
            self._gate(goal)
            return
        winner = self._goals.propose(goal)
        if winner is not None:
            self._gate(winner)
        elif goal.source is not self._goals.winning_source:
            self._publish_outranked(goal)

    def _on_preference(self, _topic: str, hint: PreferenceHint) -> None:
        """What an occupant said, as a proposal (FR-53), then gated (FR-45)."""
        winner = self._goals.consider(hint)
        if winner is not None:
            self._gate(winner)

    def _publish_outranked(self, goal: Goal) -> None:
        verdict = ValidationVerdict(
            ts=self._clock.now(),
            proposed={SETPOINT_KEY: goal.setpoint_c},
            verdict=Verdict.BLOCKED,
            reason=ReasonCode.OUTRANKED,
            applied={SETPOINT_KEY: self.setpoint_c},
        )
        self._blackboard.publish(topics.AUDIT_VALIDATION, verdict)
        LOGGER.info(
            "goal %.2f from %s outranked by %s; setpoint stays %.2f",
            goal.setpoint_c,
            goal.source.value,
            self._goals.winning_source.value,
            self.setpoint_c,
        )

    def _gate(self, goal: Goal) -> None:
        """Gate a winning setpoint and adopt what survives (FR-13, FR-14).

        The verdict is published whatever it says. A clamped proposal is the
        gate working and belongs in the audit trail; hiding it would remove the
        only evidence that the gate runs at all (section 5.4).
        """
        verdict = self._goal_validator.validate(goal)
        self._blackboard.publish(topics.AUDIT_VALIDATION, verdict)
        self._publish_active_goal(goal)
        LOGGER.info(
            "goal %.2f from %s: %s (%s), setpoint now %.2f",
            goal.setpoint_c,
            goal.source.value,
            verdict.verdict.value,
            verdict.reason.value,
            self.setpoint_c,
        )

    def _publish_active_goal(self, goal: Goal) -> None:
        """Retain what is actually in force, which is not what was proposed."""
        self._blackboard.publish(
            topics.GOAL_ACTIVE,
            Goal(
                ts=self._clock.now(),
                source=goal.source,
                setpoint_c=self.setpoint_c,
                mode=self._mode,
                rationale=goal.rationale,
                expires_ts=goal.expires_ts,
            ),
        )

    # --- the loop -----------------------------------------------------

    def tick(self) -> Command | None:
        """Run one control cycle and publish the command that survives.

        :returns: the command published, or None when there is nothing to
            control on yet. Optional is honest: before the first reading there
            is no measurement, and a loop that commanded from a default would
            drive a room it has never observed.
        """
        # Proposals age out on the clock, not on a message: a source going
        # silent is exactly the case where nothing arrives, and a supervisor
        # that crashed mid-proposal must not keep steering the room.
        successor = self._goals.expire()
        if successor is not None:
            self._gate(successor)

        if self._measured_c is None:
            if not self._warned_about_no_reading:
                LOGGER.warning(
                    "no reading from %s yet; holding off until the room is "
                    "observed", self._config.estimator.indoor_sensor_id
                )
                self._warned_about_no_reading = True
            return None

        proposed = self._controller.tick(
            measured_c=self._measured_c,
            predicted_c=self._effective_prediction_c(),
            setpoint_c=self.setpoint_c,
            mode=self._mode,
        )
        verdict = self._command_validator.validate(proposed, self._mode)
        admitted = self._admitted_command(proposed, verdict)

        self._blackboard.publish(topics.AUDIT_VALIDATION, verdict)
        self._blackboard.publish(
            topics.ACTUATOR_COMMAND, admitted, actuator_id=admitted.actuator_id
        )
        if admitted.kind not in _QUIET_COMMANDS:
            LOGGER.info(
                "%s at setpoint %.2f in %s",
                admitted.kind.value,
                self.setpoint_c,
                self._mode.value,
            )
        return admitted

    def _effective_prediction_c(self) -> float:
        """What to substitute when the sensor cannot be trusted (FR-27).

        With no estimate available the measurement is handed back, which makes
        the substitution a no-op rather than a crash. That is the right failure:
        a dead estimator should cost the system its degraded-mode capability,
        not its regulatory loop (FR-47).
        """
        if self._predicted_c is not None:
            return self._predicted_c
        if self._mode is Mode.DEGRADED_SENSOR:
            LOGGER.warning(
                "in %s with no model prediction; controlling on the reading "
                "the detectors distrust", self._mode.value
            )
        return self._measured_c if self._measured_c is not None else 0.0

    def _admitted_command(
        self, proposed: Command, verdict: ValidationVerdict
    ) -> Command:
        """The command the validator actually allows through.

        The validator reports the kind it admitted; this rebuilds a Command
        from it so that what is published and what the audit trail says are
        the same thing by construction.
        """
        applied_kind = CommandKind(verdict.applied[COMMAND_KIND_KEY])
        if applied_kind is proposed.kind:
            return proposed
        return Command(
            ts=proposed.ts,
            actuator_id=proposed.actuator_id,
            kind=applied_kind,
            setpoint_c=proposed.setpoint_c if applied_kind is CommandKind.COOL else None,
        )


def build_service(
    config: Config, clock: Clock, blackboard: Blackboard
) -> ControlService:
    """Assemble the loop and both validators from configuration."""
    return ControlService(
        config=config,
        clock=clock,
        blackboard=blackboard,
        controller=RegulatoryController(
            config=config.controller,
            clock=clock,
            actuator_id=topics.AIR_CONDITIONER_ID,
        ),
        goal_validator=GoalValidator(
            config=config.validator,
            clock=clock,
            initial_setpoint_c=config.controller.default_setpoint_c,
        ),
        command_validator=CommandValidator(config=config.validator, clock=clock),
        goals=GoalManager(config, clock),
    )
