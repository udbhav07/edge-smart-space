"""The Environmental Supervisor (sections 5.7.1 to 5.7.3; FR-40, FR-41).

The one open-ended tool loop in the system: the model decides how many times
to look before it proposes, bounded by ``supervisor.max_steps``. It reads the
room through four tools, proposes one setpoint through a fifth, and is told
what the safety validator made of it. Its entire influence on the plant is
that proposal (FR-45).

It runs every ``supervisor.period_s`` and on the events FR-41 names --
occupancy transition, tariff transition, fault confirmation -- with a
hold-off so a flapping sensor cannot run it every tick. When it fails in any
way -- no server, no proposal, a proposal that makes no sense -- nothing is
published and the previous goal stands (section 5.7.1). Every cycle is
recorded on ``space/audit/reasoning`` whatever happened (FR-46, FR-63).
"""

from __future__ import annotations

import logging
from collections import deque

from src.common import topics
from src.common.clock import Clock
from src.common.config import SupervisorConfig
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    SETPOINT_KEY,
    SOURCE_KEY,
    Goal,
    GoalSource,
    ReasoningCaller,
    ReasoningRecord,
    ValidationVerdict,
)
from src.common.tools import ToolArgumentError
from src.reasoning.assistant import local_now
from src.reasoning.chat import ChatClient, ChatTurn, assistant_message, tool_message
from src.reasoning.single_shot import ReasoningUnavailableError
from src.reasoning.supervisor_tools import (
    PROPOSE_SETPOINT,
    SUPERVISOR_TOOLS,
    SemanticRejection,
    SupervisorState,
)

LOGGER = logging.getLogger(__name__)

#: Seconds to wait for the validator's verdict on a proposal.
_VERDICT_WAIT_S = 3.0
_POLL_S = 0.05
_VERDICTS_KEPT = 32

SUPERVISOR_PROMPT = (
    "You supervise the temperature of one room. It is now {now}. Each cycle: "
    "call get_thermal_state, get_occupancy, get_tariff_state and "
    "get_active_faults, then call propose_setpoint exactly once and stop. "
    "Policy, applied in this order: (1) if get_active_faults lists any fault, "
    "propose t_setpoint_c unchanged; (2) otherwise, if get_occupancy says "
    "setback_applies is true, propose {vacant:g}; (3) otherwise propose "
    "{occupied:g}. Do not adjust for peak tariff -- the system does that "
    "itself. Pass the mode exactly as get_active_faults reports it. The "
    "rationale is one sentence naming the rule you applied."
)


class SupervisorAgent:
    """Runs supervisory cycles on a cadence and on events."""

    def __init__(
        self,
        config: SupervisorConfig,
        clock: Clock,
        blackboard: Blackboard,
        chat: ChatClient,
        state: SupervisorState,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._chat = chat
        self._state = state
        self._tools = tuple(spec.as_schema() for spec in SUPERVISOR_TOOLS)
        self._last_run_s: float | None = None
        self._verdicts: deque[ValidationVerdict] = deque(maxlen=_VERDICTS_KEPT)

    def subscribe(self) -> None:
        self._state.subscribe(self._blackboard)
        self._blackboard.subscribe(
            topics.AUDIT_VALIDATION, ValidationVerdict, self._on_verdict
        )

    def _on_verdict(self, _topic: str, verdict: ValidationVerdict) -> None:
        self._verdicts.append(verdict)

    # --- scheduling ---------------------------------------------------

    def due(self) -> str | None:
        """The reason to run a cycle now, or None (FR-41)."""
        if not self._state.thermal_known:
            return None
        now_s = self._clock.monotonic()
        if self._last_run_s is None:
            return "startup"
        since = now_s - self._last_run_s
        changes = self._state.take_changes()
        if changes and since >= self._config.event_holdoff_s:
            return changes[0]
        if since >= self._config.period_s:
            return "periodic"
        return None

    def maybe_run(self) -> ReasoningRecord | None:
        trigger = self.due()
        if trigger is None:
            return None
        return self.run_cycle(trigger)

    # --- one cycle ----------------------------------------------------

    def run_cycle(self, trigger: str) -> ReasoningRecord:
        """Read, propose once, and record what happened."""
        self._last_run_s = self._clock.monotonic()
        instruction = f"Cycle trigger: {trigger}. Read the room, then propose."
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": SUPERVISOR_PROMPT.format(
                    now=local_now(self._clock),
                    vacant=self._config.vacant_setpoint_c,
                    occupied=self._config.occupied_setpoint_c,
                ),
            },
            {"role": "user", "content": instruction},
        ]
        turns: list[ChatTurn] = []
        verdict_text = "no proposal; previous goal retained"
        applied = ""
        try:
            for _ in range(self._config.max_steps):
                turn = self._chat.complete(messages, self._tools)
                turns.append(turn)
                if not turn.tool_calls:
                    break
                messages.append(assistant_message(turn))
                proposed = False
                for call in turn.tool_calls:
                    if call.name == PROPOSE_SETPOINT.name:
                        answer, verdict_text, applied = self._propose(call.arguments, trigger)
                        proposed = True
                    else:
                        answer = self._read(call.name)
                    messages.append(tool_message(call, answer))
                if proposed:
                    break
        except ReasoningUnavailableError as exc:
            LOGGER.warning("supervisor unavailable, previous goal retained: %s", exc)
            verdict_text = "reasoning unavailable; previous goal retained"

        record = ReasoningRecord(
            ts=self._clock.now(),
            caller=ReasoningCaller.SUPERVISOR,
            trigger=trigger,
            model=self._chat.model,
            inputs=instruction,
            raw_output="\n---\n".join(turn.raw for turn in turns),
            verdict=verdict_text,
            applied=applied,
            tool_calls=tuple(call.name for turn in turns for call in turn.tool_calls),
            latency_s=sum(turn.latency_s for turn in turns),
            prompt_tokens=sum(turn.prompt_tokens for turn in turns),
            completion_tokens=sum(turn.completion_tokens for turn in turns),
        )
        self._blackboard.publish(topics.AUDIT_REASONING, record)
        LOGGER.info("supervisor cycle (%s): %s", trigger, verdict_text)
        return record

    def _read(self, name: str) -> str:
        try:
            return self._state.read(name)
        except ToolArgumentError as exc:
            return f"error: {exc}"

    def _propose(self, arguments, trigger: str) -> tuple[str, str, str]:
        """Check, publish, and report the validator's verdict.

        :returns: what the model is told, the audit verdict, and what applied.
        """
        try:
            setpoint_c, mode, rationale = self._state.check_proposal(arguments)
        except (ToolArgumentError, SemanticRejection, ValueError) as exc:
            LOGGER.warning("supervisor proposal discarded: %s", exc)
            return f"discarded: {exc}", f"discarded ({exc}); previous goal retained", ""
        now = self._clock.now()
        goal = Goal(
            ts=now,
            source=GoalSource.SUPERVISOR,
            setpoint_c=setpoint_c,
            mode=mode,
            rationale=f"{trigger}: {rationale}",
            expires_ts=now + self._config.goal_ttl_s,
        )
        self._blackboard.publish(topics.GOAL_PROPOSED, goal)
        verdict = self._await_verdict(since_ts=now)
        if verdict is None:
            return (
                "proposal sent; no verdict heard",
                f"proposed {setpoint_c:g} C; no verdict heard",
                "",
            )
        applied = verdict.applied.get(SETPOINT_KEY)
        summary = f"{verdict.verdict.value} ({verdict.reason.value}); in force {applied}"
        return summary, f"proposed {setpoint_c:g} C: {summary}", f"setpoint {applied}"

    def _await_verdict(self, since_ts: float) -> ValidationVerdict | None:
        deadline = self._clock.monotonic() + _VERDICT_WAIT_S
        while True:
            for verdict in reversed(self._verdicts):
                if (
                    verdict.ts >= since_ts
                    and verdict.proposed.get(SOURCE_KEY) == GoalSource.SUPERVISOR.value
                ):
                    return verdict
            if self._clock.monotonic() >= deadline:
                return None
            self._clock.sleep(_POLL_S)
