"""The Environmental Supervisor: the one tool-using agent (FR-40, FR-41).

Of the three call sites in section 5.7.1 this is the only one with an
open-ended tool loop. It decides for itself how much to read before it
proposes, using the server's native tool-call template (section 5.7.3), and it
ends by calling ``propose_setpoint`` -- which proposes, and cannot command.

**What can go wrong, and what happens when it does.** Every exit keeps the
regulatory loop running on the last validated setpoint (FR-11, FR-47), and
every exit is recorded (FR-46):

* the server is down or slow: ``UNAVAILABLE``, nothing proposed;
* the model answers in words without proposing: ``NO_ACTION``;
* it proposes something the post-decode check rejects: ``DISCARDED`` (FR-44),
  and the run ends there -- a discarded output is discarded, not negotiated;
* it runs out of rounds without proposing: ``DISCARDED``;
* it proposes and the gate answers: ``APPLIED``, with the gate's verdict in
  the record. "Applied" means the proposal went to the gate, not that the gate
  accepted it: a clamped proposal is still the supervisor's goal, bounded.

**When it runs** is :class:`SupervisorSchedule`'s business: on a cadence, and
on the three events FR-41 names, with a minimum spacing so a flapping sensor
cannot turn into a queue of model calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.common.clock import Clock
from src.common.config import Config
from src.common.localtime import local_time
from src.common.schemas import (
    CallSite,
    Goal,
    ReasoningOutcome,
    ReasoningRecord,
    ValidationVerdict,
)
from src.reasoning.audit import ReasoningAudit, Trace
from src.reasoning.endpoint import ChatEndpoint, ReasoningUnavailableError
from src.reasoning.supervisor_tools import PROPOSE_SETPOINT, SupervisorTools

LOGGER = logging.getLogger(__name__)


def supervisor_prompt(config: Config) -> str:
    """The system prompt, with the policy numbers taken from configuration.

    The policy is the occupant's, not the model's: a comfort target, how far
    to relax an empty room, how far to shift at peak. The model's job is to
    read the room and apply that policy in words a person can check.
    """
    comfort_c = config.controller.default_setpoint_c
    relax_c = config.reasoning.vacancy_relax_c
    return (
        "You are the supervisor of one room's air conditioning. You never "
        "control the air conditioner. You read the room with your tools and "
        "then propose one setpoint goal with propose_setpoint, which a safety "
        "validator checks and may clamp or refuse.\n"
        "Policy:\n"
        f"- While the room is occupied, aim for {comfort_c:.1f} C.\n"
        f"- While it is empty, you may relax it by up to {relax_c:.1f} C "
        "warmer to save energy.\n"
        "- While the tariff is peak, shift the goal up by the offset "
        "get_tariff_state reports.\n"
        "- While any fault is active, hold the setpoint already in force "
        "rather than move it.\n"
        "Read what you need first. Then call propose_setpoint exactly once, "
        "passing the mode get_active_faults reports, and a rationale of one "
        "plain sentence citing what you read."
    )


@dataclass(frozen=True)
class SupervisorRun:
    """What one run produced, for the service and for tests."""

    record: ReasoningRecord
    goal: Goal | None = None
    verdict: ValidationVerdict | None = None


class EnvironmentalSupervisor:
    """Reads the room through its tools and proposes a goal."""

    def __init__(
        self,
        config: Config,
        clock: Clock,
        endpoint: ChatEndpoint,
        tools: SupervisorTools,
        audit: ReasoningAudit,
    ) -> None:
        self._config = config
        self._clock = clock
        self._endpoint = endpoint
        self._tools = tools
        self._audit = audit

    def run(self, trigger: str) -> SupervisorRun:
        """One supervisory cycle. Never raises for a model or server failure."""
        now = local_time(self._clock.now(), self._config.site.utc_offset_h)
        request = (
            f"Trigger: {trigger}. Local time is "
            f"{now.isoformat(timespec='minutes')}. Decide the setpoint goal."
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": supervisor_prompt(self._config)},
            {"role": "user", "content": request},
        ]
        trace = Trace(CallSite.SUPERVISOR, trigger, request)
        schemas = self._tools.schemas()

        for _ in range(self._config.reasoning.supervisor_max_rounds):
            try:
                completion = self._endpoint.complete(messages, tools=schemas)
            except ReasoningUnavailableError as exc:
                trace.add_failure(exc.latency_s)
                return self._finish(trace, ReasoningOutcome.UNAVAILABLE, str(exc))
            trace.add(completion)
            messages.append(completion.as_message())

            if not completion.tool_calls:
                return self._finish(
                    trace,
                    ReasoningOutcome.NO_ACTION,
                    "answered without proposing; the previous goal stands",
                )

            for call in completion.tool_calls:
                answer = self._tools.execute(call.name, call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": answer.as_text(),
                    }
                )
                if answer.discarded:
                    return self._finish(
                        trace, ReasoningOutcome.DISCARDED, answer.discarded
                    )
                if call.name == PROPOSE_SETPOINT.name and answer.proposed is not None:
                    return self._finish_proposed(trace, answer.proposed, answer.verdict)

        return self._finish(
            trace,
            ReasoningOutcome.DISCARDED,
            f"no proposal within {self._config.reasoning.supervisor_max_rounds} "
            f"rounds; the previous goal stands",
        )

    def _finish(
        self, trace: Trace, outcome: ReasoningOutcome, reason: str
    ) -> SupervisorRun:
        return SupervisorRun(record=self._audit.publish(trace, outcome, reason=reason))

    def _finish_proposed(
        self, trace: Trace, goal: Goal, verdict: ValidationVerdict | None
    ) -> SupervisorRun:
        applied = f"proposed {goal.setpoint_c:.1f} C in {goal.mode.value}"
        if verdict is not None:
            applied += (
                f"; gate {verdict.verdict.value} ({verdict.reason.value}), "
                f"applied {verdict.applied.get('setpoint_c')}"
            )
        record = self._audit.publish(
            trace, ReasoningOutcome.APPLIED, reason=goal.rationale, applied=applied
        )
        return SupervisorRun(record=record, goal=goal, verdict=verdict)


class SupervisorSchedule:
    """When the supervisor runs (FR-41).

    On its cadence, and on the events it is handed -- an occupancy
    transition, a tariff transition, a newly confirmed fault -- but never two
    runs closer than the configured minimum. An event that arrives inside the
    spacing is not lost: it waits for the next opportunity and names itself
    as the trigger then.
    """

    def __init__(self, config: Config, clock: Clock) -> None:
        self._period_s = config.reasoning.supervisor_period_s
        self._min_interval_s = config.reasoning.supervisor_min_interval_s
        self._clock = clock
        self._last_run_s: float | None = None
        self._pending: list[str] = []

    def notice(self, events: list[str]) -> None:
        self._pending.extend(events)
        # Bounded: only the most recent few are worth naming as a trigger.
        del self._pending[:-8]

    def due(self) -> str | None:
        """The trigger to run for now, or None if it is not time."""
        now = self._clock.monotonic()
        if self._last_run_s is None:
            return self._start("startup", now)
        since = now - self._last_run_s
        if self._pending and since >= self._min_interval_s:
            return self._start("; ".join(self._pending), now)
        if since >= self._period_s:
            return self._start("cadence", now)
        return None

    def _start(self, trigger: str, now: float) -> str:
        self._last_run_s = now
        self._pending = []
        return trigger
