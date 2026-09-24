"""The append-only record of every reasoning invocation (FR-46, FR-63).

One :class:`ReasoningRecord` per call, published to ``space/audit/reasoning``
whatever happened: applied, discarded, asked for nothing, or never reached the
model at all. The recorder writes the topic to disk (FR-62), so "append-only"
is a property of the blackboard rather than of a file this process could
rewrite.

A :class:`Trace` gathers one invocation's rounds as they happen, so a call
site that runs three model turns and four tools reports them as one decision
with one latency, rather than as seven fragments nobody can reassemble.
"""

from __future__ import annotations

import itertools
import logging

from src.common import topics
from src.common.clock import Clock
from src.common.mqtt_client import Blackboard
from src.common.schemas import CallSite, ReasoningOutcome, ReasoningRecord
from src.reasoning.endpoint import Completion

LOGGER = logging.getLogger(__name__)

#: Appended where text was cut, so a truncated record never reads as whole.
_TRUNCATION_MARK = " [...]"

#: Separates the turns of a multi-round call in ``raw_output``.
_TURN_SEPARATOR = "\n---\n"

#: Most turns one trace keeps text for. A call site is bounded in rounds
#: already; this is the collection bound that holds even if one were not.
MAX_TURNS_KEPT = 16


class Trace:
    """One invocation, accumulated round by round."""

    def __init__(self, call_site: CallSite, trigger: str, inputs: str) -> None:
        self.call_site = call_site
        self.trigger = trigger
        self.inputs = inputs
        self.rounds = 0
        self.latency_s = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._outputs: list[str] = []
        self._tool_calls: list[str] = []

    def add(self, completion: Completion) -> None:
        """Account for one model turn."""
        self.rounds += 1
        self.latency_s += completion.latency_s
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        if len(self._outputs) < MAX_TURNS_KEPT:
            text = completion.content
            if completion.tool_calls:
                calls = "; ".join(
                    f"{call.name}({call.arguments})" for call in completion.tool_calls
                )
                text = f"{text}\n[tool calls] {calls}".strip()
            self._outputs.append(text)
        for call in completion.tool_calls:
            if len(self._tool_calls) < MAX_TURNS_KEPT * 4:
                self._tool_calls.append(call.name)

    @property
    def tool_calls(self) -> tuple[str, ...]:
        return tuple(self._tool_calls)

    @property
    def raw_output(self) -> str:
        return _TURN_SEPARATOR.join(self._outputs)


class ReasoningAudit:
    """Publishes the record of each invocation."""

    def __init__(self, clock: Clock, blackboard: Blackboard, max_chars: int) -> None:
        if max_chars <= len(_TRUNCATION_MARK):
            raise ValueError(f"max_chars must leave room for text, got {max_chars}")
        self._clock = clock
        self._blackboard = blackboard
        self._max_chars = max_chars
        self._sequence = itertools.count()

    def publish(
        self,
        trace: Trace,
        outcome: ReasoningOutcome,
        reason: str = "",
        applied: str = "",
    ) -> ReasoningRecord:
        """Record one finished invocation on the blackboard."""
        now = self._clock.now()
        record = ReasoningRecord(
            ts=now,
            invocation_id=(
                f"rsn_{int(now)}_{trace.call_site.value}_{next(self._sequence)}"
            ),
            call_site=trace.call_site,
            trigger=trace.trigger,
            inputs=self._bounded(trace.inputs),
            raw_output=self._bounded(trace.raw_output),
            tool_calls=trace.tool_calls,
            rounds=trace.rounds,
            outcome=outcome,
            reason=self._bounded(reason),
            applied=self._bounded(applied),
            latency_s=trace.latency_s,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
        )
        self._blackboard.publish(topics.AUDIT_REASONING, record)
        LOGGER.info(
            "%s (%s): %s in %.2f s, %d+%d tokens%s",
            trace.call_site.value,
            trace.trigger,
            outcome.value,
            trace.latency_s,
            trace.prompt_tokens,
            trace.completion_tokens,
            f" -- {reason}" if reason else "",
        )
        return record

    def _bounded(self, text: str) -> str:
        """Cut to the configured size, saying so. Bounds every record."""
        if len(text) <= self._max_chars:
            return text
        keep = self._max_chars - len(_TRUNCATION_MARK)
        return text[:keep] + _TRUNCATION_MARK
