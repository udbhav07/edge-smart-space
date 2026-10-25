"""Overheard speech: a plan pencilled in, a fact offered (FR-76 to FR-78).

When the room listens continuously, what is said without the wake word
arrives on ``space/context/ambient``. Nobody asked for anything, so nothing
here answers, and nothing here can book, send or set: the only tools it
invokes are ``pencil_event`` -- a calendar entry marked tentative, which the
occupant keeps or discards -- and ``suggest_fact`` -- a fact held until the
person keeps it. Both are ``write`` tools and cross the blackboard to the
executor like every other invocation (FR-71); neither is offered to the
assistant.

One single-shot call per overheard line, schema-constrained and validated
after decoding (FR-44), with the voice's last few lines as context, so "we
should see the dentist" followed by "Thursday at four works" is one plan.
Calls are rate-limited: a busy room must not starve the assistant and the
supervisor of the one inference server. A line skipped for the rate stays in
the context and is read with the next.

The day a plan falls on is taken from the words where they name one, not
from the model (FR-44, as for the assistant), and a plan already in the past
is dropped rather than written.

MQTT callbacks only queue work; the calls run on the supervisory thread, so
the assistant answering someone who did address it never waits behind one.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.common import topics
from src.common.clock import Clock
from src.common.config import AmbientConfig
from src.common.localtime import local_time
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    GUEST_PROFILE_ID,
    AmbientUtterance,
    ReasoningCaller,
    ReasoningRecord,
)
from src.common.tools import PENCIL_EVENT, SUGGEST_FACT, ToolInvocation, ToolRequester
from src.reasoning.chat import ChatClient
from src.reasoning.dates import annotation, on_day, the_one_day
from src.reasoning.single_shot import ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Overheard lines waiting for a call. Bounded: in a burst the oldest go.
_QUEUE_LENGTH = 32

#: Voices whose context is kept at once. Bounded, like everything else.
_VOICES_KEPT = 16

#: What one call may yield. More than this from one sentence is a model
#: inventing, not a person planning.
_MAX_PLANS = 3
_MAX_FACTS = 3

_SECONDS_PER_MINUTE = 60.0

AMBIENT_PROMPT = (
    "You listen to a conversation in someone's home. Nobody is talking to "
    "you; do not reply to them. Find only two things. (1) Plans: a specific "
    "meeting, appointment, trip, flight, train or event they intend to "
    "attend, with a day you can name. (2) Facts: something lasting about "
    "the person speaking -- a preference, a routine, an allergy -- worth "
    "remembering for later conversations. Ignore small talk, opinions about "
    "the moment, things about other people, and anything hypothetical or "
    "already past. Most of the time there is nothing: then return empty "
    "lists. Reply with a JSON object and nothing else: "
    '{"plans": [{"subject": short title, "starts_at": ISO-8601 local date '
    'and time, "duration_min": minutes or null}], "facts": [one short '
    'sentence about them in the third person, without their name]}. If a '
    "plan names a day but no time, use 09:00."
)


class _Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    subject: str = Field(min_length=1, max_length=80)
    starts_at: datetime
    duration_min: int | None = Field(default=None, ge=5, le=1440)


class _Findings(BaseModel):
    """What one call may say, checked after decoding (FR-44)."""

    model_config = ConfigDict(extra="ignore")

    plans: list[_Plan] = Field(default_factory=list, max_length=_MAX_PLANS)
    facts: list[str] = Field(default_factory=list, max_length=_MAX_FACTS)


class AmbientMiner:
    """Looks in overheard speech for plans and facts, and proposes nothing else."""

    def __init__(
        self,
        config: AmbientConfig,
        clock: Clock,
        blackboard: Blackboard,
        chat: ChatClient,
        utc_offset_h: float,
        confirmation_window_s: float,
        max_fact_chars: int,
    ) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._chat = chat
        self._utc_offset_h = utc_offset_h
        self._window_s = confirmation_window_s
        self._max_fact_chars = max_fact_chars
        self._pending: deque[AmbientUtterance] = deque(maxlen=_QUEUE_LENGTH)
        #: Each voice's recent lines, as (ts, words). Guest lines share one.
        self._heard: dict[str, deque[tuple[float, str]]] = {}
        self._calls: deque[float] = deque()
        self._invocations = 0

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.CONTEXT_AMBIENT, AmbientUtterance, self._on_heard)

    def _on_heard(self, _topic: str, heard: AmbientUtterance) -> None:
        self._pending.append(heard)

    @property
    def pending(self) -> int:
        return len(self._pending)

    def process_pending(self) -> list[ToolInvocation]:
        """Mine every line waiting. Returns what was proposed."""
        proposed: list[ToolInvocation] = []
        while self._pending:
            proposed.extend(self.mine(self._pending.popleft()))
        return proposed

    # --- one line ------------------------------------------------------

    def mine(self, heard: AmbientUtterance) -> list[ToolInvocation]:
        """Read one overheard line, with what came before it from that voice."""
        voice = heard.profile_id or GUEST_PROFILE_ID
        earlier = self._context_of(voice)
        self._remember_line(voice, heard)
        if not self._may_call():
            LOGGER.info("overheard speech not read now: the call rate is spent")
            return []

        now = local_time(self._clock.now(), self._utc_offset_h)
        inputs = self._inputs(heard.transcript, earlier, now)
        raw, latency_s, tokens, verdict = "", 0.0, (0, 0), ""
        findings: _Findings | None = None
        try:
            turn = self._chat.complete(
                [{"role": "system", "content": AMBIENT_PROMPT}, {"role": "user", "content": inputs}],
                json_only=True,
            )
            raw, latency_s = turn.content, turn.latency_s
            tokens = (turn.prompt_tokens, turn.completion_tokens)
            findings = _validated(raw)
            if findings is None:
                verdict = "discarded: failed validation"
        except ReasoningUnavailableError as exc:
            LOGGER.warning("overheard speech not read: %s", exc)
            verdict = "reasoning unavailable"

        proposed: list[ToolInvocation] = []
        if findings is not None:
            said = " ".join([*earlier, heard.transcript])
            proposed += self._pencil(findings.plans, heard.transcript, said, now)
            proposed += self._suggest(findings.facts, heard.transcript, heard.profile_id)
            verdict = _summary(findings, proposed, heard.profile_id)
        self._record(inputs, raw, verdict, proposed, latency_s, tokens)
        return proposed

    def _pencil(self, plans: list[_Plan], line: str, said: str, now: datetime) -> list[ToolInvocation]:
        """Each plan on the day the words name, and only if still ahead."""
        named = the_one_day(said, now.date())
        proposed = []
        for plan in plans:
            starts = plan.starts_at.replace(tzinfo=None)
            if named is not None:
                moved = on_day(starts.isoformat(timespec="seconds"), named)
                starts = datetime.fromisoformat(moved) if moved else starts
            if starts < now:
                LOGGER.info("not pencilling %r: %s is already past", plan.subject, starts)
                continue
            arguments = {
                "subject": plan.subject.strip(),
                "starts_at": starts.isoformat(timespec="seconds"),
                "heard": line,
            }
            if plan.duration_min:
                arguments["duration_min"] = plan.duration_min
            proposed.append(self._propose(PENCIL_EVENT.name, arguments, line))
        return proposed

    def _suggest(self, facts: list[str], line: str, profile_id: str) -> list[ToolInvocation]:
        """Offer each fact to a recognised voice; a guest has nowhere to keep one (FR-78)."""
        if not profile_id or profile_id == GUEST_PROFILE_ID:
            return []
        proposed = []
        for fact in facts:
            text = " ".join(str(fact).split())
            if not text or len(text) > self._max_fact_chars:
                continue
            arguments = {"profile_id": profile_id, "fact": text, "heard": line}
            proposed.append(self._propose(SUGGEST_FACT.name, arguments, line))
        return proposed

    def _propose(self, tool: str, arguments: dict, line: str) -> ToolInvocation:
        now = self._clock.now()
        self._invocations += 1
        invocation = ToolInvocation(
            ts=now,
            invocation_id=f"amb_{int(now)}_{self._invocations}",
            tool=tool,
            arguments=arguments,
            requester=ToolRequester.AMBIENT,
            rationale=line,
            expires_ts=now + self._window_s,
        )
        self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)
        LOGGER.info("overheard -> %s %s", tool, arguments)
        return invocation

    # --- context and rate ------------------------------------------------

    def _context_of(self, voice: str) -> list[str]:
        """The voice's earlier lines still within the window, oldest first."""
        lines = self._heard.get(voice)
        if not lines:
            return []
        horizon = self._clock.now() - self._config.context_s
        while lines and lines[0][0] < horizon:
            lines.popleft()
        return [words for _, words in lines]

    def _remember_line(self, voice: str, heard: AmbientUtterance) -> None:
        if voice not in self._heard and len(self._heard) >= _VOICES_KEPT:
            # The voice heard from longest ago makes room.
            oldest = min(self._heard, key=lambda v: self._heard[v][-1][0] if self._heard[v] else 0.0)
            del self._heard[oldest]
        lines = self._heard.setdefault(voice, deque(maxlen=self._config.max_lines))
        lines.append((self._clock.now(), heard.transcript))

    def _may_call(self) -> bool:
        now = self._clock.monotonic()
        while self._calls and now - self._calls[0] >= _SECONDS_PER_MINUTE:
            self._calls.popleft()
        if len(self._calls) >= self._config.max_calls_per_min:
            return False
        self._calls.append(now)
        return True

    @staticmethod
    def _inputs(line: str, earlier: list[str], now: datetime) -> str:
        parts = [f"Now: {now:%A} {now.isoformat(timespec='minutes')}."]
        if earlier:
            parts.append("Said earlier: " + " / ".join(earlier))
        parts.append(f"Just said: {line}")
        dates = annotation(" ".join([*earlier, line]), now.date(), now)
        if dates:
            parts.append(dates)
        return "\n".join(parts)

    def _record(self, inputs: str, raw: str, verdict: str, proposed, latency_s: float, tokens) -> None:
        self._blackboard.publish(
            topics.AUDIT_REASONING,
            ReasoningRecord(
                ts=self._clock.now(),
                caller=ReasoningCaller.AMBIENT,
                trigger="overheard",
                model=self._chat.model,
                inputs=inputs,
                raw_output=raw,
                verdict=verdict,
                applied="; ".join(f"{i.tool} {i.arguments.get('subject') or i.arguments.get('fact')}" for i in proposed),
                tool_calls=tuple(invocation.tool for invocation in proposed),
                latency_s=latency_s,
                prompt_tokens=tokens[0],
                completion_tokens=tokens[1],
            ),
        )


def _validated(raw: str) -> _Findings | None:
    """Post-decode validation (FR-44): anything malformed is nothing."""
    try:
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            return None
        return _Findings.model_validate(decoded)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        LOGGER.warning("discarding what was made of overheard speech: %s", exc)
        return None


def _summary(findings: _Findings, proposed: list[ToolInvocation], profile_id: str) -> str:
    plans = sum(1 for invocation in proposed if invocation.tool == PENCIL_EVENT.name)
    facts = sum(1 for invocation in proposed if invocation.tool == SUGGEST_FACT.name)
    if not findings.plans and not findings.facts:
        return "accepted: nothing to note"
    notes = [f"{plans} plan(s) pencilled", f"{facts} fact(s) offered"]
    if len(findings.plans) > plans:
        notes.append(f"{len(findings.plans) - plans} past plan(s) dropped")
    if findings.facts and not profile_id:
        notes.append("facts from a guest not held")
    return "accepted: " + ", ".join(notes)
