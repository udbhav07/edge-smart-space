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
import re
from collections import deque
from datetime import datetime, timedelta

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
from src.reasoning.dates import annotation, from_now, named_days, on_day, the_one_day
from src.reasoning.single_shot import ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Overheard lines waiting for a call. Bounded: in a burst the oldest go.
_QUEUE_LENGTH = 32

#: Voices whose context is kept at once. Bounded, like everything else.
_VOICES_KEPT = 16

#: A time said outright: "at four", "by 10:30", "6 pm", "noon". A plan needs
#: one of these or a named day in the words themselves (S-01): live, room audio
#: with neither -- "We'll get to them", "good night" -- became calendar entries.
_TIME_SAID = re.compile(
    r"\b(?:at|by|from|around|before|after|until)\s+(?:\d{1,2}(?:[:.]\d{2})?|one|two|three|four|five|six|"
    r"seven|eight|nine|ten|eleven|twelve|half past \w+|quarter (?:past|to) \w+)\b"
    r"|\b\d{1,2}(?:[:.]\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|o'clock)\b"
    r"|\b(?:noon|midday|midnight)\b",
    re.IGNORECASE,
)

#: The nearest "in N minutes" that counts as a plan rather than a turn of phrase.
_SHORTEST_PLAN_AHEAD = timedelta(minutes=30)

#: Words too common to tell two plans apart.
_FILLER = frozenset({"the", "and", "with", "for", "our", "my", "to", "at", "on", "in", "a", "an"})

#: What one call may yield. More than this from one sentence is a model
#: inventing, not a person planning.
_MAX_PLANS = 3
_MAX_FACTS = 3

_SECONDS_PER_MINUTE = 60.0

#: Live, a 7B model echoed one plan twice ("Dentist", then "appointment")
#: and inferred a routine nobody stated. The rules and examples below are
#: what stopped it; the per-day guard in ``_pencil`` catches what slips by.
AMBIENT_PROMPT = (
    "You listen to a conversation in someone's home. Nobody is talking to "
    "you; do not reply to them. Find only two things.\n"
    "(1) Plans: a specific meeting, appointment, trip, flight, train or "
    "event the speaker intends to attend, on a day you can name. Name each "
    "plan once, by what it is (\"Dentist\", \"Flight to Goa\"), never as a "
    "generic \"appointment\" or \"event\". Use the time they said, as people "
    "mean it without am or pm: dinner at eight is 20:00, the dentist at four "
    "is 16:00, breakfast at eight is 08:00. If they named a day but no time, "
    "use 09:00.\n"
    "(2) Facts: something lasting the speaker says outright about "
    "themselves -- a preference, an allergy, a routine -- in their own "
    "words. Never infer a fact from a plan, and never guess.\n"
    "Ignore small talk, the moment's opinions, other people, and anything "
    "hypothetical or past. Most of the time there is nothing: return empty "
    "lists.\n"
    "Report only what the line marked \"Just said\" adds. Earlier lines are "
    "there to complete it -- a day, a time -- and were already read: never "
    "report a plan or fact from them again.\n"
    "Reply with a JSON object and nothing else: "
    '{"plans": [{"subject": short title, "starts_at": ISO-8601 local date '
    'and time, "duration_min": minutes or null}], "facts": [one short '
    "sentence about them in the third person, without their name]}.\n"
    "Examples. \"I've got the dentist on Thursday at four\" -> one plan, "
    "Dentist, that Thursday at 16:00; no facts. \"I'm vegetarian, so no "
    "meat for me\" -> no plans; facts [\"Is vegetarian.\"]. \"Lovely "
    'weather today" -> {"plans": [], "facts": []}.'
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
        #: What each voice's window has already produced, as (ts, key), so a
        #: plan or fact re-read from an earlier line is not proposed twice.
        #: Live, the model re-reported the earlier lines it was given as context.
        self._noted: dict[str, deque[tuple[float, str]]] = {}
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
            proposed += self._pencil(findings.plans, heard.transcript, said, now, voice)
            proposed += self._suggest(findings.facts, heard.transcript, heard.profile_id, said)
            verdict = _summary(findings, proposed, heard.profile_id)
        self._record(inputs, raw, verdict, proposed, latency_s, tokens)
        return proposed

    def _pencil(self, plans: list[_Plan], line: str, said: str, now: datetime, voice: str) -> list[ToolInvocation]:
        """Each plan on the day the words name, and only if still ahead."""
        # The day the words name corrects a single plan; with several, which
        # one it belongs to is not knowable from the words alone.
        named = the_one_day(said, now.date()) if len(plans) == 1 else None
        proposed = []
        days: set = set()
        for plan in plans:
            starts = plan.starts_at.replace(tzinfo=None)
            if named is not None:
                moved = on_day(starts.isoformat(timespec="seconds"), named)
                starts = datetime.fromisoformat(moved) if moved else starts
            if starts < now:
                LOGGER.info("not pencilling %r: %s is already past", plan.subject, starts)
                continue
            if starts.date() in days:
                # One overheard line almost never holds two plans on one day;
                # live, the second was the first echoed under another name.
                LOGGER.info("not pencilling %r: this answer already has a plan that day", plan.subject)
                continue
            if not anchored(said, now):
                # A plan is a day or a time somebody said. Without one it is
                # a phrase that sounded like a plan (S-01).
                LOGGER.info("not pencilling %r: no day or time was said", plan.subject)
                continue
            if not grounded(plan.subject, said):
                # Live, "what a long day" came back as the prompt's own
                # example dentist: a plan must be about words actually said.
                LOGGER.info("not pencilling %r: nothing said names it", plan.subject)
                continue
            if self._already_noted(voice, _plan_key(plan.subject)):
                LOGGER.info("not pencilling %r: an earlier line already gave it", plan.subject)
                continue
            days.add(starts.date())
            arguments = {
                "subject": plan.subject.strip(),
                "starts_at": starts.isoformat(timespec="seconds"),
                "heard": line,
            }
            if plan.duration_min:
                arguments["duration_min"] = plan.duration_min
            proposed.append(self._propose(PENCIL_EVENT.name, arguments, line))
        return proposed

    def _suggest(self, facts: list[str], line: str, profile_id: str, said: str) -> list[ToolInvocation]:
        """Offer each fact to a recognised voice; a guest has nowhere to keep one (FR-78)."""
        if not profile_id or profile_id == GUEST_PROFILE_ID:
            return []
        proposed = []
        for fact in facts:
            text = " ".join(str(fact).split())
            if not text or len(text) > self._max_fact_chars or not grounded(text, said):
                continue
            if self._already_noted(profile_id, f"fact:{text.casefold().rstrip('.')}"):
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

    def _already_noted(self, voice: str, key: str) -> bool:
        """Whether this voice's window already produced it; notes it if not.

        Plans match on their telling words either way round, so "dentist"
        and "dentist appointment" are one plan.
        """
        if not key:
            return False
        noted = self._noted.setdefault(voice, deque(maxlen=self._config.max_lines * (_MAX_PLANS + _MAX_FACTS)))
        horizon = self._clock.now() - self._config.context_s
        while noted and noted[0][0] < horizon:
            noted.popleft()
        for _, seen in noted:
            if seen == key or (key.startswith("plan:") and seen.startswith("plan:") and _same_words(seen, key)):
                return True
        noted.append((self._clock.now(), key))
        return False

    def _remember_line(self, voice: str, heard: AmbientUtterance) -> None:
        if voice not in self._heard and len(self._heard) >= _VOICES_KEPT:
            # The voice heard from longest ago makes room.
            oldest = min(self._heard, key=lambda v: self._heard[v][-1][0] if self._heard[v] else 0.0)
            del self._heard[oldest]
            self._noted.pop(oldest, None)
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


#: Letters two words must share at the start to count as the same word, so
#: "flying" grounds "flight"'s plan through "goa", and "allergic" matches
#: "allergy". Short enough for inflection, long enough not to match by chance.
_STEM = 4


def _telling_words(text: str) -> set[str]:
    words = "".join(c if c.isalnum() else " " for c in text.lower()).split()
    return {word for word in words if word not in _FILLER and len(word) > 2}


def anchored(said: str, now: datetime) -> bool:
    """Whether the words name a day or a time a plan could be on (S-01)."""
    ahead = from_now(said)
    # "In a minute" is an idiom, not an appointment; "in two hours" can be one.
    soon_enough = ahead is not None and ahead >= _SHORTEST_PLAN_AHEAD
    return bool(named_days(said, now.date()) or soon_enough or _TIME_SAID.search(said))


def grounded(claim: str, said: str) -> bool:
    """Whether a plan or fact is about something actually said (FR-44).

    At least one of its telling words must share its first letters with a
    word heard. It cannot prove the model right; it does stop a model
    reporting what nobody said at all.
    """
    heard = {word[:_STEM] for word in _telling_words(said)}
    return any(word[:_STEM] in heard for word in _telling_words(claim))


def _plan_key(subject: str) -> str:
    """A plan's telling words, as a key: lower case, filler dropped, sorted."""
    words = "".join(c if c.isalnum() else " " for c in subject.lower()).split()
    telling = sorted({word for word in words if word not in _FILLER and len(word) > 2})
    return "plan:" + " ".join(telling) if telling else ""


def _same_words(one: str, other: str) -> bool:
    mine, theirs = set(one[5:].split()), set(other[5:].split())
    return bool(mine and theirs) and (mine <= theirs or theirs <= mine)


def _validated(raw: str) -> _Findings | None:
    """Post-decode validation (FR-44), item by item.

    A plan that does not validate is dropped and the rest kept: live, the
    model answered one real plan and an empty ``{}`` beside it, and losing
    the dentist over the stray object helped nobody. An answer that is not
    the shape asked for at all, or holds more than one line could, is
    nothing.
    """
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        LOGGER.warning("discarding what was made of overheard speech: %s", exc)
        return None
    if not isinstance(decoded, dict):
        return None
    plans, facts = decoded.get("plans") or [], decoded.get("facts") or []
    if not isinstance(plans, list) or not isinstance(facts, list):
        return None
    if len(plans) > _MAX_PLANS or len(facts) > _MAX_FACTS:
        LOGGER.warning("discarding what was made of overheard speech: more than one line holds")
        return None
    kept: list[_Plan] = []
    for item in plans:
        try:
            kept.append(_Plan.model_validate(item))
        except (ValidationError, ValueError, TypeError) as exc:
            LOGGER.info("dropping one plan made of overheard speech: %s", str(exc).splitlines()[0])
    return _Findings(plans=kept, facts=[fact for fact in facts if isinstance(fact, str) and fact.strip()])


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
