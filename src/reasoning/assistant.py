"""Acting on what an occupant said, and saying what happened (sections 5.7.1, 5.7.6).

The speech pipeline turns an utterance into a :class:`PreferenceHint` and
publishes it; that is where it stops, and it is left exactly so. This module
listens for those hints and does the rest:

* **A service request** -- a meeting, a question about the calendar, a
  flight -- gets one bounded round of tool calls (FR-42,
  ``assistance.max_tool_rounds``), each published to ``space/assist/proposed``
  and answered by the executor, never by a provider held here (FR-71). Then a
  second completion, with no tools, turns the results into a reply.
* **A temperature request** is proposed by the control service's goal path,
  not here (FR-45, FR-53). This only waits for the validator's verdict on it
  and says what became of it -- including, plainly, when the gate clamped it.
  That reply is written from the verdict rather than by the model: a sentence
  about a safety decision must not be able to misstate it.

Every completion is recorded on ``space/audit/reasoning`` with its inputs,
raw output, verdict, latency and token counts (FR-46, FR-63), and every reply
goes to ``space/context/reply``.

**Who is speaking personalises, and permits nothing (FR-52).** A hint names
the recognised voice, and the person's context -- what they asked to be
remembered, and the conversation so far -- is read off
``space/context/profile/{profile_id}``, never out of a file. The model is
told it and may keep a new fact with ``remember``; whose memory that is, is
filled in here from the voice and never by the model. A guest is offered no
memory tools. Every tool, setpoint and booking is gated exactly as for
anyone, and the first reply of a conversation says so.

MQTT callbacks only queue work. Waiting for a tool result inside a callback
would wait for a message the same thread has to deliver.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from datetime import datetime, timedelta

from src.common import topics
from src.common.clock import Clock
from src.common.config import AssistanceConfig, Bounds
from src.common.localtime import local_time
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    GUEST_PROFILE_ID,
    SETPOINT_KEY,
    SOURCE_KEY,
    THERMAL_SUBJECTS,
    AssistantReply,
    ProfileContext,
    GoalSource,
    Intent,
    Mode,
    PreferenceHint,
    ReasonCode,
    ReasoningCaller,
    ReasoningRecord,
    ValidationVerdict,
    Verdict,
)
from src.common.tools import (
    ASSISTANCE_TOOLS,
    PROFILE_TOOLS,
    REMEMBER,
    ToolInvocation,
    ToolRequester,
    ToolResult,
    ToolStatus,
)
from src.reasoning.chat import (
    ChatClient,
    ChatTurn,
    ToolCall,
    assistant_message,
    tool_message,
)
from src.reasoning.dates import annotation, from_now, named_days, on_day, the_one_day, time_said
from src.reasoning.room_state import RoomState
from src.reasoning.understanding import memory_request
from src.reasoning.single_shot import ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Utterances waiting to be answered. Bounded: a burst of speech while the
#: model is slow must not grow without limit (NFR-05).
_QUEUE_LENGTH = 16

#: Results and verdicts kept while a caller waits for its own. Bounded.
_MAILBOX_LENGTH = 64

#: People whose context is held. Bounded; the store bounds them too.
_PROFILES_KEPT = 32

#: Said once per conversation to a recognised person (FR-52): that the voice
#: was recognised, and that recognition unlocks nothing.
_RECOGNITION_NOTE = (
    "(I recognised your voice as {name}. That only personalises my replies; "
    "it doesn't unlock anything.)"
)
#: The same, for words typed in the console as someone (S-12).
_CHATTING_AS_NOTE = (
    "(You're chatting as {name}. That only personalises my replies; "
    "it doesn't unlock anything.)"
)

#: "What do you know about me?" -- asked of the speaker's own memory.
_ABOUT_ME = re.compile(r"\b(me|my|myself|i)\b|\bwhat\s+do\s+you\s+remember\b", re.IGNORECASE)

#: How the model is told who it is speaking with.
_PERSON_PROMPT = "You are speaking with {name}."
_FACTS_PROMPT = "They asked you to remember:\n{facts}"
_PREFERENCES_PROMPT = "Their stated preferences:\n{preferences}"
_RECENT_PROMPT = "Earlier in this conversation:\n{turns}"
_MEMORY_PROMPT = (
    "If they ask you to remember something about them, or state a lasting "
    "preference, keep it with remember; if they ask you to forget, use forget."
)

#: How often a wait re-checks its mailbox, in seconds.
_POLL_S = 0.05

TRIGGER = "utterance"

ASSISTANT_PROMPT = (
    "You are the voice assistant of a smart room, acting for the person who "
    "lives in it. It is now {now}. The coming days are:\n{days}\n"
    "Take every date from that list -- never work a weekday out yourself -- "
    "and pass dates and times to tools as ISO-8601 local date-times such as "
    "2026-10-01T15:00:00. A weekday means the next one in the list; 'next "
    "Monday' means the same. When a time of day is not said, use 09:00 for a "
    "start, and for a question about a day use 00:00 to 23:59 of that day. "
    "Use a tool whenever the request needs one: "
    "schedule_event to add to their calendar, get_events to answer what is "
    "planned, cancel_event or move_event to cancel or move an entry -- never "
    "schedule an entry to record a cancellation -- book_travel for a flight "
    "or hotel, and remember or forget when "
    "they ask you to keep or drop something about them. Call each tool at "
    "most once. "
    "If the request cannot be done with these tools, say so in one sentence "
    "without calling any. Answer a question about the room, its temperature "
    "or the system from 'The room now' below and nothing else, and reply to "
    "a greeting in one friendly sentence."
)

ANSWER_PROMPT = (
    "You are the voice assistant of a smart room. It is now {now}. Tell the "
    "person, in one or two short spoken sentences, what happened with their "
    "request, using only the tool results given. Say times the way people "
    "speak them. Never claim something was done if the result does not say "
    "so. If a booking needs their confirmation, say that nothing has been "
    "booked yet and ask whether to go ahead. If a result says it is a mock, "
    "say it was a mock booking and no real reservation was made."
)

_NO_TOOL_REPLY = "Sorry, I could not do that."

#: A reply written as though something was done.
_CLAIMS_DONE = re.compile(
    r"\b(cancell?ed|added|scheduled|rescheduled|booked|moved|removed|deleted|"
    r"remembered|forgotten|reserved|put\s+(it|that)\s+in)\b"
    # A confirmation question is the system's to ask, from a real refusal;
    # written by the model with nothing behind it, it promises a booking.
    r"|\bshall\s+i\s+go\s+ahead\b|\bmock\s+booking\b",
    re.IGNORECASE,
)

_NO_TOOL_NUDGE = (
    "You answered without calling a tool. If the occupant asked you to do "
    "something -- book, schedule, cancel, move, remember -- call the tool now "
    "with its arguments. Only the system asks for confirmation, after the "
    "tool is called. If nothing was asked to be done, answer plainly and do "
    "not describe anything as prepared or done."
)
_NOTHING_DONE = (
    "I haven't changed anything: I couldn't carry that out. "
    "Could you say it another way?"
)
_NO_RESULT = "The request got no answer in time, so nothing is confirmed."


#: Days listed for the model to take dates from, today included.
_DAYS_LISTED = 8


def local_now(clock: Clock, utc_offset_h: float) -> str:
    """The present at the site, as the model is told it: weekday, date and time."""
    moment = local_time(clock.now(), utc_offset_h)
    return f"{moment:%A} {moment.day} {moment:%B %Y}, {moment:%H:%M}"


def coming_days(clock: Clock, utc_offset_h: float) -> str:
    """A lookup table of weekday to date, for the model to copy from.

    A 7B model asked for "Thursday" from a Saturday answered Sunday. Working
    out a weekday is arithmetic it is bad at; reading one off a list is not.
    """
    today = local_time(clock.now(), utc_offset_h).date()
    lines = []
    for offset in range(_DAYS_LISTED):
        day = today + timedelta(days=offset)
        label = " (today)" if offset == 0 else " (tomorrow)" if offset == 1 else ""
        lines.append(f"- {day:%A} {day.day} {day:%B}{label}: {day.isoformat()}")
    return "\n".join(lines)


class Assistant:
    """Answers utterances, acting through the tool surface where needed."""

    def __init__(
        self,
        config: AssistanceConfig,
        clock: Clock,
        blackboard: Blackboard,
        chat: ChatClient,
        safe_range_c: Bounds,
        utc_offset_h: float,
    ) -> None:
        self._config = config
        self._utc_offset_h = utc_offset_h
        self._safe_range_c = safe_range_c
        self._clock = clock
        self._blackboard = blackboard
        self._chat = chat
        self._pending: deque[PreferenceHint] = deque(maxlen=_QUEUE_LENGTH)
        self._results: dict[str, ToolResult] = {}
        self._result_order: deque[str] = deque(maxlen=_MAILBOX_LENGTH)
        self._verdicts: deque[ValidationVerdict] = deque(maxlen=_MAILBOX_LENGTH)
        self._invocations = 0
        self._invocations_by_id: dict[str, ToolInvocation] = {}
        self._corrections: list[str] = []
        self._specs = {spec.name: spec for spec in ASSISTANCE_TOOLS}
        #: What is true of the room, so a reply never claims otherwise.
        self._room = RoomState()
        self._profiles: dict[str, ProfileContext] = {}
        #: The entry each voice last put in or moved, so "move it" means that
        #: one. Bounded by the voices kept.
        self._last_entry: dict[str, tuple[str, str, str]] = {}

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        self._blackboard.subscribe(
            topics.CONTEXT_PREFERENCE, PreferenceHint, self._on_hint
        )
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)
        self._blackboard.subscribe(
            topics.AUDIT_VALIDATION, ValidationVerdict, self._on_verdict
        )
        self._blackboard.subscribe(topics.CONTEXT_PROFILE, ProfileContext, self._on_profile)
        self._room.subscribe(self._blackboard)

    def _on_profile(self, _topic: str, context: ProfileContext) -> None:
        if context.profile_id not in self._profiles and len(self._profiles) >= _PROFILES_KEPT:
            self._profiles.pop(next(iter(self._profiles)))
        self._profiles[context.profile_id] = context

    def _on_hint(self, _topic: str, hint: PreferenceHint) -> None:
        if len(self._pending) == self._pending.maxlen:
            LOGGER.warning("assistant queue full; dropping the oldest utterance")
        self._pending.append(hint)

    def _on_result(self, _topic: str, result: ToolResult) -> None:
        if result.invocation_id not in self._results:
            if len(self._result_order) == self._result_order.maxlen:
                self._results.pop(self._result_order[0], None)
            self._result_order.append(result.invocation_id)
        self._results[result.invocation_id] = result

    def _on_verdict(self, _topic: str, verdict: ValidationVerdict) -> None:
        self._verdicts.append(verdict)

    # --- the work -----------------------------------------------------

    @property
    def pending(self) -> int:
        return len(self._pending)

    def process_pending(self) -> list[AssistantReply]:
        """Answer every queued utterance, oldest first. Called from the main loop."""
        replies = []
        while self._pending:
            hint = self._pending.popleft()
            reply = self.answer(hint)
            if reply is not None:
                replies.append(reply)
        return replies

    def answer(self, hint: PreferenceHint) -> AssistantReply | None:
        """Act on one hint and publish what was said back.

        :returns: the reply, or None for a hint that asked for nothing.
        """
        if hint.intent is Intent.QUESTION and hint.subject == "memory" and _ABOUT_ME.search(hint.transcript or hint.rationale):
            # Answered from the profile as it stands, never by the model:
            # live, it recited a fact forgotten a minute earlier, copied from
            # its own earlier reply, and told a guest about Ravi's meeting
            # as something it knew about them (S-05, S-06).
            reply = self._what_i_know(hint)
        elif hint.intent in (Intent.SERVICE, Intent.QUESTION):
            # A question is answered the same way a request is served: with
            # the room's state, the person's context and the read tools.
            reply = self._serve(hint)
        elif hint.intent is Intent.ENVIRONMENT:
            reply = self._report_setpoint(hint)
        else:
            return None
        reply = self._personalised(reply, hint)
        self._blackboard.publish(topics.CONTEXT_REPLY, reply)
        LOGGER.info("replied: %s", reply.reply)
        return reply

    def _what_i_know(self, hint: PreferenceHint) -> AssistantReply:
        """What is remembered about the speaker, read from their profile now."""
        person = self._recognised(hint)
        if person is None:
            text = (
                "I don't know who you are, so I keep nothing about you. "
                "Someone can enrol your voice, and then I can remember things you ask me to."
            )
        else:
            name = person.display_name or person.profile_id
            known = [_as_you(fact) for fact in person.facts]
            comfort = person.preferences.get("comfort_c")
            if comfort:
                known.insert(0, f"you like the room at {comfort} degrees")
            if known:
                listed = known[0] if len(known) == 1 else ", ".join(known[:-1]) + " and " + known[-1]
                text = f"Here is what I remember about you, {name}: {listed}. Ask me to forget any of it."
            else:
                text = f"I don't have anything remembered about you yet, {name}."
            waiting = len(person.suggestions)
            if waiting:
                text += f" {waiting} thing{'s' if waiting != 1 else ''} I overheard {'are' if waiting != 1 else 'is'} waiting for you to keep or dismiss on Chat."
        return AssistantReply(
            ts=self._clock.now(),
            transcript=hint.transcript or hint.rationale,
            intent=hint.intent,
            reply=text,
        )

    # --- who is speaking (FR-52) -------------------------------------

    def _context_of(self, hint: PreferenceHint) -> ProfileContext | None:
        return self._profiles.get(hint.profile_id or GUEST_PROFILE_ID)

    def _recognised(self, hint: PreferenceHint) -> ProfileContext | None:
        """The speaker's context, if they are a recognised person."""
        if not hint.profile_id:
            return None
        return self._profiles.get(hint.profile_id)

    def _personalised(self, reply: AssistantReply, hint: PreferenceHint) -> AssistantReply:
        """Name whose reply it is, and say once that recognition permits nothing."""
        update: dict[str, object] = {"profile_id": hint.profile_id}
        person = self._recognised(hint)
        if person is not None and not person.recent:
            note = (_CHATTING_AS_NOTE if hint.typed else _RECOGNITION_NOTE).format(
                name=person.display_name or person.profile_id
            )
            update["reply"] = f"{reply.reply} {note}"
        return reply.model_copy(update=update)

    def _person_prompt(self, hint: PreferenceHint) -> str:
        """What the model is told about the speaker, if anything is known."""
        context = self._context_of(hint)
        if context is None:
            return ""
        parts = []
        person = self._recognised(hint)
        if person is not None:
            parts.append(_PERSON_PROMPT.format(name=person.display_name or person.profile_id))
            if person.preferences:
                listed = "\n".join(f"- {key}: {value}" for key, value in person.preferences.items())
                parts.append(_PREFERENCES_PROMPT.format(preferences=listed))
            if person.facts:
                parts.append(_FACTS_PROMPT.format(facts="\n".join(f"- {fact}" for fact in person.facts)))
            parts.append(_MEMORY_PROMPT)
        if context.recent:
            turns = "\n".join(
                f"- They said: {turn.said} / You replied: {turn.replied}" for turn in context.recent
            )
            parts.append(_RECENT_PROMPT.format(turns=turns))
        return "\n".join(parts)

    def _tools_for(self, hint: PreferenceHint) -> tuple[dict[str, object], ...]:
        """The surface this speaker is offered: memory tools for a person only,
        and never the tools that only overheard speech or the console use."""
        recognised = self._recognised(hint) is not None
        return tuple(
            spec.as_schema()
            for name, spec in self._specs.items()
            if spec.offered_to_assistant and (recognised or name not in PROFILE_TOOLS)
        )

    def _memory_asked(self, hint: PreferenceHint, utterance: str) -> ToolCall | None:
        """The memory tool the words plainly ask for, for a recognised person."""
        if hint.subject != "memory" or self._recognised(hint) is None:
            return None
        request = memory_request(utterance)
        if request is None:
            return None
        tool, text = request
        argument = "fact" if tool == REMEMBER.name else "about"
        return ToolCall(call_id="from-the-words", name=tool, arguments={argument: text})

    def _entry_asked(self, hint: PreferenceHint, utterance: str) -> ToolCall | None:
        """The calendar change the words plainly ask for about the entry just
        discussed: "actually make it half past two", "cancel it" (S-03, S-04).

        Only a follow-up -- "it", "that", or "actually" -- naming no day, from
        a voice whose last entry is known. Anything else is the model's.
        """
        voice = hint.profile_id or GUEST_PROFILE_ID
        last = self._last_entry.get(voice)
        said = utterance.lower()
        if last is None or not _FOLLOW_UP.search(said) or named_days(said, self._today()):
            return None
        subject, starts_at, _event_id = last
        if _CANCEL.search(said):
            return ToolCall(call_id="from-the-words", name="cancel_event", arguments={"subject": subject, "on": starts_at[:10]})
        when = time_said(said)
        if when is None:
            return None
        entry = datetime.fromisoformat(starts_at)
        hour, minute = when.nearest_to(entry.hour)
        moved = entry.replace(hour=hour, minute=minute).isoformat(timespec="seconds")
        return ToolCall(call_id="from-the-words", name="move_event", arguments={"subject": subject, "starts_at": moved})

    def _as_asked(self, call: ToolCall, asked: ToolCall | None, utterance: str = "") -> ToolCall:
        """A call the words contradict is replaced by the one they ask for.

        "Please remember that I am vegetarian" was answered with ``forget``;
        "actually make it half past eleven" with a move to the wrong time, or
        a new entry. The model's own phrasing is kept when it chose right.
        """
        if asked is not None and asked.name in ("move_event", "cancel_event"):
            chosen = str(call.arguments.get("subject", "")).lower()
            if chosen != str(asked.arguments["subject"]).lower() and any(
                word in utterance.lower() for word in chosen.split() if len(word) > 2
            ):
                # "Move the design review to five, it clashes": the entry is
                # named in the words, so the model's choice of it stands.
                return call
            if call.name in ("move_event", "schedule_event", "cancel_event") and (call.name != asked.name or call.arguments != asked.arguments):
                LOGGER.info("model called %s for a follow-up the words decide; using %s", call.name, asked.name)
                self._corrections.append(f"{call.name} for a follow-up became {asked.name} from the words")
                return asked
            return call
        if asked is None or call.name not in PROFILE_TOOLS or call.name == asked.name:
            return call
        LOGGER.info("model called %s for a request to %s; using %s", call.name, asked.name, asked.name)
        self._corrections.append(f"{call.name} for a request to {asked.name} became {asked.name}")
        return asked

    def _system_supplied(self, tool: str, arguments: dict, hint: PreferenceHint) -> dict:
        """Fill what the model is never shown -- whose memory -- from the voice.

        Overwrites anything the model wrote there: a model naming a profile
        could otherwise put a fact into somebody else's file.
        """
        spec = self._specs.get(tool)
        if spec is None:
            return arguments
        filled = dict(arguments)
        for parameter in spec.parameters:
            if parameter.supplied_by_system and parameter.name == "profile_id":
                filled["profile_id"] = hint.profile_id or GUEST_PROFILE_ID
        return filled

    # --- temperature --------------------------------------------------

    def _report_setpoint(self, hint: PreferenceHint) -> AssistantReply:
        """Say what the gate did with the occupant's request.

        The verdict is looked for among those published since the hint; a
        request the goal path did not turn into a proposal -- the lights,
        say -- gets no verdict, and is answered as such.
        """
        thermal = hint.subject.strip().lower() in THERMAL_SUBJECTS
        verdict = self._await_verdict(since_ts=hint.ts) if thermal else None
        return AssistantReply(
            ts=self._clock.now(),
            transcript=hint.transcript,
            intent=hint.intent,
            reply=_with_caveat(
                _describe_verdict(verdict, hint, self._safe_range_c, self._room.cooling_on_hold),
                self._room.caveat(verdict.applied.get(SETPOINT_KEY)) if verdict else "",
            ),
        )

    def _await_verdict(self, since_ts: float) -> ValidationVerdict | None:
        deadline = self._clock.monotonic() + self._config.result_timeout_s
        while True:
            for verdict in reversed(self._verdicts):
                if (
                    verdict.ts >= since_ts
                    and verdict.proposed.get(SOURCE_KEY) == GoalSource.PREFERENCE.value
                ):
                    return verdict
            if self._clock.monotonic() >= deadline:
                return None
            self._clock.sleep(_POLL_S)

    # --- service ------------------------------------------------------

    def _serve(self, hint: PreferenceHint) -> AssistantReply:
        utterance = hint.transcript or hint.rationale
        now = local_now(self._clock, self._utc_offset_h)
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": "\n".join(
                    part
                    for part in (
                        ASSISTANT_PROMPT.format(
                            now=now, days=coming_days(self._clock, self._utc_offset_h)
                        ),
                        self._person_prompt(hint),
                        self._room.summary(),
                    )
                    if part
                ),
            },
            {
                "role": "user",
                "content": f"{utterance}\n{annotation(utterance, self._today(), self._local_now())}".strip(),
            },
        ]
        turns: list[ChatTurn] = []
        results: list[ToolResult] = []
        asked = self._memory_asked(hint, utterance) or self._entry_asked(hint, utterance)
        try:
            for _ in range(self._config.max_tool_rounds):
                turn = self._chat.complete(messages, self._tools_for(hint))
                turns.append(turn)
                if not turn.tool_calls:
                    break
                messages.append(assistant_message(turn))
                for call in turn.tool_calls:
                    result = self._run(self._as_asked(call, asked, utterance), utterance, hint)
                    results.append(result)
                    messages.append(tool_message(call, _for_the_model(result)))
            if (
                hint.intent is Intent.SERVICE
                and not results
                and asked is None
                and turns
                and not turns[-1].tool_calls
                and _CLAIMS_DONE.search(turns[-1].content)
            ):
                # Live: asked to book, the model wrote "That would be a mock
                # booking ... Shall I go ahead?" and called nothing, copying
                # the system's own confirmation prompt from earlier in the
                # conversation. It is told so once, and given the chance to
                # do what was asked (FR-44, FR-74).
                LOGGER.info("service request answered without a tool; asking the model again")
                self._corrections.append("a service request answered without a tool was asked again")
                messages.append(assistant_message(turns[-1]))
                messages.append({"role": "user", "content": _NO_TOOL_NUDGE})
                turn = self._chat.complete(messages, self._tools_for(hint))
                turns.append(turn)
                if turn.tool_calls:
                    messages.append(assistant_message(turn))
                    for call in turn.tool_calls:
                        result = self._run(call, utterance, hint)
                        results.append(result)
                        messages.append(tool_message(call, _for_the_model(result)))
            done = PROFILE_TOOLS if asked is not None and asked.name in PROFILE_TOOLS else {asked.name if asked else ""}
            if asked is not None and not any(result.tool in done for result in results):
                # The model said "I'll remember that", or "I'll update the
                # plumber to 11:30", and called nothing: the words asked
                # plainly, so the request is carried out rather than claimed
                # (FR-45, U-07, S-03).
                LOGGER.info("request the model did not act on: %s", asked.name)
                self._corrections.append(f"{asked.name} run for a request the model left undone")
                results.append(self._run(asked, utterance, hint))
            confirmation = next(
                (
                    result
                    for result in results
                    if result.status is ToolStatus.CONFIRMATION_REQUIRED
                ),
                None,
            )
            if confirmation is not None:
                reply_text = self._ask_to_confirm(confirmation)
            elif results and not _needs_phrasing(results):
                # The provider's own sentence already says what happened;
                # a second completion would only add latency and a chance
                # to misstate it.
                reply_text = _fallback(results)
            elif results:
                reply_text = self._phrase(utterance, results, now, turns)
            else:
                reply_text = turns[-1].content if turns else ""
        except ReasoningUnavailableError as exc:
            LOGGER.warning("assistant unavailable: %s", exc)
            reply_text = _fallback(results) if results else hint.spoken_reply
        reply_text = reply_text or _fallback(results) or _NO_TOOL_REPLY
        if hint.intent is Intent.SERVICE and not results and _CLAIMS_DONE.search(reply_text):
            # Live: "cancel my meeting with the CEO" -- there was none -- was
            # answered "-cancelled the meeting with the CEO" with no tool
            # called. A claim no tool stands behind is not passed on (FR-45).
            LOGGER.warning("reply claimed an action no tool performed: %r", reply_text[:120])
            self._corrections.append("a claimed action with no tool behind it was withheld")
            reply_text = _NOTHING_DONE
        if self._room.mode not in (None, Mode.NORMAL) and _ALL_FINE.search(reply_text):
            # Live, during a sensor fault: "Everything seems fine, just a
            # note that the temperature sensor isn't working". An active
            # fault is never "fine"; the status is said as it is (S-13).
            LOGGER.info("reply called a faulted room fine: %r", reply_text[:120])
            self._corrections.append("'everything is fine' during a fault replaced by the status")
            reply_text = self._room.status()
        if _REMINDER.search(utterance.lower()) and any(
            result.status is ToolStatus.OK and result.tool == "schedule_event" for result in results
        ):
            # An entry is not an alarm; saying "added" alone promised one (U-17).
            reply_text = f"{reply_text} {_REMINDER_NOTE}"

        awaiting = next(
            (
                result.invocation_id
                for result in results
                if result.status is ToolStatus.CONFIRMATION_REQUIRED
            ),
            "",
        )
        self._record(utterance, turns, results, reply_text)
        return AssistantReply(
            ts=self._clock.now(),
            transcript=utterance,
            intent=hint.intent,
            reply=reply_text,
            invocation_ids=tuple(result.invocation_id for result in results),
            awaiting_confirmation=awaiting,
        )

    def _local_now(self) -> datetime:
        return local_time(self._clock.now(), self._utc_offset_h)

    def _from_now(self, tool: str, arguments: dict, utterance: str) -> dict:
        """ "In 10 minutes" is arithmetic on the clock, so it is done here (FR-44).

        A duration equal to the delay is dropped: it was the delay misread as
        how long the entry lasts.
        """
        ahead = from_now(utterance)
        if ahead is None or not isinstance(arguments.get("starts_at"), str):
            return arguments
        corrected = dict(arguments)
        corrected["starts_at"] = (self._local_now() + ahead).replace(second=0, microsecond=0).isoformat(
            timespec="seconds"
        )
        if corrected.get("duration_min") == ahead.total_seconds() / 60:
            corrected.pop("duration_min")
        if corrected != arguments:
            LOGGER.info("%s: time put %s from now: %s", tool, ahead, corrected["starts_at"])
            self._corrections.append(f"{tool} time computed from 'in {ahead}'")
        return corrected

    def _today(self):
        return local_time(self._clock.now(), self._utc_offset_h).date()

    def _which_week(self, utterance: str, moment: datetime) -> str:
        """"Next Monday", said the day before, may mean the week after (S-11).

        The date is resolved as the coming one, and a booking says which
        day it is; within two days of today it also says so plainly, so the
        occupant can catch the other reading before saying yes.
        """
        if not _NEXT_WEEKDAY.search(utterance or ""):
            return ""
        ahead = (moment.date() - self._today()).days
        if ahead == 0:
            return " (that's today; say if you meant the week after)"
        if ahead == 1:
            return " (that's tomorrow; say if you meant the week after)"
        if ahead == 2:
            return " (that's the day after tomorrow; say if you meant the week after)"
        return ""

    def _ask_to_confirm(self, result: ToolResult) -> str:
        """The question put before a booking, written from its own arguments.

        Not left to the model: what would be booked, and that it is a mock,
        must be stated exactly (FR-54, FR-55).
        """
        invocation = self._invocations_by_id.get(result.invocation_id)
        if invocation is None:
            return "That needs your confirmation first, and nothing is booked yet."
        arguments = invocation.arguments
        if invocation.tool == "send_mail":
            recipient = str(arguments.get("to", "")).strip() or "someone"
            subject = str(arguments.get("subject", "")).strip()
            about = f" about “{subject}”" if subject else ""
            return (
                f"That would be a mock email to {recipient}{about}; nothing is "
                f"actually sent. Nothing has gone yet. Shall I go ahead?"
            )
        kind = str(arguments.get("kind", "booking"))
        destination = str(arguments.get("destination", "")).strip()
        origin = str(arguments.get("origin") or "").strip()
        nights = arguments.get("nights")
        if kind == "flight":
            what = f"a flight {'from ' + origin + ' ' if origin else ''}to {destination}"
        elif kind == "hotel":
            what = f"a hotel in {destination}"
            if nights:
                what += f" for {nights} night{'s' if str(nights) != '1' else ''}"
        else:
            what = f"a {kind}"
        when = ""
        departs = arguments.get("depart_on")
        if departs:
            try:
                moment = datetime.fromisoformat(str(departs)).replace(tzinfo=None)
                when = f" on {moment:%A} {moment.day} {moment:%B}"
                when += self._which_week(invocation.rationale, moment)
            except ValueError:
                when = f" on {departs}"
        return (
            f"That would be a mock booking of {what}{when}; no real reservation "
            f"is made. Nothing is booked yet. Shall I go ahead?"
        )

    def _run(self, call: ToolCall, utterance: str, hint: PreferenceHint) -> ToolResult:
        """Publish one invocation and wait for the executor's answer."""
        now = self._clock.now()
        invocation_id = f"inv_{int(now)}_{self._invocations}"
        self._invocations += 1
        if call.malformed:
            # FR-44: output failing post-decode validation is not acted on,
            # and the refusal is still an outcome to report.
            return ToolResult(
                ts=now,
                invocation_id=invocation_id,
                tool=call.name if call.name.isidentifier() else "unknown",
                status=ToolStatus.BAD_ARGUMENTS,
                message=f"the arguments were not readable: {call.malformed[:80]}",
            )
        spec = self._specs.get(call.name)
        if spec is not None and not spec.offered_to_assistant:
            # Named in bare JSON though never offered: pencilling or settling
            # belongs to overheard speech and the console, not to a reply.
            return ToolResult(
                ts=now,
                invocation_id=invocation_id,
                tool=call.name,
                status=ToolStatus.UNKNOWN_TOOL,
                message=f"{call.name} is not available here",
            )
        arguments = {key: _flat(value) for key, value in call.arguments.items()}
        arguments = self._system_supplied(call.name, arguments, hint)
        if call.name == "move_event" and not named_days(utterance, self._today()) and from_now(utterance) is None:
            # "Move it to four" names no day: the entry keeps its own, and the
            # date the model wrote -- today's, live -- is not used (FR-44).
            arguments["keep_day"] = True
        arguments = self._the_one_meant(call.name, arguments, utterance, hint)
        arguments = self._exact_entry(call.name, arguments, hint)
        corrected = _onto_named_day(call.name, arguments, the_one_day(utterance, self._today()))
        corrected = self._from_now(call.name, corrected, utterance)
        if corrected != arguments:
            LOGGER.info("%s: dates moved to the day named: %s", call.name, corrected)
            self._corrections.append(f"{call.name} date corrected to the day named")
        tool, corrected = _as_cancellation(call.name, corrected)
        if tool != call.name:
            LOGGER.info("%s for a cancellation became %s", call.name, tool)
            self._corrections.append(f"{call.name} of a cancellation became {tool}")
        corrected, refusal = self._not_in_the_past(tool, corrected, utterance)
        if refusal:
            return ToolResult(
                ts=now,
                invocation_id=invocation_id,
                tool=tool,
                status=ToolStatus.BAD_ARGUMENTS,
                message=refusal,
            )
        invocation = ToolInvocation(
            ts=now,
            invocation_id=invocation_id,
            tool=tool,
            arguments=corrected,
            requester=ToolRequester.PERSONAL_CONTEXT,
            rationale=utterance,
            expires_ts=now + self._config.confirmation_window_s,
        )
        self._invocations_by_id[invocation_id] = invocation
        self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)
        result = self._await_result(invocation)
        voice = hint.profile_id or GUEST_PROFILE_ID
        if result.status is ToolStatus.OK and tool in ("schedule_event", "move_event"):
            starts_at = str(result.detail.get("starts_at") or corrected.get("starts_at") or "")
            self._last_entry.pop(voice, None)
            if starts_at:
                self._last_entry[voice] = (
                    str(corrected.get("subject", "")), starts_at, str(result.detail.get("event_id") or "")
                )
            while len(self._last_entry) > _VOICES_REMEMBERED:
                self._last_entry.pop(next(iter(self._last_entry)))
        elif result.status is ToolStatus.OK and tool == "cancel_event":
            self._last_entry.pop(voice, None)
        return result

    def _the_one_meant(self, tool: str, arguments: dict, utterance: str, hint: PreferenceHint) -> dict:
        """FR-44 for a pronoun: "move it" is the entry this voice last touched.

        Live, "put the budget meeting in tomorrow at two" then "move it to
        three" moved the design review from three turns back. When the words
        say "it" or "that" and name none of the subject the model chose, the
        subject is the last entry this voice put in or moved.
        """
        if tool not in ("move_event", "cancel_event") or not _PRONOUN.search(utterance):
            return arguments
        entry = self._last_entry.get(hint.profile_id or GUEST_PROFILE_ID)
        last = entry[0] if entry else None
        chosen = str(arguments.get("subject", ""))
        said = utterance.lower()
        if not last or chosen.lower() == last.lower():
            return arguments
        if any(word in said for word in chosen.lower().split() if len(word) > 2):
            return arguments
        self._corrections.append(f"{tool} of 'it' moved to the entry last touched: {last}")
        return {**arguments, "subject": last}

    def _exact_entry(self, tool: str, arguments: dict, hint: PreferenceHint) -> dict:
        """Name the entry by id when it is the one this voice last touched (S-03).

        Live, two entries called "plumber" at 11:00 made "actually make it
        half past eleven" and "cancel it" both answer "which one?".
        """
        if tool not in ("move_event", "cancel_event"):
            return arguments
        entry = self._last_entry.get(hint.profile_id or GUEST_PROFILE_ID)
        if entry is None or not entry[2] or str(arguments.get("subject", "")).lower() != entry[0].lower():
            return arguments
        return {**arguments, "event_id": entry[2]}

    def _not_in_the_past(self, tool: str, arguments: dict, utterance: str) -> tuple[dict, str]:
        """FR-44 against the clock: an entry or booking must not start in the past.

        "Thursday at 3", said on Thursday evening, means next Thursday, and
        is moved there. Anything else already past is refused with a
        question, rather than written silently as it once was (U-17).

        :returns: the arguments, possibly moved a week on, and a refusal to
            say back instead, or "" when the call may go ahead.
        """
        name = _MOMENT_OF.get(tool)
        value = arguments.get(name) if name else None
        if not isinstance(value, str) or arguments.get("keep_day"):
            # Kept on its own day, the date written here is not the one used.
            return arguments, ""
        try:
            moment = datetime.fromisoformat(value).replace(tzinfo=None)
        except ValueError:
            return arguments, ""
        now = local_time(self._clock.now(), self._utc_offset_h)
        if moment >= now:
            return arguments, ""
        today = now.strftime("%A")
        if today in named_days(utterance, now.date()) and moment.date() == now.date():
            moved = dict(arguments)
            moved[name] = (moment + timedelta(days=7)).isoformat(timespec="seconds")
            self._corrections.append(f"{tool} moved to next {today}: today's time had passed")
            return moved, ""
        return arguments, (
            f"That would be {moment:%A} {moment.day} {moment:%B} at {moment:%H:%M}, "
            f"which has already passed, so I haven't done it. Which day did you mean?"
        )

    def _await_result(self, invocation: ToolInvocation) -> ToolResult:
        deadline = self._clock.monotonic() + self._config.result_timeout_s
        while self._clock.monotonic() < deadline:
            result = self._results.get(invocation.invocation_id)
            if result is not None:
                return result
            self._clock.sleep(_POLL_S)
        result = self._results.get(invocation.invocation_id)
        if result is not None:
            return result
        LOGGER.warning("no result for %s within the timeout", invocation.invocation_id)
        return ToolResult(
            ts=self._clock.now(),
            invocation_id=invocation.invocation_id,
            tool=invocation.tool,
            status=ToolStatus.UNAVAILABLE,
            message=_NO_RESULT,
        )

    def _phrase(
        self,
        utterance: str,
        results: list[ToolResult],
        now: str,
        turns: list[ChatTurn],
    ) -> str:
        """Turn tool results into a spoken reply, with no tools on offer."""
        summary = "\n".join(_for_the_model(result) for result in results)
        turn = self._chat.complete(
            [
                {"role": "system", "content": ANSWER_PROMPT.format(now=now)},
                {
                    "role": "user",
                    "content": f"They said: {utterance}\nTool results:\n{summary}",
                },
            ]
        )
        turns.append(turn)
        return turn.content

    def _record(
        self,
        utterance: str,
        turns: list[ChatTurn],
        results: list[ToolResult],
        reply: str,
    ) -> None:
        verdict = (
            "tool results " + ", ".join(result.status.value for result in results)
            if results
            else "answered without tools"
        )
        if self._corrections:
            verdict += "; " + "; ".join(self._corrections)
            self._corrections = []
        record = ReasoningRecord(
            ts=self._clock.now(),
            caller=ReasoningCaller.ASSISTANT,
            trigger=TRIGGER,
            model=self._chat.model,
            inputs=utterance,
            raw_output="\n---\n".join(turn.raw for turn in turns),
            verdict=verdict if turns else "reasoning unavailable",
            applied=reply,
            tool_calls=tuple(call.name for turn in turns for call in turn.tool_calls),
            latency_s=sum(turn.latency_s for turn in turns),
            prompt_tokens=sum(turn.prompt_tokens for turn in turns),
            completion_tokens=sum(turn.completion_tokens for turn in turns),
        )
        self._blackboard.publish(topics.AUDIT_REASONING, record)


#: A reply calling the room fine (S-13).
_ALL_FINE = re.compile(
    r"\b(everything|all)\s+(is|seems|looks|appears)\s+(to\s+be\s+)?(fine|ok|okay|good|well|alright|normal)\b"
    r"|\ball\s+good\b",
    re.IGNORECASE,
)

#: "Next Monday": ambiguous between the coming one and the week after's (S-11).
_NEXT_WEEKDAY = re.compile(r"\bnext\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE)

#: "Move it", "cancel that": a follow-up about the entry just discussed.
_PRONOUN = re.compile(r"\b(it|that|this one|that one)\b", re.IGNORECASE)

#: A follow-up about the entry just discussed: a pronoun, or "actually ...".
_FOLLOW_UP = re.compile(r"\b(it|that|this one|that one)\b|^\s*actually\b", re.IGNORECASE)

#: Cancelling the entry just discussed, in the words people use.
_CANCEL = re.compile(r"\b(cancel|delete|remove|scrap|drop|call\s+off)\b", re.IGNORECASE)

#: Voices whose last entry is remembered for "it".
_VOICES_REMEMBERED = 16


def _as_you(fact: str) -> str:
    """A third-person fact, said to its person: "Likes tea." -> "you like tea"."""
    text = fact.strip().rstrip(".")
    first, _, rest = text.partition(" ")
    verbs = {"Is": "are", "Has": "have", "Their": "your"}
    if first in verbs:
        return f"{'you ' if first != 'Their' else ''}{verbs[first]} {rest}".strip()
    if first.endswith("s") and first[:1].isupper() and rest:
        return f"you {first[:-1].lower()} {rest}"
    return text[:1].lower() + text[1:]


def _flat(value: object) -> bool | int | float | str | None:
    """Tool arguments are flat (``ArgumentValue``); anything else is stringified
    so the argument check refuses it with the parameter named, rather than the
    invocation failing to be built at all."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _for_the_model(result: ToolResult) -> str:
    mock = " (mock endpoint, no real reservation)" if result.simulated else ""
    return f"{result.tool}: {result.status.value}{mock}. {result.message}"


#: The argument holding when each tool's effect happens, checked against now.
_MOMENT_OF = {"schedule_event": "starts_at", "move_event": "starts_at", "book_travel": "depart_on"}

#: A subject that is itself a request to cancel, which no entry should carry.
_CANCELLATION = re.compile(r"^\s*(cancel|delete|remove|drop|unschedule|call\s+off)\b[\s:-]*", re.IGNORECASE)

_REMINDER = re.compile(r"\bremind(er)?\b")
_REMINDER_NOTE = "It's in your calendar, but I can't alert you yet, so I won't remind you by myself."


def _as_cancellation(tool: str, arguments: dict) -> tuple[str, dict]:
    """An entry titled "Cancel ..." is a cancellation the model filed as an entry (U-02)."""
    if tool != "schedule_event":
        return tool, arguments
    subject = str(arguments.get("subject", ""))
    match = _CANCELLATION.match(subject)
    if match is None or not subject[match.end():].strip():
        return tool, arguments
    cancel = {"subject": subject[match.end():].strip()}
    if isinstance(arguments.get("starts_at"), str):
        cancel["on"] = arguments["starts_at"]
    return "cancel_event", cancel


#: Timestamp arguments that name a single moment on the day asked about.
_MOMENT_ARGUMENTS = ("starts_at", "depart_on")


def _onto_named_day(tool: str, arguments: dict, day) -> dict:
    """FR-44 applied to dates: put the call on the day the occupant named.

    A moment is moved to that day, its time kept. A window (get_events) is
    shifted whole, so that one ending at midnight the next day is not
    collapsed onto a single instant.
    """
    if day is None:
        return arguments
    corrected = dict(arguments)
    for name in _MOMENT_ARGUMENTS:
        value = corrected.get(name)
        if isinstance(value, str):
            moved = on_day(value, day)
            if moved is not None:
                corrected[name] = moved
    start = corrected.get("from_time")
    if isinstance(start, str) and isinstance(corrected.get("to_time"), str):
        try:
            begin = datetime.fromisoformat(start).replace(tzinfo=None)
            end = datetime.fromisoformat(str(corrected["to_time"])).replace(tzinfo=None)
        except ValueError:
            return corrected
        shift = datetime.combine(day, begin.time()) - begin
        if shift:
            corrected["from_time"] = (begin + shift).isoformat(timespec="seconds")
            corrected["to_time"] = (end + shift).isoformat(timespec="seconds")
    return corrected


def _needs_phrasing(results: list[ToolResult]) -> bool:
    """Whether a reply needs the model: only to answer a question from a read.

    "Am I free tomorrow afternoon?" is answered by reading the calendar and
    then saying yes or no, which is language. "Added the review at 15:00" is
    already the answer.
    """
    return any(
        result.status is ToolStatus.OK and result.tool == "get_events"
        for result in results
    )


def _fallback(results: list[ToolResult]) -> str:
    """A reply written from the results alone, when the model cannot phrase one."""
    return " ".join(result.message for result in results if result.message)


def _with_caveat(reply: str, caveat: str) -> str:
    return f"{reply} {caveat}".strip() if caveat else reply


def _describe_verdict(
    verdict: ValidationVerdict | None,
    hint: PreferenceHint,
    safe_range_c: Bounds,
    on_hold: bool = False,
) -> str:
    if verdict is None:
        if hint.subject and hint.subject.lower() not in ("temperature", ""):
            return f"I can only change the temperature, not the {hint.subject}."
        return "I passed that on, but nothing changed the target temperature."
    asked = verdict.proposed.get(SETPOINT_KEY)
    applied = verdict.applied.get(SETPOINT_KEY)
    if verdict.verdict is Verdict.ACCEPTED:
        if on_hold:
            # "Done" would claim an effect nothing will produce (U-03).
            return f"I've set the target to {applied:g} degrees."
        return f"Done. The room is now set to {applied:g} degrees."
    if not safe_range_c.contains(asked):
        limit = safe_range_c.clamp(asked)
        now = (
            f"it is set to {applied:g} degrees"
            if applied == limit
            else f"it is heading there and is {applied:g} degrees for now"
        )
        return (
            f"{asked:g} degrees is outside the safe range of "
            f"{safe_range_c.low:g} to {safe_range_c.high:g}, so the closest I can "
            f"do is {limit:g} degrees; {now}."
        )
    if verdict.reason is ReasonCode.RATE_LIMIT:
        return (
            f"Heading to {asked:g} degrees. The target changes a couple of "
            f"degrees at a time, so it is {applied:g} now and will get there "
            f"over the next few minutes."
        )
    return f"I could not change it; the target stays at {applied:g} degrees."
