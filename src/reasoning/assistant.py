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

MQTT callbacks only queue work. Waiting for a tool result inside a callback
would wait for a message the same thread has to deliver.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta

from src.common import topics
from src.common.clock import Clock
from src.common.config import AssistanceConfig, Bounds
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    SETPOINT_KEY,
    SOURCE_KEY,
    THERMAL_SUBJECTS,
    AssistantReply,
    GoalSource,
    Intent,
    PreferenceHint,
    ReasonCode,
    ReasoningCaller,
    ReasoningRecord,
    ValidationVerdict,
    Verdict,
)
from src.common.tools import (
    ASSISTANCE_TOOLS,
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
from src.reasoning.dates import annotation, on_day, the_one_day
from src.reasoning.single_shot import ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Utterances waiting to be answered. Bounded: a burst of speech while the
#: model is slow must not grow without limit (NFR-05).
_QUEUE_LENGTH = 16

#: Results and verdicts kept while a caller waits for its own. Bounded.
_MAILBOX_LENGTH = 64

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
    "planned, book_travel for a flight or hotel. Call each tool at most once. "
    "If the request cannot be done with these tools, say so in one sentence "
    "without calling any."
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
_NO_RESULT = "The request got no answer in time, so nothing is confirmed."


#: Days listed for the model to take dates from, today included.
_DAYS_LISTED = 8


def local_now(clock: Clock) -> str:
    """The present, as the model is told it: weekday, date and time."""
    moment = datetime.fromtimestamp(clock.now())
    return f"{moment:%A} {moment.day} {moment:%B %Y}, {moment:%H:%M}"


def coming_days(clock: Clock) -> str:
    """A lookup table of weekday to date, for the model to copy from.

    A 7B model asked for "Thursday" from a Saturday answered Sunday. Working
    out a weekday is arithmetic it is bad at; reading one off a list is not.
    """
    today = datetime.fromtimestamp(clock.now()).date()
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
    ) -> None:
        self._config = config
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
        self._tools = tuple(spec.as_schema() for spec in ASSISTANCE_TOOLS)

    # --- wiring -------------------------------------------------------

    def subscribe(self) -> None:
        self._blackboard.subscribe(
            topics.CONTEXT_PREFERENCE, PreferenceHint, self._on_hint
        )
        self._blackboard.subscribe(topics.ASSIST_RESULT, ToolResult, self._on_result)
        self._blackboard.subscribe(
            topics.AUDIT_VALIDATION, ValidationVerdict, self._on_verdict
        )

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
        if hint.intent is Intent.SERVICE:
            reply = self._serve(hint)
        elif hint.intent is Intent.ENVIRONMENT:
            reply = self._report_setpoint(hint)
        else:
            return None
        self._blackboard.publish(topics.CONTEXT_REPLY, reply)
        LOGGER.info("replied: %s", reply.reply)
        return reply

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
            reply=_describe_verdict(verdict, hint, self._safe_range_c),
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
        now = local_now(self._clock)
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": ASSISTANT_PROMPT.format(now=now, days=coming_days(self._clock)),
            },
            {
                "role": "user",
                "content": f"{utterance}\n{annotation(utterance, self._today())}".strip(),
            },
        ]
        turns: list[ChatTurn] = []
        results: list[ToolResult] = []
        try:
            for _ in range(self._config.max_tool_rounds):
                turn = self._chat.complete(messages, self._tools)
                turns.append(turn)
                if not turn.tool_calls:
                    break
                messages.append(assistant_message(turn))
                for call in turn.tool_calls:
                    result = self._run(call, utterance)
                    results.append(result)
                    messages.append(tool_message(call, _for_the_model(result)))
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

    def _today(self):
        return datetime.fromtimestamp(self._clock.now()).date()

    def _ask_to_confirm(self, result: ToolResult) -> str:
        """The question put before a booking, written from its own arguments.

        Not left to the model: what would be booked, and that it is a mock,
        must be stated exactly (FR-54, FR-55).
        """
        invocation = self._invocations_by_id.get(result.invocation_id)
        if invocation is None:
            return "That needs your confirmation first, and nothing is booked yet."
        arguments = invocation.arguments
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
            except ValueError:
                when = f" on {departs}"
        return (
            f"That would be a mock booking of {what}{when}; no real reservation "
            f"is made. Nothing is booked yet. Shall I go ahead?"
        )

    def _run(self, call: ToolCall, utterance: str) -> ToolResult:
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
        arguments = {key: _flat(value) for key, value in call.arguments.items()}
        corrected = _onto_named_day(call.name, arguments, the_one_day(utterance, self._today()))
        if corrected != arguments:
            LOGGER.info("%s: dates moved to the day named: %s", call.name, corrected)
            self._corrections.append(f"{call.name} date corrected to the day named")
        invocation = ToolInvocation(
            ts=now,
            invocation_id=invocation_id,
            tool=call.name,
            arguments=corrected,
            requester=ToolRequester.PERSONAL_CONTEXT,
            rationale=utterance,
            expires_ts=now + self._config.confirmation_window_s,
        )
        self._invocations_by_id[invocation_id] = invocation
        self._blackboard.publish(topics.ASSIST_PROPOSED, invocation)
        return self._await_result(invocation)

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


def _describe_verdict(
    verdict: ValidationVerdict | None, hint: PreferenceHint, safe_range_c: Bounds
) -> str:
    if verdict is None:
        if hint.subject and hint.subject.lower() not in ("temperature", ""):
            return f"I can only change the temperature, not the {hint.subject}."
        return "I passed that on, but nothing changed the target temperature."
    asked = verdict.proposed.get(SETPOINT_KEY)
    applied = verdict.applied.get(SETPOINT_KEY)
    if verdict.verdict is Verdict.ACCEPTED:
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
