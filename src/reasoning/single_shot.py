"""Personal Context: what an occupant asked for, and doing the parts it may (FR-42).

Not an agent (section 5.7.1). It is one schema-constrained extraction with, at
most, ``assistance.max_tool_rounds`` rounds of assistance tools before it must
answer -- default one, which is enough to look something up and reply about
it, and which is what keeps this a bounded call rather than an open-ended loop.
It is not shown the supervisor's read tools, and the supervisor is not shown
these.

Three properties hold whatever the model says:

* **No route to the plant.** A tool can put something in a calendar; nothing
  here can move an actuator. A temperature request becomes a
  ``PreferenceHint``, which reaches the room only through arbitration and the
  validator (FR-45, FR-53).
* **No route past the executor.** Tools are invoked by publishing to
  ``space/assist/proposed``. A ``commit`` tool comes back
  ``CONFIRMATION_REQUIRED`` and nothing is booked; only an occupant, through
  the console, can republish it as confirmed (FR-54, FR-74).
* **Checked after decoding** (FR-44). The answer must parse and fit the
  schema, and two things are enforced on what is said back rather than left to
  the model's good manners: a mock result is always called a mock (FR-55), and
  an action awaiting confirmation is always put as a question, never reported
  as done.

The date and time are in the prompt, in the room's local frame, because "on
Thursday" means nothing to a model that does not know what today is.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from src.common.clock import Clock
from src.common.config import Config
from src.common.localtime import local_time
from src.common.schemas import (
    CallSite,
    Comfort,
    Intent,
    PreferenceHint,
    ReasoningOutcome,
)
from src.common.tools import ASSISTANCE_TOOLS, ArgumentValue, ToolResult, ToolStatus
from src.reasoning.audit import ReasoningAudit, Trace
from src.reasoning.endpoint import ChatEndpoint, ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Most tool calls one round may make. A model asking for twenty lookups in
#: answer to one sentence has misunderstood the task.
MAX_CALLS_PER_ROUND = 4

#: Longest rationale carried on an invocation or a hint, in characters.
_RATIONALE_CHARS = 200

#: Said when a mock ran and the model forgot to say so (FR-55).
MOCK_NOTICE = "This was a mock booking; nothing real has been reserved."

#: Said when an action awaits confirmation and the model did not ask (FR-54).
CONFIRMATION_NOTICE = (
    "That needs your confirmation before anything is booked. Confirm it on "
    "the console if you want it."
)

#: Every ``commit`` tool's name. Derived from the declarations, so a tool
#: added later is covered by what it declares (FR-74).
_COMMIT_TOOLS = frozenset(spec.name for spec in ASSISTANCE_TOOLS if spec.requires_confirmation)


def personal_context_prompt(now_text: str) -> str:
    """The system prompt, told what time it is."""
    return (
        "You understand what someone asked of their smart room. It is now "
        f"{now_text} local time. You do not control the room: a temperature "
        "request is passed on to a control system that weighs it. You may use "
        "the tools you are given to act on a request that needs one -- adding "
        "to or reading the calendar, or asking for a booking, which only the "
        "occupant can confirm. Write dates and times as ISO-8601 local times.\n"
        "When you are done, reply with a JSON object and nothing else, with "
        'keys: "intent" ("environment" for a request about the room, '
        '"service" for anything a tool was for, or "none" when nothing was '
        'asked), "subject" (a short noun such as "temperature", "calendar" or '
        '"booking", or "" for none), "comfort" ("warmer", "cooler" or '
        '"unchanged"), "target_c" (a number, or null if no temperature was '
        'named), "rationale" (a short quote of what they asked), and '
        '"spoken_reply" (one or two plain sentences to say back). Never claim '
        "the room has been changed: say the request has been passed on. Never "
        "claim a booking was made unless a tool result says so, and if a "
        "booking needs confirmation, ask for it."
    )


class ToolClient(Protocol):
    """How Personal Context reaches the executor. See ``tool_client.py``."""

    def call(
        self, tool: str, arguments: dict[str, ArgumentValue], rationale: str
    ) -> ToolResult | None: ...


def _is_empty(hint: PreferenceHint) -> bool:
    """Whether the extraction found nothing worth forwarding.

    A hint about a non-thermal subject is still worth publishing with no
    temperature in it: the goal path will not act on it, but discarding it
    here would hide from the audit log that anything was said at all.
    """
    if hint.intent is Intent.NONE:
        return True
    if hint.intent is Intent.SERVICE:
        return False
    return hint.comfort is Comfort.UNCHANGED and hint.target_c is None and not hint.subject


class PersonalContext:
    """Turns one utterance into a hint, running the tools it needs on the way.

    Stateless between utterances by design: carrying a conversation would make
    one call's output depend on an earlier one, and an earlier "yes" is
    exactly what must never confirm a later booking.
    """

    def __init__(
        self,
        config: Config,
        clock: Clock,
        endpoint: ChatEndpoint,
        audit: ReasoningAudit,
        tools: ToolClient | None = None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._endpoint = endpoint
        self._audit = audit
        self._tools = tools
        self._schemas = tuple(spec.as_schema() for spec in ASSISTANCE_TOOLS)

    def extract(self, transcript: str, trigger: str = "utterance") -> PreferenceHint | None:
        """Read one utterance, acting on it where a tool is needed.

        :returns: the hint to publish, or None when nothing was asked or the
            call failed. None is the ordinary case, not an error; every call is
            in the audit record either way.
        """
        transcript = transcript.strip()
        if not transcript:
            return None

        now = local_time(self._clock.now(), self._config.site.utc_offset_h)
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": personal_context_prompt(
                    now.strftime("%A %Y-%m-%dT%H:%M")
                ),
            },
            {"role": "user", "content": transcript},
        ]
        trace = Trace(CallSite.PERSONAL_CONTEXT, trigger, transcript)
        results: list[ToolResult] = []

        try:
            answer = self._converse(messages, trace, transcript, results)
        except ReasoningUnavailableError as exc:
            self._audit.publish(trace, ReasoningOutcome.UNAVAILABLE, reason=str(exc))
            return self._fallback(transcript, results)

        hint, problem = self._validate(answer, results)
        if hint is None:
            self._audit.publish(trace, ReasoningOutcome.DISCARDED, reason=problem)
            return self._fallback(transcript, results)
        if _is_empty(hint):
            self._audit.publish(trace, ReasoningOutcome.NO_ACTION)
            return None
        self._audit.publish(
            trace,
            ReasoningOutcome.APPLIED,
            applied=f"{hint.intent.value} hint: {hint.spoken_reply}",
        )
        return hint

    # --- the conversation ---------------------------------------------

    def _converse(
        self,
        messages: list[dict[str, object]],
        trace: Trace,
        transcript: str,
        results: list[ToolResult],
    ) -> str:
        """Up to ``max_tool_rounds`` rounds of tools, then a constrained answer."""
        if self._tools is not None:
            for _ in range(self._config.assistance.max_tool_rounds):
                completion = self._endpoint.complete(messages, tools=self._schemas)
                trace.add(completion)
                if not completion.tool_calls:
                    if _parses(completion.content):
                        return completion.content
                    break
                messages.append(completion.as_message())
                for call in completion.tool_calls[:MAX_CALLS_PER_ROUND]:
                    content = self._run_tool(call.name, call.arguments, transcript, results)
                    messages.append(
                        {"role": "tool", "tool_call_id": call.call_id, "content": content}
                    )

        completion = self._endpoint.complete(messages, json_output=True)
        trace.add(completion)
        return completion.content

    def _run_tool(
        self,
        name: str,
        raw_arguments: str,
        transcript: str,
        results: list[ToolResult],
    ) -> str:
        """Invoke one tool through the executor and describe the outcome."""
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as exc:
            return json.dumps({"status": "BAD_ARGUMENTS", "message": f"not JSON: {exc}"})
        if not isinstance(arguments, dict):
            return json.dumps(
                {"status": "BAD_ARGUMENTS", "message": "arguments must be an object"}
            )
        try:
            result = self._tools.call(name, arguments, transcript[:_RATIONALE_CHARS])
        except ValueError as exc:
            return json.dumps({"status": "BAD_ARGUMENTS", "message": str(exc)})
        if result is None:
            return json.dumps(
                {
                    "status": "NO_RESULT",
                    "message": "the assistant service did not answer in time",
                }
            )
        results.append(result)
        return json.dumps(
            {
                "status": result.status.value,
                "message": result.message,
                "detail": dict(result.detail),
                "simulated": result.simulated,
            }
        )

    # --- checking the answer ------------------------------------------

    def _validate(
        self, raw: str, results: Sequence[ToolResult]
    ) -> tuple[PreferenceHint | None, str]:
        """Post-decode semantic validation (FR-44), then the two guarantees."""
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            return None, f"not JSON: {exc}"
        if not isinstance(decoded, dict):
            return None, "not a JSON object"
        try:
            hint = PreferenceHint(
                ts=self._clock.now(),
                intent=Intent(decoded.get("intent", Intent.ENVIRONMENT.value)),
                comfort=Comfort(decoded.get("comfort", Comfort.UNCHANGED.value)),
                subject=str(decoded.get("subject", "")),
                target_c=decoded.get("target_c"),
                rationale=str(decoded.get("rationale", ""))[:_RATIONALE_CHARS],
                spoken_reply=str(decoded.get("spoken_reply", "")),
            )
        except (ValidationError, ValueError) as exc:
            return None, f"fails the schema: {exc}"
        return self._honest(hint, results), ""

    @staticmethod
    def _honest(hint: PreferenceHint, results: Sequence[ToolResult]) -> PreferenceHint:
        """Say what the tools actually did, whatever the model chose to say.

        Appended rather than substituted: the model's sentence is kept, and the
        system's own statement follows it. Enforced here because FR-54 and
        FR-55 are about what an occupant hears, and a prompt is a request, not
        a guarantee.
        """
        reply = hint.spoken_reply.strip()
        if any(r.status is ToolStatus.OK and r.simulated for r in results) and (
            "mock" not in reply.lower()
        ):
            reply = f"{reply} {MOCK_NOTICE}".strip()
        awaiting = any(
            r.status is ToolStatus.CONFIRMATION_REQUIRED and r.tool in _COMMIT_TOOLS
            for r in results
        )
        if awaiting and "confirm" not in reply.lower():
            reply = f"{reply} {CONFIRMATION_NOTICE}".strip()
        if reply == hint.spoken_reply:
            return hint
        return hint.model_copy(update={"spoken_reply": reply})

    def _fallback(
        self, transcript: str, results: Sequence[ToolResult]
    ) -> PreferenceHint | None:
        """What to say when the model's answer is gone but a tool already ran.

        If a calendar entry was written, the occupant has to hear so whatever
        became of the model's sentence (FR-75). The reply is then the system's
        own words -- the results' messages -- not the model's. With nothing
        done there is nothing to say, and the ordinary answer is silence.
        """
        if not results:
            return None
        hint = PreferenceHint(
            ts=self._clock.now(),
            intent=Intent.SERVICE,
            comfort=Comfort.UNCHANGED,
            subject=results[0].tool,
            rationale=transcript[:_RATIONALE_CHARS],
            spoken_reply=" ".join(r.message for r in results if r.message)
            or "Your request was passed on.",
        )
        return self._honest(hint, results)


def _parses(content: str) -> bool:
    try:
        return isinstance(json.loads(content), dict)
    except json.JSONDecodeError:
        return False
