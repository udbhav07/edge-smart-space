"""Schema-constrained reasoning calls (DESIGN.md section 5.7.1).

Neither call site here is an agent. Fault Diagnosis is single-shot with no
tools at all; Personal Context may invoke assistance tools, but within a
bounded number of rounds rather than an open-ended loop (FR-42). Calling
either an agent would be a naming convention rather than an architecture.

Two properties hold whatever tools are attached:

* **No route to the plant.** A tool can put something in a calendar; nothing
  here can move an actuator. The output reaches the room only if the goal
  path proposes a setpoint from it and the safety validator admits that
  setpoint (FR-45, FR-53).
* **Schema-constrained, then checked again.** The decoder is asked for JSON,
  and the result is validated afterwards regardless (FR-44). Constraining
  the decode guarantees the output *parses*; it says nothing about whether
  the values are sensible. An output failing the second stage is discarded
  and nothing is published, which is the specified behaviour rather than a
  fallback.

This module is the extraction only -- a transcript in, a PreferenceHint out
-- and holds no tools. Acting on a SERVICE intent is the assistant's job
(``src/reasoning/assistant.py``): it takes the published hint and makes the
one bounded round of tool calls FR-42 allows, through the executor (section
5.7.6). The speech pipeline depends on this module exactly as it is.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.common.clock import Clock
from src.common.config import ReasoningConfig
from src.common.schemas import Comfort, ContextTurn, Intent, PreferenceHint
from src.reasoning.understanding import understood

LOGGER = logging.getLogger(__name__)

#: The endpoint is local and ignores the key, but the client requires one.
LOCAL_API_KEY = "local"

#: Asking for a JSON object is the OpenAI-compatible equivalent of the GBNF
#: grammar in section 5.7.3. Both constrain the decode; neither makes the
#: content correct, which is why post-decode validation still runs.
_JSON_RESPONSE_FORMAT = {"type": "json_object"}

PERSONAL_CONTEXT_PROMPT = (
    "You understand what someone asked of their smart space. You do not "
    "control anything: you describe the request so the control system can "
    "weigh it. Reply with a JSON object and nothing else, with keys: "
    '"intent", one of: "environment" for a request to change the room\'s '
    'temperature or comfort; "service" for anything about their calendar, '
    "schedule, reminders, meetings or bookings -- adding, moving, cancelling "
    "or asking about them -- and for asking you to remember or forget "
    'something about them; "question" for a question about the room, its '
    "temperature, the system, or what you remember about them, and for a "
    'greeting or small talk; or "none" when nothing was said to you. '
    '"subject": a short noun for what it was about, such as "temperature", '
    '"lights", "calendar", "booking" or "memory", or "" for none. '
    '"comfort": one of "warmer", "cooler", "unchanged"; "unchanged" unless '
    "they said they are too hot or too cold, or asked for warmer or cooler. "
    "Stuffy, muggy, humid or sticky means cooler. "
    '"target_c": the exact number of degrees they asked for, or null if they '
    "named no temperature; never adjust it to something sensible or safe, "
    "because limits are enforced by the control system and it must see what "
    "was actually asked. A question never has a target. "
    '"rationale": a short quote of what they asked for. '
    '"spoken_reply": one short sentence of plain English to say back. '
    "Never claim anything has been changed or switched: say a request has "
    "been passed on, because that is all that has happened. Examples: "
    '"it\'s a bit stuffy in here" -> environment, cooler; '
    '"make it 22 degrees" -> environment, target_c 22; '
    '"why is it so warm" -> question; "what\'s the temperature" -> question; '
    '"hello" -> question; "what do you know about me" -> question; '
    '"remember that I am vegetarian" -> service, memory; '
    '"forget what I said about meetings" -> service, memory; '
    '"cancel the meeting with Ravi" -> service, calendar; '
    '"move it to 4", just after a calendar entry was discussed -> service, '
    "calendar, target_c null."
)

#: Exchanges shown to the model so a follow-up can be read against them.
_RECENT_SHOWN = 3


def _is_empty(hint: PreferenceHint) -> bool:
    """Whether the extraction found nothing worth forwarding.

    A hint about a non-thermal subject is still worth publishing even with no
    temperature in it: the goal path may not act on it, but discarding it here
    would hide from the audit log that anything was said at all.
    """
    if hint.intent is Intent.NONE:
        return True
    if hint.intent in (Intent.SERVICE, Intent.QUESTION):
        return False
    return (
        hint.comfort is Comfort.UNCHANGED
        and hint.target_c is None
        and not hint.subject
    )


#: A comfort left blank rather than answered. Read as the prompt's default.
_BLANK_COMFORT = (None, "")


def _comfort_field(decoded: dict) -> object:
    """The comfort answer, with a blank one read as "unchanged".

    A missing key is passed through as None so the enum rejects it -- for an
    environment request, the one intent where comfort is the answer. For any
    other intent comfort can only be "unchanged", and the served model leaves
    the key out: "remember that I like tea" was dropped as invalid for want
    of a field that could hold one value (U-07, measured live).
    """
    if "comfort" in decoded and decoded["comfort"] in _BLANK_COMFORT:
        return Comfort.UNCHANGED.value
    if "comfort" not in decoded and decoded.get("intent") not in (None, Intent.ENVIRONMENT.value):
        return Comfort.UNCHANGED.value
    if "comfort" not in decoded and isinstance(decoded.get("target_c"), (int, float)):
        # "Make it 22 again" came back with a target and no comfort key. A
        # named number is the answer to the question comfort asks.
        return Comfort.UNCHANGED.value
    return decoded.get("comfort")


class ReasoningUnavailableError(RuntimeError):
    """Inference could not be reached or did not answer in time.

    Raised rather than swallowed so the caller can decide. The regulatory
    loop is unaffected either way: it holds the last validated setpoint and
    keeps running when the reasoning layer is absent (FR-11, FR-47).
    """


class PersonalContext:
    """Extracts a preference from a transcript.

    Stateless between calls by design. Personal Context is invoked on a
    transcript (section 5.7.1), and carrying conversation across utterances
    would make one call's output depend on an earlier one, which an
    extraction is not.

    Holds no tool registry: the assistant acts on what this extracts.
    """

    def __init__(
        self,
        config: ReasoningConfig,
        clock: Clock,
        client=None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._client = client if client is not None else self._connect(config)

    @staticmethod
    def _connect(config: ReasoningConfig):
        """Build the default client.

        The OpenAI package is imported here rather than at module scope so
        the post-decode validation stays importable and testable without the
        inference stack installed.
        """
        from openai import OpenAI

        return OpenAI(
            base_url=config.base_url,
            api_key=LOCAL_API_KEY,
            timeout=config.timeout_s,
        )

    def extract(
        self, transcript: str, recent: tuple[ContextTurn, ...] = ()
    ) -> PreferenceHint | None:
        """Read a preference out of one utterance.

        :param recent: the conversation so far, newest last, so a follow-up
            ("move it to 4") is read against what it follows.
        :returns: the hint, or None when nothing actionable was said or the
            output failed validation. None is the ordinary case, not an
            error.
        :raises ReasoningUnavailableError: if inference could not be reached.
        """
        if not transcript.strip():
            return None

        raw = self._complete(transcript, recent[-_RECENT_SHOWN:])
        hint = self._validate(raw)
        if hint is None:
            return None
        # FR-44 again, against the words this time: the model's classification
        # is checked deterministically, and only ever moved away from the plant.
        hint = understood(hint, transcript, recent)
        if _is_empty(hint):
            return None
        # The words themselves travel with the hint, so whatever acts on it --
        # a calendar entry needs the exact time said -- need not trust a paraphrase.
        return hint.model_copy(update={"transcript": transcript.strip()})

    def _complete(self, transcript: str, recent: tuple[ContextTurn, ...] = ()) -> str:
        prompt = PERSONAL_CONTEXT_PROMPT
        if recent:
            lines = "\n".join(
                f"- They said: {turn.said} / The system replied: {turn.replied}" for turn in recent
            )
            prompt += f"\nEarlier in this conversation:\n{lines}"
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": transcript},
                ],
                response_format=_JSON_RESPONSE_FORMAT,
                # An extraction, not a composition: the same words should give
                # the same hint every time. At the default temperature a
                # calendar question was classified as a request on one run
                # and as nothing on the next, and was then silently dropped.
                temperature=0.0,
            )
        except Exception as exc:
            raise ReasoningUnavailableError(f"inference failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def _validate(self, raw: str) -> PreferenceHint | None:
        """Post-decode semantic validation (FR-44).

        Discards anything that does not parse, names a comfort outside the
        enum, or carries a target that is not a number. Nothing is published
        and the previous state stands.

        A comfort given but *blank* -- null or empty -- is read as the
        prompt's stated default, "unchanged". Served models leave it blank for
        a booking or a calendar entry, where no temperature preference exists,
        and discarding those threw away every service request the live run
        made. A comfort key that is missing, or present and wrong, is still
        discarded: the model did not answer the question it was asked.
        """
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            LOGGER.warning("discarding unparseable extraction: %s", exc)
            return None

        if not isinstance(decoded, dict):
            LOGGER.warning("discarding extraction that is not an object")
            return None

        try:
            hint = PreferenceHint(
                ts=self._clock.now(),
                intent=Intent(decoded.get("intent", Intent.ENVIRONMENT.value)),
                comfort=Comfort(_comfort_field(decoded)),
                subject=str(decoded.get("subject", "")),
                target_c=decoded.get("target_c"),
                rationale=str(decoded.get("rationale", "")),
                spoken_reply=str(decoded.get("spoken_reply", "")),
            )
        except (ValidationError, ValueError) as exc:
            LOGGER.warning("discarding extraction failing validation: %s", exc)
            return None

        return hint
