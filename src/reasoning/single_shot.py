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

**Not yet wired:** section 5.7.6's tool surface exists in
``src/common/tools.py``, and this module does not yet call it. The extraction
below is the v1.4 behaviour -- a transcript in, a PreferenceHint out -- and a
SERVICE intent is still only reported. Stated here rather than left to be
discovered, because FR-42 now permits more than this does.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from src.common.clock import Clock
from src.common.config import ReasoningConfig
from src.common.schemas import Comfort, Intent, PreferenceHint

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
    '"intent" (one of "environment" for a request about the space, '
    '"service" for an external action such as a booking, or "none" when '
    "nothing was asked), "
    '"subject" (a short noun for what it was about, such as "temperature", '
    '"lights" or "booking", or "" for none), '
    '"comfort" (one of "warmer", "cooler", "unchanged"; use "unchanged" '
    "unless a temperature preference was expressed), "
    '"target_c" (a number, or null if no temperature was named), '
    '"rationale" (a short quote of what they asked for), and '
    '"spoken_reply" (one short sentence of plain English to say back). '
    "Never claim anything has been changed or switched: say a request has "
    "been passed on, because that is all that has happened."
)


def _is_empty(hint: PreferenceHint) -> bool:
    """Whether the extraction found nothing worth forwarding.

    A hint about a non-thermal subject is still worth publishing even with no
    temperature in it: the goal path may not act on it, but discarding it here
    would hide from the audit log that anything was said at all.
    """
    if hint.intent is Intent.NONE:
        return True
    if hint.intent is Intent.SERVICE:
        return False
    return (
        hint.comfort is Comfort.UNCHANGED
        and hint.target_c is None
        and not hint.subject
    )


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

    Holds no tool registry yet; see the module docstring for what that
    changes when it does.
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

    def extract(self, transcript: str) -> PreferenceHint | None:
        """Read a preference out of one utterance.

        :returns: the hint, or None when nothing actionable was said or the
            output failed validation. None is the ordinary case, not an
            error.
        :raises ReasoningUnavailableError: if inference could not be reached.
        """
        if not transcript.strip():
            return None

        raw = self._complete(transcript)
        return self._validate(raw)

    def _complete(self, transcript: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": PERSONAL_CONTEXT_PROMPT},
                    {"role": "user", "content": transcript},
                ],
                response_format=_JSON_RESPONSE_FORMAT,
            )
        except Exception as exc:
            raise ReasoningUnavailableError(f"inference failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def _validate(self, raw: str) -> PreferenceHint | None:
        """Post-decode semantic validation (FR-44).

        Discards anything that does not parse, names a comfort outside the
        enum, or carries a target that is not a number. Nothing is published
        and the previous state stands.
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
                comfort=Comfort(decoded.get("comfort")),
                subject=str(decoded.get("subject", "")),
                target_c=decoded.get("target_c"),
                rationale=str(decoded.get("rationale", "")),
                spoken_reply=str(decoded.get("spoken_reply", "")),
            )
        except (ValidationError, ValueError) as exc:
            LOGGER.warning("discarding extraction failing validation: %s", exc)
            return None

        return None if _is_empty(hint) else hint
