"""Conversational front end for spoken requests.

This component does not act. It cannot act. Its entire output is a spoken
reply plus, at most, one *requested* environment change that the caller is
free to forward, gate, or discard.

That constraint is the design, not caution (DESIGN.md sections 1.3, 5.7,
FR-45, FR-53):

* The reasoning layer never writes to an actuator. All of its influence is
  exerted through a proposed setpoint, which the safety validator gates.
* A recognised intent is forwarded as a *supervisory input*, never as a
  direct actuator command. Voice is a request, weighed like any other.

An earlier version exposed a ``control_appliance`` tool that switched
devices, including the air conditioner, straight from the model's output.
That bypassed the setpoint bounds, the rate limit, the compressor dwell
timer and the mode interlocks, and left the question "what stops it doing
something unsafe" with no answer. The tool now records a request and
returns it; nothing downstream of this module is touched.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

LOGGER = logging.getLogger(__name__)

#: Local inference endpoint. No traffic leaves the node.
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b"

#: Ollama ignores the key but the client requires one to be present.
LOCAL_API_KEY = "ollama"

#: Seconds before a stalled inference call is abandoned. Speech is the
#: lowest-priority feature in the system; it must degrade on its own rather
#: than take the process with it.
REQUEST_TIMEOUT_S = 30.0

#: Conversation turns retained. Unbounded history grows the prompt on every
#: exchange until inference is slow and then fails outright.
MAX_HISTORY_TURNS = 20

SYSTEM_PROMPT = (
    "You are a smart space assistant. Keep spoken responses very brief, "
    "casual and helpful. You can note that a change has been requested, but "
    "never claim a device has been switched or a temperature has been "
    "reached: you request, you do not control."
)

REQUEST_TOOL_NAME = "request_environment_change"


class Comfort(str, Enum):
    """The direction a person asked for, in their own terms."""

    WARMER = "warmer"
    COOLER = "cooler"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class EnvironmentRequest:
    """A spoken preference, expressed as a supervisory input (FR-53).

    Deliberately not a command. It names a direction and, optionally, a
    target, and carries the words that produced it so the audit trail can
    show why a goal was proposed.
    """

    comfort: Comfort
    target_c: float | None
    spoken_reason: str


@dataclass(frozen=True)
class Interaction:
    """What one utterance produced.

    ``request`` is None when nothing actionable was said, which is the
    common case and is not an error.
    """

    reply: str
    request: EnvironmentRequest | None


class LlmUnavailableError(RuntimeError):
    """Inference could not be reached or did not answer in time."""


_REQUEST_TOOL = {
    "type": "function",
    "function": {
        "name": REQUEST_TOOL_NAME,
        "description": (
            "Record that the occupant asked the space to feel warmer or "
            "cooler. This only records a request for the control system to "
            "consider; it does not change any device."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "comfort": {
                    "type": "string",
                    "enum": [member.value for member in Comfort],
                },
                "target_c": {
                    "type": "number",
                    "description": "Temperature named by the occupant, if any.",
                },
                "spoken_reason": {
                    "type": "string",
                    "description": "What the occupant said, briefly.",
                },
            },
            "required": ["comfort", "spoken_reason"],
        },
    },
}


class SmartAgentLLM:
    """Turns a transcript into a reply and, at most, a request.

    Holds no reference to any device, driver, or actuator topic. There is
    nothing it could switch even if the model asked it to.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_s: float = REQUEST_TIMEOUT_S,
        max_history_turns: int = MAX_HISTORY_TURNS,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url, api_key=LOCAL_API_KEY, timeout=timeout_s
        )
        self._model = model_name
        self._max_history_turns = max_history_turns
        self._history: list[dict] = []

    @property
    def history_length(self) -> int:
        """Turns currently retained, excluding the system prompt."""
        return len(self._history)

    def chat(self, transcript: str) -> Interaction:
        """Answer one utterance.

        :raises LlmUnavailableError: if inference cannot be reached or times
            out. The caller is expected to carry on without speech rather
            than fail; voice is the lowest-priority feature in the system.
        """
        self._remember({"role": "user", "content": transcript})

        message = self._complete(tools=[_REQUEST_TOOL])
        request = self._extract_request(message)

        if request is None:
            reply = message.content or ""
            self._remember({"role": "assistant", "content": reply})
            return Interaction(reply=reply, request=None)

        self._remember(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [call.model_dump() for call in message.tool_calls],
            }
        )
        for call in message.tool_calls:
            self._remember(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": "Request recorded for the control system to weigh.",
                }
            )

        reply = self._complete(tools=None).content or ""
        self._remember({"role": "assistant", "content": reply})
        return Interaction(reply=reply, request=request)

    def _complete(self, tools: list[dict] | None):
        arguments = {"model": self._model, "messages": self._conversation()}
        if tools is not None:
            arguments["tools"] = tools
        try:
            response = self._client.chat.completions.create(**arguments)
        except Exception as exc:
            raise LlmUnavailableError(f"inference failed: {exc}") from exc
        return response.choices[0].message

    def _conversation(self) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}, *self._history]

    def _remember(self, entry: dict) -> None:
        self._history.append(entry)
        excess = len(self._history) - self._max_history_turns
        if excess > 0:
            del self._history[:excess]

    def _extract_request(self, message) -> EnvironmentRequest | None:
        """Read a request out of a tool call, or None if there wasn't one."""
        calls = getattr(message, "tool_calls", None)
        if not calls:
            return None

        call = calls[0]
        if call.function.name != REQUEST_TOOL_NAME:
            LOGGER.warning("ignoring unrecognised tool call %r", call.function.name)
            return None

        try:
            arguments = json.loads(call.function.arguments)
            return EnvironmentRequest(
                comfort=Comfort(arguments["comfort"]),
                target_c=arguments.get("target_c"),
                spoken_reason=arguments.get("spoken_reason", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            LOGGER.warning("discarding malformed request: %s", exc)
            return None
