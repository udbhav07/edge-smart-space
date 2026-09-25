"""The one inference endpoint every call site shares (section 5.7.5).

A thin wrapper, deliberately. It does three things the call sites would
otherwise each do slightly differently:

* **Measures every call** (FR-63). Latency on the injected clock's monotonic
  time, and the token counts the server reports, so NFR-03 and E7 are read off
  the audit record rather than off a stopwatch somebody forgot to start.
* **Normalises the reply** into a frozen :class:`Completion`: text, and the
  tool calls the model asked for with their arguments exactly as it wrote them.
  Parsing those arguments is the caller's job, because a malformed argument is
  a finding about the model (FR-44) and must be reported as one, not smoothed
  over here.
* **Turns every transport failure into one exception**,
  :class:`ReasoningUnavailableError`, so no call site needs to know which
  server is running. Both ``llama-server --jinja`` and Ollama speak the same
  OpenAI-compatible protocol, and nothing here depends on which.

The OpenAI client is imported lazily, so every validation path stays
importable and testable without the inference stack installed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from src.common.clock import Clock
from src.common.config import ReasoningConfig

LOGGER = logging.getLogger(__name__)

#: The endpoint is local and ignores the key, but the client requires one.
LOCAL_API_KEY = "local"

#: The OpenAI-compatible equivalent of a GBNF grammar (section 5.7.3). It
#: constrains the decode to a JSON object; it says nothing about whether the
#: object is right, which is why post-decode validation runs regardless.
_JSON_RESPONSE_FORMAT = {"type": "json_object"}


class ReasoningUnavailableError(RuntimeError):
    """Inference could not be reached or did not answer in time.

    Raised rather than swallowed so the caller can decide. The regulatory
    loop is unaffected either way: it holds the last validated setpoint and
    keeps running when the reasoning layer is absent (FR-11, FR-47).

    Carries how long the failure took. A server that times out after thirty
    seconds and one that refuses at once are different findings, and FR-63's
    latency is only honest if the slow failures are in it.
    """

    def __init__(self, message: str, latency_s: float = 0.0) -> None:
        super().__init__(message)
        self.latency_s = latency_s


@dataclass(frozen=True)
class ToolCall:
    """One tool call a model asked for, as it asked for it."""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class Completion:
    """One model turn, with what it cost."""

    content: str
    tool_calls: tuple[ToolCall, ...]
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

    def as_message(self) -> dict[str, object]:
        """This turn as an assistant message, for the next round's history."""
        message: dict[str, object] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


class ChatEndpoint:
    """Chat completions against the configured local server."""

    def __init__(self, config: ReasoningConfig, clock: Clock, client=None) -> None:
        self._config = config
        self._clock = clock
        self._client = client if client is not None else self._connect(config)

    @staticmethod
    def _connect(config: ReasoningConfig):
        from openai import OpenAI

        # No retries. The client retries twice by default, which turns the
        # configured timeout into three of them -- measured against a live
        # stack, a refused connection took over a second before failing, and
        # a hung server would hold a call for 90 s against section 7.1's 30.
        # A failed call is recorded and the next cadence tries again.
        return OpenAI(
            base_url=config.base_url,
            api_key=LOCAL_API_KEY,
            timeout=config.timeout_s,
            max_retries=0,
        )

    def complete(
        self,
        messages: Sequence[dict[str, object]],
        tools: Sequence[dict[str, object]] = (),
        json_output: bool = False,
    ) -> Completion:
        """Ask the model for one turn.

        :param tools: tool schemas, in OpenAI function-calling form. Given, the
            server's native tool template is used (section 5.7.3).
        :param json_output: constrain the decode to a JSON object.
        :raises ReasoningUnavailableError: if the server could not be reached,
            timed out, or answered with nothing usable.
        """
        request: dict[str, object] = {
            "model": self._config.model,
            "messages": list(messages),
            "temperature": self._config.temperature,
        }
        if tools:
            request["tools"] = list(tools)
        if json_output:
            request["response_format"] = _JSON_RESPONSE_FORMAT

        started = self._clock.monotonic()
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ReasoningUnavailableError(
                f"inference failed: {exc}", self._clock.monotonic() - started
            ) from exc
        latency_s = self._clock.monotonic() - started

        try:
            message = response.choices[0].message
        except (AttributeError, IndexError) as exc:
            raise ReasoningUnavailableError(
                f"inference returned no choices: {exc}", latency_s
            ) from exc

        usage = getattr(response, "usage", None)
        return Completion(
            content=message.content or "",
            tool_calls=tuple(
                ToolCall(
                    call_id=call.id or f"call_{index}",
                    name=call.function.name,
                    arguments=call.function.arguments or "{}",
                )
                for index, call in enumerate(message.tool_calls or ())
            ),
            latency_s=latency_s,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        )
