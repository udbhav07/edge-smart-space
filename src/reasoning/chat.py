"""One chat completion against the local endpoint, timed and counted.

Shared by the Environmental Supervisor and the assistant (section 5.7.1).
Personal Context keeps its own client in ``single_shot.py``, because the
speech pipeline depends on that module exactly as it is.

What this adds over calling the SDK directly is the part FR-46 and FR-63 need:
every call comes back with its latency and token counts, and tool calls come
back as plain values rather than SDK objects, so the callers can be tested
with a stub and the audit record can be written from what was actually said.

The OpenAI package is imported lazily, as in ``single_shot.py``: it ships with
the speech extra, and a core-only checkout must still import this module.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from src.common.clock import Clock
from src.common.config import ReasoningConfig
from src.reasoning.single_shot import LOCAL_API_KEY, ReasoningUnavailableError

LOGGER = logging.getLogger(__name__)

#: Qwen sometimes writes a tool call into the message text instead of the
#: tool-call field, wrapped in these tags. Recovered rather than discarded:
#: the intent is unambiguous, and the argument check still applies.
_TEXT_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass(frozen=True)
class ToolCall:
    """One tool the model asked for, decoded."""

    call_id: str
    name: str
    arguments: Mapping[str, object]
    #: Set when the arguments were not valid JSON. The call is still reported
    #: so the refusal can name what the model wrote (FR-44).
    malformed: str = ""


@dataclass(frozen=True)
class ChatTurn:
    """What one completion produced, and what it cost."""

    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: str = field(default="", repr=False)


class ChatClient:
    """Completions against the configured endpoint."""

    def __init__(self, config: ReasoningConfig, clock: Clock, client=None) -> None:
        self._config = config
        self._clock = clock
        self._client = client if client is not None else self._connect(config)

    @property
    def model(self) -> str:
        return self._config.model

    @staticmethod
    def _connect(config: ReasoningConfig):
        from openai import OpenAI

        return OpenAI(
            base_url=config.base_url, api_key=LOCAL_API_KEY, timeout=config.timeout_s
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] = (),
        json_only: bool = False,
    ) -> ChatTurn:
        """Run one completion.

        :raises ReasoningUnavailableError: if the endpoint could not be
            reached or did not answer. The caller decides what that costs;
            it never costs the regulatory loop anything (FR-47).
        """
        started = self._clock.monotonic()
        request: dict[str, object] = {
            "model": self._config.model,
            "messages": list(messages),
            "temperature": 0.0,
        }
        if tools:
            request["tools"] = list(tools)
        if json_only:
            # The constrained decode of section 5.7.3, as the OpenAI-compatible
            # endpoint exposes it. It guarantees a parse, not sense.
            request["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**request)
        except Exception as exc:
            raise ReasoningUnavailableError(f"inference failed: {exc}") from exc
        latency_s = self._clock.monotonic() - started

        message = response.choices[0].message
        content = message.content or ""
        calls = tuple(_decode(call) for call in (message.tool_calls or ()))
        if not calls:
            calls, content = _recover_text_calls(content)
        usage = getattr(response, "usage", None)
        return ChatTurn(
            content=content.strip(),
            tool_calls=calls,
            latency_s=latency_s,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            raw=_render(content, calls),
        )


def assistant_message(turn: ChatTurn) -> dict[str, object]:
    """The turn as it goes back into the conversation for the next round."""
    message: dict[str, object] = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments))},
            }
            for call in turn.tool_calls
        ]
    return message


def tool_message(call: ToolCall, content: str) -> dict[str, object]:
    """A tool's answer, as the model reads it."""
    return {"role": "tool", "tool_call_id": call.call_id, "content": content}


def _decode(call) -> ToolCall:
    text = call.function.arguments or "{}"
    try:
        arguments = json.loads(text)
    except json.JSONDecodeError:
        return ToolCall(call.id, call.function.name, {}, malformed=text)
    if not isinstance(arguments, dict):
        return ToolCall(call.id, call.function.name, {}, malformed=text)
    return ToolCall(call.id, call.function.name, arguments)


def _recover_text_calls(content: str) -> tuple[tuple[ToolCall, ...], str]:
    calls = []
    for index, match in enumerate(_TEXT_TOOL_CALL.finditer(content)):
        try:
            body = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(body, dict) and isinstance(body.get("name"), str):
            arguments = body.get("arguments") or {}
            if isinstance(arguments, dict):
                calls.append(ToolCall(f"text_{index}", body["name"], arguments))
    if not calls:
        return (), content
    LOGGER.info("recovered %d tool call(s) written as text", len(calls))
    return tuple(calls), _TEXT_TOOL_CALL.sub("", content)


def _render(content: str, calls: Sequence[ToolCall]) -> str:
    """The model's output verbatim enough to audit (FR-46)."""
    parts = [content] if content else []
    parts.extend(
        f"{call.name}({call.malformed or json.dumps(dict(call.arguments))})"
        for call in calls
    )
    return "\n".join(parts)
