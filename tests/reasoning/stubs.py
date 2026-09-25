"""A scripted model, for testing the reasoning layer without a server."""

from src.reasoning.chat import ChatTurn, ToolCall
from src.reasoning.single_shot import ReasoningUnavailableError


class ScriptedChat:
    """Answers each completion with the next scripted turn, and records the request."""

    model = "scripted"

    def __init__(self, *turns: ChatTurn | Exception) -> None:
        self._turns = list(turns)
        self.requests: list[tuple[list, tuple]] = []

    def complete(self, messages, tools=(), json_only=False):
        self.requests.append((list(messages), tuple(tools)))
        if not self._turns:
            return ChatTurn(content="(nothing scripted)")
        turn = self._turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(call_id=f"call_{name}", name=name, arguments=arguments)


def calling(*calls: ToolCall, latency_s: float = 1.0) -> ChatTurn:
    return ChatTurn(
        content="",
        tool_calls=calls,
        latency_s=latency_s,
        prompt_tokens=100,
        completion_tokens=10,
        raw="calls",
    )


def saying(text: str, latency_s: float = 1.0) -> ChatTurn:
    return ChatTurn(
        content=text, latency_s=latency_s, prompt_tokens=50, completion_tokens=5, raw=text
    )


UNAVAILABLE = ReasoningUnavailableError("no server")
