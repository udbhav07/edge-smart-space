"""A scripted stand-in for the OpenAI client, for every reasoning test.

It answers from a queue of replies written by the test, and records every
request it was sent, so a test can assert both what the model was told and
what the system did with the answer. No server, no network, no GPU.

Replies are built with :func:`text` and :func:`calls`. A reply may also be an
exception instance, which is raised instead of answered -- how a test makes
the server disappear mid-conversation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace


def text(content: str, prompt_tokens: int = 100, completion_tokens: int = 20):
    """A reply that is words only."""
    return _response(content, (), prompt_tokens, completion_tokens)


def calls(*requested: tuple[str, dict], content: str = ""):
    """A reply asking for tools: ``calls(("get_occupancy", {}), ...)``."""
    tool_calls = tuple(
        SimpleNamespace(
            id=f"call_{index}",
            function=SimpleNamespace(
                name=name,
                arguments=arguments if isinstance(arguments, str) else json.dumps(arguments),
            ),
        )
        for index, (name, arguments) in enumerate(requested)
    )
    return _response(content, tool_calls, 150, 30)


def _response(content, tool_calls, prompt_tokens, completion_tokens):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=list(tool_calls))
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


@dataclass
class ScriptedClient:
    """Answers each request with the next scripted reply."""

    replies: list = field(default_factory=list)
    requests: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        if not self.replies:
            raise RuntimeError("the script ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def tool_names_offered(self, index: int = -1) -> list[str]:
        return [
            tool["function"]["name"] for tool in self.requests[index].get("tools", [])
        ]

    def system_prompt(self, index: int = -1) -> str:
        return str(self.requests[index]["messages"][0]["content"])
