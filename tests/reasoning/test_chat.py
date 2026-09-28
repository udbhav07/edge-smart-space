"""Unit tests for the shared chat client: decoding, timing, and counting."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.common.clock import SimClock
from src.common.config import load_config
from src.reasoning.chat import ChatClient, assistant_message, tool_message
from src.reasoning.single_shot import ReasoningUnavailableError


class FakeCompletions:
    def __init__(self, message, usage=None, delay_s=0.0, clock=None, error=None):
        self._message = message
        self._usage = usage
        self._delay_s = delay_s
        self._clock = clock
        self._error = error
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if self._error:
            raise self._error
        if self._clock:
            self._clock.advance(self._delay_s)
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)], usage=self._usage)


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _function_call(name, arguments):
    return SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=arguments))


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml")).reasoning


def test_tool_calls_are_decoded_to_plain_values(config):
    message = SimpleNamespace(content="", tool_calls=[_function_call("get_events", '{"from_time": "a"}')])
    turn = ChatClient(config, SimClock(), _client(FakeCompletions(message))).complete([], [{}])
    assert turn.tool_calls[0].name == "get_events"
    assert turn.tool_calls[0].arguments == {"from_time": "a"}


def test_unreadable_arguments_are_reported_not_dropped(config):
    message = SimpleNamespace(content="", tool_calls=[_function_call("get_events", "{oops")])
    turn = ChatClient(config, SimClock(), _client(FakeCompletions(message))).complete([], [{}])
    assert turn.tool_calls[0].malformed == "{oops"


def test_a_tool_call_written_as_text_is_recovered(config):
    text = '<tool_call>{"name": "get_events", "arguments": {"from_time": "a"}}</tool_call>'
    message = SimpleNamespace(content=text, tool_calls=None)
    turn = ChatClient(config, SimClock(), _client(FakeCompletions(message))).complete([], [{}])
    assert [call.name for call in turn.tool_calls] == ["get_events"]
    assert turn.content == ""


def test_latency_is_measured_on_the_injected_clock(config):
    clock = SimClock()
    message = SimpleNamespace(content="hi", tool_calls=None)
    turn = ChatClient(config, clock, _client(FakeCompletions(message, delay_s=2.5, clock=clock))).complete([])
    assert turn.latency_s == 2.5


def test_tokens_are_counted(config):
    message = SimpleNamespace(content="hi", tool_calls=None)
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=7)
    turn = ChatClient(config, SimClock(), _client(FakeCompletions(message, usage))).complete([])
    assert (turn.prompt_tokens, turn.completion_tokens) == (120, 7)


def test_no_tools_are_sent_when_none_are_offered(config):
    completions = FakeCompletions(SimpleNamespace(content="hi", tool_calls=None))
    ChatClient(config, SimClock(), _client(completions)).complete([])
    assert "tools" not in completions.requests[0]


def test_an_unreachable_server_is_reported_as_such(config):
    completions = FakeCompletions(None, error=ConnectionError("refused"))
    with pytest.raises(ReasoningUnavailableError):
        ChatClient(config, SimClock(), _client(completions)).complete([])


def test_a_turn_goes_back_into_the_conversation_with_its_calls(config):
    message = SimpleNamespace(content="", tool_calls=[_function_call("get_events", "{}")])
    turn = ChatClient(config, SimClock(), _client(FakeCompletions(message))).complete([], [{}])
    rendered = assistant_message(turn)
    assert rendered["tool_calls"][0]["function"]["name"] == "get_events"
    assert tool_message(turn.tool_calls[0], "none")["tool_call_id"] == "c1"
