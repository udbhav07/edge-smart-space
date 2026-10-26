"""Typed console utterances take the speech path (FR-56, FR-57)."""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import AssistantReply, Comfort, ConsoleUtterance, Intent, PreferenceHint
from src.reasoning.single_shot import ReasoningUnavailableError
from src.reasoning.utterances import NO_MODEL, NOTHING_TO_DO, UtteranceInterpreter


class FakeContext:
    def __init__(self, hint=None, error=None):
        self.hint, self.error, self.calls = hint, error, []

    def extract(self, text, recent=()):
        self.calls.append(text)
        if self.error:
            raise self.error
        return self.hint


@pytest.fixture(name="room")
def _room():
    config = load_config(Path("config/default.yaml"))
    clock = SimClock()
    transport = LoopbackTransport()
    boards = [Blackboard(config.mqtt, transport) for _ in range(2)]
    for board in boards:
        transport.attach(board)
    return clock, transport, boards


def _say(board, clock, text, profile_id=""):
    board.publish(topics.CONTEXT_UTTERANCE, ConsoleUtterance(ts=clock.now(), utterance_id="u1", text=text, profile_id=profile_id))


def _published(transport, spec, schema):
    return [schema.model_validate_json(p) for t, p, _, _ in transport.published if t == spec.pattern and p]


def test_an_utterance_becomes_a_preference_hint(room):
    clock, transport, (mine, console) = room
    context = FakeContext(PreferenceHint(ts=clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.COOLER))
    interpreter = UtteranceInterpreter(clock, mine, context)
    interpreter.subscribe()
    _say(console, clock, "it's stuffy in here", "ravi")
    interpreter.process_pending()
    hint = _published(transport, topics.CONTEXT_PREFERENCE, PreferenceHint)[-1]
    assert hint.comfort is Comfort.COOLER and hint.transcript == "it's stuffy in here" and hint.profile_id == "ravi"


def test_nothing_actionable_is_still_answered(room):
    """A console has someone watching; silence would look broken."""
    clock, transport, (mine, console) = room
    interpreter = UtteranceInterpreter(clock, mine, FakeContext(hint=None))
    interpreter.subscribe()
    _say(console, clock, "blue")
    interpreter.process_pending()
    assert _published(transport, topics.CONTEXT_REPLY, AssistantReply)[-1].reply == NOTHING_TO_DO


def test_an_unavailable_model_is_said_plainly(room):
    clock, transport, (mine, console) = room
    interpreter = UtteranceInterpreter(clock, mine, FakeContext(error=ReasoningUnavailableError("down")))
    interpreter.subscribe()
    _say(console, clock, "make it cooler")
    interpreter.process_pending()
    assert _published(transport, topics.CONTEXT_REPLY, AssistantReply)[-1].reply == NO_MODEL


def test_the_console_cannot_command_the_actuator(room):
    """FR-57: whatever is typed, nothing is published on an actuator topic."""
    clock, transport, (mine, console) = room
    context = FakeContext(PreferenceHint(ts=clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.UNCHANGED, target_c=500.0))
    interpreter = UtteranceInterpreter(clock, mine, context)
    interpreter.subscribe()
    _say(console, clock, "set the AC to 500 degrees now")
    interpreter.process_pending()
    assert not any("/actuator/" in topic for topic, _, _, _ in transport.published)


def test_a_typed_request_says_it_was_typed(room):
    """S-12: so the reply says 'chatting as', not 'recognised your voice'."""
    clock, transport, (mine, console) = room
    context = FakeContext(PreferenceHint(ts=clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.COOLER))
    interpreter = UtteranceInterpreter(clock, mine, context)
    interpreter.subscribe()
    _say(console, clock, "it's stuffy in here", "ravi")
    interpreter.process_pending()
    assert _published(transport, topics.CONTEXT_PREFERENCE, PreferenceHint)[-1].typed is True
