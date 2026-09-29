"""Unit tests for the typed stand-in for a voice."""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import AssistantReply, Comfort, Intent, PreferenceHint
from src.common.tools import ToolInvocation, ToolRequester, ToolResult, ToolStatus
from tools.say import Conversation, converse


class StubContext:
    """Stands in for Personal Context: returns a fixed hint, or none."""

    def __init__(self, hint=None) -> None:
        self._hint = hint
        self.heard: list[str] = []

    def extract(self, transcript):
        self.heard.append(transcript)
        return self._hint


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


def _wired(config, hint):
    clock = SimClock()
    transport = LoopbackTransport()
    mine = Blackboard(config.mqtt, transport)
    other = Blackboard(config.mqtt, transport)
    transport.attach(mine)
    transport.attach(other)
    context = StubContext(hint)
    conversation = Conversation(config, clock, mine, context)
    conversation.subscribe()
    return conversation, transport, other, clock, context


def _hint(clock, transcript="make it 22"):
    return PreferenceHint(
        ts=clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.UNCHANGED,
        subject="temperature", target_c=22.0, transcript=transcript,
    )


def test_what_is_heard_goes_through_personal_context(config):
    clock = SimClock()
    conversation, _, _, _, context = _wired(config, _hint(clock))
    conversation.hear("make it 22")
    assert context.heard == ["make it 22"]


def test_the_hint_is_published_where_the_speech_pipeline_publishes_it(config):
    clock = SimClock()
    hint = _hint(clock)
    conversation, transport, _, _, _ = _wired(config, hint)
    conversation.hear("make it 22")
    published = [p for t, p, _, _ in transport.published if t == topics.CONTEXT_PREFERENCE.pattern]
    assert PreferenceHint.model_validate_json(published[-1]) == hint


def test_nothing_actionable_publishes_nothing(config):
    conversation, transport, _, _, _ = _wired(config, None)
    assert conversation.hear("what is the capital of France") is None
    assert not [t for t, *_ in transport.published if t == topics.CONTEXT_PREFERENCE.pattern]


def test_the_reply_to_this_utterance_is_the_one_returned(config):
    conversation, _, other, clock, _ = _wired(config, None)
    for transcript in ("something else", "make it 22"):
        other.publish(
            topics.CONTEXT_REPLY,
            AssistantReply(ts=clock.now(), transcript=transcript, intent=Intent.ENVIRONMENT, reply=f"re: {transcript}"),
        )
    assert conversation.await_reply("make it 22", timeout_s=1.0).reply == "re: make it 22"


def test_confirming_republishes_the_invocation_verbatim(config):
    """FR-74: confirmation is carried by the topic, and the invocation is unchanged."""
    conversation, transport, other, clock, _ = _wired(config, None)
    invocation = ToolInvocation(
        ts=clock.now(), invocation_id="inv_9", tool="book_travel",
        arguments={"kind": "flight", "destination": "Delhi", "depart_on": "2026-09-28T00:00:00"},
        requester=ToolRequester.PERSONAL_CONTEXT, expires_ts=clock.now() + 300.0,
    )
    other.publish(topics.ASSIST_PROPOSED, invocation)
    other.publish(
        topics.ASSIST_RESULT,
        ToolResult(ts=clock.now(), invocation_id="inv_9", tool="book_travel", status=ToolStatus.OK,
                   message="Mock booking made.", provider="mock_travel", simulated=True),
    )
    result = conversation.confirm("inv_9", timeout_s=1.0)
    confirmed = [p for t, p, _, _ in transport.published if t == topics.ASSIST_CONFIRMED.pattern]
    assert ToolInvocation.model_validate_json(confirmed[-1]) == invocation
    assert result.message == "Mock booking made."


def test_an_unknown_invocation_cannot_be_confirmed(config):
    conversation, *_ = _wired(config, None)
    assert conversation.confirm("inv_missing", timeout_s=0.1) is None


def test_declining_books_nothing(config, capsys):
    clock = SimClock()
    hint = PreferenceHint(
        ts=clock.now(), intent=Intent.SERVICE, comfort=Comfort.UNCHANGED,
        subject="booking", transcript="book a flight",
    )
    conversation, transport, other, clock, _ = _wired(config, hint)
    other.publish(
        topics.CONTEXT_REPLY,
        AssistantReply(ts=clock.now(), transcript="book a flight", intent=Intent.SERVICE,
                       reply="Shall I go ahead?", awaiting_confirmation="inv_1"),
    )
    converse(conversation, "book a flight", answer_yes=False, ask=lambda _: "no")
    assert "nothing was booked" in capsys.readouterr().out
    assert not [t for t, *_ in transport.published if t == topics.ASSIST_CONFIRMED.pattern]
