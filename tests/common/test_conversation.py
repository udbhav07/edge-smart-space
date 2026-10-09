"""Unit tests for the conversation-so-far cache (FR-52, U-01)."""

from src.common.conversation import Conversations
from src.common.schemas import ContextTurn, ProfileContext

TURN = ContextTurn(ts=1.0, said="hello", replied="hi")


def _heard(conversations, profile_id, turns):
    conversations._on_context("", ProfileContext(ts=1.0, profile_id=profile_id, recent=turns))


def test_a_speakers_conversation_is_returned():
    conversations = Conversations()
    _heard(conversations, "ravi", (TURN,))
    assert conversations.recent("ravi") == (TURN,)


def test_an_unrecognised_speaker_gets_the_guests():
    conversations = Conversations()
    _heard(conversations, "guest", (TURN,))
    assert conversations.recent("") == (TURN,)


def test_nobody_heard_of_has_no_conversation():
    assert Conversations().recent("anu") == ()


def test_the_cache_is_bounded():
    conversations = Conversations()
    for index in range(100):
        _heard(conversations, f"p{index}", (TURN,))
    assert len(conversations._recent) <= 32
