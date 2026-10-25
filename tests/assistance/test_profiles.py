"""Unit tests for remembering through the tool gates, and publishing context.

The executor is real, the store is real, the transport is in-process: a fact
reaches a person's file only by an invocation the registry admitted, and
their context reaches the blackboard only through the keeper.
"""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_keeper, build_service, build_store
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import GUEST_PROFILE_ID, AssistantReply, Intent, ProfileContext
from src.common.tools import ToolInvocation, ToolRequester, ToolResult, ToolStatus


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    profiles = config.profiles.model_copy(
        update={"directory": str(tmp_path / "profiles"), "context_directory": str(tmp_path / "context")}
    )
    assistance = config.assistance.model_copy(update={"calendar_path": str(tmp_path / "calendar.json")})
    return config.model_copy(update={"profiles": profiles, "assistance": assistance})


class Room:
    def __init__(self, config) -> None:
        self.config = config
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(3)]
        for board in boards:
            self.transport.attach(board)
        self.executor = build_service(config, self.clock, boards[0])
        self.executor.subscribe()
        self.keeper = build_keeper(config, self.clock, boards[1])
        self.keeper.subscribe()
        self.caller = boards[2]
        self.store = build_store(config, self.clock)
        self._next = 0

    def invoke(self, tool: str, **arguments) -> ToolResult:
        self._next += 1
        invocation = ToolInvocation(
            ts=self.clock.now(),
            invocation_id=f"inv_{self._next}",
            tool=tool,
            arguments=arguments,
            requester=ToolRequester.PERSONAL_CONTEXT,
            expires_ts=self.clock.now() + 300.0,
        )
        self.caller.publish(topics.ASSIST_PROPOSED, invocation)
        return self.results()[-1]

    def results(self) -> list[ToolResult]:
        return [
            ToolResult.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic == topics.ASSIST_RESULT.pattern
        ]

    def context(self, profile_id: str) -> ProfileContext:
        topic = topics.CONTEXT_PROFILE.format(profile_id=profile_id)
        payloads = [payload for name, payload, _, _ in self.transport.published if name == topic]
        return ProfileContext.model_validate_json(payloads[-1])

    def reply(self, text: str, said: str, profile_id: str = "") -> None:
        self.caller.publish(
            topics.CONTEXT_REPLY,
            AssistantReply(
                ts=self.clock.now(), transcript=said, intent=Intent.SERVICE, reply=text,
                profile_id=profile_id,
            ),
        )


@pytest.fixture(name="room")
def _room(config):
    room = Room(config)
    room.store.create("ravi", "Ravi")
    return room


class TestRememberingThroughTheGates:
    def test_a_fact_is_kept_on_the_reasoning_layers_own_authority(self, room):
        """A write tool: no confirmation (FR-74)."""
        result = room.invoke("remember", profile_id="ravi", fact="prefers meetings after ten")
        assert result.status is ToolStatus.OK
        assert room.store.load("ravi").facts == ("prefers meetings after ten",)

    def test_the_reply_names_the_person(self, room):
        assert "Ravi" in room.invoke("remember", profile_id="ravi", fact="vegetarian").message

    def test_a_guest_has_nowhere_to_keep_a_fact(self, room):
        result = room.invoke("remember", profile_id=GUEST_PROFILE_ID, fact="vegetarian")
        assert result.status is ToolStatus.FAILED
        assert "recognise" in result.message

    def test_a_missing_speaker_is_refused_before_any_file_is_touched(self, room):
        """FR-73: profile_id is required, even though the model never sees it."""
        result = room.invoke("remember", fact="vegetarian")
        assert result.status is ToolStatus.BAD_ARGUMENTS

    def test_forgetting_through_the_gate(self, room):
        room.invoke("remember", profile_id="ravi", fact="vegetarian")
        result = room.invoke("forget", profile_id="ravi", about="vegetarian")
        assert result.status is ToolStatus.OK and room.store.load("ravi").facts == ()


class TestPublishing:
    def test_a_remembered_fact_reaches_the_blackboard(self, room):
        room.invoke("remember", profile_id="ravi", fact="vegetarian")
        assert room.context("ravi").facts == ("vegetarian",)

    def test_the_context_is_retained(self, room):
        room.keeper.publish("ravi")
        topic = topics.CONTEXT_PROFILE.format(profile_id="ravi")
        assert any(name == topic and retain for name, _, _, retain in room.transport.published)

    def test_an_exchange_becomes_short_term_context(self, room):
        room.reply("Added it at 15:00.", said="put the review at three", profile_id="ravi")
        assert room.context("ravi").recent[-1].said == "put the review at three"

    def test_an_unrecognised_speakers_exchange_is_the_guests(self, room):
        room.reply("It is three.", said="what time is it")
        assert room.context(GUEST_PROFILE_ID).recent[-1].replied == "It is three."

    def test_a_reply_for_someone_since_deleted_costs_only_that_update(self, room):
        room.store.delete("ravi")
        room.reply("ok", said="hi", profile_id="ravi")  # must not raise

    def test_a_deleted_profile_is_withdrawn_from_the_blackboard(self, room):
        room.keeper.publish_all()
        room.store.delete("ravi")
        room.keeper.publish_all()
        topic = topics.CONTEXT_PROFILE.format(profile_id="ravi")
        last = [payload for name, payload, _, _ in room.transport.published if name == topic][-1]
        assert last == b""

    def test_the_sweep_runs_on_its_interval_only(self, room, config):
        room.keeper.tick()
        published = len(room.transport.published)
        room.keeper.tick()
        assert len(room.transport.published) == published
        room.clock.advance(config.profiles.publish_interval_s)
        room.keeper.tick()
        assert len(room.transport.published) > published


class TestSuggestionsThroughTheGates:
    """FR-78: overheard, offered, and kept only on the person's word."""

    def test_a_suggestion_is_held_not_remembered(self, room):
        result = room.invoke("suggest_fact", profile_id="ravi", fact="Hates the cold.", heard="I hate the cold")
        assert result.status is ToolStatus.OK
        assert room.store.load("ravi").facts == () and room.store.load("ravi").suggestions == ("Hates the cold.",)

    def test_it_reaches_the_blackboard_as_a_suggestion(self, room):
        room.invoke("suggest_fact", profile_id="ravi", fact="Hates the cold.")
        context = room.context("ravi")
        assert context.suggestions == ("Hates the cold.",) and context.facts == ()

    def test_nothing_overheard_from_a_guest_is_held(self, room):
        result = room.invoke("suggest_fact", profile_id=GUEST_PROFILE_ID, fact="Hates the cold.")
        assert result.status is ToolStatus.FAILED

    def test_kept_it_is_remembered_and_published(self, room):
        room.invoke("suggest_fact", profile_id="ravi", fact="Hates the cold.")
        result = room.invoke("settle_fact", profile_id="ravi", fact="Hates the cold.", keep=True)
        assert result.status is ToolStatus.OK
        assert room.context("ravi").facts == ("Hates the cold.",) and room.context("ravi").suggestions == ()

    def test_dismissed_it_is_gone_from_the_blackboard(self, room):
        room.invoke("suggest_fact", profile_id="ravi", fact="Hates the cold.")
        room.invoke("settle_fact", profile_id="ravi", fact="Hates the cold.", keep=False)
        assert room.context("ravi").suggestions == () and room.context("ravi").facts == ()
