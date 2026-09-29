"""Unit tests for the assistant (sections 5.7.1 and 5.7.6).

The model is scripted; the executor, the calendar and the blackboard are real,
so an invocation crosses the same topics it would against mosquitto and the
calendar file is what is asserted on.
"""

from datetime import datetime
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service as build_executor
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AssistantReply,
    Comfort,
    GoalSource,
    Intent,
    PreferenceHint,
    ReasonCode,
    ReasoningRecord,
    ValidationVerdict,
    Verdict,
)
from src.common.tools import ToolInvocation
from src.reasoning.__main__ import build_assistant
from tests.reasoning.stubs import UNAVAILABLE, ScriptedChat, call, calling, saying

#: A Saturday afternoon, in local time, so weekday resolution is deterministic.
SATURDAY_AFTERNOON = datetime(2026, 9, 26, 15, 0).timestamp()


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(
        update={"calendar_path": str(tmp_path / "calendar.json"), "result_timeout_s": 1.0}
    )
    return config.model_copy(update={"assistance": assistance})


class Room:
    """An assistant, a real executor, and a place to look at what was published."""

    def __init__(self, config, chat) -> None:
        self.clock = SimClock(start_epoch_s=SATURDAY_AFTERNOON)
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(3)]
        for board in boards:
            self.transport.attach(board)
        self.executor = build_executor(config, self.clock, boards[0])
        self.executor.subscribe()
        self.assistant = build_assistant(config, self.clock, boards[1], chat)
        self.assistant.subscribe()
        self.caller = boards[2]
        self.calendar_path = Path(config.assistance.calendar_path)

    def say(self, transcript, intent=Intent.SERVICE, **fields) -> AssistantReply:
        hint = PreferenceHint(
            ts=self.clock.now(),
            intent=intent,
            comfort=fields.pop("comfort", Comfort.UNCHANGED),
            subject=fields.pop("subject", "calendar"),
            transcript=transcript,
            **fields,
        )
        self.caller.publish(topics.CONTEXT_PREFERENCE, hint)
        replies = self.assistant.process_pending()
        return replies[-1] if replies else None

    def published(self, spec, schema):
        return [
            schema.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic == spec.pattern and payload
        ]

    def calendar(self):
        import json

        if not self.calendar_path.exists():
            return []
        return json.loads(self.calendar_path.read_text())["events"]


class TestScheduling:
    def test_a_meeting_lands_in_the_calendar(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review"))
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        assert [e["starts_at"] for e in room.calendar()] == ["2026-10-01T15:00"]

    def test_the_reply_is_the_calendars_own_sentence(self, config):
        """No second completion: the provider already said what happened."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review"))
        )
        room = Room(config, chat)
        reply = room.say("put the review on Thursday at 3")
        assert reply.reply == "Added review at 15:00 on Thursday 1 October."
        assert len(chat.requests) == 1

    def test_a_wrong_weekday_is_corrected_to_the_one_named(self, config):
        """Live, a 7B model booked 'Tuesday at 11' on a Monday (FR-44)."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-09-28T11:00:00", subject="dentist"))
        )
        room = Room(config, chat)
        room.say("dentist on Tuesday at 11")
        assert [e["starts_at"] for e in room.calendar()] == ["2026-09-29T11:00"]

    def test_the_correction_is_recorded(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-09-28T11:00:00", subject="dentist"))
        )
        room = Room(config, chat)
        room.say("dentist on Tuesday at 11")
        record = room.published(topics.AUDIT_REASONING, ReasoningRecord)[-1]
        assert "date corrected" in record.verdict

    def test_the_model_is_told_the_day_it_named(self, config):
        chat = ScriptedChat(saying("ok"))
        room = Room(config, chat)
        room.say("dentist on Tuesday at 11")
        user_turn = chat.requests[0][0][1]["content"]
        assert "Tuesday = 2026-09-29" in user_turn

    def test_the_invocation_crosses_the_blackboard(self, config):
        """FR-71: the assistant holds no provider; the executor runs it."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review"))
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        assert room.published(topics.ASSIST_PROPOSED, ToolInvocation)[-1].tool == "schedule_event"


class TestReading:
    def test_a_question_is_answered_from_the_calendar(self, config):
        chat = ScriptedChat(
            calling(call("get_events", from_time="2026-10-01T00:00:00", to_time="2026-10-01T23:59:00")),
            saying("You have nothing on Thursday."),
        )
        room = Room(config, chat)
        reply = room.say("what is on Thursday")
        assert reply.reply == "You have nothing on Thursday."

    def test_the_answer_is_phrased_without_tools_on_offer(self, config):
        chat = ScriptedChat(
            calling(call("get_events", from_time="2026-10-01T00:00:00", to_time="2026-10-01T23:59:00")),
            saying("Nothing."),
        )
        room = Room(config, chat)
        room.say("what is on Thursday")
        assert chat.requests[1][1] == ()

    def test_a_window_is_shifted_whole_not_collapsed(self, config):
        chat = ScriptedChat(
            calling(call("get_events", from_time="2026-09-30T00:00:00", to_time="2026-10-01T00:00:00")),
            saying("Nothing."),
        )
        room = Room(config, chat)
        room.say("what is on Thursday")
        invocation = room.published(topics.ASSIST_PROPOSED, ToolInvocation)[-1]
        assert invocation.arguments["from_time"] == "2026-10-01T00:00:00"
        assert invocation.arguments["to_time"] == "2026-10-02T00:00:00"


class TestBooking:
    def _flight(self):
        return calling(
            call("book_travel", kind="flight", destination="Delhi", depart_on="2026-09-28T00:00:00", origin="Hyderabad")
        )

    def test_a_flight_gets_a_question_not_a_booking(self, config):
        """The Week 6 demonstration, and FR-54."""
        room = Room(config, ScriptedChat(self._flight()))
        reply = room.say("book a flight to Delhi on Monday")
        assert reply.awaiting_confirmation
        assert "Nothing is booked yet" in reply.reply

    def test_the_question_says_what_and_that_it_is_a_mock(self, config):
        room = Room(config, ScriptedChat(self._flight()))
        reply = room.say("book a flight to Delhi on Monday")
        assert "a flight from Hyderabad to Delhi on Monday 28 September" in reply.reply
        assert "mock" in reply.reply


class TestWithoutTools:
    def test_a_request_needing_no_tool_is_answered_directly(self, config):
        room = Room(config, ScriptedChat(saying("I cannot order food.")))
        assert room.say("order a pizza").reply == "I cannot order food."

    def test_an_unreachable_model_falls_back_to_the_hints_reply(self, config):
        room = Room(config, ScriptedChat(UNAVAILABLE))
        reply = room.say("book something", spoken_reply="I have passed that on.")
        assert reply.reply == "I have passed that on."

    def test_unreadable_arguments_are_refused_not_run(self, config):
        from src.reasoning.chat import ToolCall

        bad = ToolCall("c1", "schedule_event", {}, malformed="{not json")
        room = Room(config, ScriptedChat(calling(bad)))
        room.say("put something in")
        assert room.calendar() == []


class TestTheAudit:
    def test_every_answer_is_recorded_with_its_cost(self, config):
        """FR-46, FR-63."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review"), latency_s=2.5)
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        record = room.published(topics.AUDIT_REASONING, ReasoningRecord)[-1]
        assert record.latency_s == 2.5
        assert record.prompt_tokens == 100
        assert record.tool_calls == ("schedule_event",)

    def test_every_reply_is_published(self, config):
        room = Room(config, ScriptedChat(saying("done")))
        room.say("something")
        assert room.published(topics.CONTEXT_REPLY, AssistantReply)[-1].reply == "done"


class TestTemperatureReplies:
    """The reply to a setpoint request is written from the verdict, not by the model."""

    def _verdict(self, room, asked, applied, verdict, reason):
        message = ValidationVerdict(
            ts=room.clock.now(),
            proposed={"setpoint_c": asked, "source": GoalSource.PREFERENCE.value},
            verdict=verdict,
            reason=reason,
            applied={"setpoint_c": applied},
        )
        room.caller.publish(topics.AUDIT_VALIDATION, message)

    def _ask(self, room, target):
        hint = PreferenceHint(
            ts=room.clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.UNCHANGED,
            subject="temperature", target_c=target, transcript=f"make it {target}",
        )
        return hint

    def test_an_accepted_request_is_confirmed(self, config):
        room = Room(config, ScriptedChat())
        hint = self._ask(room, 23.0)
        self._verdict(room, 23.0, 23.0, Verdict.ACCEPTED, ReasonCode.NONE)
        assert room.assistant.answer(hint).reply == "Done. The room is now set to 23 degrees."

    def test_an_unsafe_request_names_the_safe_range(self, config):
        room = Room(config, ScriptedChat())
        hint = self._ask(room, 5.0)
        self._verdict(room, 5.0, 22.0, Verdict.CLAMPED, ReasonCode.RATE_LIMIT)
        reply = room.assistant.answer(hint).reply
        assert "outside the safe range of 18 to 30" in reply
        assert "the closest I can do is 18 degrees" in reply

    def test_a_rate_limited_request_says_it_is_on_its_way(self, config):
        room = Room(config, ScriptedChat())
        hint = self._ask(room, 20.0)
        self._verdict(room, 20.0, 22.0, Verdict.CLAMPED, ReasonCode.RATE_LIMIT)
        assert "Heading to 20 degrees" in room.assistant.answer(hint).reply

    def test_a_request_about_the_lights_is_answered_at_once(self, config):
        room = Room(config, ScriptedChat())
        hint = PreferenceHint(
            ts=room.clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.UNCHANGED,
            subject="lights", transcript="turn off the lights",
        )
        started = room.clock.monotonic()
        reply = room.assistant.answer(hint).reply
        assert reply == "I can only change the temperature, not the lights."
        assert room.clock.monotonic() == started

    def test_the_model_is_never_asked(self, config):
        chat = ScriptedChat()
        room = Room(config, chat)
        hint = self._ask(room, 23.0)
        self._verdict(room, 23.0, 23.0, Verdict.ACCEPTED, ReasonCode.NONE)
        room.assistant.answer(hint)
        assert chat.requests == []
