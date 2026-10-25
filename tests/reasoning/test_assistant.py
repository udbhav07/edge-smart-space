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
from src.common.localtime import epoch_of
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
SATURDAY_AFTERNOON = epoch_of(
    datetime(2026, 9, 26, 15, 0), load_config(Path("config/default.yaml")).site.utc_offset_h
)


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



class TestRepliesTellTheTruthAboutTheRoom:
    """U-03, U-04, U-06: a reply must not claim what the room is not doing."""

    def _mode(self, room, mode, fault_ids=()):
        from src.common.schemas import ModeState

        room.caller.publish(
            topics.SYSTEM_MODE,
            ModeState(ts=room.clock.now(), mode=mode, since_ts=room.clock.now(), active_fault_ids=fault_ids),
        )

    def _accepted(self, room, target):
        replies = TestTemperatureReplies()
        hint = replies._ask(room, target)
        replies._verdict(room, target, target, Verdict.ACCEPTED, ReasonCode.NONE)
        return room.assistant.answer(hint).reply

    def test_with_the_unit_on_hold_it_does_not_say_done(self, config):
        from src.common.schemas import Mode

        room = Room(config, ScriptedChat())
        self._mode(room, Mode.DEGRADED_ACTUATOR, ("f_ac_noresponse_1",))
        reply = self._accepted(room, 20.0)
        assert not reply.startswith("Done")
        assert "on hold" in reply and "will not be cooled" in reply

    def test_with_the_sensor_out_it_says_it_is_running_on_an_estimate(self, config):
        from src.common.schemas import Mode

        room = Room(config, ScriptedChat())
        self._mode(room, Mode.DEGRADED_SENSOR, ("f_temp01_stuck_1",))
        assert "estimate" in self._accepted(room, 22.0)

    def test_during_peak_it_says_what_the_room_is_held_at(self, config):
        from src.common.schemas import TariffBand, TrackingState

        room = Room(config, ScriptedChat())
        room.caller.publish(
            topics.CONTROL_TRACKING,
            TrackingState(
                ts=room.clock.now(), setpoint_c=24.0, tracked_c=25.0, band=TariffBand.PEAK,
                reason="peak tariff: held +1 C until 22:00",
            ),
        )
        reply = self._accepted(room, 22.0)
        assert "held at 23 degrees" in reply and "22:00" in reply

    def test_in_a_healthy_room_the_reply_is_unchanged(self, config):
        room = Room(config, ScriptedChat())
        assert self._accepted(room, 23.0) == "Done. The room is now set to 23 degrees."


class TestQuestionsAreAnswered:
    """U-06, U-25: a question or a greeting gets an answer, from the facts."""

    def _question(self, room, transcript):
        hint = PreferenceHint(
            ts=room.clock.now(), intent=Intent.QUESTION, comfort=Comfort.UNCHANGED,
            subject="room", transcript=transcript,
        )
        return room.assistant.answer(hint)

    def test_a_question_is_answered(self, config):
        room = Room(config, ScriptedChat(saying("It is 26.5 degrees.")))
        assert self._question(room, "what's the temperature in here").reply == "It is 26.5 degrees."

    def test_the_model_is_told_the_room_as_it_is(self, config):
        from src.common.schemas import FaultDiagnosis, Hypothesis, DiagnosisConfidence, Mode, ModeState, ThermalEstimate, AdaptationState

        chat = ScriptedChat(saying("The sensor is out."))
        room = Room(config, chat)
        room.caller.publish(
            topics.SYSTEM_MODE,
            ModeState(ts=room.clock.now(), mode=Mode.DEGRADED_SENSOR, since_ts=room.clock.now(),
                      active_fault_ids=("f_temp01_stuck_1",)),
        )
        room.caller.publish(
            topics.DIAGNOSIS,
            FaultDiagnosis(
                ts=room.clock.now(), fault_id="f_temp01_stuck_1",
                primary_hypothesis=Hypothesis.SENSOR_STUCK, confidence=DiagnosisConfidence.HIGH,
                recommended_mode=Mode.DEGRADED_SENSOR, user_message="The temperature sensor appears stuck.",
                recommendation_legal=True,
            ),
            fault_id="f_temp01_stuck_1",
        )
        room.caller.publish(
            topics.ESTIMATE_THERMAL,
            ThermalEstimate(ts=room.clock.now(), t_in=27.0, t_pred=26.4, residual=0.6,
                            residual_sigma=0.2, model_confidence=0.9, adaptation=AdaptationState.FROZEN),
        )
        self._question(room, "is everything working ok")
        prompt = chat.requests[0][0][0]["content"]
        assert "The temperature sensor appears stuck." in prompt
        assert "26.4 C (an estimate" in prompt

    def test_a_greeting_gets_a_reply(self, config):
        room = Room(config, ScriptedChat(saying("Hello! How can I help?")))
        assert self._question(room, "hello").reply == "Hello! How can I help?"



class TestCalendarCallsAreChecked:
    """U-02, U-17: what the model wrote is checked against the clock and the words.

    The room's clock here reads Saturday 26 September 2026, 15:00.
    """

    def test_an_entry_titled_as_a_cancellation_cancels_instead(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review")),
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="Cancel review")),
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        reply = room.say("cancel the review")
        assert room.calendar() == []
        assert "Cancelled review" in reply.reply

    def test_a_time_already_past_is_refused_with_a_question(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-09-25T17:00:00", subject="dentist"))
        )
        room = Room(config, chat)
        reply = room.say("schedule the dentist yesterday at 5pm")
        assert room.calendar() == []
        assert "already passed" in reply.reply and "Which day" in reply.reply

    def test_todays_weekday_after_its_time_means_next_week(self, config):
        """'Saturday at 10', said on Saturday afternoon, is next Saturday."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-09-26T10:00:00", subject="yoga"))
        )
        room = Room(config, chat)
        room.say("put yoga on Saturday at 10")
        assert [event["starts_at"] for event in room.calendar()] == ["2026-10-03T10:00"]

    def test_a_reminder_says_it_will_not_alert(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-09-26T15:10:00", subject="call mom"))
        )
        room = Room(config, chat)
        reply = room.say("remind me to call mom in 10 minutes")
        assert "won't remind you" in reply.reply

    def test_an_ordinary_entry_says_nothing_about_alerts(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review"))
        )
        room = Room(config, chat)
        assert "remind" not in room.say("put the review on Thursday at 3").reply

    def test_a_follow_up_moves_the_entry(self, config):
        """U-01's other half: 'move it to 4' now has somewhere to go."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review")),
            calling(call("move_event", subject="review", starts_at="2026-10-01T16:00:00")),
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        reply = room.say("move it to 4")
        assert [event["starts_at"] for event in room.calendar()] == ["2026-10-01T16:00"]
        assert "Moved review" in reply.reply


class TestNoClaimWithoutATool:
    """Live: "cancel my meeting with the CEO" -- there was none -- was answered
    "-cancelled the meeting with the CEO" with no tool called."""

    def test_a_claim_repeated_after_the_nudge_is_withheld(self, config):
        claim = "Cancelled the meeting with the CEO at 16:00."
        room = Room(config, ScriptedChat(saying(claim), saying(claim)))
        reply = room.say("cancel my meeting with the CEO")
        assert "Cancelled" not in reply.reply and "haven't changed anything" in reply.reply

    def test_a_fake_confirmation_question_is_nudged_into_a_real_call(self, config):
        """Live: "That would be a mock booking ... Shall I go ahead?" with no tool."""
        fake = "That would be a mock booking of a flight to Goa; no real reservation is made. Shall I go ahead?"
        chat = ScriptedChat(saying(fake), calling(call("book_travel", kind="flight", destination="Goa", depart_on="2026-10-03")))
        room = Room(config, chat)
        reply = room.say("book a flight to Goa next Saturday")
        assert reply.awaiting_confirmation, "a real invocation now waits for the occupant"

    def test_an_honest_second_answer_is_used(self, config):
        fake = "Shall I go ahead and book it?"
        room = Room(config, ScriptedChat(saying(fake), saying("I need a destination to book a flight.")))
        assert room.say("book me a flight").reply == "I need a destination to book a flight."

    def test_a_plain_answer_is_left_alone(self, config):
        room = Room(config, ScriptedChat(saying("Which day would you like it on?")))
        assert room.say("put a meeting in with Anu").reply == "Which day would you like it on?"


class TestInSoManyMinutes:
    def test_the_entry_is_put_that_far_ahead(self, config):
        """The model put "in 10 minutes" at now, for ten minutes, and it was refused as past."""
        now = datetime(2026, 9, 26, 15, 0)
        chat = ScriptedChat(
            calling(call("schedule_event", subject="call mom", starts_at=now.isoformat(), duration_min=10)),
            saying("Added."),
        )
        room = Room(config, chat)
        room.say("remind me to call mom in 10 minutes")
        assert room.calendar()[0]["starts_at"].startswith("2026-09-26T15:10")
