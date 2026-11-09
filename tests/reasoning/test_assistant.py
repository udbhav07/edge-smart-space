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

    def __init__(self, config, chat, **executor) -> None:
        self.clock = SimClock(start_epoch_s=SATURDAY_AFTERNOON)
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(3)]
        for board in boards:
            self.transport.attach(board)
        self.executor = build_executor(config, self.clock, boards[0], **executor)
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

    def test_cooler_at_the_limit_names_no_number_nobody_said(self, config):
        """Live: "why is it not cooling?" read as "cooler" from 18 was answered about 17 degrees."""
        room = Room(config, ScriptedChat())
        hint = PreferenceHint(ts=room.clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.COOLER, subject="temperature", transcript="make it cooler")
        self._verdict(room, 17.0, 18.0, Verdict.CLAMPED, ReasonCode.BOUND_CLAMP)
        reply = room.assistant.answer(hint).reply
        assert "17" not in reply and "as cool as I can make it: 18 degrees" in reply

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

    def test_a_follow_up_naming_no_day_keeps_the_entrys_day(self, config):
        """Live: 'tomorrow at three', then 'move it to four' -- the model wrote
        today's date, and the move was refused as past."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review")),
            calling(call("move_event", subject="review", starts_at="2026-09-26T16:00:00")),
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        reply = room.say("actually move it to four")
        assert [event["starts_at"] for event in room.calendar()] == ["2026-10-01T16:00"]
        assert "Moved review" in reply.reply

    def test_a_move_naming_a_day_goes_to_that_day(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="review")),
            calling(call("move_event", subject="review", starts_at="2026-10-02T16:00:00")),
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        room.say("move the review to Friday at four")
        assert [event["starts_at"] for event in room.calendar()] == ["2026-10-02T16:00"]

    def test_it_is_the_entry_last_put_in(self, config):
        """Live: 'move it to three' moved the design review from three turns back."""
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="design review")),
            calling(call("schedule_event", starts_at="2026-10-01T14:00:00", subject="budget meeting")),
            calling(call("move_event", subject="design review", starts_at="2026-10-01T15:00:00")),
        )
        room = Room(config, chat)
        room.say("put the design review on Thursday at 3")
        room.say("put the budget meeting on Thursday at 2")
        room.say("actually move it to three")
        moved = {event["subject"]: event["starts_at"] for event in room.calendar()}
        assert moved == {"design review": "2026-10-01T15:00", "budget meeting": "2026-10-01T15:00"}

    def test_a_subject_named_in_the_words_is_left_alone(self, config):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T15:00:00", subject="design review")),
            calling(call("schedule_event", starts_at="2026-10-01T14:00:00", subject="budget meeting")),
            calling(call("move_event", subject="design review", starts_at="2026-10-01T17:00:00")),
        )
        room = Room(config, chat)
        room.say("put the design review on Thursday at 3")
        room.say("put the budget meeting on Thursday at 2")
        room.say("move the design review to five, it clashes")
        moved = {event["subject"]: event["starts_at"] for event in room.calendar()}
        assert moved["design review"] == "2026-10-01T17:00" and moved["budget meeting"] == "2026-10-01T14:00"

    def test_the_model_is_never_shown_keep_day(self, config):
        chat = ScriptedChat(saying("ok"))
        room = Room(config, chat)
        room.say("what is on Thursday")
        (move,) = [t for t in chat.requests[0][1] if t["function"]["name"] == "move_event"]
        assert "keep_day" not in move["function"]["parameters"]["properties"]


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


class TestMail:
    def test_a_mail_is_put_to_the_occupant_as_a_mock(self, config):
        """FR-54, FR-55: nothing is sent, and the question says so."""
        chat = ScriptedChat(calling(call("send_mail", to="Anu", subject="Running late", body="Ten minutes late, sorry.")))
        reply = Room(config, chat).say("email Anu that I'm running ten minutes late")
        assert "mock email to Anu" in reply.reply and "nothing is actually sent" in reply.reply
        assert reply.awaiting_confirmation


class TestOverheardToolsAreNotTheAssistants:
    """FR-77, FR-78: pencilling and settling belong to overheard speech and the console."""

    def test_they_are_never_offered(self, config):
        chat = ScriptedChat(saying("ok"))
        room = Room(config, chat)
        room.say("what is on Thursday")
        offered = {tool["function"]["name"] for tool in chat.requests[0][1]}
        assert offered.isdisjoint({"pencil_event", "keep_event", "suggest_fact", "settle_fact"})

    def test_one_named_anyway_is_refused_before_the_executor(self, config):
        chat = ScriptedChat(
            calling(call("pencil_event", starts_at="2026-10-01T15:00:00", subject="review")),
            saying("I couldn't."),
        )
        room = Room(config, chat)
        room.say("put the review on Thursday at 3")
        assert room.calendar() == []
        assert not [i for i in room.published(topics.ASSIST_PROPOSED, ToolInvocation) if i.tool == "pencil_event"]


class TestFollowUpsTheWordsDecide:
    """S-03 and S-04, as met live through the console."""

    def _room_with_dentist(self, config, *then):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-01T14:00:00", subject="dentist")),
            *then,
        )
        room = Room(config, chat)
        room.say("put the dentist in my calendar on Thursday at two")
        return room

    def test_a_claimed_move_with_no_tool_is_carried_out(self, config):
        """Live: "Sure, I'll update the plumber appointment to 11:30" -- and nothing moved."""
        room = self._room_with_dentist(config, saying("Sure, I'll update the dentist to 14:30."))
        reply = room.say("actually make it half past two")
        assert [e["starts_at"] for e in room.calendar()] == ["2026-10-01T14:30"]
        assert "Moved dentist" in reply.reply

    def test_a_move_to_the_wrong_time_is_put_right(self, config):
        room = self._room_with_dentist(
            config, calling(call("move_event", subject="dentist", starts_at="2026-10-01T02:30:00"))
        )
        room.say("actually make it half past two")
        assert [e["starts_at"] for e in room.calendar()] == ["2026-10-01T14:30"]

    def test_a_new_entry_for_a_follow_up_becomes_the_move(self, config):
        room = self._room_with_dentist(
            config, calling(call("schedule_event", subject="dentist", starts_at="2026-10-01T14:30:00"))
        )
        room.say("actually make it half past two")
        assert [e["starts_at"] for e in room.calendar()] == ["2026-10-01T14:30"]

    def test_cancel_it_cancels_the_entry_just_discussed(self, config):
        """Live: "cancel it" was answered "which entry would you like to cancel?"."""
        room = self._room_with_dentist(config, saying("Sure, which entry would you like to cancel?"))
        reply = room.say("cancel it")
        assert room.calendar() == [] and "Cancelled dentist" in reply.reply

    def test_a_follow_up_with_nothing_discussed_is_left_to_the_model(self, config):
        chat = ScriptedChat(saying("Which entry do you mean?"))
        room = Room(config, chat)
        assert room.say("actually make it half past two").reply == "Which entry do you mean?"

    def test_another_voices_entry_is_not_theirs_to_move(self, config):
        room = self._room_with_dentist(config, saying("Which entry do you mean?"))
        reply = room.say("actually make it half past two", profile_id="ravi")
        assert [e["starts_at"] for e in room.calendar()] == ["2026-10-01T14:00"]
        assert reply.reply == "Which entry do you mean?"


class TestNeverFineDuringAFault:
    """S-13, live: "Everything seems fine, just a note that the temperature sensor isn't working"."""

    def test_a_faulted_room_is_not_called_fine(self, config):
        from src.common.schemas import Mode

        room = Room(config, ScriptedChat(saying("Everything seems fine, just a note that the sensor is off.")))
        TestRepliesTellTheTruthAboutTheRoom()._mode(room, Mode.DEGRADED_SENSOR, ("f_temp01_stuck_1",))
        reply = room.say("is everything ok?", intent=Intent.QUESTION, subject="room")
        assert "fine" not in reply.reply.lower()
        assert reply.reply == "Not quite. The temperature sensor is not working, so the room is being controlled on an estimate."

    def test_a_healthy_room_may_be_called_fine(self, config):
        from src.common.schemas import Mode

        room = Room(config, ScriptedChat(saying("Everything is fine.")))
        TestRepliesTellTheTruthAboutTheRoom()._mode(room, Mode.NORMAL)
        assert room.say("is everything ok?", intent=Intent.QUESTION, subject="room").reply == "Everything is fine."


class TestWhichNextWeek:
    """S-11, live: 'next Monday' said on a Sunday became tomorrow, unremarked."""

    def test_next_weekday_within_two_days_says_so(self, config):
        chat = ScriptedChat(calling(call("book_travel", kind="flight", destination="Delhi", depart_on="2026-09-27T09:00:00")))
        room = Room(config, chat)
        reply = room.say("book me a flight to Delhi next Sunday")
        assert "that's tomorrow" in reply.reply and "week after" in reply.reply

    def test_a_weekday_without_next_says_nothing_more(self, config):
        chat = ScriptedChat(calling(call("book_travel", kind="flight", destination="Delhi", depart_on="2026-09-27T09:00:00")))
        room = Room(config, chat)
        assert "week after" not in room.say("book me a flight to Delhi on Sunday").reply

    def test_next_weekday_further_out_says_nothing_more(self, config):
        chat = ScriptedChat(calling(call("book_travel", kind="flight", destination="Delhi", depart_on="2026-10-01T09:00:00")))
        room = Room(config, chat)
        assert "week after" not in room.say("book me a flight to Delhi next Thursday").reply


class TestTwoEntriesNamedAlike:
    """S-03 re-verified live: an earlier 'Plumber appointment' and a new 'plumber'
    at the same time made 'actually make it half past eleven' and 'cancel it'
    both answer 'which one?'."""

    def _two_plumbers(self, config, *then):
        chat = ScriptedChat(
            calling(call("schedule_event", starts_at="2026-10-02T11:00:00", subject="Plumber appointment")),
            calling(call("schedule_event", starts_at="2026-10-02T11:00:00", subject="plumber")),
            *then,
        )
        room = Room(config, chat)
        room.say("put the plumber appointment in on Friday at eleven", profile_id="anu")
        room.say("put the plumber in my calendar on Friday at eleven")
        return room

    def test_the_follow_up_moves_the_one_just_put_in(self, config):
        room = self._two_plumbers(config, saying("Which one?"))
        room.say("actually make it half past eleven")
        times = {event["subject"]: event["starts_at"] for event in room.calendar()}
        assert times == {"Plumber appointment": "2026-10-02T11:00", "plumber": "2026-10-02T11:30"}

    def test_cancel_it_cancels_the_one_just_put_in(self, config):
        room = self._two_plumbers(config, saying("Which one?"))
        room.say("cancel it")
        assert [event["subject"] for event in room.calendar()] == ["Plumber appointment"]

    def test_the_model_is_never_shown_the_entry_id(self, config):
        chat = ScriptedChat(saying("ok"))
        Room(config, chat).say("what is on Thursday")
        tools = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in chat.requests[0][1]}
        assert "event_id" not in tools["move_event"] and "event_id" not in tools["cancel_event"]


class TestRepliesSayHowTheyWereAsked:
    """FR-79: a reply to a spoken request is marked to be said aloud."""

    def test_a_spoken_request_gets_a_spoken_reply(self, config):
        room = Room(config, ScriptedChat(saying("Nothing on Thursday.")))
        assert room.say("what is on Thursday", intent=Intent.QUESTION).spoken is True

    def test_a_typed_request_does_not(self, config):
        room = Room(config, ScriptedChat(saying("Nothing on Thursday.")))
        assert room.say("what is on Thursday", intent=Intent.QUESTION, typed=True).spoken is False

    def test_a_remote_request_gets_a_remote_reply(self, config):
        """FR-94: answered on the device that asked, not in the room."""
        room = Room(config, ScriptedChat(saying("Nothing on Thursday.")))
        reply = room.say("what is on Thursday", intent=Intent.QUESTION, typed=True, remote=True)
        assert reply.remote is True and reply.spoken is False


class TestTheModelsOwnMistake:
    """A call refused for its own arguments is retried, never read out (FR-44)."""

    WRONG = call("get_events", from_time="2026-10-01T00:00:00", to_time="2026-10-01T23:59:00", room="bathroom")
    RIGHT = call("get_events", from_time="2026-10-01T00:00:00", to_time="2026-10-01T23:59:00")

    def test_it_is_told_why_and_asked_again(self, config):
        chat = ScriptedChat(calling(self.WRONG), calling(self.RIGHT), saying("You have nothing on Thursday."))
        reply = Room(config, chat).say("what is on Thursday")
        assert reply.reply == "You have nothing on Thursday."
        told = chat.requests[1][0][-1]
        assert told["role"] == "tool" and "does not take ['room']" in told["content"]

    def test_the_refusal_is_never_said(self, config):
        chat = ScriptedChat(calling(self.WRONG), calling(self.WRONG))
        reply = Room(config, chat).say("what is on Thursday")
        assert "does not take" not in reply.reply and "say it another way" in reply.reply

    def test_it_is_asked_again_only_so_often(self, config):
        chat = ScriptedChat(calling(self.WRONG), calling(self.WRONG), calling(self.RIGHT))
        Room(config, chat).say("what is on Thursday")
        assert len(chat.requests) == 1 + config.assistance.argument_retries

    def test_a_refusal_about_the_request_is_still_said(self, config):
        chat = ScriptedChat(calling(call("cancel_event", subject="the plumber")))
        reply = Room(config, chat).say("cancel the plumber")
        assert "say it another way" not in reply.reply and len(chat.requests) == 1

    def test_switched_off_it_is_not_asked_again(self, config):
        quiet = config.model_copy(update={"assistance": config.assistance.model_copy(update={"argument_retries": 0})})
        chat = ScriptedChat(calling(self.WRONG), calling(self.RIGHT))
        reply = Room(quiet, chat).say("what is on Thursday")
        assert len(chat.requests) == 1 and "does not take" not in reply.reply


# --- The world outside (FR-101 to FR-103) ----------------------------------


class _Search:
    name = "searxng"

    def __init__(self, hits=None):
        from src.assistance.providers.web_search import Hit

        self.hits = hits if hits is not None else [
            Hit(title="RCB win the IPL final", url="https://www.cricbuzz.com/1", source="cricbuzz.com", snippet="By five wickets.", published_ts=SATURDAY_AFTERNOON - 3 * 3600)
        ]

    def search(self, query, *, news):
        self.asked = [*getattr(self, "asked", []), query]
        return list(self.hits)


class _Scores:
    name = "cricinfo"

    def scores(self):
        from src.assistance.providers.cricket import Match, Scores, Side

        india = Match(home=Side("India", "212/4", True), away=Side("Australia", "260/10", False), url="https://cricinfo.com/1")
        return Scores(matches=(india,), as_of=SATURDAY_AFTERNOON)


def _fetch_page(url, timeout_s, max_bytes, stop):
    from src.assistance.providers.pages import Page

    return Page(
        url=url,
        content_type="text/html",
        text="<html><title>Final report</title><article><p>RCB beat Gujarat Titans by five wickets in the final at Ahmedabad on Sunday night.</p>"
        "<p>Virat Kohli made 75 from 42 balls and was named player of the match.</p>"
        "<p>It was the franchise's first title, and the crowd stayed long after the presentation to celebrate.</p></article></html>",
    )


class WorldRoom(Room):
    """A room whose executor searches a fake web and reads a fake score feed."""

    def __init__(self, config, chat, search=None, fetch_page=None) -> None:
        from src.assistance.__main__ import WorldParts

        world = WorldParts(search=search or _Search(), scores=_Scores(), fetch_page=fetch_page or _fetch_page)
        super().__init__(config, chat, world=world)
        self.executor.publish_catalogue()


def _system(chat, request):
    return chat.requests[request][0][0]["content"]


class TestTheWorldOutside:
    def test_the_search_tools_are_offered_to_the_model(self, config):
        chat = ScriptedChat(saying("Hello."))
        WorldRoom(config, chat).say("hello", intent=Intent.QUESTION, subject="")
        offered = {tool["function"]["name"] for tool in chat.requests[0][1]}
        assert {"web_search", "news_search", "cricket_score"} <= offered

    def test_the_model_is_told_never_to_answer_the_world_from_memory(self, config):
        chat = ScriptedChat(saying("Hello."))
        WorldRoom(config, chat).say("who won the IPL", intent=Intent.QUESTION, subject="")
        assert "never answer it from memory" in _system(chat, 0)

    def test_results_are_phrased_by_the_model_not_read_out(self, config):
        chat = ScriptedChat(
            calling(call("web_search", query="who won the IPL")),
            saying("According to Cricbuzz, three hours ago, RCB won the final by five wickets."),
        )
        reply = WorldRoom(config, chat).say("who won the IPL", intent=Intent.QUESTION, subject="")
        assert reply.reply == "According to Cricbuzz, three hours ago, RCB won the final by five wickets."

    def test_the_phrasing_asks_for_the_source_and_the_age_and_no_memory(self, config):
        chat = ScriptedChat(calling(call("web_search", query="who won the IPL")), saying("RCB."))
        WorldRoom(config, chat).say("who won the IPL", intent=Intent.QUESTION, subject="")
        phrasing = _system(chat, 1)
        assert "where" in phrasing and "how old" in phrasing and "memory" in phrasing

    def test_the_results_reach_the_phrasing_with_their_source(self, config):
        chat = ScriptedChat(calling(call("web_search", query="who won the IPL")), saying("RCB."))
        WorldRoom(config, chat).say("who won the IPL", intent=Intent.QUESTION, subject="")
        assert "cricbuzz.com, 3 hours ago" in chat.requests[1][0][1]["content"]

    def test_a_live_score_is_phrased_too(self, config):
        chat = ScriptedChat(calling(call("cricket_score", team="India")), saying("India are 212 for 4, chasing 260."))
        reply = WorldRoom(config, chat).say("what's the India score", intent=Intent.QUESTION, subject="")
        assert reply.reply == "India are 212 for 4, chasing 260."

    def test_without_the_model_to_phrase_it_the_top_result_is_said_without_links(self, config):
        chat = ScriptedChat(calling(call("web_search", query="who won the IPL")), UNAVAILABLE)
        reply = WorldRoom(config, chat).say("who won the IPL", intent=Intent.QUESTION, subject="")
        assert "RCB win the IPL final" in reply.reply and "cricbuzz.com" in reply.reply
        assert "https://" not in reply.reply

    def test_nothing_found_is_said_as_the_search_said_it(self, config):
        chat = ScriptedChat(calling(call("web_search", query="zzqx")), saying("unused"))
        reply = WorldRoom(config, chat, search=_Search(hits=[])).say("search for zzqx", intent=Intent.QUESTION, subject="")
        assert "found nothing" in reply.reply


class TestDeepAnswers:
    """FR-104: snippets first; the pages, and a second search, only when they do not answer."""

    def _invocations(self, room):
        return [(invocation.tool, dict(invocation.arguments)) for invocation in room.published(topics.ASSIST_PROPOSED, ToolInvocation)]

    def test_a_question_the_snippets_answer_reads_no_page(self, config):
        chat = ScriptedChat(calling(call("web_search", query="who won the IPL")), saying("According to Cricbuzz, RCB won."))
        room = WorldRoom(config, chat)
        room.say("who won the IPL", intent=Intent.QUESTION, subject="")
        assert [tool for tool, _ in self._invocations(room)] == ["web_search"]

    def test_the_model_may_ask_for_more_when_snippets_could_be_deepened(self, config):
        chat = ScriptedChat(calling(call("web_search", query="who won the IPL")), saying("RCB."))
        WorldRoom(config, chat).say("who won the IPL", intent=Intent.QUESTION, subject="")
        assert "MORE:" in _system(chat, 1)

    def test_asked_for_more_it_searches_again_in_the_better_words(self, config):
        search = _Search()
        chat = ScriptedChat(
            calling(call("web_search", query="IPL")),
            saying("MORE: IPL 2026 final result man of the match"),
            saying("According to the final report, Kohli was player of the match."),
        )
        room = WorldRoom(config, chat, search=search)
        room.say("who was player of the match in the IPL final", intent=Intent.QUESTION, subject="")
        assert search.asked == ["IPL", "IPL 2026 final result man of the match"]

    def test_then_the_top_pages_are_read_with_the_question(self, config):
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("MORE: IPL final"), saying("Kohli."))
        room = WorldRoom(config, chat)
        room.say("who was player of the match in the IPL final", intent=Intent.QUESTION, subject="")
        tool, arguments = self._invocations(room)[-1]
        assert tool == "read_pages" and arguments["urls"] == "https://www.cricbuzz.com/1"
        assert arguments["question"] == "who was player of the match in the IPL final"

    def test_the_answer_is_written_from_the_pages(self, config):
        chat = ScriptedChat(
            calling(call("web_search", query="IPL")),
            saying("MORE:"),
            saying("According to the final report, Virat Kohli was player of the match."),
        )
        room = WorldRoom(config, chat)
        reply = room.say("who was player of the match in the IPL final", intent=Intent.QUESTION, subject="")
        assert reply.reply == "According to the final report, Virat Kohli was player of the match."
        assert "named player of the match" in chat.requests[2][0][1]["content"]

    def test_the_final_answer_may_not_ask_for_more_again(self, config):
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("MORE:"), saying("RCB."))
        WorldRoom(config, chat).say("who won", intent=Intent.QUESTION, subject="")
        assert "MORE:" not in _system(chat, 2)

    def test_asking_for_more_a_second_time_is_answered_as_not_found(self, config):
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("MORE:"), saying("MORE: something else"))
        reply = WorldRoom(config, chat).say("who won", intent=Intent.QUESTION, subject="")
        assert "couldn't find" in reply.reply and "MORE" not in reply.reply

    def test_more_after_a_sentence_is_still_asking_for_more(self, config):
        search = _Search()
        chat = ScriptedChat(
            calling(call("web_search", query="IPL")),
            saying("The results don't say.\nMORE: IPL 2026 final player of the match"),
            saying("Kohli was player of the match."),
        )
        reply = WorldRoom(config, chat, search=search).say("who was player of the match", intent=Intent.QUESTION, subject="")
        assert search.asked == ["IPL", "IPL 2026 final player of the match"]
        assert reply.reply == "Kohli was player of the match."

    def test_an_answer_that_begins_more_in_plain_words_is_said(self, config):
        chat = ScriptedChat(calling(call("web_search", query="Kerala rain")), saying("More: rain is due in Kerala tomorrow, says the IMD."))
        room = WorldRoom(config, chat)
        reply = room.say("will it rain in Kerala", intent=Intent.QUESTION, subject="")
        assert reply.reply == "More: rain is due in Kerala tomorrow, says the IMD."
        assert [tool for tool, _ in self._invocations(room)] == ["web_search"]

    def test_better_words_are_cut_to_a_search_length(self, config):
        search = _Search()
        words = " ".join(["cricket"] * 80)
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying(f"MORE: {words}"), saying("RCB."))
        WorldRoom(config, chat, search=search).say("who won", intent=Intent.QUESTION, subject="")
        assert len(search.asked[-1]) <= config.web_search.deep.max_query_chars and search.asked[-1].startswith("cricket")

    def test_not_finding_it_after_the_pages_is_recorded(self, config):
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("MORE:"), saying("MORE: something else"))
        room = WorldRoom(config, chat)
        room.say("who won", intent=Intent.QUESTION, subject="")
        assert "still not found after the pages" in room.published(topics.AUDIT_REASONING, ReasoningRecord)[-1].verdict

    def test_more_without_better_words_reads_the_pages_without_searching_again(self, config):
        search = _Search()
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("MORE:"), saying("RCB."))
        room = WorldRoom(config, chat, search=search)
        room.say("who won", intent=Intent.QUESTION, subject="")
        assert search.asked == ["IPL"] and self._invocations(room)[-1][0] == "read_pages"

    def test_the_same_words_again_are_not_searched_twice(self, config):
        search = _Search()
        chat = ScriptedChat(calling(call("web_search", query="IPL final")), saying("MORE: ipl final"), saying("RCB."))
        WorldRoom(config, chat, search=search).say("who won", intent=Intent.QUESTION, subject="")
        assert search.asked == ["IPL final"]

    def test_with_a_second_search_turned_off_only_the_pages_are_read(self, config):
        deep = config.web_search.deep.model_copy(update={"search_again": False})
        config = config.model_copy(update={"web_search": config.web_search.model_copy(update={"deep": deep})})
        search = _Search()
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("MORE: IPL final"), saying("RCB."))
        WorldRoom(config, chat, search=search).say("who won", intent=Intent.QUESTION, subject="")
        assert search.asked == ["IPL"]

    def test_a_deep_answer_is_recorded_with_why(self, config):
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("MORE: IPL final"), saying("RCB."))
        room = WorldRoom(config, chat)
        room.say("who won", intent=Intent.QUESTION, subject="")
        assert "deep answer" in room.published(topics.AUDIT_REASONING, ReasoningRecord)[-1].verdict

    def test_headlines_are_never_deepened(self, config):
        chat = ScriptedChat(calling(call("news_search", when="today")), saying("Here is the news."))
        WorldRoom(config, chat).say("what's the news today", intent=Intent.QUESTION, subject="")
        assert "MORE:" not in _system(chat, 1)

    def test_news_on_a_topic_may_be_deepened(self, config):
        chat = ScriptedChat(calling(call("news_search", topic="ISRO", when="this_week")), saying("ISRO launched."))
        WorldRoom(config, chat).say("what did ISRO do this week", intent=Intent.QUESTION, subject="")
        assert "MORE:" in _system(chat, 1)

    def test_scores_are_never_deepened(self, config):
        chat = ScriptedChat(calling(call("cricket_score", team="India")), saying("India are 212 for 4."))
        WorldRoom(config, chat).say("what's the India score", intent=Intent.QUESTION, subject="")
        assert "MORE:" not in _system(chat, 1)

    def test_with_deep_answers_off_the_model_is_not_offered_more(self, config):
        deep = config.web_search.deep.model_copy(update={"enabled": False})
        config = config.model_copy(update={"web_search": config.web_search.model_copy(update={"deep": deep})})
        chat = ScriptedChat(calling(call("web_search", query="IPL")), saying("RCB."))
        WorldRoom(config, chat).say("who won", intent=Intent.QUESTION, subject="")
        assert "MORE:" not in _system(chat, 1)
