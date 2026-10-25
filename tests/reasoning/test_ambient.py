"""Unit tests for what is made of overheard speech (FR-76 to FR-78).

The model is scripted and the executor, calendar and profiles are real, so a
pencilled plan is asserted on in the calendar file and a suggestion in the
person's profile, having crossed the blackboard like any invocation.
"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service as build_executor
from src.assistance.__main__ import build_store
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.localtime import epoch_of
from src.common.mqtt_client import Blackboard
from src.common.schemas import AmbientUtterance, ReasoningCaller, ReasoningRecord
from src.common.tools import ToolInvocation, ToolRequester
from src.reasoning.__main__ import build_miner
from tests.reasoning.stubs import UNAVAILABLE, ScriptedChat, saying

#: A Saturday afternoon, local time, so "Thursday" is 1 October.
SATURDAY_AFTERNOON = epoch_of(
    datetime(2026, 9, 26, 15, 0), load_config(Path("config/default.yaml")).site.utc_offset_h
)


def findings(plans=(), facts=()) -> str:
    return json.dumps({"plans": list(plans), "facts": list(facts)})


DENTIST = {"subject": "dentist", "starts_at": "2026-10-01T16:00:00", "duration_min": 30}


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(update={"calendar_path": str(tmp_path / "calendar.json")})
    profiles = config.profiles.model_copy(
        update={"directory": str(tmp_path / "profiles"), "context_directory": str(tmp_path / "context")}
    )
    ambient = config.ambient.model_copy(update={"max_calls_per_min": 3})
    return config.model_copy(update={"assistance": assistance, "profiles": profiles, "ambient": ambient})


class Room:
    def __init__(self, config, *answers) -> None:
        self.clock = SimClock(start_epoch_s=SATURDAY_AFTERNOON)
        self.transport = LoopbackTransport()
        boards = [Blackboard(config.mqtt, self.transport) for _ in range(3)]
        for board in boards:
            self.transport.attach(board)
        self.executor = build_executor(config, self.clock, boards[0])
        self.executor.subscribe()
        self.chat = ScriptedChat(*answers)
        self.miner = build_miner(config, self.clock, boards[1], self.chat)
        self.miner.subscribe()
        self.speech = boards[2]
        self.store = build_store(config, self.clock)
        self.store.create("ravi", "Ravi")
        self.calendar_path = Path(config.assistance.calendar_path)

    def overhear(self, words: str, profile_id: str = "ravi") -> list[ToolInvocation]:
        self.speech.publish(
            topics.CONTEXT_AMBIENT,
            AmbientUtterance(ts=self.clock.now(), transcript=words, profile_id=profile_id),
        )
        return self.miner.process_pending()

    def calendar(self) -> list[dict]:
        if not self.calendar_path.exists():
            return []
        return json.loads(self.calendar_path.read_text())["events"]

    def records(self) -> list[ReasoningRecord]:
        return [
            ReasoningRecord.model_validate_json(payload)
            for topic, payload, _, _ in self.transport.published
            if topic == topics.AUDIT_REASONING.pattern and payload
        ]


class TestPlans:
    """FR-77: a plan overheard is pencilled in as tentative, never booked."""

    def test_a_plan_is_pencilled_in(self, config):
        room = Room(config, saying(findings(plans=[DENTIST])))
        room.overhear("I've got the dentist on Thursday at four")
        (event,) = room.calendar()
        assert event["subject"] == "dentist" and event["tentative"] is True

    def test_what_was_heard_goes_with_it(self, config):
        room = Room(config, saying(findings(plans=[DENTIST])))
        room.overhear("I've got the dentist on Thursday at four")
        assert room.calendar()[0]["heard"] == "I've got the dentist on Thursday at four"

    def test_it_is_proposed_as_overheard(self, config):
        room = Room(config, saying(findings(plans=[DENTIST])))
        (invocation,) = room.overhear("I've got the dentist on Thursday at four")
        assert invocation.tool == "pencil_event" and invocation.requester is ToolRequester.AMBIENT

    def test_a_trip_is_pencilled_not_booked(self, config):
        trip = {"subject": "flight to Goa", "starts_at": "2026-10-02T09:00:00"}
        room = Room(config, saying(findings(plans=[trip])))
        proposed = room.overhear("we're flying to Goa on Friday")
        assert [i.tool for i in proposed] == ["pencil_event"]
        assert room.calendar()[0]["subject"] == "flight to Goa"

    def test_the_day_named_wins_over_the_models(self, config):
        """FR-44: the model put Thursday's dentist on Wednesday."""
        wrong = {**DENTIST, "starts_at": "2026-09-30T16:00:00"}
        room = Room(config, saying(findings(plans=[wrong])))
        room.overhear("I've got the dentist on Thursday at four")
        assert room.calendar()[0]["starts_at"] == "2026-10-01T16:00"

    def test_a_plan_already_past_is_dropped(self, config):
        past = {**DENTIST, "starts_at": "2026-09-25T16:00:00"}
        room = Room(config, saying(findings(plans=[past])))
        assert room.overhear("the dentist was awful yesterday") == []
        assert room.calendar() == []

    def test_a_guests_plan_is_still_pencilled(self, config):
        """The calendar is the room's; only memory is per person."""
        room = Room(config, saying(findings(plans=[DENTIST])))
        room.overhear("dentist on Thursday at four", profile_id="")
        assert len(room.calendar()) == 1

    def test_the_same_plan_said_twice_is_one_entry(self, config):
        room = Room(config, saying(findings(plans=[DENTIST])), saying(findings(plans=[DENTIST])))
        room.overhear("I've got the dentist on Thursday at four")
        room.overhear("don't forget, dentist Thursday at four")
        assert len(room.calendar()) == 1


class TestFacts:
    """FR-78: a fact overheard is offered to the person, not remembered."""

    def test_a_fact_is_offered_not_kept(self, config):
        room = Room(config, saying(findings(facts=["Hates the room below 23."])))
        room.overhear("I really hate it when the room is below 23")
        profile = room.store.load("ravi")
        assert profile.suggestions == ("Hates the room below 23.",) and profile.facts == ()

    def test_nothing_is_offered_for_a_guest(self, config):
        room = Room(config, saying(findings(facts=["Hates the cold."])))
        assert room.overhear("I hate the cold, honestly", profile_id="") == []

    def test_an_overlong_fact_is_not_offered(self, config):
        room = Room(config, saying(findings(facts=["x" * 500])))
        assert room.overhear("something rambling and long about everything") == []


class TestNothingToNote:
    def test_small_talk_proposes_nothing(self, config):
        room = Room(config, saying(findings()))
        assert room.overhear("nice weather we are having today") == []
        assert room.calendar() == []

    def test_malformed_output_proposes_nothing(self, config):
        room = Room(config, saying("Sure! Here's what I found: the dentist."))
        assert room.overhear("I've got the dentist on Thursday at four") == []

    def test_and_is_recorded_as_discarded(self, config):
        room = Room(config, saying("not json"))
        room.overhear("I've got the dentist on Thursday at four")
        assert room.records()[-1].verdict == "discarded: failed validation"

    def test_too_many_plans_from_one_line_is_malformed(self, config):
        many = [{**DENTIST, "subject": f"thing {n}"} for n in range(5)]
        room = Room(config, saying(findings(plans=many)))
        assert room.overhear("busy week ahead with lots going on") == []

    def test_an_unreachable_model_costs_only_this_line(self, config):
        room = Room(config, UNAVAILABLE)
        assert room.overhear("I've got the dentist on Thursday at four") == []
        assert room.records()[-1].verdict == "reasoning unavailable"


class TestContext:
    """A plan said across two sentences is read as one."""

    def test_the_earlier_line_is_shown_with_the_next(self, config):
        room = Room(config, saying(findings()), saying(findings(plans=[DENTIST])))
        room.overhear("we really should get to the dentist")
        room.overhear("Thursday at four works for me")
        user_turn = room.chat.requests[1][0][1]["content"]
        assert "we really should get to the dentist" in user_turn and "Thursday at four" in user_turn

    def test_the_day_named_earlier_still_counts(self, config):
        wrong = {**DENTIST, "starts_at": "2026-09-30T16:00:00"}
        room = Room(config, saying(findings()), saying(findings(plans=[wrong])))
        room.overhear("the dentist is on Thursday")
        room.overhear("at four, I think, so book the afternoon off")
        assert room.calendar()[0]["starts_at"] == "2026-10-01T16:00"

    def test_an_old_line_falls_out_of_the_context(self, config):
        room = Room(config, saying(findings()), saying(findings()))
        room.overhear("we really should get to the dentist")
        room.clock.advance(config.ambient.context_s + 1.0)
        room.overhear("Thursday at four works for me")
        assert "dentist" not in room.chat.requests[1][0][1]["content"]

    def test_one_voices_lines_are_not_anothers_context(self, config):
        room = Room(config, saying(findings()), saying(findings()))
        room.overhear("we really should get to the dentist", profile_id="ravi")
        room.overhear("Thursday at four works for me", profile_id="")
        assert "dentist" not in room.chat.requests[1][0][1]["content"]

    def test_the_model_is_told_the_day_named(self, config):
        room = Room(config, saying(findings()))
        room.overhear("dentist on Thursday at four")
        assert "Thursday = 2026-10-01" in room.chat.requests[0][0][1]["content"]


class TestBounds:
    def test_calls_are_rate_limited(self, config):
        room = Room(config, *[saying(findings()) for _ in range(5)])
        for n in range(5):
            room.overhear(f"line number {n} of a long chat")
        assert len(room.chat.requests) == config.ambient.max_calls_per_min

    def test_the_limit_is_per_minute(self, config):
        room = Room(config, *[saying(findings()) for _ in range(5)])
        for n in range(4):
            room.overhear(f"line number {n} of a long chat")
        room.clock.advance(61.0)
        room.overhear("and one more after a minute")
        assert len(room.chat.requests) == config.ambient.max_calls_per_min + 1

    def test_every_call_is_audited_as_ambient(self, config):
        room = Room(config, saying(findings(plans=[DENTIST])))
        room.overhear("I've got the dentist on Thursday at four")
        record = room.records()[-1]
        assert record.caller is ReasoningCaller.AMBIENT and record.tool_calls == ("pencil_event",)

    def test_nothing_overheard_is_ever_answered(self, config):
        room = Room(config, saying(findings(plans=[DENTIST], facts=["Likes tea."])))
        room.overhear("I've got the dentist on Thursday at four and I like tea")
        assert not [t for t, *_ in room.transport.published if t == topics.CONTEXT_REPLY.pattern]


class TestItemByItem:
    """FR-44, as met live: a stray item does not cost the real one."""

    def test_an_empty_plan_beside_a_real_one_is_dropped_alone(self, config):
        """qwen2.5:7b answered the dentist and an empty {} beside it."""
        room = Room(config, saying(json.dumps({"plans": [DENTIST, {}], "facts": []})))
        room.overhear("I've got the dentist on Thursday at four")
        assert [event["subject"] for event in room.calendar()] == ["dentist"]

    def test_a_fact_that_is_not_words_is_dropped_alone(self, config):
        room = Room(config, saying(json.dumps({"plans": [], "facts": [{}, "Likes tea.", ""]})))
        room.overhear("I always have tea in the afternoon")
        assert room.store.load("ravi").suggestions == ("Likes tea.",)

    def test_missing_lists_are_empty_lists(self, config):
        room = Room(config, saying(json.dumps({})))
        assert room.overhear("nothing much going on today really") == []
        assert room.records()[-1].verdict == "accepted: nothing to note"

    def test_a_plan_echoed_under_another_name_the_same_day_is_dropped(self, config):
        """Live: "Dentist" at four, then "appointment" at 09:00 the same Thursday."""
        echo = {"subject": "appointment", "starts_at": "2026-10-01T09:00:00"}
        room = Room(config, saying(findings(plans=[DENTIST, echo])))
        room.overhear("I've got the dentist on Thursday at four")
        assert [event["subject"] for event in room.calendar()] == ["dentist"]

    def test_plans_on_different_days_are_both_kept(self, config):
        friday = {"subject": "flight to Goa", "starts_at": "2026-10-02T09:00:00"}
        room = Room(config, saying(findings(plans=[DENTIST, friday])))
        room.overhear("dentist on Thursday then we fly to Goa the day after")
        assert len(room.calendar()) == 2

    def test_the_prompt_forbids_inferring_a_fact(self):
        from src.reasoning.ambient import AMBIENT_PROMPT

        assert "Never infer a fact" in AMBIENT_PROMPT


class TestEarlierLinesAreNotReportedAgain:
    """Live: the haircut came back with the peanuts, and again with the weather."""

    HAIRCUT = {"subject": "Haircut", "starts_at": "2026-09-30T18:00:00"}
    HAIRCUT_WRONG_DAY = {"subject": "Haircut", "starts_at": "2026-09-26T18:00:00"}

    def test_a_plan_from_an_earlier_line_is_not_pencilled_again(self, config):
        room = Room(
            config,
            saying(findings(plans=[self.HAIRCUT])),
            saying(findings(plans=[{**self.HAIRCUT, "starts_at": "2026-09-27T18:00:00"}], facts=["Is allergic to peanuts."])),
        )
        room.overhear("I have a haircut on Wednesday at six in the evening")
        proposed = room.overhear("by the way I'm allergic to peanuts, so no satay for me")
        assert [i.tool for i in proposed] == ["suggest_fact"]
        assert [event["subject"] for event in room.calendar()] == ["Haircut"]

    def test_a_fact_from_an_earlier_line_is_not_offered_again(self, config):
        room = Room(config, saying(findings(facts=["Is allergic to peanuts."])), saying(findings(facts=["Is allergic to peanuts"])))
        room.overhear("by the way I'm allergic to peanuts, so no satay for me")
        assert room.overhear("the weather has been lovely today hasn't it") == []

    def test_the_same_plan_under_a_longer_name_is_the_same_plan(self, config):
        room = Room(config, saying(findings(plans=[self.HAIRCUT])), saying(findings(plans=[{**self.HAIRCUT, "subject": "haircut appointment"}])))
        room.overhear("I have a haircut on Wednesday at six in the evening")
        assert room.overhear("yes that haircut on Wednesday, at six") == []

    def test_after_the_window_a_plan_may_be_noted_again(self, config):
        later = {**self.HAIRCUT, "starts_at": "2026-10-07T18:00:00"}
        room = Room(config, saying(findings(plans=[self.HAIRCUT])), saying(findings(plans=[later])))
        room.overhear("I have a haircut on Wednesday at six in the evening")
        room.clock.advance(config.ambient.context_s + 1.0)
        assert len(room.overhear("another haircut the Wednesday after, at six")) == 1

    def test_one_voices_notes_do_not_silence_anothers(self, config):
        room = Room(config, saying(findings(plans=[self.HAIRCUT])), saying(findings(plans=[{**self.HAIRCUT, "starts_at": "2026-10-01T18:00:00"}])))
        room.overhear("I have a haircut on Wednesday at six in the evening", profile_id="ravi")
        assert len(room.overhear("I have a haircut on Thursday at six too", profile_id="")) == 1

    def test_the_prompt_says_earlier_lines_were_already_read(self):
        from src.reasoning.ambient import AMBIENT_PROMPT

        assert "never report a plan or fact from them again" in AMBIENT_PROMPT


class TestGrounding:
    """Live: "what a long day" came back as the prompt's own example dentist."""

    def test_a_plan_nobody_mentioned_is_not_pencilled(self, config):
        room = Room(config, saying(findings(plans=[DENTIST])))
        assert room.overhear("what a long day it has been, honestly") == []
        assert room.calendar() == []

    def test_a_fact_nobody_stated_is_not_offered(self, config):
        room = Room(config, saying(findings(facts=["Is vegetarian."])))
        assert room.overhear("what a long day it has been, honestly") == []

    def test_a_plan_named_in_the_earlier_line_is_grounded(self, config):
        service = {"subject": "Car service", "starts_at": "2026-09-29T10:00:00"}
        room = Room(config, saying(findings()), saying(findings(plans=[service])))
        room.overhear("the car really needs its service soon")
        assert len(room.overhear("the garage said Tuesday at ten works")) == 1

    def test_an_inflected_word_still_grounds_it(self):
        from src.reasoning.ambient import grounded

        assert grounded("Flight to Goa", "we're flying to Goa next Friday")
        assert grounded("Is allergic to peanuts.", "I have a peanut allergy")

    def test_unrelated_words_do_not(self):
        from src.reasoning.ambient import grounded

        assert not grounded("Dentist", "what a long day it has been")



class TestAPlanIsADayOrATimeSomebodySaid:
    """S-01: nine calendar entries came from room audio naming no day or time."""

    LIVE_JUNK = [
        "and the pilot me, everyone is okay, I'll grab you a second.",
        "then log, log 1 need to run there.",
        "I'm going to go to the water and see if there's any water in the water.",
        "for 2 power 1. Until the remaining ending you will go to the clock.",
        "So that I'm going to put all of those in a minute.",
        "to be safe. So, good night. Thanks. I'll see you in a minute.",
        "Okay. All right. Okay. We'll get to them. There we go.",
        "Let's grab a few information.",
    ]

    @pytest.mark.parametrize("line", LIVE_JUNK)
    def test_live_junk_pencils_nothing(self, config, line):
        words = [w for w in line.lower().replace(".", " ").replace(",", " ").split() if len(w) > 3]
        plan = {"subject": words[0].capitalize(), "starts_at": "2026-09-26T23:00:00"}
        room = Room(config, saying(findings(plans=[plan])))
        assert room.overhear(line) == []
        assert room.calendar() == []

    @pytest.mark.parametrize("line", [
        "the dentist is on Thursday at four",
        "dinner with Meera at eight tonight",
        "the plumber comes at 10:30 tomorrow",
        "we fly to Goa on Friday",
        "lunch at noon with Anu",
    ])
    def test_a_day_or_a_time_said_is_an_anchor(self, line):
        from src.reasoning.ambient import anchored

        assert anchored(line, datetime(2026, 9, 26, 15, 0))

    @pytest.mark.parametrize("line", LIVE_JUNK)
    def test_live_junk_has_no_anchor(self, line):
        from src.reasoning.ambient import anchored

        assert not anchored(line, datetime(2026, 9, 26, 15, 0))
