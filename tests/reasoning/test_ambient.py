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
