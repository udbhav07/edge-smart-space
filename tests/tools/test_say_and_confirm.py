"""The two terminal tools that stand in for a microphone and a console.

Run on the in-process bus against the real executor, so confirming a booking
is shown to reach the mock endpoint and declining it is shown to reach
nothing.
"""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service as build_executor
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Comfort,
    Intent,
    PreferenceHint,
    ReasoningOutcome,
    ReasoningRecord,
    CallSite,
    Utterance,
    UtteranceSource,
)
from src.common.tools import ToolInvocation, ToolRequester, ToolResult, ToolStatus
from tools.blackboard_view import BlackboardView, describe
from tools.confirm import ConfirmationDesk, describe as describe_booking
from tools.say import Conversation

FLIGHT = {
    "kind": "flight",
    "destination": "Delhi",
    "origin": "Hyderabad",
    "depart_on": "2025-08-29T09:00:00",
}


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(
        update={"calendar_path": str(tmp_path / "calendar.json")}
    )
    return config.model_copy(update={"assistance": assistance})


class Bus:
    def __init__(self, config) -> None:
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        self.boards = []

    def board(self, config) -> Blackboard:
        board = Blackboard(config.mqtt, self.transport)
        self.boards.append(board)
        return board

    def attach_all(self) -> None:
        for board in self.boards:
            self.transport.attach(board)

    def results(self) -> list[ToolResult]:
        return [
            ToolResult.model_validate_json(payload)
            for name, payload, _, _ in self.transport.published
            if name == "space/assist/result"
        ]

    def published_on(self, topic: str) -> int:
        return sum(1 for name, _, _, _ in self.transport.published if name == topic)


def _ask_for_a_flight(bus, board, clock):
    invocation = ToolInvocation(
        ts=clock.now(),
        invocation_id="inv_flight_1",
        tool="book_travel",
        arguments=FLIGHT,
        requester=ToolRequester.PERSONAL_CONTEXT,
        rationale="book me a flight to Delhi on Friday",
        expires_ts=clock.now() + 300.0,
    )
    board.publish(topics.ASSIST_PROPOSED, invocation)
    return invocation


class TestConfirmationDesk:
    def _wired(self, config):
        bus = Bus(config)
        executor = build_executor(config, bus.clock, bus.board(config))
        executor.subscribe()
        desk = ConfirmationDesk(bus.clock, bus.board(config))
        desk.subscribe()
        asker = bus.board(config)
        bus.attach_all()
        return bus, desk, asker

    def test_a_booking_needing_confirmation_waits_at_the_desk(self, config):
        bus, desk, asker = self._wired(config)
        _ask_for_a_flight(bus, asker, bus.clock)
        assert [i.invocation_id for i in desk.waiting()] == ["inv_flight_1"]

    def test_confirming_reaches_the_mock_and_says_so(self, config):
        """FR-54 then FR-55: only after a yes, and never as a real booking."""
        bus, desk, asker = self._wired(config)
        _ask_for_a_flight(bus, asker, bus.clock)
        desk.confirm("inv_flight_1")
        result = bus.results()[-1]
        assert result.status is ToolStatus.OK and result.simulated

    def test_the_confirmation_is_the_invocation_verbatim(self, config):
        """Agreeing to a booking is agreeing to the one that was described."""
        bus, desk, asker = self._wired(config)
        asked = _ask_for_a_flight(bus, asker, bus.clock)
        assert desk.confirm("inv_flight_1") == asked

    def test_declining_publishes_nothing(self, config):
        bus, desk, asker = self._wired(config)
        _ask_for_a_flight(bus, asker, bus.clock)
        desk.decline("inv_flight_1")
        assert bus.published_on("space/assist/confirmed") == 0
        assert desk.waiting() == []

    def test_a_calendar_write_never_waits_at_the_desk(self, config):
        """Only commit tools ask (FR-74); a calendar entry just happens."""
        bus, desk, asker = self._wired(config)
        asker.publish(
            topics.ASSIST_PROPOSED,
            ToolInvocation(
                ts=bus.clock.now(),
                invocation_id="inv_cal_1",
                tool="schedule_event",
                arguments={"starts_at": "2025-08-28T15:00:00", "subject": "review"},
                requester=ToolRequester.PERSONAL_CONTEXT,
                expires_ts=bus.clock.now() + 300.0,
            ),
        )
        assert desk.waiting() == []

    def test_it_does_not_depend_on_which_message_arrives_first(self, config):
        """Two publishers promise no order between them. Here the desk hears
        the proposal before the refusal, the reverse of the fixture above."""
        bus = Bus(config)
        desk = ConfirmationDesk(bus.clock, bus.board(config))
        desk.subscribe()
        executor = build_executor(config, bus.clock, bus.board(config))
        executor.subscribe()
        asker = bus.board(config)
        bus.attach_all()
        _ask_for_a_flight(bus, asker, bus.clock)
        assert [i.invocation_id for i in desk.waiting()] == ["inv_flight_1"]

    def test_an_unknown_id_cannot_be_confirmed(self, config):
        _, desk, _ = self._wired(config)
        with pytest.raises(KeyError):
            desk.confirm("inv_never")

    def test_a_booking_is_described_in_full_before_anyone_says_yes(self, config):
        bus, desk, asker = self._wired(config)
        _ask_for_a_flight(bus, asker, bus.clock)
        line = describe_booking(desk.waiting()[0])
        assert "Delhi" in line and "Hyderabad" in line


class TestConversation:
    def _wired(self, config):
        bus = Bus(config)
        conversation = Conversation(bus.clock, bus.board(config))
        conversation.subscribe()
        room = bus.board(config)
        bus.attach_all()
        return bus, conversation, room

    def test_the_words_go_out_as_an_utterance(self, config):
        bus, conversation, _ = self._wired(config)
        conversation.say("it is too warm")
        heard = [
            Utterance.model_validate_json(payload)
            for name, payload, _, _ in bus.transport.published
            if name == "space/context/utterance"
        ]
        assert heard[-1].text == "it is too warm"
        assert heard[-1].source is UtteranceSource.OPERATOR

    def test_the_reply_is_collected(self, config):
        bus, conversation, room = self._wired(config)
        conversation.say("it is too warm")
        room.publish(
            topics.CONTEXT_PREFERENCE,
            PreferenceHint(
                ts=bus.clock.now(),
                intent=Intent.ENVIRONMENT,
                comfort=Comfort.COOLER,
                spoken_reply="I have passed that on.",
            ),
        )
        assert conversation.answered.is_set()
        assert "I have passed that on." in conversation.transcript()

    def test_a_mock_result_is_marked_as_one(self, config):
        bus, conversation, room = self._wired(config)
        conversation.say("book it")
        room.publish(
            topics.ASSIST_RESULT,
            ToolResult(
                ts=bus.clock.now(),
                invocation_id="inv_1",
                tool="book_travel",
                status=ToolStatus.OK,
                message="Simulated only.",
                provider="mock_travel",
                simulated=True,
            ),
        )
        assert "[MOCK]" in conversation.transcript()

    def test_silence_says_what_to_check(self, config):
        _, conversation, _ = self._wired(config)
        conversation.say("hello")
        assert "reasoning process" in conversation.transcript()


class TestTheViewKnowsTheNewTopics:
    def test_a_reasoning_record_is_decoded_and_described(self):
        view = BlackboardView(SimClock())
        record = ReasoningRecord(
            ts=1756032000.0,
            invocation_id="rsn_1",
            call_site=CallSite.SUPERVISOR,
            trigger="cadence",
            rounds=2,
            outcome=ReasoningOutcome.APPLIED,
            applied="proposed 25.0 C",
            latency_s=3.1,
            prompt_tokens=900,
            completion_tokens=40,
        )
        entry = view.accept("space/audit/reasoning", record.model_dump_json().encode())
        assert "supervisor APPLIED" in describe("space/audit/reasoning", entry)

    def test_no_declared_topic_is_left_undecodable(self):
        """Every topic this build declares has a schema the view knows."""
        from tools.blackboard_view import _SCHEMAS

        known = {spec.pattern for spec, _ in _SCHEMAS}
        declared = {
            value.pattern
            for value in vars(topics).values()
            if isinstance(value, topics.TopicSpec)
        }
        missing = declared - known - {
            topics.SYSTEM_RESET.pattern,
            topics.INJECT.pattern,
            topics.ACTUATOR_COMMAND.pattern,
        }
        assert missing == set()
