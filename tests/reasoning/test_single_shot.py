"""Personal Context, acting through the real executor (FR-42, FR-44, FR-54, FR-55).

The model is scripted; the executor, the registry, the calendar and the
travel mock are the real ones, on the in-process bus, with the calendar in a
temporary file. So "asking for a meeting puts it in the calendar" is asserted
by reading the calendar, not by trusting the model's reply.

The simulated clock stands at Sunday 24 August 2025, 16:10 in the room's
local time, so Thursday is the 28th.
"""

import json
from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service as build_executor
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Intent,
    ReasoningOutcome,
    ReasoningRecord,
)
from src.common.tools import ToolInvocation, ToolResult, ToolStatus
from src.reasoning.audit import ReasoningAudit
from src.reasoning.endpoint import ChatEndpoint
from src.reasoning.single_shot import (
    CONFIRMATION_NOTICE,
    MOCK_NOTICE,
    PersonalContext,
)
from src.reasoning.tool_client import MAX_PENDING, BlackboardToolClient
from tests.reasoning.fakes import ScriptedClient, calls, text

THURSDAY_3PM = "2025-08-28T15:00:00"


def answer(**fields) -> str:
    base = {
        "intent": "service",
        "subject": "calendar",
        "comfort": "unchanged",
        "target_c": None,
        "rationale": "put the design review in on Thursday at three",
        "spoken_reply": "Done.",
    }
    base.update(fields)
    return json.dumps(base)


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(
        update={"calendar_path": str(tmp_path / "calendar.json")}
    )
    return config.model_copy(update={"assistance": assistance})


class Assistant:
    """Personal Context and the real executor, on one bus."""

    def __init__(self, config, replies, with_executor=True) -> None:
        self.clock = SimClock()
        self.transport = LoopbackTransport()
        board = Blackboard(config.mqtt, self.transport)
        self.tools = BlackboardToolClient(config, self.clock, board)
        self.tools.subscribe()
        self.client = ScriptedClient(list(replies))
        self.context = PersonalContext(
            config,
            self.clock,
            ChatEndpoint(config.reasoning, self.clock, self.client),
            ReasoningAudit(self.clock, board, config.reasoning.max_audit_chars),
            tools=self.tools,
        )
        self.transport.attach(board)
        if with_executor:
            executor_board = Blackboard(config.mqtt, self.transport)
            self.executor = build_executor(config, self.clock, executor_board)
            self.executor.subscribe()
            self.transport.attach(executor_board)

    def _decoded(self, topic, schema):
        return [
            schema.model_validate_json(payload)
            for name, payload, _, _ in self.transport.published
            if name == topic and payload
        ]

    def invocations(self) -> list[ToolInvocation]:
        return self._decoded("space/assist/proposed", ToolInvocation)

    def results(self) -> list[ToolResult]:
        return self._decoded("space/assist/result", ToolResult)

    def records(self) -> list[ReasoningRecord]:
        return self._decoded("space/audit/reasoning", ReasoningRecord)


class TestAMeetingGoesInTheCalendar:
    """Week 6: asking for a meeting on Thursday puts it in the calendar."""

    def _ask(self, config):
        assistant = Assistant(
            config,
            [
                calls(
                    (
                        "schedule_event",
                        {"starts_at": THURSDAY_3PM, "subject": "design review"},
                    )
                ),
                text(answer(spoken_reply="I've put the design review in for Thursday at three.")),
            ],
        )
        hint = assistant.context.extract("put the design review in on Thursday at three")
        return assistant, hint

    def test_the_calendar_really_has_it(self, config):
        _, _ = self._ask(config)
        stored = json.loads(Path(config.assistance.calendar_path).read_text())
        assert any("design review" in json.dumps(entry) for entry in stored)

    def test_the_executor_ran_it_and_said_so(self, config):
        assistant, _ = self._ask(config)
        assert assistant.results()[-1].status is ToolStatus.OK

    def test_it_says_so_to_the_occupant(self, config):
        _, hint = self._ask(config)
        assert hint.intent is Intent.SERVICE
        assert "Thursday" in hint.spoken_reply

    def test_the_invocation_crossed_the_blackboard(self, config):
        """FR-75: no provider is reached without crossing the bus."""
        assistant, _ = self._ask(config)
        assert assistant.invocations()[-1].requester.value == "personal_context"

    def test_it_is_audited_as_applied(self, config):
        assistant, _ = self._ask(config)
        record = assistant.records()[-1]
        assert record.outcome is ReasoningOutcome.APPLIED
        assert record.tool_calls == ("schedule_event",)


class TestAskingAboutThursdayReadsItBack:
    def test_the_answer_is_read_from_the_calendar(self, config):
        writer = Assistant(
            config,
            [
                calls(("schedule_event", {"starts_at": THURSDAY_3PM, "subject": "design review"})),
                text(answer()),
            ],
        )
        writer.context.extract("put the design review in on Thursday at three")

        reader = Assistant(
            config,
            [
                calls(
                    (
                        "get_events",
                        {"from_time": "2025-08-28T00:00:00", "to_time": "2025-08-28T23:59:00"},
                    )
                ),
                text(answer(spoken_reply="On Thursday you have the design review at 15:00.")),
            ],
        )
        reader.context.extract("what have I got on Thursday?")
        result = reader.results()[-1]
        assert result.status is ToolStatus.OK and "design review" in result.message

    def test_the_model_is_given_what_the_tool_returned(self, config):
        reader = Assistant(
            config,
            [
                calls(("get_events", {"from_time": "2025-08-28T00:00:00", "to_time": "2025-08-28T23:59:00"})),
                text(answer(spoken_reply="Nothing on Thursday.")),
            ],
        )
        reader.context.extract("what have I got on Thursday?")
        tool_message = reader.client.requests[-1]["messages"][-1]
        assert tool_message["role"] == "tool"
        assert json.loads(tool_message["content"])["status"] == "OK"


class TestAFlightGetsAQuestionNotABooking:
    """Week 6: asking for a flight gets you a question, not a booking."""

    def _ask(self, config, reply="I've booked your flight to Delhi."):
        assistant = Assistant(
            config,
            [
                calls(
                    (
                        "book_travel",
                        {
                            "kind": "flight",
                            "destination": "Delhi",
                            "origin": "Hyderabad",
                            "depart_on": "2025-08-29T09:00:00",
                        },
                    )
                ),
                text(answer(subject="booking", spoken_reply=reply)),
            ],
        )
        return assistant, assistant.context.extract("book me a flight to Delhi on Friday")

    def test_nothing_is_booked(self, config):
        assistant, _ = self._ask(config)
        assert assistant.results()[-1].status is ToolStatus.CONFIRMATION_REQUIRED
        assert not any(r.status is ToolStatus.OK for r in assistant.results())

    def test_a_model_claiming_it_booked_is_corrected(self, config):
        """FR-54 is about what the occupant hears; a prompt is not a guarantee."""
        _, hint = self._ask(config)
        assert CONFIRMATION_NOTICE in hint.spoken_reply

    def test_a_model_that_asked_is_left_alone(self, config):
        _, hint = self._ask(config, reply="Shall I book it? Please confirm.")
        assert hint.spoken_reply == "Shall I book it? Please confirm."

    def test_nothing_personal_context_does_can_confirm_it(self, config):
        """Only the console republishes on assist/confirmed (FR-74)."""
        assistant, _ = self._ask(config)
        assert not any(
            name == "space/assist/confirmed" for name, _, _, _ in assistant.transport.published
        )


class TestTheBound:
    """FR-42: a bounded number of tool rounds, then it must answer."""

    def test_after_one_round_the_answer_is_constrained_and_toolless(self, config):
        assistant = Assistant(
            config,
            [calls(("get_events", {"from_time": THURSDAY_3PM, "to_time": THURSDAY_3PM})), text(answer())],
        )
        assistant.context.extract("what is on Thursday?")
        final = assistant.client.requests[-1]
        assert "tools" not in final
        assert final["response_format"] == {"type": "json_object"}

    def test_it_is_offered_only_the_assistance_surface(self, config):
        """It is not given the supervisor's read tools (FR-42)."""
        assistant = Assistant(config, [text(answer(intent="none", subject=""))])
        assistant.context.extract("hello")
        assert assistant.client.tool_names_offered(0) == [
            "schedule_event",
            "get_events",
            "book_travel",
        ]

    def test_a_flood_of_calls_is_cut_to_the_per_round_limit(self, config):
        lookups = [("get_events", {"from_time": THURSDAY_3PM, "to_time": THURSDAY_3PM})] * 10
        assistant = Assistant(config, [calls(*lookups), text(answer())])
        assistant.context.extract("look everything up")
        assert len(assistant.invocations()) == 4

    def test_no_history_is_carried_between_utterances(self, config):
        assistant = Assistant(
            config,
            [text(answer(intent="none", subject="")), text(answer(intent="none", subject=""))],
        )
        assistant.context.extract("yes")
        assistant.context.extract("yes")
        first, second = assistant.client.requests
        assert first["messages"] == second["messages"]

    def test_the_model_is_told_what_day_it_is(self, config):
        """"On Thursday" means nothing to a model that does not know today."""
        assistant = Assistant(config, [text(answer(intent="none", subject=""))])
        assistant.context.extract("hello")
        assert "Sunday 2025-08-24T16:10" in assistant.client.system_prompt(0)


class TestTemperatureRequests:
    def test_a_temperature_request_needs_no_tool(self, config):
        reply = answer(
            intent="environment",
            subject="temperature",
            comfort="cooler",
            target_c=23.0,
            spoken_reply="I've passed that on.",
        )
        assistant = Assistant(config, [text(reply)])
        hint = assistant.context.extract("it's too warm, make it 23")
        assert hint.intent is Intent.ENVIRONMENT and hint.target_c == 23.0
        assert len(assistant.client.requests) == 1

    def test_prose_instead_of_json_gets_one_constrained_retry(self, config):
        reply = answer(intent="environment", subject="temperature", comfort="cooler")
        assistant = Assistant(config, [text("Sure, cooler it is."), text(reply)])
        hint = assistant.context.extract("cooler please")
        assert hint.intent is Intent.ENVIRONMENT
        assert len(assistant.client.requests) == 2


class TestFailures:
    def test_a_server_that_is_down_yields_nothing(self, config):
        assistant = Assistant(config, [ConnectionError("refused")])
        assert assistant.context.extract("cooler please") is None
        assert assistant.records()[-1].outcome is ReasoningOutcome.UNAVAILABLE

    def test_a_tool_that_ran_is_reported_even_if_the_model_then_fails(self, config):
        """The calendar was written; the occupant must hear so (FR-75)."""
        assistant = Assistant(
            config,
            [
                calls(("schedule_event", {"starts_at": THURSDAY_3PM, "subject": "review"})),
                ConnectionError("refused"),
            ],
        )
        hint = assistant.context.extract("put the review in on Thursday at three")
        assert hint is not None and "review" in hint.spoken_reply
        assert hint.intent is Intent.SERVICE

    def test_an_unparseable_answer_is_discarded(self, config):
        assistant = Assistant(config, [text("nope"), text("still not json")])
        assert assistant.context.extract("cooler") is None
        assert assistant.records()[-1].outcome is ReasoningOutcome.DISCARDED

    def test_an_answer_outside_the_schema_is_discarded(self, config):
        assistant = Assistant(config, [text(answer(comfort="freezing"))])
        assert assistant.context.extract("cooler") is None

    def test_nothing_asked_is_nothing_published(self, config):
        assistant = Assistant(config, [text(answer(intent="none", subject=""))])
        assert assistant.context.extract("hello there") is None
        assert assistant.records()[-1].outcome is ReasoningOutcome.NO_ACTION

    def test_an_empty_transcript_makes_no_call(self, config):
        assistant = Assistant(config, [])
        assert assistant.context.extract("   ") is None
        assert assistant.client.requests == []

    def test_a_missing_executor_is_survived(self, config):
        """Section 7.1: the call answers without the tool."""
        assistant = Assistant(
            config,
            [calls(("get_events", {"from_time": THURSDAY_3PM, "to_time": THURSDAY_3PM})), text(answer())],
            with_executor=False,
        )
        assert assistant.context.extract("what is on Thursday?") is not None
        tool_message = assistant.client.requests[-1]["messages"][-1]
        assert json.loads(tool_message["content"])["status"] == "NO_RESULT"

    def test_nested_arguments_never_reach_the_blackboard(self, config):
        assistant = Assistant(
            config,
            [calls(("schedule_event", {"starts_at": {"day": "Thu"}, "subject": "x"})), text(answer())],
        )
        assistant.context.extract("put x in on Thursday")
        assert assistant.invocations() == []


class TestHonesty:
    def test_a_mock_that_ran_is_always_called_a_mock(self):
        """FR-55, enforced on the words rather than hoped for."""
        from src.common.schemas import Comfort, PreferenceHint

        hint = PreferenceHint(
            ts=1756032000.0,
            intent=Intent.SERVICE,
            comfort=Comfort.UNCHANGED,
            spoken_reply="Your flight is booked.",
        )
        result = ToolResult(
            ts=1756032000.0,
            invocation_id="inv_1",
            tool="book_travel",
            status=ToolStatus.OK,
            message="Simulated only.",
            provider="mock_travel",
            simulated=True,
        )
        assert MOCK_NOTICE in PersonalContext._honest(hint, [result]).spoken_reply


class TestToolClient:
    def test_a_result_for_someone_elses_invocation_is_ignored(self, config):
        assistant = Assistant(config, [], with_executor=False)
        stray = ToolResult(
            ts=1756032000.0, invocation_id="inv_other", tool="get_events", status=ToolStatus.OK,
            message="x", provider="local_calendar",
        )
        assistant.transport.publish(
            topics.ASSIST_RESULT.pattern, stray.model_dump_json().encode(), 1, False
        )
        assert assistant.tools.call("get_events", {}, "x") is None

    def test_the_invocation_expires_after_the_confirmation_window(self, config):
        assistant = Assistant(config, [], with_executor=False)
        assistant.tools.call("get_events", {}, "x")
        invocation = assistant.invocations()[-1]
        assert invocation.expires_ts - invocation.ts == config.assistance.confirmation_window_s

    def test_the_pending_table_is_bounded(self, config):
        assert MAX_PENDING <= 64

    def test_an_argument_that_cannot_be_carried_is_refused_before_publishing(self, config):
        assistant = Assistant(config, [], with_executor=False)
        with pytest.raises(ValueError):
            assistant.tools.call("get_events", {"from_time": [1, 2]}, "x")
        assert assistant.invocations() == []
