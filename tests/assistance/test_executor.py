"""Unit tests for the assistance executor (section 5.7.6).

Driven over an in-process blackboard, so every invocation and result crosses
the same topics and schemas it would against mosquitto.
"""

from pathlib import Path

import pytest

from eval.loopback import LoopbackTransport
from src.assistance.__main__ import build_service
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.tools import (
    ToolCatalogue,
    ToolInvocation,
    ToolRequester,
    ToolResult,
    ToolStatus,
)

FLIGHT = {"kind": "flight", "destination": "Delhi", "depart_on": "2026-10-01T09:00:00"}
MEETING = {"starts_at": "2026-10-01T15:00:00", "subject": "review"}


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    assistance = config.assistance.model_copy(
        update={"calendar_path": str(tmp_path / "calendar.json")}
    )
    return config.model_copy(update={"assistance": assistance})


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="wired")
def _wired(config, clock):
    transport = LoopbackTransport()
    executor_board = Blackboard(config.mqtt, transport)
    caller = Blackboard(config.mqtt, transport)
    transport.attach(executor_board)
    transport.attach(caller)
    service = build_service(config, clock, executor_board)
    service.subscribe()
    return service, transport, caller


def _invocation(clock, tool, arguments, invocation_id="inv_1", ttl_s=300.0):
    return ToolInvocation(
        ts=clock.now(),
        invocation_id=invocation_id,
        tool=tool,
        arguments=arguments,
        requester=ToolRequester.PERSONAL_CONTEXT,
        rationale="test",
        expires_ts=clock.now() + ttl_s,
    )


def _results(transport) -> list[ToolResult]:
    return [
        ToolResult.model_validate_json(payload)
        for topic, payload, _, _ in transport.published
        if topic == topics.ASSIST_RESULT.pattern and payload
    ]


class TestRunning:
    def test_a_calendar_write_runs_without_confirmation(self, wired, clock):
        _, transport, caller = wired
        caller.publish(topics.ASSIST_PROPOSED, _invocation(clock, "schedule_event", MEETING))
        result = _results(transport)[-1]
        assert result.status is ToolStatus.OK
        assert result.provider == "local_calendar"

    def test_the_result_is_correlated_to_the_invocation(self, wired, clock):
        _, transport, caller = wired
        caller.publish(
            topics.ASSIST_PROPOSED,
            _invocation(clock, "schedule_event", MEETING, invocation_id="inv_42"),
        )
        assert _results(transport)[-1].invocation_id == "inv_42"

    def test_a_booking_is_refused_until_confirmed(self, wired, clock):
        """FR-54, FR-74: asking for a flight gets a question, not a booking."""
        _, transport, caller = wired
        caller.publish(topics.ASSIST_PROPOSED, _invocation(clock, "book_travel", FLIGHT))
        assert _results(transport)[-1].status is ToolStatus.CONFIRMATION_REQUIRED

    def test_a_confirmed_booking_runs_and_is_marked_simulated(self, wired, clock):
        """FR-55: the mock is identifiable on the blackboard itself."""
        _, transport, caller = wired
        caller.publish(topics.ASSIST_CONFIRMED, _invocation(clock, "book_travel", FLIGHT))
        result = _results(transport)[-1]
        assert result.status is ToolStatus.OK
        assert result.simulated is True

    def test_a_repeated_confirmation_books_once(self, wired, clock):
        """A double click on the console must not book twice."""
        _, transport, caller = wired
        booking = _invocation(clock, "book_travel", FLIGHT)
        caller.publish(topics.ASSIST_CONFIRMED, booking)
        caller.publish(topics.ASSIST_CONFIRMED, booking)
        assert len(_results(transport)) == 1

    def test_a_late_confirmation_is_refused(self, wired, clock):
        _, transport, caller = wired
        booking = _invocation(clock, "book_travel", FLIGHT, ttl_s=10.0)
        clock.advance(60.0)
        caller.publish(topics.ASSIST_CONFIRMED, booking)
        assert _results(transport)[-1].status is ToolStatus.EXPIRED

    def test_bad_arguments_are_refused_with_the_parameter_named(self, wired, clock):
        """FR-73: refused before any provider is reached."""
        _, transport, caller = wired
        caller.publish(
            topics.ASSIST_PROPOSED, _invocation(clock, "schedule_event", {"subject": "x"})
        )
        result = _results(transport)[-1]
        assert result.status is ToolStatus.BAD_ARGUMENTS
        assert "starts_at" in result.message

    def test_an_unknown_tool_is_refused(self, wired, clock):
        _, transport, caller = wired
        caller.publish(topics.ASSIST_PROPOSED, _invocation(clock, "order_pizza", {}))
        assert _results(transport)[-1].status is ToolStatus.UNKNOWN_TOOL


class TestCatalogue:
    def test_the_catalogue_is_published_retained(self, wired):
        service, transport, _ = wired
        service.publish_catalogue()
        topic, payload, _, retain = transport.published[-1]
        assert topic == topics.ASSIST_CATALOGUE.pattern and retain
        names = {spec.name for spec in ToolCatalogue.model_validate_json(payload).tools}
        assert names == {"schedule_event", "get_events", "cancel_event", "move_event", "book_travel", "send_mail", "remember", "forget", "pencil_event", "keep_event", "suggest_fact", "settle_fact", "web_search", "news_search", "cricket_score", "read_pages"}


class TestOverheardPlans:
    """FR-77: a plan overheard is pencilled at once; nothing commits."""

    def test_a_pencilled_plan_runs_without_confirmation(self, wired, clock):
        _, transport, caller = wired
        invocation = _invocation(clock, "pencil_event", {**MEETING, "heard": "review on thursday"})
        caller.publish(topics.ASSIST_PROPOSED, invocation.model_copy(update={"requester": ToolRequester.AMBIENT}))
        assert _results(transport)[-1].status is ToolStatus.OK

    def test_keeping_it_runs_too(self, wired, clock):
        _, transport, caller = wired
        caller.publish(topics.ASSIST_PROPOSED, _invocation(clock, "pencil_event", MEETING))
        event_id = _results(transport)[-1].detail["event_id"]
        caller.publish(topics.ASSIST_PROPOSED, _invocation(clock, "keep_event", {"event_id": event_id}, invocation_id="inv_2"))
        assert _results(transport)[-1].status is ToolStatus.OK


class _Backend:
    """A search service that answers at once, and remembers being asked."""

    name = "searxng"

    def __init__(self):
        self.asked = []

    def search(self, query, *, news):
        from src.assistance.providers.web_search import Hit

        self.asked.append((query, news))
        return [Hit(title="RCB win the final", url="https://a.in/1", source="a.in", snippet="By five wickets.", published_ts=None)]


def _search_wired(config, clock, backend):
    transport = LoopbackTransport()
    executor_board = Blackboard(config.mqtt, transport)
    caller = Blackboard(config.mqtt, transport)
    transport.attach(executor_board)
    transport.attach(caller)
    service = build_service(config, clock, executor_board, search=backend)
    service.subscribe()
    return service, transport, caller


class TestWebSearch:
    """FR-101: on by default, off on request, and never for overheard speech."""

    def test_a_question_asked_of_the_room_is_searched(self, config, clock):
        backend = _Backend()
        _, transport, caller = _search_wired(config, clock, backend)
        caller.publish(topics.ASSIST_PROPOSED, _invocation(clock, "web_search", {"query": "who won the IPL"}))
        result = _results(transport)[-1]
        assert result.status is ToolStatus.OK and result.provider == "searxng"
        assert backend.asked == [("who won the IPL", False)]

    def test_overheard_speech_never_reaches_the_search_service(self, config, clock):
        backend = _Backend()
        _, transport, caller = _search_wired(config, clock, backend)
        overheard = _invocation(clock, "web_search", {"query": "who won the IPL"}).model_copy(
            update={"requester": ToolRequester.AMBIENT}
        )
        caller.publish(topics.ASSIST_PROPOSED, overheard)
        assert _results(transport)[-1].status is ToolStatus.NOT_ADDRESSED
        assert backend.asked == []

    def test_live_scores_are_declared_beside_search(self, config, clock):
        service, _, _ = _search_wired(config, clock, _Backend())
        assert "cricket_score" in service.registry.names

    def test_with_live_scores_off_only_that_tool_goes(self, config, clock):
        cricket = config.web_search.cricket.model_copy(update={"enabled": False})
        off = config.model_copy(update={"web_search": config.web_search.model_copy(update={"cricket": cricket})})
        service, _, _ = _search_wired(off, clock, _Backend())
        assert "cricket_score" not in service.registry.names and "web_search" in service.registry.names

    def test_with_web_search_off_the_tools_are_not_declared(self, config, clock):
        off = config.model_copy(update={"web_search": config.web_search.model_copy(update={"enabled": False})})
        service, _, _ = _search_wired(off, clock, _Backend())
        assert not {"web_search", "news_search", "cricket_score"} & set(service.registry.names)


class TestDeepAnswers:
    """FR-104: the page reader is there when deep answers are on, and only then."""

    def test_the_page_reader_is_declared_but_never_shown_to_the_model(self, config, clock):
        service, _, _ = _search_wired(config, clock, _Backend())
        assert "read_pages" in service.registry.names
        assert service.registry.spec("read_pages").offered_to_assistant is False

    def test_with_deep_answers_off_there_is_no_page_reader(self, config, clock):
        deep = config.web_search.deep.model_copy(update={"enabled": False})
        off = config.model_copy(update={"web_search": config.web_search.model_copy(update={"deep": deep})})
        service, _, _ = _search_wired(off, clock, _Backend())
        assert "read_pages" not in service.registry.names

    def test_a_page_no_search_returned_is_refused_through_the_executor(self, config, clock):
        _, transport, caller = _search_wired(config, clock, _Backend())
        caller.publish(topics.ASSIST_PROPOSED, _invocation(clock, "read_pages", {"urls": "http://127.0.0.1:8077/api/settings"}))
        result = _results(transport)[-1]
        assert result.status is ToolStatus.FAILED and "pages a search just found" in result.message
