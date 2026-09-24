"""Unit tests for the assistance executor.

Most of these are about the confirmation gate, because it is the one thing
standing between a model and somebody's money, and because the way it is built
-- authorisation carried by the topic rather than by a field -- is easy to
undo by accident.

The rest are about FR-75: every outcome reaches the blackboard, including the
ones nobody wanted.
"""

from pathlib import Path

import pytest

from src.assistance.__main__ import build_registry, build_service
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.tools import (
    BOOK_TRAVEL,
    GET_EVENTS,
    SCHEDULE_EVENT,
    ToolCatalogue,
    ToolInvocation,
    ToolRequester,
    ToolResult,
    ToolStatus,
)

FUTURE = "2026-12-01T09:00:00"
MEETING = {"starts_at": "2026-10-01T15:00:00", "subject": "design review"}
FLIGHT = {
    "kind": "flight",
    "destination": "Delhi",
    "depart_on": FUTURE,
    "origin": "Hyderabad",
}


class FakeTransport:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.subscribed: list[tuple[str, int]] = []

    def connect(self, host, port, keepalive):
        pass

    def publish(self, topic, payload, qos, retain):
        self.published.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos):
        self.subscribed.append((topic, qos))

    def loop_start(self):
        pass

    def loop_stop(self):
        pass

    def disconnect(self):
        pass

    def results(self) -> list[ToolResult]:
        return [
            ToolResult.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/assist/result"
        ]

    def catalogues(self) -> list[ToolCatalogue]:
        return [
            ToolCatalogue.model_validate_json(payload)
            for topic, payload, _, _ in self.published
            if topic == "space/assist/catalogue"
        ]


@pytest.fixture(name="config")
def _config(tmp_path):
    base = load_config(Path("config/default.yaml"))
    return base.model_copy(
        update={
            "assistance": base.assistance.model_copy(
                update={"calendar_path": str(tmp_path / "calendar.json")}
            )
        }
    )


@pytest.fixture(name="clock")
def _clock() -> SimClock:
    return SimClock()


@pytest.fixture(name="wired")
def _wired(config, clock):
    transport = FakeTransport()
    blackboard = Blackboard(config.mqtt, transport)
    service = build_service(config, clock, blackboard)
    service.subscribe()
    blackboard.on_connected()
    return service, transport, blackboard


def _invocation(clock, tool: str, arguments: dict, expires_in_s: float = 300.0):
    return ToolInvocation(
        ts=clock.now(),
        invocation_id=f"inv_{int(clock.now())}",
        tool=tool,
        arguments=arguments,
        requester=ToolRequester.PERSONAL_CONTEXT,
        rationale="because the occupant asked",
        expires_ts=clock.now() + expires_in_s,
    )


def _send(blackboard, invocation, topic: str) -> None:
    blackboard.dispatch(
        topic, invocation.model_dump_json(by_alias=True).encode()
    )


class TestTheCatalogue:
    def test_it_is_published_retained(self, wired):
        """FR-70: an examiner can read what the reasoning layer may ask for
        without running it."""
        service, transport, _ = wired
        service.publish_catalogue()
        entry = [
            e for e in transport.published if e[0] == "space/assist/catalogue"
        ][-1]
        assert entry[3] is True

    def test_it_lists_every_declared_tool(self, wired):
        service, transport, _ = wired
        service.publish_catalogue()
        names = {spec.name for spec in transport.catalogues()[-1].tools}
        assert names == {SCHEDULE_EVENT.name, GET_EVENTS.name, BOOK_TRAVEL.name}

    def test_it_says_which_tools_need_confirming(self, wired):
        service, transport, _ = wired
        service.publish_catalogue()
        needing = {
            spec.name
            for spec in transport.catalogues()[-1].tools
            if spec.requires_confirmation
        }
        assert needing == {BOOK_TRAVEL.name}


class TestRunningWhatWasProposed:
    def test_a_write_runs_without_confirmation(self, wired, clock):
        """It is the occupant's own calendar and the entry can be removed
        again, so asking would be friction without protection."""
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "schedule_event", MEETING),
              "space/assist/proposed")
        assert transport.results()[-1].status is ToolStatus.OK

    def test_a_read_runs_without_confirmation(self, wired, clock):
        _, transport, blackboard = wired
        _send(
            blackboard,
            _invocation(
                clock,
                "get_events",
                {
                    "from_time": "2026-10-01T00:00:00",
                    "to_time": "2026-10-02T00:00:00",
                },
            ),
            "space/assist/proposed",
        )
        assert transport.results()[-1].status is ToolStatus.OK

    def test_the_result_names_the_provider_that_ran_it(self, wired, clock):
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "schedule_event", MEETING),
              "space/assist/proposed")
        assert transport.results()[-1].provider == "local_calendar"

    def test_a_real_provider_is_not_labelled_simulated(self, wired, clock):
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "schedule_event", MEETING),
              "space/assist/proposed")
        assert transport.results()[-1].simulated is False


class TestTheConfirmationGate:
    def test_a_commit_proposed_is_refused(self, wired, clock):
        """Asking for a flight gets you a question, not a booking."""
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "book_travel", FLIGHT),
              "space/assist/proposed")
        assert transport.results()[-1].status is ToolStatus.CONFIRMATION_REQUIRED

    def test_nothing_is_booked_by_a_refusal(self, wired, clock):
        service, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "book_travel", FLIGHT),
              "space/assist/proposed")
        provider = service.registry._providers["book_travel"]
        assert provider.bookings == 0

    def test_the_same_invocation_confirmed_runs(self, wired, clock):
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "book_travel", FLIGHT),
              "space/assist/confirmed")
        assert transport.results()[-1].status is ToolStatus.OK

    def test_confirmation_comes_from_the_topic_not_the_message(
        self, wired, clock
    ):
        """There is no field to set. A publisher that could assert its own
        approval would make the gate advisory."""
        assert "confirmed" not in ToolInvocation.model_fields

    def test_a_confirmed_booking_is_still_labelled_simulated(self, wired, clock):
        """FR-55 does not relax because somebody agreed."""
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "book_travel", FLIGHT),
              "space/assist/confirmed")
        assert transport.results()[-1].simulated is True

    def test_confirming_a_write_changes_nothing_about_it(self, wired, clock):
        """Only a COMMIT tool asks, so confirmation is not a way to smuggle
        anything past a different gate."""
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "schedule_event", MEETING),
              "space/assist/confirmed")
        assert transport.results()[-1].status is ToolStatus.OK


class TestEveryOutcomeIsPublished:
    """FR-75. Silence leaves whoever asked waiting and the audit trail holed."""

    def test_an_unknown_tool_produces_a_result(self, wired, clock):
        _, transport, blackboard = wired
        _send(blackboard, _invocation(clock, "launch_rocket", {}),
              "space/assist/proposed")
        assert transport.results()[-1].status is ToolStatus.UNKNOWN_TOOL

    def test_a_bad_argument_produces_a_result(self, wired, clock):
        _, transport, blackboard = wired
        _send(
            blackboard,
            _invocation(clock, "book_travel", {**FLIGHT, "kind": "submarine"}),
            "space/assist/confirmed",
        )
        assert transport.results()[-1].status is ToolStatus.BAD_ARGUMENTS

    def test_an_expired_invocation_produces_a_result(self, wired, clock):
        """Agreeing to a booking is agreeing to the one that was described,
        not to whatever is still pending."""
        _, transport, blackboard = wired
        invocation = _invocation(clock, "book_travel", FLIGHT, expires_in_s=10.0)
        clock.advance(60.0)
        _send(blackboard, invocation, "space/assist/confirmed")
        assert transport.results()[-1].status is ToolStatus.EXPIRED

    def test_a_failing_provider_produces_a_result(self, wired, clock):
        """A flight with no origin: the mock refuses, and the refusal is an
        outcome rather than a crash."""
        _, transport, blackboard = wired
        _send(
            blackboard,
            _invocation(
                clock,
                "book_travel",
                {"kind": "flight", "destination": "Delhi", "depart_on": FUTURE},
            ),
            "space/assist/confirmed",
        )
        assert transport.results()[-1].status is ToolStatus.FAILED

    def test_a_failing_provider_does_not_stop_the_executor(self, wired, clock):
        _, transport, blackboard = wired
        _send(
            blackboard,
            _invocation(
                clock,
                "book_travel",
                {"kind": "flight", "destination": "Delhi", "depart_on": FUTURE},
            ),
            "space/assist/confirmed",
        )
        _send(blackboard, _invocation(clock, "schedule_event", MEETING),
              "space/assist/proposed")
        assert transport.results()[-1].status is ToolStatus.OK

    def test_every_invocation_produces_exactly_one_result(self, wired, clock):
        service, transport, blackboard = wired
        for _ in range(3):
            clock.advance(1.0)
            _send(blackboard, _invocation(clock, "schedule_event", MEETING),
                  "space/assist/proposed")
        assert len(transport.results()) == 3
        assert service.results_published == 3


class TestBinding:
    def test_every_declared_tool_has_a_provider(self, config, clock):
        """A declared tool with nothing behind it is one the reasoning layer
        can ask for and never get."""
        registry = build_registry(config, clock)
        for name in registry.names:
            assert name in registry._providers

    def test_the_calendar_serves_both_of_its_tools(self, config, clock):
        registry = build_registry(config, clock)
        assert (
            registry._providers["schedule_event"]
            is registry._providers["get_events"]
        )

    def test_the_executor_subscribes_to_both_invocation_topics(self, wired):
        _, transport, _ = wired
        patterns = {topic for topic, _ in transport.subscribed}
        assert topics.ASSIST_PROPOSED.pattern in patterns
        assert topics.ASSIST_CONFIRMED.pattern in patterns
