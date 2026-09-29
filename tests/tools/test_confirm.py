"""The terminal tool that stands in for the console's confirmations (FR-54, FR-74).

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
from src.common.tools import ToolInvocation, ToolRequester, ToolResult, ToolStatus
from tools.confirm import ConfirmationDesk, describe as describe_booking

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
