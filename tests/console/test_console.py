"""The occupant console (FR-56 to FR-58), with no broker and no browser.

What matters: it shows what happened, it can only publish what speech and
the operator tools already publish, and it never commands the actuator.
"""

import ast
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    AssistantReply,
    Comfort,
    ConsoleUtterance,
    FaultEvent,
    InjectionCommand,
    Intent,
    Mode,
    ModeState,
    PreferenceHint,
    SpeechActivity,
    SpeechStatus,
    ValidationVerdict,
    Verdict,
    ReasonCode,
)
from src.common.tools import ToolInvocation, ToolRequester, ToolResult, ToolStatus
from src.console.actions import ActionRefused, ConsoleActions
from src.console.server import Hub, Link, create_app
from src.console.state import ConsoleState

CONFIG = load_config(Path("config/default.yaml"))


@pytest.fixture(name="console")
def _console():
    clock = SimClock()
    transport = LoopbackTransport()
    board = Blackboard(CONFIG.mqtt, transport)
    transport.attach(board)
    state = ConsoleState(CONFIG.console, clock)
    actions = ConsoleActions(CONFIG, clock, board, state)
    return state, actions, transport, clock


def _feed(state, spec, message, **parameters):
    return state.accept(spec.format(**parameters) if parameters else spec.pattern, message.model_dump_json(by_alias=True).encode())


def _published(transport):
    return [topic for topic, _, _, _ in transport.published]


class TestWhatItShows:
    def test_a_spoken_request_appears_as_the_occupant(self, console):
        state, _, _, clock = console
        hint = PreferenceHint(ts=clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.COOLER, transcript="it's stuffy")
        _feed(state, topics.CONTEXT_PREFERENCE, hint)
        turn = state.snapshot()["conversation"][-1]
        assert turn["role"] == "occupant" and turn["source"] == "voice" and turn["text"] == "it's stuffy"

    def test_a_typed_request_is_not_shown_twice(self, console):
        """The utterance, then the hint made of it: one bubble, not two."""
        state, _, _, clock = console
        _feed(state, topics.CONTEXT_UTTERANCE, ConsoleUtterance(ts=clock.now(), utterance_id="u1", text="make it cooler"))
        _feed(state, topics.CONTEXT_PREFERENCE, PreferenceHint(ts=clock.now(), intent=Intent.ENVIRONMENT, comfort=Comfort.COOLER, transcript="make it cooler"))
        occupant = [turn for turn in state.snapshot()["conversation"] if turn["role"] == "occupant"]
        assert len(occupant) == 1 and occupant[0]["source"] == "console"

    def test_the_reply_appears_as_the_assistant(self, console):
        state, _, _, clock = console
        _feed(state, topics.CONTEXT_REPLY, AssistantReply(ts=clock.now(), transcript="x", intent=Intent.QUESTION, reply="It is 24 degrees."))
        assert state.snapshot()["conversation"][-1] == {**state.snapshot()["conversation"][-1], "role": "assistant", "text": "It is 24 degrees."}

    def test_a_clamp_is_an_agent_step(self, console):
        state, _, _, clock = console
        verdict = ValidationVerdict(ts=clock.now(), proposed={"setpoint_c": 40.0}, verdict=Verdict.CLAMPED, reason=ReasonCode.BOUND_CLAMP, applied={"setpoint_c": 30.0})
        _feed(state, topics.AUDIT_VALIDATION, verdict)
        step = state.snapshot()["activity"][-1]
        assert step["kind"] == "safety" and "40" in step["text"] and "30" in step["text"]

    def test_routine_command_verdicts_are_not_shown(self, console):
        """A MAINTAIN accepted every 5 s is not something a person needs to see."""
        state, _, _, clock = console
        verdict = ValidationVerdict(ts=clock.now(), proposed={"kind": "MAINTAIN"}, verdict=Verdict.ACCEPTED, reason=ReasonCode.NONE, applied={"kind": "MAINTAIN"})
        assert _feed(state, topics.AUDIT_VALIDATION, verdict) == []

    def test_a_mode_change_is_announced(self, console):
        state, _, _, clock = console
        _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.NORMAL, since_ts=clock.now()))
        _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.DEGRADED_SENSOR, since_ts=clock.now(), reason="D1 on temp_01"))
        assert "degraded sensor" in state.snapshot()["conversation"][-1]["text"]

    def test_a_retired_fault_leaves_the_list(self, console):
        state, _, _, clock = console
        fault = FaultEvent(fault_id="f1", detector="D1_DROPOUT", subject="temp_01", **{"class": "sensor"}, confidence=1.0, detected_ts=clock.now(), mode_impact=Mode.DEGRADED_SENSOR)
        _feed(state, topics.FAULT, fault, fault_id="f1")
        state.accept(topics.FAULT.format(fault_id="f1"), b"")
        assert state.snapshot()["faults"] == []

    def test_speech_status_is_kept(self, console):
        state, _, _, clock = console
        _feed(state, topics.SPEECH_STATUS, SpeechStatus(ts=clock.now(), activity=SpeechActivity.LISTENING, heartbeat_s=15.0))
        assert state.snapshot()["speech"]["activity"] == "listening"

    def test_a_booking_waits_until_answered(self, console):
        state, _, _, clock = console
        invocation = ToolInvocation(ts=clock.now(), invocation_id="b1", tool="book_travel", arguments={"kind": "flight", "destination": "Delhi"}, requester=ToolRequester.PERSONAL_CONTEXT, expires_ts=clock.now() + 60)
        _feed(state, topics.ASSIST_PROPOSED, invocation)
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id="b1", tool="book_travel", status=ToolStatus.CONFIRMATION_REQUIRED))
        assert state.snapshot()["waiting"][0]["summary"] == "A mock flight to Delhi"

    def test_a_calendar_read_fills_the_calendar(self, console):
        state, actions, transport, clock = console
        invocation_id = actions.refresh_calendar()
        result = ToolResult(ts=clock.now(), invocation_id=invocation_id, tool="get_events", status=ToolStatus.OK, provider="local_calendar", detail={"count": 1, "entries": "evt_1 2026-10-14T15:00:00 meeting with Anu"})
        _feed(state, topics.ASSIST_RESULT, result)
        assert state.snapshot()["calendar"]["entries"] == [{"event_id": "evt_1", "starts_at": "2026-10-14T15:00:00", "subject": "meeting with Anu"}]

    def test_a_malformed_message_is_ignored(self, console):
        state, _, _, _ = console
        assert state.accept(topics.SYSTEM_MODE.pattern, b"{not json") == []


class TestWhatItCanDo:
    def test_typing_publishes_words_only(self, console):
        _, actions, transport, _ = console
        actions.say("make it cooler", "ravi")
        assert _published(transport) == [topics.CONTEXT_UTTERANCE.pattern]

    def test_a_comfort_setting_is_a_preference_not_a_command(self, console):
        """FR-57: gated by the validator exactly as a spoken one."""
        _, actions, transport, _ = console
        actions.set_target(23.0)
        assert _published(transport) == [topics.CONTEXT_PREFERENCE.pattern]

    def test_nothing_it_does_reaches_an_actuator_topic(self, console):
        state, actions, transport, clock = console
        actions.say("turn the AC to 18")
        actions.set_target(18.0)
        actions.refresh_calendar()
        actions.remove_entry("meeting with Anu", "2026-10-14T15:00:00")
        actions.reset("checked")
        actions.inject("temp_01", "dropout")
        assert not any("/actuator/" in topic for topic in _published(transport))

    def test_a_confirmed_booking_is_republished_verbatim(self, console):
        state, actions, transport, clock = console
        invocation = ToolInvocation(ts=clock.now(), invocation_id="b1", tool="book_travel", arguments={"kind": "hotel", "destination": "Pune"}, requester=ToolRequester.PERSONAL_CONTEXT, expires_ts=clock.now() + 60)
        _feed(state, topics.ASSIST_PROPOSED, invocation)
        actions.answer("b1", yes=True)
        confirmed = [payload for topic, payload, _, _ in transport.published if topic == topics.ASSIST_CONFIRMED.pattern]
        assert ToolInvocation.model_validate_json(confirmed[0]) == invocation

    def test_a_declined_booking_publishes_nothing(self, console):
        state, actions, transport, clock = console
        invocation = ToolInvocation(ts=clock.now(), invocation_id="b1", tool="book_travel", arguments={"kind": "hotel"}, requester=ToolRequester.PERSONAL_CONTEXT, expires_ts=clock.now() + 60)
        _feed(state, topics.ASSIST_PROPOSED, invocation)
        actions.answer("b1", yes=False)
        assert topics.ASSIST_CONFIRMED.pattern not in _published(transport)

    def test_calendar_requests_come_from_the_console(self, console):
        _, actions, transport, _ = console
        actions.remove_entry("dentist", "")
        sent = ToolInvocation.model_validate_json(transport.published[-1][1])
        assert sent.tool == "cancel_event" and sent.requester is ToolRequester.CONSOLE

    def test_an_unsupported_injection_is_refused(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.inject("pir_01", "drift", 0.1)

    def test_an_injection_needing_a_value_is_refused_without_one(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.inject("temp_01", "stuck_at")

    def test_operator_controls_can_be_switched_off(self):
        config = CONFIG.model_copy(update={"console": CONFIG.console.model_copy(update={"operator_controls": False})})
        clock = SimClock()
        board = Blackboard(config.mqtt, LoopbackTransport())
        actions = ConsoleActions(config, clock, board, ConsoleState(config.console, clock))
        with pytest.raises(ActionRefused):
            actions.reset("anything")


class TestTheWebInterface:
    @pytest.fixture(name="client")
    def _client(self, console):
        state, actions, transport, _ = console
        lock = threading.Lock()
        hub = Hub()
        link = Link(state, hub, lock, Blackboard(CONFIG.mqtt, transport))
        return TestClient(create_app(state, actions, hub, link, lock)), transport

    def test_the_state_can_be_pulled(self, client):
        http, _ = client
        body = http.get("/api/state").json()
        assert {"room", "conversation", "activity", "waiting", "speech"} <= set(body)

    def test_a_snapshot_is_pushed_on_connect(self, client):
        http, _ = client
        with http.websocket_connect("/api/events") as socket:
            assert socket.receive_json()["type"] == "snapshot"

    def test_saying_something_publishes_it(self, client):
        http, transport = client
        assert http.post("/api/say", json={"text": "hello"}).json()["ok"] is True
        assert transport.published[-1][0] == topics.CONTEXT_UTTERANCE.pattern

    def test_a_refusal_is_explained(self, client):
        http, _ = client
        response = http.post("/api/operator/inject", json={"subject": "nowhere", "fault": "dropout"})
        assert response.status_code == 400 and "not a sensor" in response.json()["error"]

    def test_an_unsafe_profile_id_is_refused(self, client):
        http, _ = client
        assert http.post("/api/say", json={"text": "hi", "profile_id": "../x"}).status_code == 400

    def test_the_page_is_served_built_or_not(self, client):
        http, _ = client
        assert http.get("/").status_code == 200


def test_the_console_imports_only_the_common_contracts():
    """Layer 1: it reaches the system through the blackboard and nothing else."""
    for path in Path("src/console").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
                assert node.module.startswith(("src.common", "src.console")), f"{path.name} imports {node.module}"
