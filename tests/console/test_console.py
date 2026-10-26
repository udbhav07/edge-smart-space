"""The occupant console (FR-56 to FR-58), with no broker and no browser.

What matters: it shows what happened, it can only publish what speech and
the operator tools already publish, and it never commands the actuator.
"""

import ast
import json
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
    ProfileContext,
    SensorReading,
    SpeechActivity,
    SpeechStatus,
    ThermalEstimate,
    Unit,
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
        assert [event["type"] for event in _feed(state, topics.AUDIT_VALIDATION, verdict)] == ["mqtt"]

    def test_a_reasserted_command_is_not_a_new_decision(self, console):
        """COOL restated every 60 s flooded the decision log."""
        state, _, _, clock = console
        verdict = ValidationVerdict(ts=clock.now(), proposed={"kind": "COOL"}, verdict=Verdict.ACCEPTED, reason=ReasonCode.NONE, applied={"kind": "COOL"})
        _feed(state, topics.AUDIT_VALIDATION, verdict)
        _feed(state, topics.AUDIT_VALIDATION, verdict)
        assert [step["text"] for step in state.snapshot()["activity"]] == ["Air conditioner: COOL"]

    def test_a_change_of_command_is_shown_as_one(self, console):
        state, _, _, clock = console
        for kind in ("COOL", "OFF"):
            _feed(state, topics.AUDIT_VALIDATION, ValidationVerdict(ts=clock.now(), proposed={"kind": kind}, verdict=Verdict.ACCEPTED, reason=ReasonCode.NONE, applied={"kind": kind}))
        assert state.snapshot()["activity"][-1]["text"] == "Air conditioner: COOL → OFF"

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

    def test_the_temperature_history_is_given_to_a_new_browser(self, console):
        """A chart that starts empty on every page load shows nothing useful."""
        from src.common.schemas import AdaptationState, ThermalEstimate
        state, _, _, clock = console
        for value in (24.0, 24.2):
            _feed(state, topics.ESTIMATE_THERMAL, ThermalEstimate(ts=clock.now(), t_in=value, t_pred=value, residual=0.0, residual_sigma=0.1, model_confidence=0.9, adaptation=AdaptationState.ACTIVE))
        assert [value for _, value in state.snapshot()["series"]["t_in"]] == [24.0, 24.2]

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

    def test_a_hotel_is_summarised_as_a_stay(self, console):
        state, _, _, clock = console
        invocation = ToolInvocation(ts=clock.now(), invocation_id="b2", tool="book_travel", arguments={"kind": "hotel", "destination": "Pune", "nights": 2, "depart_on": "2026-10-17"}, requester=ToolRequester.PERSONAL_CONTEXT, expires_ts=clock.now() + 60)
        _feed(state, topics.ASSIST_PROPOSED, invocation)
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id="b2", tool="book_travel", status=ToolStatus.CONFIRMATION_REQUIRED))
        assert state.snapshot()["waiting"][0]["summary"] == "A mock hotel in Pune, 2 nights from 2026-10-17"

    def test_a_confirmed_bookings_outcome_is_said(self, console):
        """Without this the occupant said yes and never heard it was booked."""
        state, actions, _, clock = console
        invocation = ToolInvocation(ts=clock.now(), invocation_id="b3", tool="book_travel", arguments={"kind": "hotel", "destination": "Pune"}, requester=ToolRequester.PERSONAL_CONTEXT, expires_ts=clock.now() + 60)
        _feed(state, topics.ASSIST_PROPOSED, invocation)
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id="b3", tool="book_travel", status=ToolStatus.CONFIRMATION_REQUIRED))
        actions.answer("b3", yes=True)
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id="b3", tool="book_travel", status=ToolStatus.OK, provider="mock_booking", message="Mock booking MOCK-0001 recorded."))
        assert state.snapshot()["conversation"][-1]["text"] == "Mock booking MOCK-0001 recorded."

    def test_a_calendar_read_fills_the_calendar(self, console):
        state, actions, transport, clock = console
        invocation_id = actions.refresh_calendar()
        result = ToolResult(ts=clock.now(), invocation_id=invocation_id, tool="get_events", status=ToolStatus.OK, provider="local_calendar", detail={"count": 1, "entries": "evt_1 2026-10-14T15:00:00 meeting with Anu"})
        _feed(state, topics.ASSIST_RESULT, result)
        assert state.snapshot()["calendar"]["entries"] == [{"event_id": "evt_1", "starts_at": "2026-10-14T15:00:00", "subject": "meeting with Anu", "tentative": False, "google": False}]

    def test_a_malformed_message_is_ignored(self, console):
        state, _, _, _ = console
        assert [event["type"] for event in state.accept(topics.SYSTEM_MODE.pattern, b"{not json")] == ["mqtt"]
        assert state.snapshot()["room"]["mode"] is None


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


class TestTheDashboard:
    """The pages beyond the conversation: board, series, bookings, mail."""

    def test_every_message_reaches_the_board(self, console):
        state, _, _, clock = console
        state.accept("space/sensor/temp_01/state", b'{"anything": 1}')
        state.accept("space/sensor/temp_01/state", b'{"anything": 2}')
        row = next(row for row in state.snapshot()["topics"] if row["topic"] == "space/sensor/temp_01/state")
        assert row["count"] == 2 and row["payload"] == {"anything": 2}

    def test_a_large_message_reaches_the_board_whole(self, console):
        state, _, _, _ = console
        tools = [{"name": f"tool_{index}", "purpose": "x" * 200} for index in range(40)]
        state.accept("space/assist/catalogue", json.dumps({"tools": tools}).encode())
        row = state.snapshot()["topics"][0]
        assert len(row["payload"]["tools"]) == 40

    def test_a_message_too_long_to_keep_is_cut_as_text(self, console):
        state, _, _, _ = console
        state.accept("space/assist/catalogue", json.dumps({"blob": "x" * 40000}).encode())
        row = state.snapshot()["topics"][0]
        assert isinstance(row["payload"], str) and len(row["payload"]) == 32000

    def test_the_board_knows_what_is_retained(self, console):
        state, _, _, _ = console
        state.accept(topics.SYSTEM_MODE.pattern, b"{}")
        state.accept(topics.AUDIT_VALIDATION.pattern, b"{}")
        retained = {row["topic"]: row["retained"] for row in state.snapshot()["topics"]}
        assert retained == {topics.SYSTEM_MODE.pattern: True, topics.AUDIT_VALIDATION.pattern: False}

    def test_the_board_measures_a_rate(self, console):
        state, _, _, clock = console
        for _ in range(7):
            state.accept("space/estimate/thermal", b"{}")
            clock.advance(5.0)
        row = state.snapshot()["topics"][0]
        assert 11.0 < row["per_minute"] < 13.0

    def test_a_withdrawn_topic_leaves_the_board(self, console):
        state, _, _, _ = console
        state.accept("space/fault/f1", b"{}")
        state.accept("space/fault/f1", b"")
        assert state.snapshot()["topics"] == []

    def test_readings_become_a_series(self, console):
        state, _, _, clock = console
        reading = SensorReading(ts=clock.now(), sensor_id="hum_01", value=48.0, unit=Unit.PERCENT_RH)
        _feed(state, topics.SENSOR_STATE, reading, sensor_id="hum_01")
        assert state.snapshot()["series"]["hum_01"][-1][1] == 48.0

    def test_a_series_is_bounded(self, console):
        state, _, _, clock = console
        for value in range(CONFIG.console.history_kept + 10):
            _feed(state, topics.SENSOR_STATE, SensorReading(ts=clock.now() + value, sensor_id="hum_01", value=float(value % 90), unit=Unit.PERCENT_RH), sensor_id="hum_01")
        assert len(state.snapshot()["series"]["hum_01"]) == CONFIG.console.history_kept

    def test_a_confirmed_booking_is_kept(self, console):
        state, _, _, clock = console
        invocation = ToolInvocation(ts=clock.now(), invocation_id="b9", tool="book_travel", arguments={"kind": "flight", "destination": "Goa", "depart_on": "2026-10-31", "origin": "Hyderabad"}, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 60)
        _feed(state, topics.ASSIST_PROPOSED, invocation)
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id="b9", tool="book_travel", status=ToolStatus.OK, provider="mock_travel", simulated=True, message="Mock booking MOCK-0007 recorded.", detail={"reference": "MOCK-0007", "kind": "flight", "destination": "Goa"}))
        booking = state.snapshot()["bookings"][-1]
        assert booking["reference"] == "MOCK-0007" and booking["origin"] == "Hyderabad"
        assert state.snapshot()["inbox"][-1]["sender"] == "Bookings"

    def test_a_sent_mail_is_kept(self, console):
        state, _, _, clock = console
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id="m1", tool="send_mail", status=ToolStatus.OK, provider="mock_mail", simulated=True, message="Mock email MAIL-0001", detail={"reference": "MAIL-0001", "to": "Anu", "subject": "Late", "body": "Ten minutes."}))
        assert state.snapshot()["sent"][-1]["to"] == "Anu"

    def test_a_diagnosis_lands_in_the_inbox(self, console):
        from src.common.schemas import DiagnosisConfidence, FaultDiagnosis, Hypothesis
        state, _, _, clock = console
        diagnosis = FaultDiagnosis(ts=clock.now(), fault_id="f1", primary_hypothesis=Hypothesis.SENSOR_DROPOUT, confidence=DiagnosisConfidence.HIGH, recommended_mode=Mode.DEGRADED_SENSOR, user_message="The thermometer went quiet.")
        _feed(state, topics.DIAGNOSIS, diagnosis, fault_id="f1")
        assert state.snapshot()["inbox"][-1]["body"] == "The thermometer went quiet."

    def test_an_entry_is_a_write_from_the_console(self, console):
        _, actions, transport, _ = console
        actions.add_entry("Dentist", "2026-10-28T17:00:00", 30)
        sent = ToolInvocation.model_validate_json(transport.published[-1][1])
        assert sent.tool == "schedule_event" and sent.requester is ToolRequester.CONSOLE and sent.arguments["duration_min"] == 30

    def test_a_booking_request_goes_to_the_executor_not_a_provider(self, console):
        _, actions, transport, _ = console
        actions.book("hotel", "Pune", "2026-10-31", nights=2)
        assert transport.published[-1][0] == topics.ASSIST_PROPOSED.pattern

    def test_an_unknown_booking_kind_is_refused(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.book("train", "Pune", "2026-10-31")

    def test_mail_needs_a_recipient(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.send_mail("  ", "hi", "body")

    def test_mail_goes_to_the_executor_as_a_commit(self, console):
        _, actions, transport, _ = console
        actions.send_mail("Anu", "Late", "Ten minutes late.")
        assert ToolInvocation.model_validate_json(transport.published[-1][1]).tool == "send_mail"


class TestOverheard:
    """FR-77, FR-78: what was overheard is shown apart, and kept only on a word from here."""

    def test_a_pencilled_entry_is_shown_as_tentative_with_what_was_heard(self, console):
        state, actions, _, clock = console
        invocation_id = actions.refresh_calendar()
        detail = {
            "count": 2,
            "entries": "ev_0001 2026-10-14T15:00 review; ev_0002 2026-10-15T16:00 dentist",
            "tentative_ids": "ev_0002",
            "heard_ev_0002": "dentist on thursday at four",
        }
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id=invocation_id, tool="get_events", status=ToolStatus.OK, provider="local_calendar", detail=detail))
        firm, pencilled = state.snapshot()["calendar"]["entries"]
        assert firm["tentative"] is False and "heard" not in firm
        assert pencilled["tentative"] is True and pencilled["heard"] == "dentist on thursday at four"

    def test_a_pencilled_entry_is_announced_and_refreshes_the_calendar(self, console):
        state, _, _, clock = console
        result = ToolResult(ts=clock.now(), invocation_id="amb_1", tool="pencil_event", status=ToolStatus.OK, provider="local_calendar", message="Pencilled in dentist at 16:00 on Thursday 15 October, tentatively.", detail={"event_id": "ev_0002", "added": True})
        types = [event["type"] for event in _feed(state, topics.ASSIST_RESULT, result)]
        assert "calendar_stale" in types and "inbox" in types

    def test_a_repeat_mention_is_not_announced(self, console):
        state, _, _, clock = console
        result = ToolResult(ts=clock.now(), invocation_id="amb_2", tool="pencil_event", status=ToolStatus.OK, provider="local_calendar", detail={"event_id": "ev_0002", "added": False})
        assert "inbox" not in [event["type"] for event in _feed(state, topics.ASSIST_RESULT, result)]

    def test_a_profiles_suggestions_are_shown(self, console):
        state, _, _, clock = console
        _feed(state, topics.CONTEXT_PROFILE, ProfileContext(ts=clock.now(), profile_id="ravi", display_name="Ravi", suggestions=("Hates the cold.",)), profile_id="ravi")
        (profile,) = [p for p in state.snapshot()["profiles"] if p["profile_id"] == "ravi"]
        assert profile["suggestions"] == ["Hates the cold."]

    def test_keeping_an_entry_publishes_keep_event(self, console):
        _, actions, transport, _ = console
        actions.keep_entry("ev_0002")
        invocation = ToolInvocation.model_validate_json(transport.published[-1][1])
        assert invocation.tool == "keep_event" and invocation.arguments["event_id"] == "ev_0002"

    def test_settling_a_suggestion_publishes_settle_fact(self, console):
        _, actions, transport, _ = console
        actions.settle_fact("ravi", "Hates the cold.", keep=False)
        invocation = ToolInvocation.model_validate_json(transport.published[-1][1])
        assert invocation.tool == "settle_fact" and invocation.arguments == {"profile_id": "ravi", "fact": "Hates the cold.", "keep": False}

    def test_an_empty_keep_is_refused(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.keep_entry(" ")

    def test_both_are_reachable_over_http(self, console):
        state, actions, transport, _ = console
        lock = threading.Lock()
        hub = Hub()
        http = TestClient(create_app(state, actions, hub, Link(state, hub, lock, Blackboard(CONFIG.mqtt, transport)), lock))
        assert http.post("/api/calendar/keep", json={"event_id": "ev_0002"}).json()["ok"] is True
        assert http.post("/api/memory/settle", json={"profile_id": "ravi", "fact": "Likes tea.", "keep": True}).json()["ok"] is True


class TestStaleHistory:
    """S-07: a retained estimate from before a restart was charted as a climb from 24 to 29 C."""

    def test_a_stale_retained_estimate_is_not_charted(self, console):
        state, _, _, clock = console
        old = ThermalEstimate(ts=clock.now() - 120.0, t_in=24.0, t_pred=24.0, residual=0.0, residual_sigma=0.1, model_confidence=0.9, adaptation="active")
        _feed(state, topics.ESTIMATE_THERMAL, old)
        assert "t_in" not in state.snapshot()["series"]

    def test_it_still_shows_as_the_latest_value(self, console):
        state, _, _, clock = console
        old = ThermalEstimate(ts=clock.now() - 120.0, t_in=24.0, t_pred=24.0, residual=0.0, residual_sigma=0.1, model_confidence=0.9, adaptation="active")
        _feed(state, topics.ESTIMATE_THERMAL, old)
        assert state.snapshot()["room"]["thermal"]["t_in"] == 24.0

    def test_a_current_estimate_is_charted(self, console):
        state, _, _, clock = console
        now = ThermalEstimate(ts=clock.now(), t_in=29.0, t_pred=29.0, residual=0.0, residual_sigma=0.1, model_confidence=0.9, adaptation="active")
        _feed(state, topics.ESTIMATE_THERMAL, now)
        assert state.snapshot()["series"]["t_in"][-1][1] == 29.0

    def test_an_old_setpoint_still_draws_its_step(self, console):
        """On-change series keep their history: the chart extends the last step."""
        from src.common.schemas import TrackingState

        state, _, _, clock = console
        _feed(state, topics.CONTROL_TRACKING, TrackingState(ts=clock.now() - 600.0, setpoint_c=24.0, tracked_c=24.0, band="normal"))
        assert state.snapshot()["series"]["held_c"][-1][1] == 24.0


class TestTheLowFindings:
    """S-08, S-09, S-10, S-14 from the customer check of 2026-10-25."""

    def test_two_messages_are_not_a_rate(self, console):
        """S-08: two health messages seconds apart showed as 19.6/min."""
        state, _, _, clock = console
        state.accept(topics.SYSTEM_MODE.pattern, b"{}")
        clock.advance(3.0)
        state.accept(topics.SYSTEM_MODE.pattern, b"{}")
        assert state.snapshot()["topics"][0]["per_minute"] == 0.0

    def test_a_mail_waiting_says_who_it_is_for(self, console):
        """S-09: the card read "A mock send_mail"."""
        state, _, _, clock = console
        invocation = ToolInvocation(ts=clock.now(), invocation_id="m1", tool="send_mail", arguments={"to": "anu@example.com", "subject": "Dinner", "body": "x"}, requester=ToolRequester.CONSOLE, expires_ts=clock.now() + 300)
        _feed(state, topics.ASSIST_PROPOSED, invocation)
        _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id="m1", tool="send_mail", status=ToolStatus.CONFIRMATION_REQUIRED))
        assert state.snapshot()["waiting"][0]["summary"] == "A mock email to anu@example.com: Dinner"

    def test_the_inbox_survives_a_restart(self, tmp_path):
        """S-10: the inbox was empty after every restart."""
        clock = SimClock()
        path = tmp_path / "inbox.json"
        first = ConsoleState(CONFIG.console, clock, inbox_path=path)
        _feed(first, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.NORMAL, since_ts=clock.now()))
        _feed(first, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.DEGRADED_SENSOR, since_ts=clock.now(), reason="sensor out"))
        again = ConsoleState(CONFIG.console, clock, inbox_path=path)
        assert [item["subject"] for item in again.snapshot()["inbox"]] == [item["subject"] for item in first.snapshot()["inbox"]] != []

    def test_without_a_path_nothing_is_written(self, console, tmp_path):
        state, _, _, clock = console
        _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.NORMAL, since_ts=clock.now()))
        _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.DEGRADED_SENSOR, since_ts=clock.now()))
        assert list(tmp_path.iterdir()) == []

    def test_an_unreadable_inbox_starts_empty(self, tmp_path):
        path = tmp_path / "inbox.json"
        path.write_text("{not json", encoding="utf-8")
        assert ConsoleState(CONFIG.console, SimClock(), inbox_path=path).snapshot()["inbox"] == []

    def test_a_hold_says_what_to_do_next(self, console):
        """S-14: nothing told the customer to press Reset after the repair."""
        state, _, _, clock = console
        _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.NORMAL, since_ts=clock.now()))
        _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.DEGRADED_ACTUATOR, since_ts=clock.now(), reason="the unit is not cooling"))
        assert "press Reset on Faults & control" in state.snapshot()["inbox"][-1]["body"]


def test_a_mode_note_is_written_for_a_person(console):
    """S-14 re-verified live: the inbox opened 'D5_ACTUATOR_NO_RESPONSE on ac'."""
    state, _, _, clock = console
    _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.NORMAL, since_ts=clock.now()))
    _feed(state, topics.SYSTEM_MODE, ModeState(ts=clock.now(), mode=Mode.DEGRADED_ACTUATOR, since_ts=clock.now(), reason="D5_ACTUATOR_NO_RESPONSE on ac"))
    body = state.snapshot()["inbox"][-1]["body"]
    assert "D5_" not in body and body.startswith("The air conditioner isn't cooling the room")


def test_a_retained_estimate_from_just_before_start_is_not_charted(console):
    """S-07 re-verified live: only 5 s old, it still drew a climb from the old room."""
    state, _, _, clock = console
    clock.advance(10.0)
    _feed(state, topics.ESTIMATE_THERMAL, ThermalEstimate(ts=clock.now() - 15.0, t_in=23.2, t_pred=23.2, residual=0.0, residual_sigma=0.1, model_confidence=0.9, adaptation="active"))
    assert "t_in" not in state.snapshot()["series"]
