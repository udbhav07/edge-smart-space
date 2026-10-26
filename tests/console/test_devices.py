"""Devices in the console (FR-91): shown as they announce themselves, called
through the executor like any tool, and a music file kept by name only."""

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.devices import DeviceAvailability, DeviceState, air_conditioner_manifest
from src.common.mqtt_client import Blackboard
from src.common.tools import ToolInvocation, ToolRequester, ToolResult, ToolStatus
from src.console.actions import ActionRefused, ConsoleActions
from src.console.server import Hub, Link, create_app
from src.console.state import ConsoleState
from src.devices.fan import manifest as fan_manifest

BASE = load_config(Path("config/default.yaml"))


@pytest.fixture(name="console")
def _console(tmp_path):
    music = BASE.actuators.music.model_copy(update={"uploads_path": str(tmp_path / "uploads"), "max_upload_mb": 0.001})
    config = BASE.model_copy(update={"actuators": BASE.actuators.model_copy(update={"music": music})})
    clock = SimClock(start_epoch_s=1000.0)
    transport = LoopbackTransport()
    board = Blackboard(config.mqtt, transport)
    transport.attach(board)
    state = ConsoleState(config.console, clock)
    actions = ConsoleActions(config, clock, board, state)
    return state, actions, transport, clock, tmp_path


def _feed(state, spec, message, **parameters):
    return state.accept(spec.format(**parameters), message.model_dump_json().encode())


def _announce(state, clock, online=True):
    _feed(state, topics.DEVICE_MANIFEST, fan_manifest(BASE.actuators.fan, clock.now()), device_id="fan")
    _feed(state, topics.DEVICE_AVAILABILITY, DeviceAvailability(ts=clock.now(), device_id="fan", online=online), device_id="fan")


def _proposed(transport):
    return [ToolInvocation.model_validate_json(p) for t, p, _, _ in transport.published if t == topics.ASSIST_PROPOSED.pattern]


class TestWhatItShows:
    def test_an_announced_device_is_listed_with_its_capabilities(self, console):
        state, _, _, clock, _ = console
        _announce(state, clock)
        (device,) = state.snapshot()["devices"]
        assert device["device_id"] == "fan" and device["online"] is True
        assert {c["name"] for c in device["manifest"]["capabilities"]} >= {"turn_on", "set_speed"}

    def test_its_state_follows_the_board(self, console):
        state, _, _, clock, _ = console
        _announce(state, clock)
        _feed(state, topics.DEVICE_STATE, DeviceState(ts=clock.now(), device_id="fan", values={"on": True, "speed": 2}, simulated=True), device_id="fan")
        assert state.snapshot()["devices"][0]["state"] == {"on": True, "speed": 2}

    def test_going_offline_is_shown(self, console):
        state, _, _, clock, _ = console
        _announce(state, clock, online=False)
        assert state.snapshot()["devices"][0]["online"] is False

    def test_state_without_a_manifest_shows_nothing(self, console):
        state, _, _, clock, _ = console
        _feed(state, topics.DEVICE_STATE, DeviceState(ts=clock.now(), device_id="ghost", values={"x": 1}, simulated=True), device_id="ghost")
        assert state.snapshot()["devices"] == []


class TestCalling:
    def test_a_call_goes_through_the_executor_as_its_tool(self, console):
        state, actions, transport, clock, _ = console
        _announce(state, clock)
        actions.device("fan", "set_speed", {"speed": 2})
        (invocation,) = _proposed(transport)
        assert invocation.tool == "fan_set_speed" and invocation.arguments == {"speed": 2} and invocation.requester is ToolRequester.CONSOLE

    def test_its_answer_reaches_the_page(self, console):
        state, actions, _, clock, _ = console
        _announce(state, clock)
        invocation_id = actions.device("fan", "turn_on", {})
        events = _feed(state, topics.ASSIST_RESULT, ToolResult(ts=clock.now(), invocation_id=invocation_id, tool="fan_turn_on", status=ToolStatus.OK, message="The fan is on.", provider="fan", simulated=True))
        assert [e for e in events if e["type"] == "device_result"][0]["data"]["message"] == "The fan is on."

    def test_an_unknown_device_is_refused(self, console):
        with pytest.raises(ActionRefused, match="no device called kettle"):
            console[1].device("kettle", "boil", {})

    def test_an_offline_device_is_refused(self, console):
        state, actions, _, clock, _ = console
        _announce(state, clock, online=False)
        with pytest.raises(ActionRefused, match="offline"):
            actions.device("fan", "turn_on", {})

    def test_an_unknown_capability_is_refused(self, console):
        state, actions, _, clock, _ = console
        _announce(state, clock)
        with pytest.raises(ActionRefused, match="can't fly"):
            actions.device("fan", "fly", {})

    def test_the_air_conditioner_is_never_called_as_a_device(self, console):
        state, actions, transport, clock, _ = console
        _feed(state, topics.DEVICE_MANIFEST, air_conditioner_manifest(clock.now(), True), device_id="ac")
        with pytest.raises(ActionRefused, match="temperature controls"):
            actions.device("ac", "set_temperature", {"target_c": 20})
        assert _proposed(transport) == []


class TestMusicUpload:
    def _music_online(self, state, clock):
        from src.devices.music import manifest as music_manifest

        _feed(state, topics.DEVICE_MANIFEST, music_manifest(BASE.actuators.music, clock.now()), device_id="music")

    def test_a_file_is_kept_and_played_by_name(self, console):
        state, actions, transport, clock, folder = console
        self._music_online(state, clock)
        actions.upload_music("My Song.mp3", b"ID3data")
        assert (folder / "uploads" / "My Song.mp3").read_bytes() == b"ID3data"
        (invocation,) = _proposed(transport)
        assert invocation.tool == "music_play" and invocation.arguments == {"file": "My Song.mp3"}

    def test_a_path_in_the_name_is_dropped(self, console):
        state, actions, _, clock, folder = console
        self._music_online(state, clock)
        actions.upload_music("../../etc/evil.mp3", b"x")
        assert (folder / "uploads" / "evil.mp3").exists() and not (folder / "etc").exists()

    def test_a_file_that_is_not_music_is_refused(self, console):
        with pytest.raises(ActionRefused, match="must be one of"):
            console[1].upload_music("setup.exe", b"MZ")

    def test_an_empty_file_is_refused(self, console):
        with pytest.raises(ActionRefused, match="empty"):
            console[1].upload_music("a.mp3", b"")

    def test_a_file_too_large_is_refused(self, console):
        with pytest.raises(ActionRefused, match="larger than"):
            console[1].upload_music("a.mp3", b"x" * 2000)


class TestEndpoints:
    @pytest.fixture(name="http")
    def _http(self, console):
        state, actions, transport, clock, folder = console
        _announce(state, clock)
        lock = threading.Lock()
        hub = Hub()
        return TestClient(create_app(state, actions, hub, Link(state, hub, lock, Blackboard(BASE.mqtt, transport)), lock)), transport

    def test_a_device_call(self, http):
        client, transport = http
        assert client.post("/api/devices/fan/set_speed", json={"arguments": {"speed": 1}}).json()["ok"] is True
        assert _proposed(transport)[-1].tool == "fan_set_speed"

    def test_a_refused_device_call_is_a_400(self, http):
        client, _ = http
        assert client.post("/api/devices/kettle/boil", json={}).status_code == 400

    def test_an_upload_is_the_raw_body(self, http, console):
        client, _ = http
        state, _, _, clock, folder = console
        from src.devices.music import manifest as music_manifest

        _feed(state, topics.DEVICE_MANIFEST, music_manifest(BASE.actuators.music, clock.now()), device_id="music")
        response = client.post("/api/music/upload", params={"name": "tune.mp3"}, content=b"abc", headers={"content-type": "application/octet-stream"})
        assert response.json()["ok"] is True and (folder / "uploads" / "tune.mp3").read_bytes() == b"abc"
