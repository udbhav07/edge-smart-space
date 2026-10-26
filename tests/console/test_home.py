"""Home's controls, energy today, the voice settings and Google sign-in (FR-80 to FR-86).

Each control publishes a preference the safety gate still decides on; none
of them commands the actuator. The sign-in writes a token file and nothing
about it reaches the board.
"""

import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import GoogleConfig, load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import (
    Comfort,
    GoogleStatus,
    PreferenceHint,
    SensorReading,
    SpeechSettings,
    TariffBand,
    TariffState,
    Unit,
)
from src.common.tools import ToolResult, ToolStatus
from src.console.actions import ActionRefused, ConsoleActions
from src.console.google_signin import GoogleSignIn, SignInError, _email_of
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


def _last(transport, spec, model):
    for topic, payload, _, _ in reversed(transport.published):
        if topic == spec.pattern:
            return model.model_validate_json(payload)
    raise AssertionError(f"nothing published on {spec.pattern}")


def _watts(state, clock, value, after_s):
    clock.advance(after_s)
    _feed(state, topics.SENSOR_STATE, SensorReading(ts=clock.now(), sensor_id="pwr_01", value=value, unit=Unit.WATT), sensor_id="pwr_01")


class TestClimateControls:
    def test_a_typed_target_is_marked_typed(self, console):
        _, actions, transport, _ = console
        actions.set_target(23.0)
        assert _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint).typed is True

    def test_warmer_publishes_a_direction_not_a_number(self, console):
        _, actions, transport, _ = console
        actions.comfort("warmer")
        hint = _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint)
        assert hint.comfort is Comfort.WARMER and hint.target_c is None

    def test_an_unknown_direction_is_refused(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.comfort("sideways")

    def test_unchanged_is_not_a_direction(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.comfort("unchanged")

    def test_back_to_automatic_releases_the_hold(self, console):
        _, actions, transport, _ = console
        actions.automatic()
        hint = _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint)
        assert hint.release is True and hint.target_c is None

    def test_cooling_off_asks_for_the_upper_bound_for_that_long(self, console):
        _, actions, transport, _ = console
        actions.cooling_off(3)
        hint = _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint)
        assert hint.target_c == CONFIG.console.cooling_off_c and hint.hold_s == 3 * 3600.0

    @pytest.mark.parametrize("hours", [0, -1, 24.5])
    def test_cooling_off_out_of_range_is_refused(self, console, hours):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.cooling_off(hours)

    def test_cooling_off_for_a_whole_day_is_allowed(self, console):
        _, actions, transport, _ = console
        actions.cooling_off(24)
        assert _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint).hold_s == 86400.0

    def test_nothing_here_commands_the_actuator(self, console):
        _, actions, transport, _ = console
        actions.comfort("cooler")
        actions.automatic()
        actions.cooling_off(1)
        assert all(not topic.startswith("space/actuator") for topic, _, _, _ in transport.published)


class TestSpeechSettings:
    def test_a_change_keeps_the_rest(self, console):
        state, actions, transport, clock = console
        _feed(state, topics.SPEECH_SETTINGS, SpeechSettings(ts=clock.now(), speak_typed=True, volume=0.5))
        actions.speech_settings({"listening": False})
        settings = _last(transport, topics.SPEECH_SETTINGS, SpeechSettings)
        assert settings.listening is False and settings.speak_typed is True and settings.volume == 0.5

    def test_with_none_seen_the_defaults_are_the_base(self, console):
        _, actions, transport, _ = console
        actions.speech_settings({"speak_alerts": False})
        settings = _last(transport, topics.SPEECH_SETTINGS, SpeechSettings)
        assert settings.speak_alerts is False and settings.listening is True

    def test_an_unknown_setting_is_refused(self, console):
        _, actions, _, _ = console
        with pytest.raises(ActionRefused):
            actions.speech_settings({"shout": True})

    def test_the_settings_are_retained(self, console):
        _, actions, transport, _ = console
        actions.speech_settings({"volume": 0.3})
        assert [retain for topic, _, _, retain in transport.published if topic == topics.SPEECH_SETTINGS.pattern] == [True]

    def test_settings_seen_reach_the_snapshot(self, console):
        state, _, _, clock = console
        _feed(state, topics.SPEECH_SETTINGS, SpeechSettings(ts=clock.now(), listening=False))
        assert state.snapshot()["speech_settings"]["listening"] is False


class TestEnergyToday:
    def test_one_reading_counts_nothing(self, console):
        state, _, _, clock = console
        _watts(state, clock, 1000.0, 0)
        assert state.snapshot()["energy"]["kwh"] == 0.0

    def test_a_steady_kilowatt_for_a_minute(self, console):
        state, _, _, clock = console
        _watts(state, clock, 1000.0, 0)
        _watts(state, clock, 1000.0, 60)
        energy = state.snapshot()["energy"]
        assert energy["kwh"] == pytest.approx(1000.0 * 60 / 3_600_000, abs=1e-4)
        assert energy["cost"] == pytest.approx(energy["kwh"] * CONFIG.console.price_per_kwh["normal"], abs=1e-3)

    def test_the_peak_band_is_priced_at_peak(self, console):
        state, _, _, clock = console
        _feed(state, topics.TARIFF_STATE, TariffState(ts=clock.now(), band=TariffBand.PEAK, next_transition_ts=clock.now() + 3600))
        _watts(state, clock, 3600.0, 0)
        _watts(state, clock, 3600.0, 100)
        energy = state.snapshot()["energy"]
        assert energy["cost"] == pytest.approx(0.1 * CONFIG.console.price_per_kwh["peak"], abs=1e-3)

    def test_a_long_silence_is_not_bridged(self, console):
        state, _, _, clock = console
        _watts(state, clock, 1000.0, 0)
        _watts(state, clock, 1000.0, 3600)
        assert state.snapshot()["energy"]["kwh"] == 0.0

    def test_a_reading_out_of_order_counts_nothing(self, console):
        state, _, _, clock = console
        _watts(state, clock, 1000.0, 60)
        clock_then = clock.now()
        _feed(state, topics.SENSOR_STATE, SensorReading(ts=clock_then - 30, sensor_id="pwr_01", value=1000.0, unit=Unit.WATT), sensor_id="pwr_01")
        assert state.snapshot()["energy"]["kwh"] == 0.0

    def test_a_new_day_starts_from_zero(self, console):
        state, _, _, clock = console
        _watts(state, clock, 1000.0, 0)
        _watts(state, clock, 1000.0, 60)
        clock.advance(86400)
        _watts(state, clock, 1000.0, 0)
        assert state.snapshot()["energy"]["kwh"] == 0.0

    def test_other_sensors_count_nothing(self, console):
        state, _, _, clock = console
        for after in (0, 60):
            clock.advance(after)
            _feed(state, topics.SENSOR_STATE, SensorReading(ts=clock.now(), sensor_id="temp_01", value=25.0, unit=Unit.CELSIUS), sensor_id="temp_01")
        assert state.snapshot()["energy"]["kwh"] == 0.0


class TestWhatGoogleShows:
    def test_the_status_reaches_the_snapshot(self, console):
        state, _, _, clock = console
        _feed(state, topics.ASSIST_GOOGLE, GoogleStatus(ts=clock.now(), configured=True, connected=True, account="me@example.com", outbox=2))
        google = state.snapshot()["google"]
        assert google["account"] == "me@example.com" and google["outbox"] == 2

    def test_a_queued_mail_is_shown_queued(self, console):
        state, _, _, clock = console
        result = ToolResult(
            ts=clock.now(), invocation_id="m1", tool="send_mail", status=ToolStatus.OK, provider="gmail",
            simulated=False, detail={"to": "me@example.com", "subject": "hi", "body": "there", "queued": True},
        )
        _feed(state, topics.ASSIST_RESULT, result)
        assert state.snapshot()["sent"][-1]["queued"] is True

    def test_a_google_entry_is_marked(self, console):
        state, actions, _, clock = console
        invocation_id = actions.refresh_calendar()
        result = ToolResult(
            ts=clock.now(), invocation_id=invocation_id, tool="get_events", status=ToolStatus.OK, provider="local_calendar",
            detail={"count": 1, "entries": "evt_9 2026-10-14T15:00:00 dentist", "google_ids": "evt_9"},
        )
        _feed(state, topics.ASSIST_RESULT, result)
        assert state.snapshot()["calendar"]["entries"][0]["google"] is True


class _FakeFlow:
    def __init__(self, id_token=None, fail=False):
        self.id_token = id_token
        self.fail = fail
        self.credentials = self

    def authorization_url(self, **_):
        return "https://accounts.google.com/o/oauth2/auth?x=1", "state-1"

    def fetch_token(self, code):
        if self.fail:
            raise ValueError("invalid_grant")
        assert code == "the-code"

    def to_json(self):
        return json.dumps({"token": "t", "refresh_token": "r"})


def _token_for(email):
    import base64

    payload = base64.urlsafe_b64encode(json.dumps({"email": email}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


@pytest.fixture(name="signin")
def _signin(tmp_path, monkeypatch):
    secret = tmp_path / "client_secret.json"
    secret.write_text("{}", encoding="utf-8")
    config = GoogleConfig(client_secret_path=str(secret), token_path=str(tmp_path / "token.json"))
    signin = GoogleSignIn(config)
    flow = _FakeFlow(id_token=_token_for("me@example.com"))
    monkeypatch.setattr(signin, "_flow", lambda: flow)
    return signin, flow, tmp_path


class TestGoogleSignIn:
    def test_without_a_client_secret_it_says_what_to_do(self, tmp_path):
        signin = GoogleSignIn(GoogleConfig(client_secret_path=str(tmp_path / "absent.json"), token_path=str(tmp_path / "t.json")))
        with pytest.raises(SignInError, match="README"):
            signin.start()

    def test_finishing_writes_the_token_and_the_address(self, signin):
        google, _, folder = signin
        google.start()
        assert google.finish("state-1", "the-code") == "me@example.com"
        assert json.loads((folder / "token.json").read_text())["refresh_token"] == "r"
        assert json.loads((folder / "account.json").read_text()) == {"email": "me@example.com"}

    def test_an_unknown_state_is_refused(self, signin):
        google, _, _ = signin
        google.start()
        with pytest.raises(SignInError, match="expired"):
            google.finish("someone-elses", "the-code")

    def test_a_state_finishes_only_once(self, signin):
        google, _, _ = signin
        google.start()
        google.finish("state-1", "the-code")
        with pytest.raises(SignInError):
            google.finish("state-1", "the-code")

    def test_a_refused_code_writes_nothing(self, signin):
        google, flow, folder = signin
        flow.fail = True
        google.start()
        with pytest.raises(SignInError, match="did not accept"):
            google.finish("state-1", "the-code")
        assert not (folder / "token.json").exists()

    def test_disconnect_forgets_both_files(self, signin):
        google, _, folder = signin
        google.start()
        google.finish("state-1", "the-code")
        assert google.disconnect() is True
        assert not (folder / "token.json").exists() and not (folder / "account.json").exists()

    def test_disconnect_with_nothing_connected(self, signin):
        google, _, _ = signin
        assert google.disconnect() is False

    @pytest.mark.parametrize("token", [None, "", "only-one-part", "a.!!!.c"])
    def test_an_unreadable_id_token_gives_no_address(self, token):
        assert _email_of(token) == ""


class TestTheNewEndpoints:
    @pytest.fixture(name="client")
    def _client(self, console, signin):
        state, actions, transport, _ = console
        lock = threading.Lock()
        hub = Hub()
        link = Link(state, hub, lock, Blackboard(CONFIG.mqtt, transport))
        return TestClient(create_app(state, actions, hub, link, lock, signin=signin[0]), follow_redirects=False), transport

    def test_comfort(self, client):
        http, transport = client
        assert http.post("/api/comfort", json={"direction": "cooler"}).json()["ok"] is True
        assert _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint).comfort is Comfort.COOLER

    def test_a_bad_direction_is_a_400(self, client):
        http, _ = client
        assert http.post("/api/comfort", json={"direction": "up"}).status_code == 400

    def test_automatic(self, client):
        http, transport = client
        assert http.post("/api/automatic", json={}).json()["ok"] is True
        assert _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint).release is True

    def test_cooling_off(self, client):
        http, transport = client
        assert http.post("/api/cooling-off", json={"hours": 2}).json()["ok"] is True
        assert _last(transport, topics.CONTEXT_PREFERENCE, PreferenceHint).hold_s == 7200.0

    def test_speech_settings(self, client):
        http, transport = client
        assert http.post("/api/speech/settings", json={"listening": False}).json()["ok"] is True
        assert _last(transport, topics.SPEECH_SETTINGS, SpeechSettings).listening is False

    def test_a_bad_speech_setting_is_a_400(self, client):
        http, _ = client
        assert http.post("/api/speech/settings", json={"volume": 7}).status_code == 400

    def test_connect_redirects_to_google(self, client):
        http, _ = client
        response = http.get("/api/google/connect")
        assert response.status_code == 302 and response.headers["location"].startswith("https://accounts.google.com/")

    def test_the_callback_connects(self, client, signin):
        http, _ = client
        http.get("/api/google/connect")
        response = http.get("/api/google/callback", params={"state": "state-1", "code": "the-code"})
        assert response.status_code == 200 and "me@example.com" in response.text
        assert (signin[2] / "token.json").exists()

    def test_a_cancelled_sign_in_connects_nothing(self, client, signin):
        http, _ = client
        response = http.get("/api/google/callback", params={"error": "access_denied"})
        assert response.status_code == 400 and not (signin[2] / "token.json").exists()

    def test_nothing_about_the_token_is_published(self, client):
        http, transport = client
        http.get("/api/google/connect")
        http.get("/api/google/callback", params={"state": "state-1", "code": "the-code"})
        assert all(b"refresh_token" not in payload for _, payload, _, _ in transport.published)

    def test_disconnect(self, client):
        http, _ = client
        assert http.post("/api/google/disconnect").json()["ok"] is True

    def test_the_status_endpoint(self, client):
        http, _ = client
        assert http.get("/api/google").json()["configured"] is True
