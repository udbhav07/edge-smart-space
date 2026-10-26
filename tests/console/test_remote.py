"""Talking to the room from a phone away from it (FR-94).

What matters: a recording is decoded and transcribed and goes on as a remote
request, never kept; the reply is matched back to the device that asked and
spoken for it to play; and the console's own certificate covers every address
a phone might use.
"""

import datetime as dt
import io
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import AssistantReply, ConsoleUtterance, Intent
from src.console.actions import ConsoleActions
from src.console.remote_voice import RemoteVoice, RemoteVoiceUnavailable
from src.console.server import Hub, Link, create_app
from src.console.state import ConsoleState
from src.console.tls import ensure_certificate

CONFIG = load_config(Path("config/default.yaml"))


def _wav(seconds: float, rate: int = 16000) -> bytes:
    samples = (np.sin(2 * np.pi * 220 * np.arange(int(rate * seconds)) / rate) * 8000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes(samples.tobytes())
    return buffer.getvalue()


class _Transcriber:
    def __init__(self, text="make it cooler"):
        self.text, self.heard = text, []

    def transcribe_audio(self, audio):
        self.heard.append(audio.size)
        return self.text


class _Speaker:
    def render(self, text, volume=1.0):
        return b"\x00\x01" * 2205, 22050


def _voice(**kwargs):
    return RemoteVoice(CONFIG.speech, max_seconds=30, transcriber=kwargs.get("transcriber", _Transcriber()), speaker=_Speaker())


class TestHearing:
    def test_a_recording_is_decoded_to_16k_and_transcribed(self):
        transcriber = _Transcriber()
        assert _voice(transcriber=transcriber).transcribe(_wav(2.0, rate=44100)) == "make it cooler"
        assert abs(transcriber.heard[0] - 32000) < 400  # two seconds at 16 kHz, whatever it was recorded at

    def test_nothing_recorded_is_said_plainly(self):
        with pytest.raises(ValueError, match="Nothing was recorded"):
            _voice().transcribe(b"")

    def test_too_short_is_said_plainly(self):
        with pytest.raises(ValueError, match="too short"):
            _voice().transcribe(_wav(0.1))

    def test_too_long_is_refused(self):
        with pytest.raises(ValueError, match="keep it under 30 s"):
            _voice().transcribe(_wav(31.0))

    def test_an_unreadable_recording_is_said_plainly(self):
        with pytest.raises(ValueError, match="could not be read"):
            _voice().transcribe(b"this is not audio at all" * 10)


class TestSpeaking:
    def test_the_reply_is_a_wav_the_device_can_play(self):
        audio = _voice().speak("Sure.")
        with wave.open(io.BytesIO(audio)) as file:
            assert (file.getframerate(), file.getnchannels(), file.getnframes()) == (22050, 1, 2205)

    def test_no_voice_is_said_plainly(self, monkeypatch):
        voice = RemoteVoice(CONFIG.speech.model_copy(update={"voice_model_path": "no/such/voice.onnx"}), 30, transcriber=_Transcriber())
        with pytest.raises(RemoteVoiceUnavailable, match="voice is not available"):
            voice.speak("Sure.")


class TestCertificate:
    NOW = dt.datetime(2026, 10, 27, tzinfo=dt.timezone.utc)

    def _sans(self, path):
        from cryptography import x509

        certificate = x509.load_pem_x509_certificate(path.read_bytes())
        names = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        return set(names.get_values_for_type(x509.DNSName)), {str(a) for a in names.get_values_for_type(x509.IPAddress)}

    def test_it_names_every_address(self, tmp_path):
        cert, key = ensure_certificate(tmp_path, {"localhost", "room"}, {"127.0.0.1", "192.168.1.20"}, self.NOW)
        assert self._sans(cert) == ({"localhost", "room"}, {"127.0.0.1", "192.168.1.20"}) and key.exists()

    def test_it_is_kept_while_it_still_fits(self, tmp_path):
        cert, _ = ensure_certificate(tmp_path, {"localhost"}, {"127.0.0.1"}, self.NOW)
        before = cert.read_bytes()
        ensure_certificate(tmp_path, {"localhost"}, {"127.0.0.1"}, self.NOW + dt.timedelta(days=10))
        assert cert.read_bytes() == before

    def test_a_new_address_makes_a_new_one(self, tmp_path):
        cert, _ = ensure_certificate(tmp_path, {"localhost"}, {"127.0.0.1"}, self.NOW)
        ensure_certificate(tmp_path, {"localhost"}, {"127.0.0.1", "10.0.0.5"}, self.NOW)
        assert "10.0.0.5" in self._sans(cert)[1]

    def test_one_near_expiry_is_made_again(self, tmp_path):
        cert, _ = ensure_certificate(tmp_path, {"localhost"}, {"127.0.0.1"}, self.NOW)
        before = cert.read_bytes()
        ensure_certificate(tmp_path, {"localhost"}, {"127.0.0.1"}, self.NOW + dt.timedelta(days=790))
        assert cert.read_bytes() != before

    def test_the_key_is_never_in_the_certificate(self, tmp_path):
        cert, _ = ensure_certificate(tmp_path, {"localhost"}, {"127.0.0.1"}, self.NOW)
        assert b"PRIVATE KEY" not in cert.read_bytes()


@pytest.fixture(name="console")
def _console():
    clock = SimClock(start_epoch_s=1000.0)
    transport = LoopbackTransport()
    board = Blackboard(CONFIG.mqtt, transport)
    transport.attach(board)
    state = ConsoleState(CONFIG.console, clock)
    actions = ConsoleActions(CONFIG, clock, board, state)
    return state, actions, transport, clock


def _reply(clock, transcript, remote=True):
    return AssistantReply(ts=clock.now(), transcript=transcript, intent=Intent.SERVICE, reply="Where to?", remote=remote)


class TestTheReplyGoesBack:
    def test_a_remote_reply_is_matched_to_the_device_that_asked(self, console):
        state, actions, _, clock = console
        utterance = actions.say("Book me a flight", remote=True)
        events = state.accept(topics.CONTEXT_REPLY.pattern, _reply(clock, "book me  a flight").model_dump_json().encode())
        answer = [e for e in events if e["type"] == "remote_reply"][0]["data"]
        assert answer["utterance_id"] == utterance.utterance_id and answer["reply"] == "Where to?"

    def test_a_reply_in_the_room_is_not_sent_to_a_device(self, console):
        state, actions, _, clock = console
        actions.say("book me a flight", remote=True)
        events = state.accept(topics.CONTEXT_REPLY.pattern, _reply(clock, "book me a flight", remote=False).model_dump_json().encode())
        assert not any(e["type"] == "remote_reply" for e in events)

    def test_a_reply_nobody_is_waiting_for_goes_nowhere(self, console):
        state, _, _, clock = console
        events = state.accept(topics.CONTEXT_REPLY.pattern, _reply(clock, "something else").model_dump_json().encode())
        assert not any(e["type"] == "remote_reply" for e in events)

    def test_a_remote_request_is_published_as_remote(self, console):
        _, actions, transport, _ = console
        actions.say("make it cooler", remote=True)
        payload = [p for t, p, _, _ in transport.published if t == topics.CONTEXT_UTTERANCE.pattern][-1]
        assert ConsoleUtterance.model_validate_json(payload).remote is True


class TestEndpoints:
    @pytest.fixture(name="http")
    def _http(self, console):
        state, actions, transport, _ = console
        lock = threading.Lock()
        hub = Hub()
        app = create_app(state, actions, hub, Link(state, hub, lock, Blackboard(CONFIG.mqtt, transport)), lock, remote_voice=_voice(), talk_max_bytes=2_000_000)
        return TestClient(app), transport

    def _utterances(self, transport):
        return [ConsoleUtterance.model_validate_json(p) for t, p, _, _ in transport.published if t == topics.CONTEXT_UTTERANCE.pattern]

    def test_talking_publishes_a_remote_request_and_says_what_was_heard(self, http):
        client, transport = http
        body = client.post("/api/talk", content=_wav(1.5), headers={"content-type": "audio/wav"}).json()
        assert body["ok"] and body["result"]["transcript"] == "make it cooler"
        assert self._utterances(transport)[-1].remote is True and self._utterances(transport)[-1].text == "make it cooler"

    def test_no_audio_is_published(self, http):
        client, transport = http
        client.post("/api/talk", content=_wav(1.5), headers={"content-type": "audio/wav"})
        assert all(b"RIFF" not in payload for _, payload, _, _ in transport.published)

    def test_a_bad_recording_is_a_400(self, http):
        client, _ = http
        assert client.post("/api/talk", content=b"", headers={"content-type": "audio/webm"}).status_code == 400

    def test_too_large_is_a_413(self, http):
        client, _ = http
        assert client.post("/api/talk", content=b"x" * 2_000_001, headers={"content-type": "audio/webm"}).status_code == 413

    def test_typing_on_a_device_is_remote_too(self, http):
        client, transport = http
        assert client.post("/api/talk/text", json={"text": "what's on today"}).json()["ok"] is True
        assert self._utterances(transport)[-1].remote is True

    def test_the_reply_comes_back_as_audio(self, http):
        client, _ = http
        response = client.post("/api/voice/speak", json={"text": "Where to?"})
        assert response.status_code == 200 and response.headers["content-type"] == "audio/wav" and response.content[:4] == b"RIFF"

    def test_without_a_remote_voice_it_says_so(self, console):
        state, actions, transport, _ = console
        lock = threading.Lock()
        hub = Hub()
        client = TestClient(create_app(state, actions, hub, Link(state, hub, lock, Blackboard(CONFIG.mqtt, transport)), lock))
        assert client.post("/api/talk", content=_wav(1.0)).status_code == 503
