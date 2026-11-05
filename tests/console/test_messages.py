"""Messages home from someone away (FR-95), as the console takes them in.

What matters: a recording is kept as a file in the messages directory and
only its words and file name reach the board; the sender's page and the
room's card follow its retained status; its recordings can be fetched only
by a message id, never a path; and nothing is kept past its time.
"""

import io
import os
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import MessagesConfig, load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import HomeMessage, MessageState, MessageStatus
from src.console.actions import ConsoleActions
from src.console.messages import MessageBox
from src.console.remote_voice import RemoteVoice, RemoteVoiceUnavailable
from src.console.server import Hub, Link, create_app
from src.console.state import ConsoleState

CONFIG = load_config(Path("config/default.yaml"))
DAY_S = 86400.0


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
    def __init__(self, text="Call me back, Amma"):
        self.text = text

    def transcribe_audio(self, audio):
        if self.text is None:
            raise RemoteVoiceUnavailable("no Whisper here")
        return self.text


def _voice(text="Call me back, Amma"):
    return RemoteVoice(CONFIG.speech, max_seconds=30, transcriber=_Transcriber(text), speaker=None)


@pytest.fixture(name="room")
def _room(tmp_path):
    clock = SimClock(start_epoch_s=1_800_000_000.0)
    transport = LoopbackTransport()
    board = Blackboard(CONFIG.mqtt, transport)
    transport.attach(board)
    state = ConsoleState(CONFIG.console, clock)
    config = MessagesConfig(directory=str(tmp_path), max_seconds=10, max_bytes=500_000, max_kept=3, keep_days=7)
    box = MessageBox(config, clock, board, _voice())
    return box, state, transport, clock, tmp_path


def _published(transport, spec):
    return [(t, p, retain) for t, p, _, retain in transport.published if t.startswith(spec.pattern.split("{")[0])]


def _fold(state, transport):
    """What the console would have heard back from the broker."""
    for topic, payload, _, _ in list(transport.published):
        state.accept(topic, payload)


class TestSending:
    def test_the_recording_is_kept_and_the_room_is_told(self, room):
        box, _, transport, _, folder = room
        status = box.send("Ravi", _wav(2.0, rate=44100))
        assert (folder / f"{status.message_id}.wav").is_file()
        incoming = [HomeMessage.model_validate_json(p) for _, p, _ in _published(transport, topics.MESSAGE_INCOMING)]
        assert incoming[-1].audio_file == f"{status.message_id}.wav" and incoming[-1].sender == "Ravi"
        assert incoming[-1].transcript == "Call me back, Amma" and abs(incoming[-1].duration_s - 2.0) < 0.05

    def test_it_is_kept_as_16k_mono_whatever_was_recorded(self, room):
        box, _, _, _, folder = room
        status = box.send("Ravi", _wav(1.0, rate=48000))
        with wave.open(str(folder / f"{status.message_id}.wav"), "rb") as file:
            assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 16000)

    def test_its_status_starts_as_sent_and_is_retained(self, room):
        box, _, transport, _, _ = room
        status = box.send("Ravi", _wav(1.0))
        (topic, payload, retain), = _published(transport, topics.MESSAGE_STATUS)
        assert retain and topic.endswith(status.message_id) and MessageStatus.model_validate_json(payload).state is MessageState.SENT

    def test_no_audio_reaches_the_board(self, room):
        box, _, transport, _, _ = room
        box.send("Ravi", _wav(3.0))
        assert all(b"RIFF" not in payload and len(payload) < 4000 for _, payload, _, _ in transport.published)

    def test_a_sender_is_needed(self, room):
        box, *_ = room
        with pytest.raises(ValueError, match="who the message is from"):
            box.send("   ", _wav(1.0))

    def test_too_long_is_refused_and_nothing_kept(self, room):
        box, _, transport, _, folder = room
        with pytest.raises(ValueError, match="keep it under 10 s"):
            box.send("Ravi", _wav(11.0))
        assert not list(folder.iterdir()) and transport.published == []

    def test_without_whisper_it_is_still_sent_without_words(self, room, tmp_path):
        _, _, transport, clock, _ = room
        box = MessageBox(MessagesConfig(directory=str(tmp_path / "x")), clock, Blackboard(CONFIG.mqtt, transport), _voice(text=None))
        assert box.send("Ravi", _wav(1.0)).transcript == ""


class TestTheRoomsView:
    def test_the_console_lists_messages_newest_first(self, room):
        box, state, transport, clock, _ = room
        first = box.send("Ravi", _wav(1.0))
        clock.advance(60.0)
        second = box.send("Meera", _wav(1.0))
        _fold(state, transport)
        assert [m["message_id"] for m in state.snapshot()["messages"]] == [second.message_id, first.message_id]

    def test_a_reply_is_shown_and_said_in_the_activity(self, room):
        box, state, transport, clock, _ = room
        sent = box.send("Ravi", _wav(1.0))
        _fold(state, transport)
        replied = sent.model_copy(update={"state": MessageState.REPLIED, "reply_transcript": "Okay, tonight", "reply_file": f"{sent.message_id}-reply.wav", "replied_ts": clock.now()})
        events = state.accept(topics.MESSAGE_STATUS.format(message_id=sent.message_id), replied.model_dump_json().encode())
        assert state.message(sent.message_id)["reply_transcript"] == "Okay, tonight"
        assert any(e["type"] == "step" and "Answered Ravi" in e["data"]["text"] for e in events)

    def test_a_withdrawn_status_leaves_the_list(self, room):
        box, state, transport, _, _ = room
        sent = box.send("Ravi", _wav(1.0))
        _fold(state, transport)
        state.accept(topics.MESSAGE_STATUS.format(message_id=sent.message_id), b"")
        assert state.snapshot()["messages"] == []


class TestKeeping:
    def _known(self, *statuses):
        return [status.model_dump(mode="json") for status in statuses]

    def test_past_keep_days_the_audio_and_status_go(self, room):
        box, _, transport, clock, folder = room
        old = box.send("Ravi", _wav(1.0))
        (folder / f"{old.message_id}-reply.wav").write_bytes(_wav(0.5))
        clock.advance(8 * DAY_S)
        assert box.prune(self._known(old)) == [old.message_id]
        assert not list(folder.glob("*.wav"))
        assert (topics.MESSAGE_STATUS.format(message_id=old.message_id), b"", True) in _published(transport, topics.MESSAGE_STATUS)

    def test_within_keep_days_nothing_goes(self, room):
        box, _, _, clock, folder = room
        recent = box.send("Ravi", _wav(1.0))
        clock.advance(6 * DAY_S)
        assert box.prune(self._known(recent)) == [] and (folder / f"{recent.message_id}.wav").is_file()

    def test_beyond_max_kept_the_oldest_go_first(self, room):
        box, _, _, clock, _ = room
        sent = []
        for _ in range(4):
            sent.append(box.send("Ravi", _wav(1.0)))
            clock.advance(10.0)
        assert box.prune(self._known(*sent)) == [sent[0].message_id]

    def test_a_file_nobody_knows_goes_once_it_is_old(self, room):
        box, _, _, clock, folder = room
        stray = folder / "msg_1_abc.wav"
        stray.write_bytes(_wav(0.5))
        os.utime(stray, (clock.now() - 8 * DAY_S, clock.now() - 8 * DAY_S))
        fresh = folder / "msg_2_abc.wav"
        fresh.write_bytes(_wav(0.5))
        os.utime(fresh, (clock.now(), clock.now()))
        box.prune([])
        assert not stray.exists() and fresh.exists()


@pytest.fixture(name="http")
def _http(room):
    box, state, transport, clock, folder = room
    lock = threading.Lock()
    hub = Hub()
    board = Blackboard(CONFIG.mqtt, transport)
    actions = ConsoleActions(CONFIG, clock, board, state)
    app = create_app(state, actions, hub, Link(state, hub, lock, board), lock, messages=box)
    return TestClient(app), box, state, transport, folder


class TestEndpoints:
    def test_sending_says_what_was_kept(self, http):
        client, *_ = http
        body = client.post("/api/messages?sender=Ravi", content=_wav(1.5), headers={"content-type": "audio/webm"}).json()
        assert body["ok"] and body["result"]["state"] == "sent" and body["result"]["transcript"] == "Call me back, Amma"

    def test_a_bad_recording_is_a_400(self, http):
        client, *_ = http
        assert client.post("/api/messages?sender=Ravi", content=b"").status_code == 400

    def test_no_sender_is_a_400(self, http):
        client, *_ = http
        assert client.post("/api/messages", content=_wav(1.0)).status_code == 400

    def test_too_large_is_a_413(self, http):
        client, *_ = http
        assert client.post("/api/messages?sender=Ravi", content=b"x" * 500_001).status_code == 413

    def test_the_recording_and_its_reply_can_be_heard_again(self, http):
        client, _, _, _, folder = http
        message_id = client.post("/api/messages?sender=Ravi", content=_wav(1.0)).json()["result"]["message_id"]
        response = client.get(f"/api/messages/{message_id}/audio")
        assert response.status_code == 200 and response.headers["content-type"] == "audio/wav" and response.content[:4] == b"RIFF"
        assert client.get(f"/api/messages/{message_id}/reply/audio").status_code == 404
        (folder / f"{message_id}-reply.wav").write_bytes(_wav(0.5))
        assert client.get(f"/api/messages/{message_id}/reply/audio").status_code == 200

    @pytest.mark.parametrize("message_id", ["..%2F..%2Fconfig", "msg_..", "MSG_1", "x"])
    def test_only_a_message_id_names_a_file(self, http, message_id):
        client, *_ = http
        response = client.get(f"/api/messages/{message_id}/audio")
        # Not found, or the page itself for a path that is no message at all: never a file.
        assert response.status_code == 404 or response.headers["content-type"].startswith("text/html")

    def test_a_message_can_be_played_again(self, http):
        client, _, state, transport, _ = http
        message_id = client.post("/api/messages?sender=Ravi", content=_wav(1.0)).json()["result"]["message_id"]
        _fold(state, transport)
        before = len(_published(transport, topics.MESSAGE_INCOMING))
        assert client.post(f"/api/messages/{message_id}/replay").json()["ok"]
        assert len(_published(transport, topics.MESSAGE_INCOMING)) == before + 1

    def test_an_unknown_message_cannot_be_played(self, http):
        client, *_ = http
        assert client.post("/api/messages/msg_404/replay").status_code == 404

    def test_without_messages_it_says_so(self, room):
        _, state, transport, clock, _ = room
        lock, hub = threading.Lock(), Hub()
        board = Blackboard(CONFIG.mqtt, transport)
        client = TestClient(create_app(state, ConsoleActions(CONFIG, clock, board, state), hub, Link(state, hub, lock, board), lock))
        assert client.post("/api/messages?sender=Ravi", content=_wav(1.0)).status_code == 503
