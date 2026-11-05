"""Messages home in the room (FR-95): played in the sender's own voice, answered by voice.

What matters: a message plays as one piece -- chime, who it is from, their
recording -- with the microphone held throughout; the room then listens for
an answer longer than after the wake word; that one answer's audio is kept
and never treated as a request; and every step is published as the
message's status.
"""

from pathlib import Path

import numpy as np
import pytest

from eval.loopback import LoopbackTransport
from src.common import topics
from src.common.clock import SimClock
from src.common.config import MessagesConfig, load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import HomeMessage, MessageState, MessageStatus, SpeechSettings
from src.speech.messages import MessageDesk, read_recording, write_recording
from src.speech.pipeline import MessageReply, SpeechPipeline
from src.speech.voice import Voice

CONFIG = load_config(Path("config/default.yaml"))
SPEECH = CONFIG.speech


class _Speaker:
    name = "recording"

    def __init__(self):
        self.said = []

    def say(self, text, volume):
        self.said.append(text)


class _Player:
    def __init__(self):
        self.played: list[tuple[int, int]] = []

    def __call__(self, pcm, rate_hz):
        self.played.append((len(pcm), rate_hz))


def _voice(player=None, speaker=None, clock=None):
    clock = clock or SimClock(start_epoch_s=1000.0)
    voice = Voice(speaker or _Speaker(), clock, 5.5, 0.4, player=player if player is not None else _Player())
    voice.on_settings("t", SpeechSettings(ts=1.0, volume=1.0))
    return voice, clock


def _pcm(seconds=1.0, rate=16000):
    return (np.sin(np.arange(int(rate * seconds)) / 10.0) * 5000).astype(np.int16).tobytes()


class TestPlayingAMessage:
    def test_chime_then_who_it_is_from_then_their_voice(self):
        player, speaker = _Player(), _Speaker()
        voice, _ = _voice(player, speaker)
        voice.play_message("msg_1", "Message from Ravi.", _pcm(), 16000)
        voice.speak_next(timeout_s=0.0)
        assert speaker.said == ["Message from Ravi."]
        assert [rate for _, rate in player.played] == [22050, 16000]  # the chime, then the recording
        assert player.played[1][0] == len(_pcm())

    def test_the_microphone_is_held_while_it_plays(self):
        held = []

        class Watching(_Player):
            def __call__(self, pcm, rate_hz):
                held.append(voice.busy)
                super().__call__(pcm, rate_hz)

        voice, _ = _voice(Watching())
        voice.play_message("msg_1", "", _pcm(), 16000)
        voice.speak_next(timeout_s=0.0)
        assert held and all(held)

    def test_start_and_end_are_reported(self):
        seen = []
        voice, _ = _voice()
        voice.play_message("msg_1", "", _pcm(), 16000, on_start=lambda: seen.append("start"), on_end=lambda: seen.append("end"))
        voice.speak_next(timeout_s=0.0)
        assert seen == ["start", "end"]

    def test_it_follows_the_occupants_volume(self):
        recorded = []
        voice, _ = _voice(lambda pcm, rate: recorded.append(np.abs(np.frombuffer(pcm, dtype=np.int16)).max()))
        voice.on_settings("t", SpeechSettings(ts=2.0, volume=0.5))
        voice.play_message("msg_1", "", _pcm(), 16000)
        voice.speak_next(timeout_s=0.0)
        assert recorded[-1] <= 2600  # half of the 5000 peak, give or take rounding

    def test_without_a_player_it_is_not_queued(self):
        voice = Voice(_Speaker(), SimClock(), 5.5, 0.4, player=None)
        assert voice.play_message("msg_1", "", _pcm(), 16000) is False

    def test_without_a_voice_the_recording_still_plays(self):
        player = _Player()
        voice = Voice(None, SimClock(), 5.5, 0.4, player=player)
        assert voice.play_message("msg_1", "Message from Ravi.", _pcm(), 16000) is True
        voice.speak_next(timeout_s=0.0)
        assert player.played[-1][1] == 16000


class TestTheReplyWindow:
    def test_offered_once_the_room_is_quiet_again(self):
        voice, clock = _voice()
        voice.play_message("msg_1", "", _pcm(), 16000)
        voice.speak_next(timeout_s=0.0)
        assert voice.take_message_window() is None  # still within the tail
        clock.advance(0.5)
        assert voice.take_message_window() == "msg_1" and voice.take_message_window() is None

    def test_not_opened_long_after(self):
        voice, clock = _voice()
        voice.play_message("msg_1", "", _pcm(), 16000)
        voice.speak_next(timeout_s=0.0)
        clock.advance(60.0)
        assert voice.take_message_window() is None

    def test_a_reply_to_a_request_is_not_a_message_window(self):
        from src.common.schemas import AssistantReply, Intent

        voice, clock = _voice()
        voice.on_reply("t", AssistantReply(ts=1.0, transcript="x", intent=Intent.QUESTION, reply="Sure.", spoken=True))
        voice.speak_next(timeout_s=0.0)
        clock.advance(1.0)
        assert voice.take_message_window() is None


class _Detector:
    def heard(self, frame):
        return False

    def reset(self):
        pass


class _Quiet:
    """An endpointer that says nobody spoke from the first frame, then hears speech when told."""

    def __init__(self):
        self.silence_timed_out = True
        self.speech_started = False
        self.end_after = None
        self.frames = 0

    def reset(self):
        self.frames = 0

    def accept(self, frame):
        self.frames += 1
        return self.end_after is not None and self.frames >= self.end_after


class _Transcriber:
    def __init__(self, text="I'll call you tonight"):
        self.text = text

    def transcribe(self, frames, confident_only=False):
        return self.text


class _NeverAsked:
    def extract(self, transcript, recent=()):
        raise AssertionError("a reply to a message is never a request")


class _Capture:
    def drain(self):
        pass


def _pipeline(reply_wait_s=1.0, transcriber=None, utterance=None):
    return SpeechPipeline(
        SPEECH, SimClock(), _Capture(), _Detector(), utterance or _Quiet(), transcriber or _Transcriber(), _NeverAsked(),
        reply_wait_s=reply_wait_s,
    )


FRAME = np.zeros(SPEECH.chunk_samples, dtype=np.int16)
FRAMES_PER_S = SPEECH.sample_rate_hz / SPEECH.chunk_samples


class TestListeningForAReply:
    def test_it_waits_longer_than_after_the_wake_word(self):
        pipeline = _pipeline(reply_wait_s=1.0)
        pipeline.listen_for_reply("msg_1")
        results = [pipeline.accept(FRAME) for _ in range(int(FRAMES_PER_S * 0.8))]
        assert all(result is None for result in results) and pipeline.listening_for_the_assistant

    def test_then_it_gives_up_quietly(self):
        pipeline = _pipeline(reply_wait_s=0.5)
        pipeline.listen_for_reply("msg_1")
        results = [pipeline.accept(FRAME) for _ in range(int(FRAMES_PER_S * 1.0))]
        assert all(result is None for result in results) and not pipeline.listening_for_the_assistant

    def test_an_answer_is_kept_with_its_audio_and_never_asked_of_the_assistant(self):
        utterance = _Quiet()
        utterance.silence_timed_out, utterance.end_after = False, 5
        pipeline = _pipeline(utterance=utterance)
        pipeline.listen_for_reply("msg_1")
        results = [pipeline.accept(FRAME) for _ in range(5)]
        reply = results[-1]
        assert isinstance(reply, MessageReply) and reply.message_id == "msg_1" and reply.transcript == "I'll call you tonight"
        assert len(reply.pcm) == 5 * SPEECH.chunk_samples * 2 and reply.rate_hz == SPEECH.sample_rate_hz

    def test_the_next_window_is_an_ordinary_one(self):
        utterance = _Quiet()
        utterance.silence_timed_out, utterance.end_after = False, 2
        pipeline = _pipeline(utterance=utterance)
        pipeline.listen_for_reply("msg_1")
        pipeline.accept(FRAME)
        pipeline.accept(FRAME)
        pipeline.listen_for_answer()
        assert pipeline._reply_to == ""


@pytest.fixture(name="desk")
def _desk(tmp_path):
    clock = SimClock(start_epoch_s=1000.0)
    transport = LoopbackTransport()
    board = Blackboard(CONFIG.mqtt, transport)
    transport.attach(board)
    voice, _ = _voice(clock=clock)
    config = MessagesConfig(directory=str(tmp_path))
    desk = MessageDesk(config, clock, board, voice)
    desk.subscribe()
    return desk, voice, board, transport, tmp_path


def _arrive(board, tmp_path, message_id="msg_1", write=True):
    if write:
        write_recording(tmp_path / f"{message_id}.wav", _pcm(), 16000)
    board.publish(topics.MESSAGE_INCOMING, HomeMessage(ts=999.0, message_id=message_id, sender="Ravi", transcript="Call me, Amma", duration_s=1.0, audio_file=f"{message_id}.wav"))


def _statuses(transport, message_id="msg_1"):
    topic = topics.MESSAGE_STATUS.format(message_id=message_id)
    return [MessageStatus.model_validate_json(p) for t, p, _, _ in transport.published if t == topic]


class TestTheDesk:
    def test_a_message_is_played_and_its_status_follows(self, desk):
        _, voice, board, transport, folder = desk
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        assert [s.state for s in _statuses(transport)] == [MessageState.PLAYING, MessageState.PLAYED]
        assert _statuses(transport)[-1].played_ts is not None and _statuses(transport)[-1].sender == "Ravi"

    def test_statuses_are_retained(self, desk):
        _, voice, board, transport, folder = desk
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        topic = topics.MESSAGE_STATUS.format(message_id="msg_1")
        assert all(retain for t, _, _, retain in transport.published if t == topic)

    def test_a_missing_recording_is_a_failure_said_plainly(self, desk):
        _, _, board, transport, folder = desk
        _arrive(board, folder, write=False)
        assert _statuses(transport)[-1].state is MessageState.FAILED and "could not be read" in _statuses(transport)[-1].detail

    def test_a_reply_is_kept_and_sent_back(self, desk):
        d, voice, board, transport, folder = desk
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        d.reply(MessageReply("msg_1", "I'll call you tonight", _pcm(2.0), 16000, 2.0))
        last = _statuses(transport)[-1]
        assert last.state is MessageState.REPLIED and last.reply_transcript == "I'll call you tonight" and last.reply_file == "msg_1-reply.wav"
        pcm, rate = read_recording(folder / "msg_1-reply.wav")
        assert rate == 16000 and len(pcm) == len(_pcm(2.0))

    def test_silence_in_the_window_leaves_it_played(self, desk):
        d, voice, board, transport, folder = desk
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        d.reply(MessageReply("msg_1", "", b"", 16000, 0.0))
        assert _statuses(transport)[-1].state is MessageState.PLAYED and not (folder / "msg_1-reply.wav").exists()

    def test_no_audio_ever_reaches_the_board(self, desk):
        d, voice, board, transport, folder = desk
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        d.reply(MessageReply("msg_1", "Okay", _pcm(1.0), 16000, 1.0))
        assert all(b"RIFF" not in payload and len(payload) < 4000 for _, payload, _, _ in transport.published)

    def test_played_again_it_stays_answered(self, desk):
        d, voice, board, transport, folder = desk
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        d.reply(MessageReply("msg_1", "Okay", _pcm(1.0), 16000, 1.0))
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        last = _statuses(transport)[-1]
        assert last.state is MessageState.REPLIED and last.reply_transcript == "Okay"

    def test_after_a_restart_the_retained_status_is_remembered(self, desk):
        d, voice, board, transport, folder = desk
        answered = MessageStatus(ts=5.0, message_id="msg_1", sender="Ravi", duration_s=1.0, state=MessageState.REPLIED, sent_ts=4.0, replied_ts=5.0, reply_transcript="Okay", reply_file="msg_1-reply.wav")
        board.publish(topics.MESSAGE_STATUS, answered, message_id="msg_1")
        _arrive(board, folder)
        voice.speak_next(timeout_s=0.0)
        assert _statuses(transport)[-1].reply_transcript == "Okay"

    def test_a_reply_to_an_unknown_message_publishes_nothing(self, desk):
        d, _, _, transport, _ = desk
        d.reply(MessageReply("msg_404", "hello", _pcm(), 16000, 1.0))
        assert _statuses(transport, "msg_404") == []


class TestRecordings:
    def test_only_16_bit_mono_is_read(self, tmp_path):
        import wave

        path = tmp_path / "stereo.wav"
        with wave.open(str(path), "wb") as file:
            file.setnchannels(2)
            file.setsampwidth(2)
            file.setframerate(16000)
            file.writeframes(b"\x00\x00" * 200)
        with pytest.raises(ValueError):
            read_recording(path)

    def test_a_recording_is_written_whole(self, tmp_path):
        write_recording(tmp_path / "a.wav", _pcm(0.5), 16000)
        assert read_recording(tmp_path / "a.wav") == (_pcm(0.5), 16000) and not list(tmp_path.glob("*.part"))
