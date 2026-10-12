"""Unit tests for enrolling a voice and managing what is kept (FR-52, NFR-05)."""

import wave
from pathlib import Path

import numpy as np
import pytest

from eval.loopback import LoopbackTransport
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.mqtt_client import Blackboard
from src.common.schemas import GUEST_PROFILE_ID, PreferenceHint
from src.speech.speaker_profile import SpeakerIdentifier
from tests.speech.voices import ANU, RAVI, SAMPLE_RATE_HZ, utterance
from tools import enroll as enroll_tool
from tools import profiles as profiles_tool
from tools.enroll import RecordingError, enroll, read_wav
from tools.say import Conversation


#: Seconds of speech the stand-in meter hears in every take. Synthetic test
#: voices are not speech to Silero, so the real meter is replaced here and
#: tested for what it refuses separately.
HEARD_S = 2.0


@pytest.fixture(autouse=True)
def _a_meter_that_hears_speech(monkeypatch):
    monkeypatch.setattr(enroll_tool, "silero_speech_seconds", lambda _rate: (lambda _audio: HEARD_S))


@pytest.fixture(name="config")
def _config(tmp_path):
    config = load_config(Path("config/default.yaml"))
    profiles = config.profiles.model_copy(
        update={"directory": str(tmp_path / "profiles"), "context_directory": str(tmp_path / "context")}
    )
    speech = config.speech.model_copy(update={"voiceprints_path": str(tmp_path / "voiceprints.json")})
    return config.model_copy(update={"profiles": profiles, "speech": speech})


def _wav(path: Path, audio: np.ndarray, rate: int = SAMPLE_RATE_HZ, channels: int = 1) -> Path:
    samples = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(channels)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes(np.repeat(samples, channels).tobytes())
    return path


def _clips(voice, seeds=range(3)):
    return [utterance(*voice, seed) for seed in seeds]


class TestReadingRecordings:
    def test_a_16k_mono_wav_is_read(self, tmp_path):
        audio = utterance(*RAVI, seed=0)
        read = read_wav(_wav(tmp_path / "a.wav", audio), SAMPLE_RATE_HZ)
        assert np.allclose(read, audio, atol=1e-3)

    def test_another_rate_is_refused_rather_than_resampled(self, tmp_path):
        with pytest.raises(RecordingError, match="Hz"):
            read_wav(_wav(tmp_path / "a.wav", utterance(*RAVI, seed=0), rate=8000), SAMPLE_RATE_HZ)

    def test_stereo_is_refused(self, tmp_path):
        with pytest.raises(RecordingError, match="mono"):
            read_wav(_wav(tmp_path / "a.wav", utterance(*RAVI, seed=0), channels=2), SAMPLE_RATE_HZ)

    def test_a_file_that_is_not_a_wav_is_refused(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"not audio")
        with pytest.raises(RecordingError):
            read_wav(tmp_path / "a.wav", SAMPLE_RATE_HZ)


class TestEnrolment:
    def test_enrolling_starts_the_profile_and_the_voiceprint(self, config):
        lines = enroll(config, "ravi", "Ravi", _clips(RAVI))
        assert (Path(config.profiles.directory) / "ravi.md").is_file()
        assert SpeakerIdentifier(config.speech).matchable("ravi")
        assert any("can be recognised" in line for line in lines)

    def test_too_few_utterances_says_how_many_more(self, config):
        lines = enroll(config, "ravi", "Ravi", _clips(RAVI, seeds=range(1)))
        assert any("more before the voice can be recognised" in line for line in lines)

    def test_enrolling_again_adds_to_the_same_voice(self, config):
        enroll(config, "ravi", "Ravi", _clips(RAVI))
        enroll(config, "ravi", "Ravi", _clips(RAVI, seeds=range(3, 5)))
        assert SpeakerIdentifier(config.speech).enrolled == {"ravi": 5}

    def test_closeness_to_others_is_reported(self, config):
        """A confusable household shows up at enrolment, not later."""
        enroll(config, "ravi", "Ravi", _clips(RAVI))
        lines = enroll(config, "anu", "Anu", _clips(ANU))
        assert any("closeness to ravi" in line for line in lines)

    def test_it_says_recognition_grants_nothing(self, config):
        assert any("unlocks nothing" in line for line in enroll(config, "ravi", "Ravi", _clips(RAVI)))

    def test_the_command_line_enrols_from_wav_files(self, config, tmp_path, monkeypatch):
        monkeypatch.setattr(enroll_tool, "load_config", lambda _path: config)
        files = [str(_wav(tmp_path / f"{index}.wav", clip)) for index, clip in enumerate(_clips(RAVI))]
        assert enroll_tool.main(["ravi", "--name", "Ravi", *files]) == 0
        assert SpeakerIdentifier(config.speech).matchable("ravi")

    def test_the_command_line_refuses_nothing_to_enrol(self, config, monkeypatch):
        monkeypatch.setattr(enroll_tool, "load_config", lambda _path: config)
        assert enroll_tool.main(["ravi"]) == 2


class TestEnrolmentNeedsSpeech:
    """U-05: three takes of an empty room were enrolled as a voice."""

    def test_a_take_without_speech_is_refused_and_nothing_is_enrolled(self, config):
        with pytest.raises(RecordingError, match="0.0 s of speech"):
            enroll(config, "ravi", "Ravi", _clips(RAVI), speech_seconds=lambda _audio: 0.0)
        assert SpeakerIdentifier(config.speech).enrolled == {}
        assert not (Path(config.profiles.directory) / "ravi.md").exists()

    def test_one_bad_take_refuses_the_lot(self, config):
        takes = iter([2.0, 0.3, 2.0])
        with pytest.raises(RecordingError, match="recording 2"):
            enroll(config, "ravi", "Ravi", _clips(RAVI), speech_seconds=lambda _audio: next(takes))

    def test_the_speech_heard_is_reported(self, config):
        lines = enroll(config, "ravi", "Ravi", _clips(RAVI), speech_seconds=lambda _audio: 2.4)
        assert lines[0] == "speech heard: 2.4 s, 2.4 s, 2.4 s"

    def test_recording_without_a_terminal_does_not_wait_for_enter(self, config, monkeypatch, capsys):
        """U-24: --record crashed with EOFError when nothing could press Enter."""
        monkeypatch.setattr(enroll_tool, "load_config", lambda _path: config)
        monkeypatch.setattr(enroll_tool, "record", lambda _config, _seconds: utterance(*RAVI, 7))
        monkeypatch.setattr("sys.stdin", None)
        assert enroll_tool.main(["ravi", "--record", "1"]) == 0
        assert "recording now" in capsys.readouterr().out


@pytest.fixture(name="managed")
def _managed(config):
    enroll(config, "ravi", "Ravi", _clips(RAVI))
    return profiles_tool.build(config)


class TestManagement:
    def test_a_profile_can_be_created_without_a_voice(self, config):
        """U-24: only enrolment, which needs audio, used to make a profile."""
        managed = profiles_tool.build(config)
        managed.create("anu", "Anu")
        assert any(line.startswith("anu") and "not yet recognisable" in line for line in managed.list())

    def test_list_shows_each_person_and_the_guest(self, managed):
        lines = managed.list()
        assert any(line.startswith("ravi") for line in lines)
        assert any(line.startswith(GUEST_PROFILE_ID) for line in lines)

    def test_remember_and_show(self, managed):
        managed.remember("ravi", "vegetarian")
        assert any("vegetarian" in line for line in managed.show("ravi"))

    def test_forget(self, managed):
        managed.remember("ravi", "vegetarian")
        assert managed.forget("ravi", "veg") == ["forgot: vegetarian"]

    def test_prefer(self, managed):
        managed.prefer("ravi", "comfort_c", "23")
        assert any("comfort_c: 23" in line for line in managed.show("ravi"))

    def test_delete_forgets_memory_context_and_voice(self, managed, config):
        managed.remember("ravi", "vegetarian")
        managed.delete("ravi")
        assert not (Path(config.profiles.directory) / "ravi.md").exists()
        assert SpeakerIdentifier(config.speech).enrolled == {}

    def test_prune_removes_a_voiceprint_kept_for_nobody(self, managed, config):
        (Path(config.profiles.directory) / "ravi.md").unlink()
        assert any("voiceprint with no profile" in line for line in managed.prune())
        assert SpeakerIdentifier(config.speech).enrolled == {}

    def test_prune_removes_expired_context(self, config):
        clock = SimClock()
        from src.assistance.memory import ProfileStore

        store = ProfileStore(config.profiles, clock, 5.5)
        store.record_turn(GUEST_PROFILE_ID, "hello", "hi")
        clock.advance(config.profiles.short_term_ttl_s + 1.0)
        managed = profiles_tool.Profiles(config, store, SpeakerIdentifier(config.speech))
        assert any("expired" in line for line in managed.prune())

    def test_delete_from_the_command_line_asks_first(self, managed, config, monkeypatch):
        monkeypatch.setattr(profiles_tool, "load_config", lambda _path: config)
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")
        assert profiles_tool.main(["delete", "ravi"]) == 1
        assert (Path(config.profiles.directory) / "ravi.md").exists()

    def test_an_unknown_person_is_an_error_not_a_crash(self, managed, config, monkeypatch):
        monkeypatch.setattr(profiles_tool, "load_config", lambda _path: config)
        assert profiles_tool.main(["show", "nobody"]) == 2


class TestSpeakingAs:
    """tools.say --as stands in for a recognised voice."""

    class Context:
        def __init__(self, hint):
            self.hint = hint

        def extract(self, _transcript, recent=()):
            return self.hint

    def _published(self, config, speaker):
        from src.common import topics
        from src.common.schemas import Comfort, Intent

        clock = SimClock()
        transport = LoopbackTransport()
        board = Blackboard(config.mqtt, transport)
        transport.attach(board)
        hint = PreferenceHint(ts=clock.now(), intent=Intent.SERVICE, comfort=Comfort.UNCHANGED, subject="x")
        Conversation(config, clock, board, self.Context(hint), speaker=speaker).hear("hello")
        payload = [p for t, p, _, _ in transport.published if t == topics.CONTEXT_PREFERENCE.pattern][-1]
        return PreferenceHint.model_validate_json(payload)

    def test_speaking_as_a_person_names_them_on_the_hint(self, config):
        assert self._published(config, "ravi").profile_id == "ravi"

    def test_by_default_it_speaks_as_a_guest(self, config):
        assert self._published(config, "").profile_id == ""
