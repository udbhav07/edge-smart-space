"""Unit tests for endpointing and transcription.

Runs with no GPU and no models loaded. The voice-activity detector and the
Whisper model are both injected, so what is tested here is the part that
actually decides behaviour: when an utterance is judged to have ended, and
the two FR-50 timeouts around it.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.config import SpeechConfig, load_config
from src.speech.asr import (
    INT16_FULL_SCALE,
    Transcriber,
    UtteranceDetector,
    to_float_audio,
)

FRAME = np.zeros(512, dtype=np.int16)


@pytest.fixture(name="config")
def _config() -> SpeechConfig:
    return load_config(Path("config/default.yaml")).speech


class FakeVoiceActivity:
    """Emits scripted speech events, one per accepted frame."""

    def __init__(self, events: list[dict | None] | None = None) -> None:
        self.events = list(events or [])
        self.reset_count = 0
        self.frames_seen = 0

    def __call__(self, frame):
        self.frames_seen += 1
        return self.events.pop(0) if self.events else None

    def reset(self) -> None:
        self.reset_count += 1


class FakeWhisper:
    def __init__(self, text: str = "make it cooler") -> None:
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio, beam_size):
        self.calls.append({"samples": len(audio), "beam_size": beam_size})
        segments = [type("Segment", (), {"text": part})() for part in self.text.split("|")]
        return segments, None


def _detector(config, events=None) -> tuple[UtteranceDetector, FakeVoiceActivity]:
    activity = FakeVoiceActivity(events)
    return UtteranceDetector(config, voice_activity=activity), activity


class TestAudioConversion:
    def test_int16_is_scaled_into_the_unit_interval(self):
        loud = np.array([INT16_FULL_SCALE - 1], dtype=np.int16)
        assert to_float_audio([loud])[0] == pytest.approx(1.0, abs=1e-4)

    def test_silence_converts_to_zero(self):
        assert to_float_audio([FRAME]).max() == 0.0

    def test_frames_are_concatenated_in_order(self):
        first = np.array([1, 2], dtype=np.int16)
        second = np.array([3, 4], dtype=np.int16)
        assert len(to_float_audio([first, second])) == 4

    def test_the_result_is_float32_as_the_models_require(self):
        assert to_float_audio([FRAME]).dtype == np.float32


class TestEndpointing:
    def test_an_utterance_has_not_ended_before_speech_starts(self, config):
        detector, _ = _detector(config)
        assert detector.accept(FRAME) is False

    def test_speech_start_is_recorded(self, config):
        detector, _ = _detector(config, [{"start": 0}])
        detector.accept(FRAME)
        assert detector.speech_started is True

    def test_speech_end_ends_the_utterance(self, config):
        detector, _ = _detector(config, [{"start": 0}, {"end": 1}])
        detector.accept(FRAME)
        assert detector.accept(FRAME) is True

    def test_frames_are_counted(self, config):
        detector, _ = _detector(config)
        for _ in range(5):
            detector.accept(FRAME)
        assert detector.frames_seen == 5

    def test_reset_clears_the_detector_and_the_counters(self, config):
        detector, activity = _detector(config, [{"start": 0}])
        detector.accept(FRAME)
        detector.reset()
        assert detector.frames_seen == 0
        assert detector.speech_started is False
        assert activity.reset_count == 1


class TestSilenceTimeout:
    def _frames_for(self, config, seconds: float) -> int:
        return int(seconds * config.sample_rate_hz / config.chunk_samples)

    def test_silence_does_not_time_out_immediately(self, config):
        detector, _ = _detector(config)
        detector.accept(FRAME)
        assert detector.silence_timed_out is False

    def test_silence_times_out_after_the_configured_wait(self, config):
        detector, _ = _detector(config)
        for _ in range(self._frames_for(config, config.no_speech_timeout_s) + 1):
            detector.accept(FRAME)
        assert detector.silence_timed_out is True

    def test_silence_never_times_out_once_someone_speaks(self, config):
        detector, _ = _detector(config, [{"start": 0}])
        for _ in range(self._frames_for(config, config.no_speech_timeout_s) + 5):
            detector.accept(FRAME)
        assert detector.silence_timed_out is False


class TestUtteranceTimeout:
    def _frames_for(self, config, seconds: float) -> int:
        return int(seconds * config.sample_rate_hz / config.chunk_samples)

    def test_an_utterance_is_capped_even_if_speech_never_ends(self, config):
        """FR-50: capture stops at end of utterance or a hard timeout."""
        detector, _ = _detector(config, [{"start": 0}])
        ended = False
        for _ in range(self._frames_for(config, config.command_timeout_s) + 2):
            ended = detector.accept(FRAME)
        assert ended is True

    def test_the_cap_does_not_apply_before_speech_starts(self, config):
        detector, _ = _detector(config)
        for _ in range(self._frames_for(config, config.command_timeout_s) + 2):
            detector.accept(FRAME)
        assert detector.utterance_timed_out is False

    def test_a_short_utterance_is_not_capped(self, config):
        detector, _ = _detector(config, [{"start": 0}])
        detector.accept(FRAME)
        assert detector.utterance_timed_out is False


class TestTranscriber:
    def test_transcribes_buffered_audio(self, config):
        whisper = FakeWhisper("make it cooler")
        assert Transcriber(config, model=whisper).transcribe([FRAME]) == "make it cooler"

    def test_segments_are_joined(self, config):
        whisper = FakeWhisper("make it| cooler")
        assert Transcriber(config, model=whisper).transcribe([FRAME]) == "make it cooler"

    def test_an_empty_buffer_transcribes_to_nothing(self, config):
        whisper = FakeWhisper()
        assert Transcriber(config, model=whisper).transcribe([]) == ""

    def test_an_empty_buffer_never_reaches_the_model(self, config):
        whisper = FakeWhisper()
        Transcriber(config, model=whisper).transcribe([])
        assert whisper.calls == []

    def test_the_model_receives_converted_audio(self, config):
        whisper = FakeWhisper()
        Transcriber(config, model=whisper).transcribe([FRAME, FRAME])
        assert whisper.calls[0]["samples"] == 2 * len(FRAME)

    def test_surrounding_whitespace_is_trimmed(self, config):
        whisper = FakeWhisper("  spaced out  ")
        assert Transcriber(config, model=whisper).transcribe([FRAME]) == "spaced out"
