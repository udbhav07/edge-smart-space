"""Unit tests for the speech pipeline's state machine and timeouts.

These run with no microphone, no GPU, and no inference server. That is the
point of separating orchestration from the models it drives: the parts that
actually break — the state transitions and the timeout arithmetic — are the
parts that were previously untestable.
"""

from pathlib import Path

import pytest

from src.common.clock import SimClock
from src.common.config import SpeechConfig, load_config
from src.common.schemas import Comfort, PreferenceHint
from src.speech.pipeline import PipelineState, SpeechPipeline

FRAME = object()
TRANSCRIPT = "make it cooler in here"


@pytest.fixture(name="config")
def _config() -> SpeechConfig:
    return load_config(Path("config/default.yaml")).speech


class FakeCapture:
    def __init__(self) -> None:
        self.drain_count = 0

    def drain(self) -> None:
        self.drain_count += 1


class FakeDetector:
    def __init__(self, *, hears: bool = False) -> None:
        self.hears = hears
        self.reset_count = 0

    def heard(self, frame) -> bool:
        return self.hears

    def reset(self) -> None:
        self.reset_count += 1


class FakeUtterance:
    def __init__(self, *, ends_after: int | None = None) -> None:
        self.ends_after = ends_after
        self.silence_timed_out = False
        self.frames = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1
        self.frames = 0

    def accept(self, frame) -> bool:
        self.frames += 1
        return self.ends_after is not None and self.frames >= self.ends_after


class FakeTranscriber:
    def __init__(self, text: str = TRANSCRIPT) -> None:
        self.text = text
        self.calls: list[int] = []

    def transcribe(self, frames) -> str:
        self.calls.append(len(frames))
        return self.text


class FakePersonalContext:
    def __init__(self, hint: PreferenceHint | None = None, error: Exception | None = None):
        self.hint = hint
        self.error = error
        self.transcripts: list[str] = []

    def extract(self, transcript: str):
        self.transcripts.append(transcript)
        if self.error is not None:
            raise self.error
        return self.hint


def _hint(clock: SimClock) -> PreferenceHint:
    return PreferenceHint(ts=clock.now(), comfort=Comfort.COOLER, rationale=TRANSCRIPT)


def _pipeline(config, clock, **overrides) -> tuple[SpeechPipeline, dict]:
    parts = {
        "capture": FakeCapture(),
        "detector": FakeDetector(),
        "utterance": FakeUtterance(),
        "transcriber": FakeTranscriber(),
        "personal_context": FakePersonalContext(),
    }
    parts.update(overrides)
    return SpeechPipeline(config, clock, **parts), parts


class TestWakeWordGate:
    def test_starts_waiting_for_the_wake_word(self, config):
        pipeline, _ = _pipeline(config, SimClock())
        assert pipeline.state is PipelineState.WAITING_FOR_WAKE_WORD

    def test_audio_is_not_buffered_before_the_wake_word(self, config):
        """FR-50: capture begins only after detection."""
        pipeline, _ = _pipeline(config, SimClock())
        for _ in range(50):
            pipeline.accept(FRAME)
        assert pipeline.buffered_frames == 0

    def test_the_wake_word_opens_the_capture_window(self, config):
        pipeline, _ = _pipeline(config, SimClock(), detector=FakeDetector(hears=True))
        pipeline.accept(FRAME)
        assert pipeline.state is PipelineState.LISTENING

    def test_waking_resets_the_detector_and_the_endpointer(self, config):
        detector = FakeDetector(hears=True)
        utterance = FakeUtterance()
        pipeline, _ = _pipeline(
            config, SimClock(), detector=detector, utterance=utterance
        )
        pipeline.accept(FRAME)
        assert detector.reset_count == 1
        assert utterance.reset_count == 1

    def test_nothing_is_returned_merely_by_waking(self, config):
        pipeline, _ = _pipeline(config, SimClock(), detector=FakeDetector(hears=True))
        assert pipeline.accept(FRAME) is None


class TestListening:
    def _woken(self, config, clock, **overrides):
        detector = FakeDetector(hears=True)
        pipeline, parts = _pipeline(config, clock, detector=detector, **overrides)
        pipeline.accept(FRAME)
        detector.hears = False
        return pipeline, parts

    def test_frames_are_buffered_while_listening(self, config):
        pipeline, _ = self._woken(config, SimClock())
        pipeline.accept(FRAME)
        pipeline.accept(FRAME)
        assert pipeline.buffered_frames == 2

    def test_end_of_utterance_returns_to_waiting(self, config):
        pipeline, _ = self._woken(
            config, SimClock(), utterance=FakeUtterance(ends_after=2)
        )
        pipeline.accept(FRAME)
        pipeline.accept(FRAME)
        assert pipeline.state is PipelineState.WAITING_FOR_WAKE_WORD

    def test_end_of_utterance_transcribes_what_was_buffered(self, config):
        transcriber = FakeTranscriber()
        pipeline, _ = self._woken(
            config,
            SimClock(),
            utterance=FakeUtterance(ends_after=3),
            transcriber=transcriber,
        )
        for _ in range(3):
            pipeline.accept(FRAME)
        assert transcriber.calls == [3]

    def test_a_completed_utterance_yields_the_extracted_hint(self, config):
        clock = SimClock()
        expected = _hint(clock)
        pipeline, _ = self._woken(
            config,
            clock,
            utterance=FakeUtterance(ends_after=1),
            personal_context=FakePersonalContext(hint=expected),
        )
        assert pipeline.accept(FRAME) is expected

    def test_the_transcript_reaches_extraction_verbatim(self, config):
        context = FakePersonalContext()
        pipeline, _ = self._woken(
            config,
            SimClock(),
            utterance=FakeUtterance(ends_after=1),
            personal_context=context,
        )
        pipeline.accept(FRAME)
        assert context.transcripts == [TRANSCRIPT]


class TestAudioIsNotRetained:
    """FR-51 and section 5.8: the buffer goes as soon as the work is done."""

    def _woken(self, config, **overrides):
        detector = FakeDetector(hears=True)
        pipeline, parts = _pipeline(config, SimClock(), detector=detector, **overrides)
        pipeline.accept(FRAME)
        detector.hears = False
        return pipeline, parts

    def test_the_buffer_is_empty_after_a_completed_utterance(self, config):
        pipeline, _ = self._woken(config, utterance=FakeUtterance(ends_after=1))
        pipeline.accept(FRAME)
        assert pipeline.buffered_frames == 0

    def test_the_buffer_is_empty_after_a_silence_timeout(self, config):
        utterance = FakeUtterance()
        pipeline, _ = self._woken(config, utterance=utterance)
        pipeline.accept(FRAME)
        utterance.silence_timed_out = True
        pipeline.accept(FRAME)
        assert pipeline.buffered_frames == 0

    def test_stale_capture_is_drained_when_the_window_closes(self, config):
        capture = FakeCapture()
        pipeline, _ = self._woken(
            config, capture=capture, utterance=FakeUtterance(ends_after=1)
        )
        pipeline.accept(FRAME)
        assert capture.drain_count == 1


class TestSilenceTimeout:
    def test_silence_closes_the_window_without_transcribing(self, config):
        transcriber = FakeTranscriber()
        utterance = FakeUtterance()
        detector = FakeDetector(hears=True)
        pipeline, _ = _pipeline(
            config,
            SimClock(),
            detector=detector,
            utterance=utterance,
            transcriber=transcriber,
        )
        pipeline.accept(FRAME)
        detector.hears = False
        utterance.silence_timed_out = True
        assert pipeline.accept(FRAME) is None
        assert transcriber.calls == []

    def test_silence_returns_the_pipeline_to_waiting(self, config):
        utterance = FakeUtterance()
        detector = FakeDetector(hears=True)
        pipeline, _ = _pipeline(
            config, SimClock(), detector=detector, utterance=utterance
        )
        pipeline.accept(FRAME)
        detector.hears = False
        utterance.silence_timed_out = True
        pipeline.accept(FRAME)
        assert pipeline.state is PipelineState.WAITING_FOR_WAKE_WORD


class TestDegradation:
    def test_an_unavailable_reasoning_layer_does_not_propagate(self, config):
        """Speech is cuttable; it must not take the process with it."""
        pipeline, _ = _pipeline(
            config,
            SimClock(),
            detector=FakeDetector(hears=True),
            utterance=FakeUtterance(ends_after=1),
            personal_context=FakePersonalContext(error=RuntimeError("server down")),
        )
        pipeline.accept(FRAME)
        assert pipeline.accept(FRAME) is None

    def test_the_pipeline_still_returns_to_waiting_after_a_failure(self, config):
        pipeline, _ = _pipeline(
            config,
            SimClock(),
            detector=FakeDetector(hears=True),
            utterance=FakeUtterance(ends_after=1),
            personal_context=FakePersonalContext(error=RuntimeError("server down")),
        )
        pipeline.accept(FRAME)
        pipeline.accept(FRAME)
        assert pipeline.state is PipelineState.WAITING_FOR_WAKE_WORD

    def test_an_empty_transcript_yields_nothing(self, config):
        context = FakePersonalContext()
        pipeline, _ = _pipeline(
            config,
            SimClock(),
            detector=FakeDetector(hears=True),
            utterance=FakeUtterance(ends_after=1),
            transcriber=FakeTranscriber(text=""),
            personal_context=context,
        )
        pipeline.accept(FRAME)
        assert pipeline.accept(FRAME) is None
        assert context.transcripts == []


class TestNoActuation:
    def test_the_pipeline_holds_no_driver_or_actuator_reference(self, config):
        """FR-45: the reasoning layer never writes to an actuator topic."""
        pipeline, _ = _pipeline(config, SimClock())
        forbidden = {"driver", "actuator", "device", "gpio", "relay"}
        assert not any(
            word in name.lower() for name in vars(pipeline) for word in forbidden
        )
