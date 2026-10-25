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
from src.common.schemas import AmbientUtterance, Comfort, ListenMode, PreferenceHint
from src.speech.pipeline import PipelineState, SpeechPipeline

FRAME = object()
TRANSCRIPT = "make it cooler in here"


@pytest.fixture(name="config")
def _config() -> SpeechConfig:
    """FR-50's mode: capture only after the wake word. Continuous is tested below."""
    speech = load_config(Path("config/default.yaml")).speech
    return speech.model_copy(update={"listen_mode": ListenMode.WAKE_WORD})


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

    def transcribe(self, frames, confident_only=False) -> str:
        self.calls.append(len(frames))
        return self.text


class FakePersonalContext:
    def __init__(self, hint: PreferenceHint | None = None, error: Exception | None = None):
        self.hint = hint
        self.error = error
        self.transcripts: list[str] = []

    def extract(self, transcript: str, recent=()):
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


#: Real audio frames, for the tests that judge a voice: the other tests feed
#: an opaque placeholder, which recognition would (rightly) reject.
AUDIO_FRAME = __import__("numpy").zeros(512, dtype=__import__("numpy").int16)


class FakeSpeaker:
    def __init__(self, profile_id: str = "", error: Exception | None = None) -> None:
        from src.speech.speaker_profile import Identification

        self.result = Identification(profile_id=profile_id, score=0.5, judged=True)
        self.error = error

    def identify(self, audio):
        if self.error is not None:
            raise self.error
        return self.result


class TestWhoIsSpeaking:
    """FR-52: the hint names whose voice it was, and nothing else changes."""

    def _woken(self, config, speaker, hint):
        detector = FakeDetector(hears=True)
        pipeline, _ = _pipeline(
            config,
            SimClock(),
            detector=detector,
            utterance=FakeUtterance(ends_after=1),
            personal_context=FakePersonalContext(hint=hint),
            speaker=speaker,
        )
        pipeline.accept(AUDIO_FRAME)
        detector.hears = False
        return pipeline

    def test_a_recognised_voice_is_named_on_the_hint(self, config):
        pipeline = self._woken(config, FakeSpeaker("ravi"), _hint(SimClock()))
        assert pipeline.accept(AUDIO_FRAME).profile_id == "ravi"

    def test_an_unrecognised_voice_is_a_guest(self, config):
        pipeline = self._woken(config, FakeSpeaker(""), _hint(SimClock()))
        assert pipeline.accept(AUDIO_FRAME).profile_id == ""

    def test_the_rest_of_the_hint_is_untouched(self, config):
        hint = _hint(SimClock())
        pipeline = self._woken(config, FakeSpeaker("ravi"), hint)
        assert pipeline.accept(AUDIO_FRAME).model_copy(update={"profile_id": ""}) == hint

    def test_a_recognition_failure_costs_only_the_name(self, config):
        pipeline = self._woken(config, FakeSpeaker(error=RuntimeError("boom")), _hint(SimClock()))
        assert pipeline.accept(AUDIO_FRAME).profile_id == ""

    def test_the_buffer_is_still_dropped(self, config):
        """FR-51: judging the voice keeps no audio."""
        pipeline = self._woken(config, FakeSpeaker("ravi"), _hint(SimClock()))
        pipeline.accept(AUDIO_FRAME)
        assert pipeline.buffered_frames == 0

    def test_without_a_recogniser_everyone_is_a_guest(self, config):
        pipeline = self._woken(config, None, _hint(SimClock()))
        assert pipeline.accept(AUDIO_FRAME).profile_id == ""



class TestTheConversationSoFar:
    """U-01: the extraction is told the conversation, so a follow-up reads as one."""

    def test_the_speakers_conversation_reaches_the_extraction(self, config):
        from src.common.schemas import ContextTurn

        turn = ContextTurn(ts=1.0, said="put a meeting at 3", replied="Added it.")

        class Conversations:
            def recent(self, profile_id):
                return (turn,) if profile_id == "ravi" else ()

        class Remembering(FakePersonalContext):
            def extract(self, transcript, recent=()):
                self.recent = recent
                return super().extract(transcript, recent)

        context = Remembering(hint=_hint(SimClock()))
        detector = FakeDetector(hears=True)
        pipeline, _ = _pipeline(
            config, SimClock(), detector=detector, utterance=FakeUtterance(ends_after=1),
            personal_context=context, speaker=FakeSpeaker("ravi"), conversations=Conversations(),
        )
        pipeline.accept(AUDIO_FRAME)
        detector.hears = False
        pipeline.accept(AUDIO_FRAME)
        assert context.recent == (turn,)


class TestReportingActivity:
    """FR-60: what speech is doing is shown, and showing it never stops it."""

    def _run(self, config, utterance, transcriber=None, raise_in_report=False):
        seen = []

        def report(activity, heard):
            if raise_in_report:
                raise OSError("broker gone")
            seen.append((activity.value, heard))

        parts = {"detector": FakeDetector(hears=True), "utterance": utterance, "on_activity": report}
        if transcriber is not None:
            parts["transcriber"] = transcriber
        pipeline, _ = _pipeline(config, SimClock(), **parts)
        for _ in range(5):
            pipeline.accept(FRAME)
        return seen, pipeline

    def test_a_full_utterance_walks_every_activity(self, config):
        seen, _ = self._run(config, FakeUtterance(ends_after=2))
        assert seen[:4] == [
            ("listening", ""), ("transcribing", ""), ("understanding", TRANSCRIPT), ("waiting", ""),
        ]

    def test_silence_after_the_wake_word_returns_to_waiting(self, config):
        quiet = FakeUtterance()
        quiet.silence_timed_out = True
        seen, _ = self._run(config, quiet)
        assert seen[:2] == [("listening", ""), ("waiting", "")]

    def test_a_failing_report_does_not_stop_the_pipeline(self, config):
        transcriber = FakeTranscriber()
        self._run(config, FakeUtterance(ends_after=2), transcriber=transcriber, raise_in_report=True)
        assert transcriber.calls, "the utterance was still transcribed"


# --- listening continuously (FR-76) -------------------------------------------

OVERHEARD = "I've got the dentist on Thursday at four"
SPEECH = "speech"
SILENCE = "silence"


@pytest.fixture(name="continuous")
def _continuous() -> SpeechConfig:
    speech = load_config(Path("config/default.yaml")).speech
    return speech.model_copy(update={"listen_mode": ListenMode.CONTINUOUS})


class FakeVoice:
    """An endpointer driven by the frames themselves: speech starts on a
    SPEECH frame and ends on the first SILENCE after it."""

    def __init__(self) -> None:
        self.speech_started = False
        self.silence_timed_out = False
        self.frames = 0
        self.reset_count = 0
        self.counting_begun = 0

    def reset(self) -> None:
        self.reset_count += 1
        self.speech_started = False
        self.frames = 0

    def begin_counting(self) -> None:
        self.counting_begun += 1
        self.frames = 1

    def accept(self, frame) -> bool:
        self.frames += 1
        if frame == SPEECH:
            self.speech_started = True
            return False
        return self.speech_started


class ScriptedDetector(FakeDetector):
    """Hears the wake word only on the frames marked for it."""

    def heard(self, frame) -> bool:
        return frame == "wake"


def _listening(config, clock, text=OVERHEARD, **overrides):
    parts = {
        "detector": ScriptedDetector(),
        "utterance": FakeVoice(),
        "transcriber": FakeTranscriber(text),
        "personal_context": FakePersonalContext(_hint(clock)),
    }
    parts.update(overrides)
    return _pipeline(config, clock, **parts)


def _feed(pipeline, *frames):
    results = [pipeline.accept(frame) for frame in frames]
    return [result for result in results if result is not None]


class TestListeningContinuously:
    def test_speech_without_the_wake_word_is_overheard(self, continuous):
        pipeline, _ = _listening(continuous, SimClock())
        (heard,) = _feed(pipeline, SILENCE, SPEECH, SPEECH, SILENCE)
        assert isinstance(heard, AmbientUtterance) and heard.transcript == OVERHEARD

    def test_it_is_never_made_a_request(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        _feed(pipeline, SPEECH, SILENCE)
        assert parts["personal_context"].transcripts == []

    def test_the_wake_word_still_makes_a_request(self, continuous):
        pipeline, _ = _listening(continuous, SimClock())
        (heard,) = _feed(pipeline, "wake", SPEECH, SPEECH, SILENCE)
        assert isinstance(heard, PreferenceHint)

    def test_the_wake_word_mid_speech_makes_it_a_request(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        (heard,) = _feed(pipeline, SPEECH, SPEECH, "wake", SPEECH, SILENCE)
        assert isinstance(heard, PreferenceHint)

    def test_and_what_came_before_it_is_dropped(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        _feed(pipeline, SPEECH, SPEECH, SPEECH, "wake", SPEECH, SILENCE)
        assert parts["transcriber"].calls == [2]

    def test_too_few_words_are_dropped(self, continuous):
        pipeline, _ = _listening(continuous, SimClock(), text="uh huh")
        assert _feed(pipeline, SPEECH, SILENCE) == []

    def test_nothing_heard_is_nothing_published(self, continuous):
        pipeline, _ = _listening(continuous, SimClock(), text="")
        assert _feed(pipeline, SPEECH, SILENCE) == []

    def test_the_first_word_is_kept_by_the_preroll(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        _feed(pipeline, SILENCE, SILENCE, SPEECH, SILENCE)
        assert parts["transcriber"].calls[0] >= 3

    def test_the_preroll_is_bounded(self, continuous):
        pipeline, _ = _listening(continuous, SimClock())
        _feed(pipeline, *([SILENCE] * 500))
        bound = int(continuous.ambient_preroll_ms / 1000 * continuous.sample_rate_hz / continuous.chunk_samples)
        assert pipeline.preroll_frames <= max(bound, 1)

    def test_no_audio_is_held_after_an_utterance(self, continuous):
        """FR-51: the buffer and the pre-roll are dropped on close."""
        pipeline, _ = _listening(continuous, SimClock())
        _feed(pipeline, SILENCE, SPEECH, SILENCE)
        assert pipeline.buffered_frames == 0 and pipeline.preroll_frames == 0

    def test_the_hard_cap_runs_from_when_speech_began(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        _feed(pipeline, SILENCE, SILENCE, SPEECH)
        assert parts["utterance"].counting_begun == 1

    def test_the_endpointer_starts_clean_for_the_next_one(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        _feed(pipeline, SPEECH, SILENCE)
        assert parts["utterance"].speech_started is False

    def test_two_utterances_are_two_overheard(self, continuous):
        pipeline, _ = _listening(continuous, SimClock())
        heard = _feed(pipeline, SPEECH, SILENCE, SILENCE, SPEECH, SILENCE)
        assert len(heard) == 2

    def test_the_speaker_is_judged(self, continuous, monkeypatch):
        # The scripted frames are markers, not audio; the voice is FakeSpeaker's to judge.
        monkeypatch.setattr("src.speech.pipeline.to_float_audio", lambda frames: frames)
        pipeline, _ = _listening(continuous, SimClock(), speaker=FakeSpeaker("ravi"))
        (heard,) = _feed(pipeline, SPEECH, SILENCE)
        assert heard.profile_id == "ravi"

    def test_what_was_overheard_is_not_put_in_the_status(self, continuous):
        """The retained status never carries the room's conversation."""
        told = []
        pipeline, _ = _listening(continuous, SimClock(), on_activity=lambda activity, heard="": told.append(heard))
        _feed(pipeline, SPEECH, SILENCE)
        assert all(heard == "" for heard in told)

    def test_in_wake_word_mode_nothing_is_overheard(self, config):
        pipeline, _ = _listening(config, SimClock())
        assert _feed(pipeline, SPEECH, SPEECH, SILENCE) == []

    def test_noise_transcribed_as_one_word_repeated_is_dropped(self, continuous):
        """Live: room noise came back from Whisper as "No. No. No. No."."""
        pipeline, _ = _listening(continuous, SimClock(), text="No. No. No. No.")
        assert _feed(pipeline, SPEECH, SILENCE) == []


class TestHallucinationLoops:
    """S-01: live, Whisper looped on room noise and it became calendar entries."""

    LOOP = "We are going to go home We are going home No we are going home We are going home We are going home"

    def test_a_transcript_looping_on_itself_is_dropped(self, continuous):
        pipeline, _ = _listening(continuous, SimClock(), text=self.LOOP)
        assert _feed(pipeline, SPEECH, SILENCE) == []

    def test_ordinary_speech_with_a_repeat_is_kept(self, continuous):
        pipeline, _ = _listening(continuous, SimClock(), text="no, no, the dentist is on Thursday at four")
        assert len(_feed(pipeline, SPEECH, SILENCE)) == 1

    def test_overheard_speech_asks_for_confident_segments_only(self, continuous):
        asked = []

        class Recording(FakeTranscriber):
            def transcribe(self, frames, confident_only=False):
                asked.append(confident_only)
                return OVERHEARD

        pipeline, _ = _listening(continuous, SimClock(), transcriber=Recording())
        _feed(pipeline, SPEECH, SILENCE)
        assert asked == [True]


class TestPause:
    """FR-79: a window half-filled with the room's own voice is dropped."""

    def test_pausing_mid_utterance_drops_the_window(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        _feed(pipeline, SPEECH, SPEECH)
        pipeline.pause()
        assert pipeline.buffered_frames == 0 and pipeline.state is PipelineState.WAITING_FOR_WAKE_WORD

    def test_nothing_paused_is_transcribed_later(self, continuous):
        pipeline, parts = _listening(continuous, SimClock())
        _feed(pipeline, SPEECH, SPEECH)
        pipeline.pause()
        _feed(pipeline, SILENCE)
        assert parts["transcriber"].calls == []
