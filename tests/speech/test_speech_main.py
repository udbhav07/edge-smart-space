"""Unit tests for the speech service runner.

Driven with a fake capture queue and a stub pipeline, so none of this needs
a microphone, a GPU, or an inference server.
"""

import queue
from pathlib import Path

import pytest

from src.common import topics
from src.common.clock import SimClock
from src.common.config import load_config
from src.common.schemas import Comfort, PreferenceHint
from src.reasoning.single_shot import ReasoningUnavailableError
from src.speech import __main__ as runner
from src.speech.audio_capture import AudioCapture

TS = 1756032000.0


@pytest.fixture(name="config")
def _config():
    return load_config(Path("config/default.yaml"))


class FakeCapture:
    """Hands out preloaded frames, then nothing.

    ``frames`` is a property because it is a property on AudioCapture. An
    earlier version declared it a method, so every test here passed while the
    runner raised TypeError on its first frame: a double that does not match
    the real interface tests nothing.
    """

    def __init__(self, frames):
        self._queue = queue.Queue()
        for frame in frames:
            self._queue.put(frame)

    @property
    def frames(self):
        return self._queue


class FakePipeline:
    """Returns a scripted result per frame."""

    def __init__(self, results):
        self._results = list(results)
        self.seen = 0

    def accept(self, frame):
        self.seen += 1
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class RecordingBlackboard:
    def __init__(self):
        self.published = []

    def publish(self, spec, message):
        self.published.append((spec, message))


def _hint(**overrides) -> PreferenceHint:
    return PreferenceHint(
        **{
            "ts": TS,
            "comfort": Comfort.COOLER,
            "target_c": 24.0,
            "rationale": "too warm",
            **overrides,
        }
    )


def _run(config, results, frames=None):
    capture = FakeCapture(["f"] * len(results))
    pipeline = FakePipeline(results)
    blackboard = RecordingBlackboard()
    processed = runner.run(
        config,
        SimClock(),
        capture,
        pipeline,
        blackboard,
        frames=len(results) if frames is None else frames,
    )
    return processed, pipeline, blackboard


class TestFramePump:
    def test_every_frame_reaches_the_pipeline(self, config):
        processed, pipeline, _ = _run(config, [None, None, None])
        assert processed == 3
        assert pipeline.seen == 3

    def test_stops_after_the_requested_frame_count(self, config):
        processed, _, _ = _run(config, [None, None, None], frames=2)
        assert processed == 2

    def test_a_frameless_run_does_nothing(self, config):
        processed, _, blackboard = _run(config, [], frames=0)
        assert processed == 0
        assert blackboard.published == []


class TestPublishing:
    def test_a_hint_is_published_to_the_preference_topic(self, config):
        _, _, blackboard = _run(config, [_hint()])
        assert len(blackboard.published) == 1
        spec, message = blackboard.published[0]
        assert spec is topics.CONTEXT_PREFERENCE
        assert message.target_c == 24.0

    def test_frames_producing_no_hint_publish_nothing(self, config):
        _, _, blackboard = _run(config, [None, None])
        assert blackboard.published == []

    def test_the_preference_topic_is_not_an_actuator_topic(self):
        """FR-45: the reasoning layer never writes to an actuator."""
        assert "actuator" not in topics.CONTEXT_PREFERENCE.pattern


class TestReasoningFailure:
    def test_an_unavailable_reasoning_server_drops_only_the_utterance(self, config):
        """FR-47: losing reasoning costs a hint, not the pipeline."""
        results = [ReasoningUnavailableError("server down"), _hint()]
        processed, _, blackboard = _run(config, results)
        assert processed == 2
        assert len(blackboard.published) == 1

    def test_the_loop_survives_repeated_reasoning_failures(self, config):
        results = [ReasoningUnavailableError("down")] * 5
        processed, _, blackboard = _run(config, results)
        assert processed == 5
        assert blackboard.published == []


class TestEmptyQueue:
    def test_an_empty_queue_does_not_end_the_run(self, config, monkeypatch):
        """A quiet microphone is normal; the loop waits rather than exiting."""
        monkeypatch.setattr(runner, "_FRAME_WAIT_S", 0.01)
        capture = FakeCapture([])
        pipeline = FakePipeline([])
        processed = runner.run(
            config, SimClock(), capture, pipeline, RecordingBlackboard(), frames=0
        )
        assert processed == 0


class TestWiringAgainstTheRealInterfaces:
    """Regressions for two bugs that shipped because the doubles drifted.

    Both would have crashed `python -m src.speech` on startup while every
    test in this file passed, which is the failure mode a test double
    invites when it does not match the class it stands in for.
    """

    def test_the_runner_reads_frames_from_a_real_capture_object(self, config):
        """AudioCapture.frames is a property; calling it raises TypeError."""
        capture = AudioCapture(
            config.speech, SimClock(), stream=object(), audio=object()
        )
        capture.frames.put("frame")

        processed = runner.run(
            config,
            SimClock(),
            capture,
            FakePipeline([None]),
            RecordingBlackboard(),
            frames=1,
        )
        assert processed == 1

    def test_build_pipeline_gives_personal_context_a_clock(self, config, monkeypatch):
        """PersonalContext takes a clock; omitting it raises at construction."""
        captured = {}

        class StubModel:
            def __init__(self, *args, **kwargs):
                pass

        def _personal_context(reasoning_config, clock, *args, **kwargs):
            captured["clock"] = clock
            return object()

        for name in ("WakeWordDetector", "UtteranceDetector", "Transcriber"):
            monkeypatch.setattr(runner, name, StubModel)
        monkeypatch.setattr(runner, "PersonalContext", _personal_context)
        monkeypatch.setattr(runner, "SpeechPipeline", lambda **kwargs: kwargs)

        clock = SimClock()
        runner.build_pipeline(config, clock, FakeCapture([]))
        assert captured["clock"] is clock
