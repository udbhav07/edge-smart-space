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
from src.common.schemas import Utterance, UtteranceSource
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

    def utterances(self) -> list[Utterance]:
        return [m for spec, m in self.published if spec is topics.CONTEXT_UTTERANCE]


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
    def test_a_transcript_is_published_as_an_utterance(self, config):
        _, _, blackboard = _run(config, ["it is too warm in here"])
        (utterance,) = blackboard.utterances()
        assert utterance.text == "it is too warm in here"

    def test_the_utterance_says_it_was_spoken(self, config):
        _, _, blackboard = _run(config, ["hello"])
        assert blackboard.utterances()[0].source is UtteranceSource.SPEECH

    def test_frames_producing_no_transcript_publish_nothing(self, config):
        _, _, blackboard = _run(config, [None, None])
        assert blackboard.published == []

    def test_nothing_but_utterances_is_published(self, config):
        """Speech is Layer 1: it reports what was heard and decides nothing."""
        _, _, blackboard = _run(config, ["a", None, "b"])
        assert {spec for spec, _ in blackboard.published} == {topics.CONTEXT_UTTERANCE}

    def test_an_overlong_transcript_is_cut_to_what_the_topic_carries(self, config):
        _, _, blackboard = _run(config, ["x" * 5000])
        assert len(blackboard.utterances()[0].text) == 2000


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

    def test_build_pipeline_wires_no_reasoning_call(self, config, monkeypatch):
        """Understanding happens in the reasoning process, not here."""

        class StubModel:
            def __init__(self, *args, **kwargs):
                pass

        for name in ("WakeWordDetector", "UtteranceDetector", "Transcriber"):
            monkeypatch.setattr(runner, name, StubModel)
        monkeypatch.setattr(runner, "SpeechPipeline", lambda **kwargs: kwargs)

        parts = runner.build_pipeline(config, SimClock(), FakeCapture([]))
        assert "personal_context" not in parts
