"""Unit tests for microphone capture.

The device stream is injected, so no PortAudio is involved. What is tested
is the queue policy and the failure handling: the two things that decide
whether this survives a long run or a mic being unplugged.
"""

import queue
from pathlib import Path

import numpy as np
import pytest

from src.common.clock import SimClock
from src.common.config import SpeechConfig, load_config
from src.speech.audio_capture import MAX_CONSECUTIVE_ERRORS, AudioCapture

FRAME_BYTES = np.zeros(512, dtype=np.int16).tobytes()


@pytest.fixture(name="config")
def _config() -> SpeechConfig:
    return load_config(Path("config/default.yaml")).speech


class FakeStream:
    """Returns scripted reads, then raises to end the loop."""

    def __init__(self, reads: int = 0, errors: int = 0) -> None:
        self.remaining_reads = reads
        self.errors = errors
        self.closed = False
        self.stopped = False

    def read(self, frames, exception_on_overflow=False):
        if self.errors > 0:
            self.errors -= 1
            raise OSError("input overflowed")
        if self.remaining_reads <= 0:
            raise StopIteration
        self.remaining_reads -= 1
        return FRAME_BYTES

    def stop_stream(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


class FakeAudio:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True


def _capture(config, stream=None, audio=None) -> AudioCapture:
    return AudioCapture(
        config,
        SimClock(),
        stream=stream if stream is not None else FakeStream(),
        audio=audio if audio is not None else FakeAudio(),
    )


class TestQueueSizing:
    def test_the_queue_is_bounded(self, config):
        """An unbounded queue moves the overflow from the sound card to RAM."""
        assert _capture(config).frames.maxsize > 0

    def test_the_bound_matches_the_configured_seconds_of_audio(self, config):
        expected = int(
            (config.sample_rate_hz / config.chunk_samples) * config.queue_seconds
        )
        assert _capture(config).frames.maxsize == expected

    def test_the_bound_is_never_zero(self, config):
        tiny = config.model_copy(update={"queue_seconds": 0.0001})
        assert _capture(tiny).frames.maxsize >= 1


class TestDropOldest:
    def _fill(self, capture) -> None:
        for index in range(capture.frames.maxsize):
            capture.frames.put_nowait(np.full(1, index, dtype=np.int16))

    def test_a_full_queue_drops_its_oldest_frame(self, config):
        capture = _capture(config)
        self._fill(capture)
        newest = np.full(1, 999, dtype=np.int16)
        capture._offer(newest)
        assert capture.frames.qsize() == capture.frames.maxsize

    def test_the_newest_frame_survives(self, config):
        capture = _capture(config)
        self._fill(capture)
        capture._offer(np.full(1, 999, dtype=np.int16))
        held = [capture.frames.get_nowait()[0] for _ in range(capture.frames.qsize())]
        assert held[-1] == 999

    def test_the_oldest_frame_is_the_one_lost(self, config):
        capture = _capture(config)
        self._fill(capture)
        capture._offer(np.full(1, 999, dtype=np.int16))
        assert capture.frames.get_nowait()[0] == 1


class TestDrain:
    def test_drain_empties_the_queue(self, config):
        capture = _capture(config)
        for _ in range(5):
            capture.frames.put_nowait(np.zeros(1, dtype=np.int16))
        capture.drain()
        assert capture.frames.empty()

    def test_draining_an_empty_queue_is_harmless(self, config):
        capture = _capture(config)
        capture.drain()
        assert capture.frames.empty()

    def test_drain_uses_the_public_interface_and_leaves_the_queue_usable(self, config):
        capture = _capture(config)
        capture.frames.put_nowait(np.zeros(1, dtype=np.int16))
        capture.drain()
        capture.frames.put_nowait(np.zeros(1, dtype=np.int16))
        assert capture.frames.qsize() == 1


class TestReadLoop:
    def test_frames_read_from_the_device_reach_the_queue(self, config):
        capture = _capture(config, stream=FakeStream(reads=3))
        capture._running = True
        with pytest.raises(StopIteration):
            capture._read_loop()
        assert capture.frames.qsize() == 3

    def test_a_transient_error_does_not_stop_capture(self, config):
        capture = _capture(config, stream=FakeStream(reads=2, errors=1))
        capture._running = True
        with pytest.raises(StopIteration):
            capture._read_loop()
        assert capture.frames.qsize() == 2

    def test_repeated_failures_give_up_rather_than_spinning_forever(self, config):
        """An unplugged mic will not recover; spinning on it hides the fault."""
        capture = _capture(
            config, stream=FakeStream(reads=99, errors=MAX_CONSECUTIVE_ERRORS)
        )
        capture._running = True
        capture._read_loop()
        assert capture.is_running is False

    def test_giving_up_leaves_the_queue_empty_rather_than_partial(self, config):
        capture = _capture(
            config, stream=FakeStream(reads=99, errors=MAX_CONSECUTIVE_ERRORS)
        )
        capture._running = True
        capture._read_loop()
        assert capture.frames.empty()


class TestShutdown:
    def test_stop_releases_the_device(self, config):
        stream, audio = FakeStream(), FakeAudio()
        capture = _capture(config, stream=stream, audio=audio)
        capture.stop()
        assert stream.stopped and stream.closed and audio.terminated

    def test_stop_clears_the_running_flag(self, config):
        capture = _capture(config)
        capture.stop()
        assert capture.is_running is False

    def test_a_failure_while_closing_does_not_raise(self, config):
        class AwkwardStream(FakeStream):
            def stop_stream(self):
                raise OSError("device already gone")

        capture = _capture(config, stream=AwkwardStream())
        capture.stop()
        assert capture.is_running is False
