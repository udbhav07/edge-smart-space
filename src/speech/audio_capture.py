"""Microphone capture (Layer 1).

Runs the device read on its own thread so a slow consumer cannot overrun the
driver's buffer. The queue is bounded and drops its oldest frame when full:
without that the overflow simply moves from the sound card into RAM, which
matters on a node with a fixed memory budget.
"""

from __future__ import annotations

import logging
import queue
import threading

import numpy as np

from src.common.clock import Clock
from src.common.config import SpeechConfig

LOGGER = logging.getLogger(__name__)

#: Consecutive read failures tolerated before capture gives up. A mic that
#: has been unplugged will not recover, and spinning on it forever hides the
#: failure from everything upstream.
MAX_CONSECUTIVE_ERRORS = 10

#: Pause after a failed read, so a persistent fault does not burn a core.
ERROR_BACKOFF_S = 0.1

_JOIN_TIMEOUT_S = 2.0


def _open_default_stream(config: SpeechConfig):
    """Open the microphone.

    PyAudio is imported here rather than at module scope so the queue and
    error-handling logic stay importable without PortAudio installed.

    :returns: the stream and the PyAudio instance owning it.
    """
    import pyaudio

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=config.channels,
        rate=config.sample_rate_hz,
        input=True,
        input_device_index=config.device_index,
        frames_per_buffer=config.chunk_samples,
    )
    return stream, audio


class AudioCapture:
    """Background microphone reader feeding a bounded frame queue."""

    def __init__(self, config: SpeechConfig, clock: Clock, stream=None, audio=None) -> None:
        self._config = config
        self._clock = clock
        if stream is None:
            stream, audio = _open_default_stream(config)
        self._stream = stream
        self._audio = audio
        frames_per_second = config.sample_rate_hz / config.chunk_samples
        self._queue: queue.Queue = queue.Queue(
            maxsize=max(1, int(frames_per_second * config.queue_seconds))
        )
        self._running = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)

    @property
    def frames(self) -> queue.Queue:
        """The bounded queue capture writes into."""
        return self._queue

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        """Stop capture and release the device.

        Without this the microphone stays held for the life of the process,
        which blocks anything else that wants it.
        """
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
        try:
            self._stream.stop_stream()
            self._stream.close()
        except OSError as exc:
            LOGGER.warning("closing the audio stream failed: %s", exc)
        if self._audio is not None:
            self._audio.terminate()

    def drain(self) -> None:
        """Discard buffered audio, through the queue's own interface.

        Called after transcription so ambient noise captured while the model
        was busy is not mistaken for the next utterance.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def _read_loop(self) -> None:
        consecutive_errors = 0
        while self._running:
            try:
                data = self._stream.read(
                    self._config.chunk_samples, exception_on_overflow=False
                )
            except (OSError, IOError) as exc:
                consecutive_errors += 1
                LOGGER.warning(
                    "microphone read failed (%d/%d): %s",
                    consecutive_errors,
                    MAX_CONSECUTIVE_ERRORS,
                    exc,
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    LOGGER.error("giving up on the microphone after repeated failures")
                    self._running = False
                    return
                self._clock.sleep(ERROR_BACKOFF_S)
                continue

            consecutive_errors = 0
            self._offer(np.frombuffer(data, dtype=np.int16))

    def _offer(self, frame: np.ndarray) -> None:
        """Enqueue a frame, dropping the oldest if the buffer is full."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(frame)
            except queue.Full:
                LOGGER.debug("dropped a frame: consumer is not keeping up")
