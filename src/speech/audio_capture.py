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


class MicrophoneError(OSError):
    """No usable input device, with the ones that exist named."""


#: Host APIs tried in this order. MME resamples to 16 kHz itself; WASAPI and
#: WDM-KS refuse any rate but the device's own, which on test day failed every
#: microphone except a Bluetooth headset.
_HOST_API_PREFERENCE = ("MME", "Windows DirectSound", "Windows WASAPI")


def input_devices(audio) -> list[tuple[int, str, str]]:
    """Every input device as (index, host API, name)."""
    found = []
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        if info.get("maxInputChannels", 0) > 0:
            host = audio.get_host_api_info_by_index(info["hostApi"])["name"]
            found.append((index, host, str(info["name"])))
    return found


def choose_device(devices: list[tuple[int, str, str]], config: SpeechConfig) -> int | None:
    """The device index to open: configured index, else name, else the default.

    :returns: None to ask PortAudio for the system default.
    :raises MicrophoneError: when a name is configured and nothing matches.
    """
    if config.device_index is not None:
        return config.device_index
    if not config.device_name:
        return None
    wanted = config.device_name.casefold()
    matches = [device for device in devices if wanted in device[2].casefold()]
    if not matches:
        raise MicrophoneError(
            f"no input device matches speech.device_name {config.device_name!r}; "
            f"inputs: {', '.join(sorted({name for _, _, name in devices}))}"
        )

    def preference(device: tuple[int, str, str]) -> int:
        host = device[1]
        return _HOST_API_PREFERENCE.index(host) if host in _HOST_API_PREFERENCE else len(_HOST_API_PREFERENCE)

    return min(matches, key=preference)[0]


def _open_default_stream(config: SpeechConfig):
    """Open the configured microphone.

    PyAudio is imported here rather than at module scope so the queue and
    error-handling logic stay importable without PortAudio installed.

    :returns: the stream and the PyAudio instance owning it.
    :raises MicrophoneError: when no device can be opened, naming the inputs
        that exist rather than PortAudio's "Invalid input device".
    """
    import pyaudio

    audio = pyaudio.PyAudio()
    devices = input_devices(audio)
    try:
        index = choose_device(devices, config)
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=config.channels,
            rate=config.sample_rate_hz,
            input=True,
            input_device_index=index,
            frames_per_buffer=config.chunk_samples,
        )
    except OSError as exc:
        audio.terminate()
        if isinstance(exc, MicrophoneError):
            raise
        raise MicrophoneError(
            f"cannot open the microphone ({exc}); set speech.device_name to one of: "
            f"{', '.join(sorted({name for _, _, name in devices}))}"
        ) from exc
    name = next((name for i, _, name in devices if i == index), "the system default")
    LOGGER.info("listening on %s", name)
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
