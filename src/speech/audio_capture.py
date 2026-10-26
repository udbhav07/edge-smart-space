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


def candidate_devices(devices: list[tuple[int, str, str]], config: SpeechConfig) -> list[int | None]:
    """Devices to try, best first: the configured index, else every name match
    ordered by host API, else the system default (None).

    Every match is offered, not only the first: on test day the API that
    resamples was absent from one enumeration, and the first match left was
    one that refuses 16 kHz.

    :raises MicrophoneError: when a name is configured and nothing matches.
    """
    if config.device_index is not None:
        return [config.device_index]
    if not config.device_name:
        return [None]
    wanted = config.device_name.casefold()
    matches = [device for device in devices if wanted in device[2].casefold()]
    if not matches:
        raise MicrophoneError(
            f"no input device matches speech.device_name {config.device_name!r}; "
            f"inputs: {_names(devices)}"
        )

    def preference(device: tuple[int, str, str]) -> int:
        host = device[1]
        return _HOST_API_PREFERENCE.index(host) if host in _HOST_API_PREFERENCE else len(_HOST_API_PREFERENCE)

    return [device[0] for device in sorted(matches, key=preference)]


def choose_device(devices: list[tuple[int, str, str]], config: SpeechConfig) -> int | None:
    """The first device that would be tried; None for the system default."""
    return candidate_devices(devices, config)[0]


def _names(devices: list[tuple[int, str, str]]) -> str:
    return ", ".join(sorted({" ".join(name.split()) for _, _, name in devices}))


def _open_default_stream(config: SpeechConfig):
    """Open the configured microphone, at the pipeline's rate if it allows,
    else at its own rate for :class:`Resampler` to convert.

    PyAudio is imported here rather than at module scope so the queue and
    error-handling logic stay importable without PortAudio installed.

    :returns: the stream, the PyAudio instance owning it, and the rate it runs at.
    :raises MicrophoneError: when nothing can be opened, naming the inputs
        that exist rather than PortAudio's "Invalid input device".
    """
    import pyaudio

    audio = pyaudio.PyAudio()
    devices = input_devices(audio)
    failures = []
    try:
        candidates = candidate_devices(devices, config)
    except MicrophoneError:
        audio.terminate()
        raise
    for index in candidates:
        try:
            info = audio.get_device_info_by_index(index) if index is not None else audio.get_default_input_device_info()
        except OSError as exc:
            failures.append(f"default: {exc}")
            continue
        native_hz = int(info.get("defaultSampleRate") or config.sample_rate_hz)
        for rate in dict.fromkeys((config.sample_rate_hz, native_hz)):
            frames = max(1, round(config.chunk_samples * rate / config.sample_rate_hz))
            try:
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=config.channels,
                    rate=rate,
                    input=True,
                    input_device_index=index,
                    frames_per_buffer=frames,
                )
            except OSError as exc:
                failures.append(f"{index} at {rate} Hz: {exc}")
                continue
            LOGGER.info(
                "listening on %s at %d Hz%s",
                " ".join(str(info.get("name", "the default")).split()),
                rate,
                "" if rate == config.sample_rate_hz else f", resampled to {config.sample_rate_hz} Hz",
            )
            return stream, audio, rate, " ".join(str(info.get("name", "")).split())
    audio.terminate()
    raise MicrophoneError(
        f"cannot open the microphone ({'; '.join(failures[-3:])}); "
        f"set speech.device_name to one of: {_names(devices)}"
    )


class Resampler:
    """Convert a stream of samples from one rate to another, chunk by chunk.

    Stateful, so chunk edges leave no click: a windowed-sinc low-pass at 90%
    of the output Nyquist keeps what cannot be represented from aliasing, and
    a fractional read position carries from one chunk into the next. numpy
    only; the speech extra does not declare scipy.
    """

    _TAPS = 63

    def __init__(self, in_rate_hz: int, out_rate_hz: int) -> None:
        self._step = in_rate_hz / out_rate_hz
        cutoff = 0.45 * min(1.0, out_rate_hz / in_rate_hz)
        offsets = np.arange(self._TAPS) - (self._TAPS - 1) / 2
        taps = 2 * cutoff * np.sinc(2 * cutoff * offsets) * np.hamming(self._TAPS)
        self._taps = taps / taps.sum()
        self._history = np.zeros(self._TAPS - 1)
        self._filtered = np.zeros(0)
        self._position = 0.0

    def process(self, samples: np.ndarray) -> np.ndarray:
        """The output-rate samples these input samples complete, as float."""
        padded = np.concatenate([self._history, samples.astype(np.float64)])
        self._history = padded[-(self._TAPS - 1):]
        filtered = np.concatenate([self._filtered, np.convolve(padded, self._taps, mode="valid")])
        last = len(filtered) - 1
        if last < self._position:
            self._filtered = filtered
            return np.zeros(0)
        count = int((last - self._position) // self._step) + 1
        points = self._position + self._step * np.arange(count)
        out = np.interp(points, np.arange(len(filtered)), filtered)
        self._position += self._step * count
        consumed = int(self._position)
        self._filtered = filtered[consumed:]
        self._position -= consumed
        return out


class AudioCapture:
    """Background microphone reader feeding a bounded frame queue."""

    def __init__(
        self, config: SpeechConfig, clock: Clock, stream=None, audio=None, stream_rate_hz: int | None = None
    ) -> None:
        self._config = config
        self._clock = clock
        #: The device in use, for whoever reports it; empty when injected.
        self.microphone = ""
        if stream is None:
            stream, audio, stream_rate_hz, self.microphone = _open_default_stream(config)
        self._stream = stream
        self._audio = audio
        rate = stream_rate_hz or config.sample_rate_hz
        self._read_samples = max(1, round(config.chunk_samples * rate / config.sample_rate_hz))
        self._resampler = Resampler(rate, config.sample_rate_hz) if rate != config.sample_rate_hz else None
        self._pending = np.zeros(0)
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
                data = self._stream.read(self._read_samples, exception_on_overflow=False)
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
            samples = np.frombuffer(data, dtype=np.int16)
            if self._resampler is None:
                self._offer(samples)
                continue
            for frame in self._reframed(self._resampler.process(samples)):
                self._offer(frame)

    def _reframed(self, samples: np.ndarray) -> list[np.ndarray]:
        """Resampled audio cut back into the pipeline's frame size."""
        self._pending = np.concatenate([self._pending, samples])
        size = self._config.chunk_samples
        frames = []
        while len(self._pending) >= size:
            chunk, self._pending = self._pending[:size], self._pending[size:]
            frames.append(np.clip(np.round(chunk), -32768, 32767).astype(np.int16))
        return frames

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
