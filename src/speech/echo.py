"""Hearing the wake word over the room's own music (FR-98).

The room knows what it is playing -- the music, and Jarvis's own voice -- so
it can take that out of what the microphone hears: acoustic echo
cancellation, with what the speakers play as the reference. Here the
reference is recorded straight from the speakers' output (WASAPI loopback on
Windows), which carries every sound the node makes without any service
having to share it, and WebRTC's echo canceller (AEC3, through LiveKit's
bindings) removes it.

Measured offline on two songs (``tools/wake_eval.py`` and the probes behind
DESIGN section 5.8.8): the canceller removed 33 to 46 dB of music, but while
someone spoke over it, it also took 15 to 22 dB off their voice, and on its
own it caught fewer wakes than the raw microphone. So it is used beside the
microphone, not instead of it: one detector listens to the raw signal, a
second to the cancelled one, and either wakes the room. With the music 5 dB
louder than the voice that caught 26 of 27 against 22 for the raw microphone;
10 dB louder, 21 against 12. The cancelled signal goes nowhere else: what is
transcribed is the microphone as it was, with the music turned down by then.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections import deque

import numpy as np

LOGGER = logging.getLogger(__name__)

RATE_HZ = 16000
#: WebRTC's processing block: exactly 10 ms.
_BLOCK = RATE_HZ // 100
#: Most reference audio held waiting for the microphone, in s; beyond it the
#: oldest is dropped, so a stall never leaves the two seconds apart.
_REFERENCE_KEPT_S = 0.5
#: Reference left waiting once a frame is taken, in s: enough for the
#: loopback's 10 ms bursts to arrive unevenly, little enough that the
#: reference is what the speakers played just now. Live, a backlog of half a
#: second left it 320 ms behind the microphone -- beyond what the canceller
#: can line up -- and it removed half a decibel.
_REFERENCE_CUSHION_S = 0.03


class EchoUnavailable(RuntimeError):
    """Echo cancelling cannot work on this node, said plainly."""


class EchoCanceller:
    """WebRTC AEC3: the microphone, with what the speakers played taken out."""

    def __init__(self, delay_ms: int, apm=None) -> None:
        """
        :param delay_ms: a first guess at how much later the microphone hears
            the speakers than the loopback records them; AEC3 refines it.
        :raises EchoUnavailable: the canceller is not installed.
        """
        if apm is None:
            try:
                from livekit import rtc
            except ImportError as exc:
                raise EchoUnavailable("echo cancelling is not installed (livekit)") from exc
            apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
            self._frame = lambda pcm: rtc.AudioFrame(pcm.tobytes(), RATE_HZ, 1, _BLOCK)
        else:
            self._frame = lambda pcm: _Frame(pcm)
        self._apm = apm
        self._delay_ms = delay_ms
        self._mic = np.zeros(0, dtype=np.int16)
        self._far = np.zeros(0, dtype=np.int16)
        # One block of lead, so a frame always has a full frame of output to return.
        self._out = np.zeros(_BLOCK, dtype=np.int16)

    def process(self, mic: np.ndarray, far: np.ndarray) -> np.ndarray:
        """The microphone frame with the reference taken out; as long as the frame given."""
        self._mic = np.concatenate([self._mic, mic.astype(np.int16)])
        self._far = np.concatenate([self._far, far.astype(np.int16)])
        cleaned = []
        while self._mic.size >= _BLOCK and self._far.size >= _BLOCK:
            near, self._mic = self._mic[:_BLOCK], self._mic[_BLOCK:]
            reference, self._far = self._far[:_BLOCK], self._far[_BLOCK:]
            self._apm.process_reverse_stream(self._frame(reference))
            frame = self._frame(near)
            self._apm.set_stream_delay_ms(self._delay_ms)
            self._apm.process_stream(frame)
            cleaned.append(np.frombuffer(bytes(frame.data), dtype=np.int16))
        self._out = np.concatenate([self._out, *cleaned]) if cleaned else self._out
        result, self._out = self._out[: mic.size], self._out[mic.size:]
        return np.pad(result, (0, mic.size - result.size))


class _Frame:
    """An audio frame for a canceller given in tests, which edit it in place."""

    def __init__(self, pcm: np.ndarray) -> None:
        self.data = bytearray(pcm.astype(np.int16).tobytes())


class LoopbackReference:
    """What the speakers are playing, as 16 kHz mono, recorded from the output itself."""

    def __init__(self, source=None) -> None:
        """
        :param source: something calling ``feed(pcm, rate, channels)``; the
            default records the default speakers through WASAPI loopback.
        :raises EchoUnavailable: no loopback on this platform, or none found.
        """
        self._lock = threading.Lock()
        self._held: deque[np.ndarray] = deque()
        self._samples = 0
        self._stream = None
        if source is None:
            self._stream = self._open()

    def feed(self, pcm: np.ndarray, rate: int, channels: int) -> None:
        """Speaker audio as it is played: int16, interleaved."""
        mono = pcm.astype(np.float32).reshape(-1, channels).mean(axis=1)
        resampled = to_16k(mono, rate).astype(np.int16)
        with self._lock:
            self._held.append(resampled)
            self._samples += resampled.size
            while self._samples > _REFERENCE_KEPT_S * RATE_HZ and self._held:
                self._samples -= self._held.popleft().size

    def take(self, count: int) -> np.ndarray:
        """The next ``count`` samples the speakers played; silence if they played nothing."""
        out = np.zeros(count, dtype=np.int16)
        filled = 0
        with self._lock:
            # Played longer ago than the microphone's frame: it is not what the frame heard.
            stale = self._samples - count - int(_REFERENCE_CUSHION_S * RATE_HZ)
            while stale > 0 and self._held:
                head = self._held[0]
                if head.size <= stale:
                    self._held.popleft()
                    dropped = head.size
                else:
                    self._held[0] = head[stale:]
                    dropped = stale
                self._samples -= dropped
                stale -= dropped
            while filled < count and self._held:
                head = self._held[0]
                used = min(count - filled, head.size)
                out[filled: filled + used] = head[:used]
                filled += used
                if used == head.size:
                    self._held.popleft()
                else:
                    self._held[0] = head[used:]
                self._samples -= used
        return out

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

    def _open(self):
        if sys.platform != "win32":
            # On Linux, PipeWire's module-echo-cancel does this in the sound
            # server instead; point the capture at its source.
            raise EchoUnavailable("speaker loopback is recorded here only on Windows")
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise EchoUnavailable("speaker loopback is not installed (PyAudioWPatch)") from exc
        audio = pyaudio.PyAudio()
        try:
            wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = audio.get_device_info_by_index(wasapi["defaultOutputDevice"])
            loopback = next(
                (device for device in audio.get_loopback_device_info_generator() if speakers["name"] in device["name"]), None
            )
        except OSError as exc:
            raise EchoUnavailable(f"no speaker loopback: {exc}") from exc
        if loopback is None:
            raise EchoUnavailable(f"no loopback for {speakers['name']}")
        rate, channels = int(loopback["defaultSampleRate"]), int(loopback["maxInputChannels"])

        def callback(data, frames, _time, _status):
            self.feed(np.frombuffer(data, dtype=np.int16), rate, channels)
            return (None, pyaudio.paContinue)

        LOGGER.info("echo reference: %s at %d Hz", loopback["name"], rate)
        return audio.open(
            format=pyaudio.paInt16, channels=channels, rate=rate, input=True,
            input_device_index=loopback["index"], frames_per_buffer=rate // 100, stream_callback=callback,
        )


def to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """Downsampled to 16 kHz, low-passed first so nothing above 8 kHz folds back."""
    if rate == RATE_HZ:
        return audio
    cutoff = 0.45 * RATE_HZ / rate
    taps = np.arange(-24, 25)
    kernel = np.sinc(2 * cutoff * taps) * np.hamming(taps.size)
    filtered = np.convolve(audio, kernel / kernel.sum(), mode="same")
    if rate % RATE_HZ == 0:
        return filtered[:: rate // RATE_HZ]
    count = int(audio.size * RATE_HZ / rate)
    return np.interp(np.linspace(0, audio.size, count, endpoint=False), np.arange(audio.size), filtered)


class ListeningTwice:
    """A wake detector on the microphone, and another on it with the room's own sound taken out.

    Either waking wakes the room. See the module's note for why both.
    """

    def __init__(self, raw, cleaned, canceller: EchoCanceller, reference: LoopbackReference) -> None:
        self._raw, self._cleaned = raw, cleaned
        self._canceller, self._reference = canceller, reference

    def heard(self, frame: np.ndarray) -> bool:
        clean = self._canceller.process(frame, self._reference.take(frame.size))
        heard = self._raw.heard(frame) | self._cleaned.heard(clean)  # both hear every frame
        if heard:
            self.reset()
        return heard

    def reset(self) -> None:
        self._raw.reset()
        self._cleaned.reset()
