"""Talking to the room from a phone or laptop away from it (FR-94).

The room's microphone is on the machine in the room. Someone elsewhere holds
a button in the console, their browser records, and the recording comes here:
it is decoded and transcribed in memory with the same Whisper the room uses,
and the words go on as a typed request would, marked remote. The reply is
spoken with the room's own Piper voice -- but rendered here and sent back to
the device that asked, so the room stays silent.

No recording is kept, published or written anywhere; it lives for the length
of one transcription (FR-51). The models load on first use, not at start, so a
console nobody talks to remotely costs nothing.

The console is a Layer 1 client that reaches the rest of the system through
the blackboard only, so it loads the two models itself, as the speech
service does -- offline, on the configured device -- rather than borrowing
the speech service's code.
"""

from __future__ import annotations

import io
import logging
import threading
import wave

import numpy as np

from src.common.config import SpeechConfig
from src.common.device import resolve

LOGGER = logging.getLogger(__name__)

#: Whisper's own sample rate; a phone's recording is decoded to it.
_SAMPLE_RATE_HZ = 16000
SAMPLE_RATE_HZ = _SAMPLE_RATE_HZ
#: As the speech service decodes, so a request from a phone is heard as well as one in the room.
_BEAM_SIZE = 5
_INT16_FULL_SCALE = 32767


class RemoteVoiceUnavailable(RuntimeError):
    """Talking from a device cannot work on this node, in words for the occupant."""


class RemoteVoice:
    """Hears a device's recording, and speaks the answer back to it."""

    def __init__(self, config: SpeechConfig, max_seconds: float, transcriber=None, speaker=None) -> None:
        """
        :param transcriber: something with ``transcribe_audio(float16k)``;
            loaded from the speech configuration on first use if not given.
        :param speaker: something with ``render(text) -> (pcm, rate)``; the
            Piper voice, loaded on first use if not given.
        """
        self._config = config
        self._max_seconds = max_seconds
        self._transcriber = transcriber
        self._speaker = speaker
        self._lock = threading.Lock()

    def transcribe(self, recording: bytes) -> str:
        """The words in a recording from a browser: webm, ogg, mp4 or wav.

        :raises ValueError: the recording is empty, unreadable or too long --
            said so the occupant can try again.
        :raises RemoteVoiceUnavailable: no speech recognition on this node.
        """
        return self.transcribe_samples(self.decode(recording))

    def decode(self, recording: bytes, max_seconds: float | None = None) -> np.ndarray:
        """A browser's recording as float samples at 16 kHz, checked for length.

        :param max_seconds: the longest accepted; this voice's own limit if not given.
        :raises ValueError: empty, unreadable, too short or too long.
        :raises RemoteVoiceUnavailable: nothing to decode with on this node.
        """
        if not recording:
            raise ValueError("Nothing was recorded; hold the button while you speak.")
        limit = self._max_seconds if max_seconds is None else max_seconds
        audio = self._decode(recording)
        seconds = audio.size / _SAMPLE_RATE_HZ
        if seconds > limit:
            raise ValueError(f"That was {seconds:.0f} s; keep it under {limit:.0f} s.")
        if seconds < 0.3:
            raise ValueError("That was too short to hear; hold the button while you speak.")
        return audio.astype(np.float32)

    def transcribe_samples(self, audio: np.ndarray) -> str:
        """The words in samples already decoded at 16 kHz.

        :raises RemoteVoiceUnavailable: no speech recognition on this node.
        """
        with self._lock:
            transcriber = self._transcriber_ready()
            return transcriber.transcribe_audio(audio).strip()

    def speak(self, text: str) -> bytes:
        """The reply, spoken in the room's voice, as a WAV file for the device to play.

        :raises RemoteVoiceUnavailable: no voice on this node.
        """
        with self._lock:
            speaker = self._speaker_ready()
            pcm, rate = speaker.render(text)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(rate)
            file.writeframes(pcm)
        return buffer.getvalue()

    def _decode(self, recording: bytes) -> np.ndarray:
        try:
            from faster_whisper import decode_audio
        except ImportError as exc:
            raise RemoteVoiceUnavailable("Speech recognition is not installed on this node.") from exc
        try:
            return decode_audio(io.BytesIO(recording), sampling_rate=_SAMPLE_RATE_HZ)
        except (ValueError, OSError, IndexError) as exc:
            # PyAV raises its own InvalidDataError, an OSError and ValueError.
            LOGGER.info("a recording could not be decoded: %s", exc)
            raise ValueError("That recording could not be read; try again.") from exc

    def _transcriber_ready(self):
        if self._transcriber is None:
            try:
                self._transcriber = _Whisper(self._config)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise RemoteVoiceUnavailable(f"Speech recognition is not available here: {exc}") from exc
        return self._transcriber

    def _speaker_ready(self):
        if self._speaker is None:
            try:
                self._speaker = _Piper(self._config)
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                raise RemoteVoiceUnavailable("The room's voice is not available here; the reply is on screen.") from exc
        return self._speaker


class _Whisper:
    """The room's Whisper model, loaded here: offline, on the configured device (NFR-06)."""

    def __init__(self, config: SpeechConfig) -> None:
        from faster_whisper import WhisperModel

        selection = resolve(config.asr_device, config.asr_compute_type)
        LOGGER.info("loading Whisper %s on %s/%s for devices away from the room", config.asr_model, selection.device, selection.compute_type)
        self._model = WhisperModel(config.asr_model, device=selection.device, compute_type=selection.compute_type, local_files_only=True)

    def transcribe_audio(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, beam_size=_BEAM_SIZE)
        return " ".join(segment.text.strip() for segment in segments).strip()


class _Piper:
    """The room's Piper voice, rendering a reply rather than playing it."""

    def __init__(self, config: SpeechConfig) -> None:
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        self._voice = PiperVoice.load(config.voice_model_path)
        self._synthesis = SynthesisConfig(length_scale=config.voice_length_scale)

    def render(self, text: str) -> tuple[bytes, int]:
        chunks = list(self._voice.synthesize(text, syn_config=self._synthesis))
        if not chunks:
            return b"", 22050
        samples = np.concatenate([chunk.audio_float_array for chunk in chunks])
        return (np.clip(samples, -1.0, 1.0) * _INT16_FULL_SCALE).astype(np.int16).tobytes(), chunks[0].sample_rate
