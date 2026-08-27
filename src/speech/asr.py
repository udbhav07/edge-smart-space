"""On-device transcription and utterance endpointing (FR-51).

Audio is transcribed on the node and never leaves it. The buffer is
discarded as soon as transcription finishes, so a recording is not held
beyond the work that needed it (DESIGN.md section 5.8).

Endpointing is separated from transcription because they fail differently:
the voice-activity detector decides *when* someone stopped speaking, and
Whisper decides *what* they said. Keeping them apart makes the timeout
arithmetic testable without a GPU.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from faster_whisper import WhisperModel
from silero_vad import VADIterator, load_silero_vad

from src.common.config import SpeechConfig

LOGGER = logging.getLogger(__name__)

#: Full-scale value for signed 16-bit audio. Both Silero and Whisper expect
#: float32 in [-1, 1], so every conversion divides by this.
INT16_FULL_SCALE = 32768.0

#: Whisper beam width. Larger is slower for little gain on short commands.
_BEAM_SIZE = 5

_SPEECH_START = "start"
_SPEECH_END = "end"


def to_float_audio(frames: list[np.ndarray]) -> np.ndarray:
    """Concatenate int16 frames into the float32 form the models expect."""
    return np.concatenate(frames).astype(np.float32) / INT16_FULL_SCALE


class UtteranceDetector:
    """Decides when an utterance has started and finished.

    Wraps Silero's voice-activity detector and adds the two timeouts FR-50
    requires: give up if nobody speaks, and stop unconditionally once the
    utterance has run long enough.
    """

    def __init__(self, config: SpeechConfig, iterator: VADIterator | None = None) -> None:
        self._config = config
        self._iterator = iterator if iterator is not None else VADIterator(
            load_silero_vad(),
            sampling_rate=config.sample_rate_hz,
            min_silence_duration_ms=config.vad_silence_ms,
        )
        self._frames_seen = 0
        self._speech_started = False

    @property
    def speech_started(self) -> bool:
        return self._speech_started

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def _frames_for(self, seconds: float) -> int:
        return int(seconds * self._config.sample_rate_hz / self._config.chunk_samples)

    @property
    def silence_timed_out(self) -> bool:
        """Nobody spoke after the wake word."""
        return not self._speech_started and self._frames_seen > self._frames_for(
            self._config.no_speech_timeout_s
        )

    @property
    def utterance_timed_out(self) -> bool:
        """The utterance has run past its hard cap (FR-50)."""
        return self._speech_started and self._frames_seen > self._frames_for(
            self._config.command_timeout_s
        )

    def reset(self) -> None:
        self._iterator.reset_states()
        self._frames_seen = 0
        self._speech_started = False

    def accept(self, frame: np.ndarray) -> bool:
        """Feed one frame. Returns True when the utterance has ended."""
        self._frames_seen += 1
        tensor = torch.from_numpy(frame.astype(np.float32) / INT16_FULL_SCALE)
        event = self._iterator(tensor, return_seconds=False)

        if event:
            if _SPEECH_START in event:
                self._speech_started = True
            elif _SPEECH_END in event:
                return True
        return self.utterance_timed_out


class Transcriber:
    """Whisper, on this node, with nothing leaving it (FR-51)."""

    def __init__(self, config: SpeechConfig, model: WhisperModel | None = None) -> None:
        self._config = config
        self._model = model if model is not None else WhisperModel(
            config.asr_model,
            device=config.asr_device,
            compute_type=config.asr_compute_type,
        )

    def transcribe(self, frames: list[np.ndarray]) -> str:
        """Transcribe buffered audio. Returns an empty string for silence."""
        if not frames:
            return ""
        segments, _ = self._model.transcribe(to_float_audio(frames), beam_size=_BEAM_SIZE)
        return " ".join(segment.text for segment in segments).strip()
