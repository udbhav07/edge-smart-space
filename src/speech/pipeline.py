"""Speech pipeline orchestration (DESIGN.md section 5.8).

Wires the Layer 1 pieces together and stops at text. An utterance becomes a
transcript and nothing more: understanding it is Personal Context's job, in
the reasoning process, reached over ``space/context/utterance`` (section
5.7.1). Layer 1 turns sound into text; it does not import the reasoning layer,
and it holds no reference to a driver or an actuator topic (FR-45).

The capture window opens only after wake-word detection and closes at end of
utterance or a hard timeout (FR-50). Audio is transcribed on the node and
the buffer is dropped as soon as transcription finishes (FR-51).
"""

from __future__ import annotations

import logging
from enum import Enum

from src.common.clock import Clock
from src.common.config import SpeechConfig

LOGGER = logging.getLogger(__name__)


class PipelineState(str, Enum):
    """Where the pipeline is.

    An enum rather than a string: a mistyped state would otherwise be a
    silent no-op that wedges the pipeline in a state nothing handles.
    """

    WAITING_FOR_WAKE_WORD = "WAITING_FOR_WAKE_WORD"
    LISTENING = "LISTENING"


class SpeechPipeline:
    """Wake word, then capture, then transcribe.

    Every collaborator is injected so the state machine and the timeout
    arithmetic can be tested with synthetic frames and no microphone or GPU.
    """

    def __init__(
        self,
        config: SpeechConfig,
        clock: Clock,
        capture,
        detector,
        utterance,
        transcriber,
    ) -> None:
        self._config = config
        self._clock = clock
        self._capture = capture
        self._detector = detector
        self._utterance = utterance
        self._transcriber = transcriber
        self._state = PipelineState.WAITING_FOR_WAKE_WORD
        self._buffer: list = []

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def buffered_frames(self) -> int:
        """Frames currently held. Zero outside an utterance (FR-51)."""
        return len(self._buffer)

    def accept(self, frame) -> str | None:
        """Feed one audio frame.

        :returns: the transcript when an utterance completed and said
            something; None otherwise, which is the common case.
        """
        if self._state is PipelineState.WAITING_FOR_WAKE_WORD:
            self._maybe_wake(frame)
            return None
        return self._listen(frame)

    def _maybe_wake(self, frame) -> None:
        if not self._detector.heard(frame):
            return
        LOGGER.info("wake word detected; opening capture window")
        self._detector.reset()
        self._utterance.reset()
        self._buffer = []
        self._state = PipelineState.LISTENING

    def _listen(self, frame) -> str | None:
        self._buffer.append(frame)
        finished = self._utterance.accept(frame)

        if self._utterance.silence_timed_out:
            LOGGER.info("no speech after the wake word; closing capture window")
            return self._close(transcribe=False)

        if finished:
            return self._close(transcribe=True)
        return None

    def _close(self, transcribe: bool) -> str | None:
        """Close the capture window and discard the audio either way."""
        frames, self._buffer = self._buffer, []
        self._state = PipelineState.WAITING_FOR_WAKE_WORD
        self._capture.drain()

        if not transcribe:
            return None
        transcript = self._transcriber.transcribe(frames).strip()
        if not transcript:
            return None
        LOGGER.info("heard: %s", transcript)
        return transcript
