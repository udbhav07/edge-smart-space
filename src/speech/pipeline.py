"""Speech pipeline orchestration (DESIGN.md section 5.8).

Wires the Layer 1 pieces together and stops. What it deliberately does not
do is act: an utterance becomes at most a ``PreferenceHint``, which is a
supervisory input for the goal path to weigh, never a device command
(FR-45, FR-53). This class holds no reference to a driver or an actuator
topic.

The capture window opens only after wake-word detection and closes at end of
utterance or a hard timeout (FR-50). Audio is transcribed on the node and
the buffer is dropped as soon as transcription finishes (FR-51).

Who was speaking is judged from the same buffer, in memory, before it is
dropped (FR-52). It selects a personalisation profile and nothing else: the
hint carries the profile id, and the hint is still only a supervisory input.
"""

from __future__ import annotations

import logging
from enum import Enum

from src.common.clock import Clock
from src.common.config import SpeechConfig
from src.common.schemas import PreferenceHint, SpeechActivity
from src.speech.asr import to_float_audio

LOGGER = logging.getLogger(__name__)


class PipelineState(str, Enum):
    """Where the pipeline is.

    An enum rather than a string: a mistyped state would otherwise be a
    silent no-op that wedges the pipeline in a state nothing handles.
    """

    WAITING_FOR_WAKE_WORD = "WAITING_FOR_WAKE_WORD"
    LISTENING = "LISTENING"


class SpeechPipeline:
    """Wake word, then capture, then transcribe, then extract.

    Every collaborator is injected so the state machine and the timeout
    arithmetic can be tested with synthetic frames and no microphone, GPU or
    inference server.
    """

    def __init__(
        self,
        config: SpeechConfig,
        clock: Clock,
        capture,
        detector,
        utterance,
        transcriber,
        personal_context,
        speaker=None,
        conversations=None,
        on_activity=None,
    ) -> None:
        self._config = config
        self._clock = clock
        self._capture = capture
        self._detector = detector
        self._utterance = utterance
        self._transcriber = transcriber
        self._personal_context = personal_context
        #: Optional: without it every speaker is a guest, as before FR-52.
        self._speaker = speaker
        #: Optional: the conversation so far, so a follow-up is read as one.
        self._conversations = conversations
        #: Optional: told of every change of activity, so it can be shown
        #: (FR-60). The pipeline does not depend on anyone listening.
        self._on_activity = on_activity
        self._state = PipelineState.WAITING_FOR_WAKE_WORD
        self._buffer: list = []

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def buffered_frames(self) -> int:
        """Frames currently held. Zero outside an utterance (FR-51)."""
        return len(self._buffer)

    def accept(self, frame) -> PreferenceHint | None:
        """Feed one audio frame.

        :returns: a hint when an utterance completed and yielded one; None
            otherwise, which is the common case.
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
        self._tell(SpeechActivity.LISTENING)

    def _listen(self, frame) -> PreferenceHint | None:
        self._buffer.append(frame)
        finished = self._utterance.accept(frame)

        if self._utterance.silence_timed_out:
            LOGGER.info("no speech after the wake word; closing capture window")
            return self._close(transcribe=False)

        if finished:
            return self._close(transcribe=True)
        return None

    def _close(self, transcribe: bool) -> PreferenceHint | None:
        """Close the capture window and discard the audio either way."""
        frames, self._buffer = self._buffer, []
        self._state = PipelineState.WAITING_FOR_WAKE_WORD
        self._capture.drain()

        if not transcribe:
            self._tell(SpeechActivity.WAITING)
            return None
        self._tell(SpeechActivity.TRANSCRIBING)
        try:
            return self._understand(frames)
        finally:
            self._tell(SpeechActivity.WAITING)

    def _understand(self, frames: list) -> PreferenceHint | None:
        transcript = self._transcriber.transcribe(frames)
        if not transcript:
            return None

        LOGGER.info("heard: %s", transcript)
        self._tell(SpeechActivity.UNDERSTANDING, transcript)
        profile_id = self._speaker_of(frames)
        try:
            recent = self._conversations.recent(profile_id) if self._conversations else ()
            hint = self._personal_context.extract(transcript, recent)
        except Exception as exc:
            # Speech is the lowest-priority feature set and is cuttable
            # without affecting the main results. It degrades on its own
            # rather than taking the process with it.
            LOGGER.warning("preference extraction unavailable: %s", exc)
            return None
        if hint is None or not profile_id:
            return hint
        return hint.model_copy(update={"profile_id": profile_id})

    def _tell(self, activity: SpeechActivity, heard: str = "") -> None:
        if self._on_activity is None:
            return
        try:
            self._on_activity(activity, heard)
        except (OSError, ValueError) as exc:
            # Showing what speech is doing must never stop it doing it.
            LOGGER.warning("could not report speech activity: %s", exc)

    def _speaker_of(self, frames: list) -> str:
        """Whose voice it was, or "" for a guest (FR-52).

        Never fatal: recognition is a nicety, and a failure in it makes the
        speaker a guest rather than losing what they said.
        """
        if self._speaker is None:
            return ""
        try:
            return self._speaker.identify(to_float_audio(frames)).profile_id
        except Exception as exc:
            LOGGER.warning("speaker recognition failed; treating as a guest: %s", exc)
            return ""
