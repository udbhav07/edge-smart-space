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

**Listening continuously** (FR-76, ``listen_mode: continuous``), the
endpointer also runs while waiting, and speech without the wake word opens a
window of its own. Such a window is *ambient*: closed, it is transcribed and
returned as an :class:`AmbientUtterance` -- words for the reasoning layer to
look through, never a request. The wake word heard during one turns it into
an addressed window, begun afresh, exactly as if it had been heard first. A
short pre-roll keeps the first word; it is a rolling few hundred
milliseconds, and like every buffer here it is dropped, never kept (FR-51).
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.common.clock import Clock
from src.common.config import SpeechConfig
from src.common.schemas import AmbientUtterance, ListenMode, PreferenceHint, SpeechActivity
from src.speech.asr import to_float_audio

LOGGER = logging.getLogger(__name__)

#: Below this many words a repeated one is ordinary speech ("no, no, I said").
_LOOP_CHECK_WORDS = 6


@dataclass(frozen=True)
class MessageReply:
    """What someone at home said back to a message (FR-95).

    The one utterance whose audio is kept: an answer they were invited to
    give, recorded to be heard by the person who sent the message. Every
    other window's audio is still dropped when it closes (FR-51).
    """

    message_id: str
    transcript: str
    pcm: bytes
    rate_hz: int
    duration_s: float


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
        reply_wait_s: float = 8.0,
    ) -> None:
        """
        :param reply_wait_s: how long a window opened for a reply to a message
            waits for someone to start speaking (FR-95); longer than after the
            wake word, because the answer is to something just heard.
        """
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
        self._continuous = config.listen_mode is ListenMode.CONTINUOUS
        #: Whether the open window was begun by the wake word.
        self._addressed = False
        preroll_frames = int(config.ambient_preroll_ms / 1000.0 * config.sample_rate_hz / config.chunk_samples)
        self._preroll: deque = deque(maxlen=max(preroll_frames, 1))
        #: The message the open window is a reply to, if it is one (FR-95).
        self._reply_to = ""
        self._reply_wait_frames = int(reply_wait_s * config.sample_rate_hz / config.chunk_samples)
        self._window_frames = 0

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def listening_for_the_assistant(self) -> bool:
        """Whether a window is open for the assistant: after the wake word, or after a reply."""
        return self._state is PipelineState.LISTENING and self._addressed

    @property
    def buffered_frames(self) -> int:
        """Frames currently held. Zero outside an utterance (FR-51)."""
        return len(self._buffer)

    @property
    def preroll_frames(self) -> int:
        """Frames held from just before speech. Bounded by configuration."""
        return len(self._preroll)

    def accept(self, frame) -> PreferenceHint | AmbientUtterance | None:
        """Feed one audio frame.

        :returns: a hint when an addressed utterance completed and yielded
            one, the words of an unaddressed one when listening continuously,
            or None, which is the common case.
        """
        if self._state is PipelineState.WAITING_FOR_WAKE_WORD:
            if self._detector.heard(frame):
                self._wake()
            elif self._continuous:
                self._maybe_overhear(frame)
            return None
        return self._listen(frame)

    def pause(self) -> None:
        """Drop whatever is being heard and wait afresh (FR-79, FR-82).

        Called for every frame while the room is speaking or the occupant has
        muted it: a window half-filled with the room's own voice must not be
        transcribed, and a muted room must not keep the start of a sentence.
        """
        if self._state is PipelineState.LISTENING or self._buffer or self._preroll:
            self._buffer = []
            self._preroll.clear()
            self._addressed = False
            self._state = PipelineState.WAITING_FOR_WAKE_WORD
            self._utterance.reset()
            self._detector.reset()

    def listen_for_reply(self, message_id: str) -> None:
        """Open a window for an answer to a message just played (FR-95): its audio is kept."""
        LOGGER.info("listening for a reply to %s", message_id)
        self._open_addressed()
        self._reply_to = message_id

    def listen_for_answer(self) -> None:
        """Open a window without the wake word: the room just asked or answered (FR-93).

        Whatever was being overheard is let go -- what is said now is meant
        for the assistant.
        """
        LOGGER.info("listening for an answer to the reply")
        self._open_addressed()

    def _wake(self) -> None:
        LOGGER.info("wake word detected; opening capture window")
        self._open_addressed()

    def _open_addressed(self) -> None:
        self._reply_to = ""
        self._window_frames = 0
        self._detector.reset()
        self._utterance.reset()
        self._buffer = []
        self._preroll.clear()
        self._addressed = True
        self._state = PipelineState.LISTENING
        self._tell(SpeechActivity.LISTENING)

    def _maybe_overhear(self, frame) -> None:
        """Speech nobody woke the assistant for opens an ambient window (FR-76)."""
        self._utterance.accept(frame)
        if not self._utterance.speech_started:
            self._preroll.append(frame)
            return
        self._buffer = [*self._preroll, frame]
        self._preroll.clear()
        # The endpointer has counted every waiting frame; the hard cap on an
        # utterance runs from here.
        self._utterance.begin_counting()
        self._addressed = False
        self._state = PipelineState.LISTENING
        self._tell(SpeechActivity.OVERHEARING)

    def _listen(self, frame) -> PreferenceHint | AmbientUtterance | None:
        if not self._addressed and self._detector.heard(frame):
            # Addressed after all: what came before was not for the assistant.
            self._wake()
            return None
        self._buffer.append(frame)
        self._window_frames += 1
        finished = self._utterance.accept(frame)

        waiting_longer = self._reply_to and self._window_frames <= self._reply_wait_frames
        if self._utterance.silence_timed_out and not waiting_longer:
            LOGGER.info("no speech in the capture window; closing it")
            return self._close(transcribe=False)

        if finished:
            return self._close(transcribe=True)
        return None

    def _close(self, transcribe: bool) -> PreferenceHint | AmbientUtterance | MessageReply | None:
        """Close the capture window and discard the audio either way -- unless
        it was a reply to a message, which is kept to be sent back (FR-95)."""
        frames, self._buffer = self._buffer, []
        addressed, self._addressed = self._addressed, False
        reply_to, self._reply_to = self._reply_to, ""
        self._state = PipelineState.WAITING_FOR_WAKE_WORD
        self._capture.drain()
        if self._continuous:
            # Waiting runs the endpointer too, so it starts the next utterance clean.
            self._utterance.reset()

        if not transcribe:
            self._tell(SpeechActivity.WAITING)
            return None
        self._tell(SpeechActivity.TRANSCRIBING if addressed else SpeechActivity.OVERHEARING)
        try:
            if reply_to:
                return self._reply(reply_to, frames)
            return self._understand(frames) if addressed else self._overheard(frames)
        finally:
            self._tell(SpeechActivity.WAITING)

    def _reply(self, message_id: str, frames: list) -> MessageReply:
        """The answer to a message, words and voice, never sent to the assistant."""
        transcript = self._transcriber.transcribe(frames) or ""
        LOGGER.info("a reply to %s: %d words", message_id, len(transcript.split()))
        pcm = np.concatenate([np.asarray(frame, dtype=np.int16) for frame in frames]).tobytes() if frames else b""
        duration_s = len(frames) * self._config.chunk_samples / self._config.sample_rate_hz
        return MessageReply(message_id, transcript, pcm, self._config.sample_rate_hz, duration_s)

    def _overheard(self, frames: list) -> AmbientUtterance | None:
        """Words nobody addressed to the assistant, for the reasoning layer to read.

        Never shown in the retained status: the room's conversation is not
        state to be kept for whoever connects next (FR-76).
        """
        transcript = self._transcriber.transcribe(frames, confident_only=True)
        # Distinct words, not words: live, room noise came back from Whisper
        # as "No. No. No. No." -- four words, one of them, and nothing said.
        words = [word.strip(".,!?;:'\"").lower() for word in (transcript or "").split()]
        words = [word for word in words if word]
        distinct = set(words)
        if len(distinct) < self._config.ambient_min_words:
            # Counted, never quoted: what the room said is not for the log (FR-51).
            LOGGER.info("overheard speech dropped: %d distinct confident words, fewer than %d", len(distinct), self._config.ambient_min_words)
            return None
        if len(words) >= _LOOP_CHECK_WORDS and len(distinct) / len(words) < self._config.ambient_min_distinct_ratio:
            # "We are going home We are going home No we are going home ...":
            # a transcript looping on itself is Whisper's, not anyone's (S-01).
            LOGGER.info("overheard speech dropped: it repeats itself")
            return None
        LOGGER.info("overheard %d words", len(transcript.split()))
        duration_s = len(frames) * self._config.chunk_samples / self._config.sample_rate_hz
        return AmbientUtterance(
            ts=self._clock.now(),
            transcript=transcript[:1000],
            profile_id=self._speaker_of(frames),
            duration_s=duration_s,
        )

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
