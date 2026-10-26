"""Messages home, as they come in from someone away (FR-95).

A recording made on the Talk page arrives here. It is decoded, transcribed
for the screen, and kept as a 16-bit mono WAV in the messages directory --
the one recording the console keeps, because keeping it is the point: it is
to be played in the room in the sender's own voice. The board carries only
its words and its file name (``space/message/incoming``); the speech service
reads the file and plays it.

Recordings and replies are deleted after ``keep_days``, and the oldest go
first beyond ``max_kept``; each deleted message's retained status is
withdrawn with it, so nothing about it outlives its audio.
"""

from __future__ import annotations

import logging
import re
import secrets
import wave
from pathlib import Path

import numpy as np

from src.common import topics
from src.common.clock import Clock
from src.common.config import MessagesConfig
from src.common.mqtt_client import Blackboard
from src.common.schemas import MESSAGE_ID_PATTERN, HomeMessage, MessageState, MessageStatus
from src.console.remote_voice import SAMPLE_RATE_HZ, RemoteVoice, RemoteVoiceUnavailable

LOGGER = logging.getLogger(__name__)

_INT16_FULL_SCALE = 32767
_SECONDS_PER_DAY = 86400.0
_MESSAGE_ID = re.compile(MESSAGE_ID_PATTERN)


class MessageBox:
    """Takes a message from someone away, keeps it, and hands it to the room."""

    def __init__(self, config: MessagesConfig, clock: Clock, blackboard: Blackboard, voice: RemoteVoice) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._voice = voice
        self._directory = Path(config.directory)

    @property
    def max_bytes(self) -> int:
        return self._config.max_bytes

    def send(self, sender: str, recording: bytes) -> MessageStatus:
        """Keep a recording and hand it to the room to play now.

        Slow -- it transcribes -- so it is called outside the console's lock.

        :raises ValueError: no sender, or the recording is empty, unreadable,
            too short or too long; said so the sender can try again.
        :raises RemoteVoiceUnavailable: nothing to decode with on this node.
        """
        sender = " ".join(sender.split())[:40]
        if not sender:
            raise ValueError("Say who the message is from.")
        audio = self._voice.decode(recording, max_seconds=self._config.max_seconds)
        try:
            transcript = self._voice.transcribe_samples(audio)[:1000]
        except RemoteVoiceUnavailable as exc:
            # The message is the voice; its words on screen are a courtesy.
            LOGGER.info("a message is sent without its words: %s", exc)
            transcript = ""
        now = self._clock.now()
        message_id = f"msg_{int(now)}_{secrets.token_hex(3)}"
        pcm = (np.clip(audio, -1.0, 1.0) * _INT16_FULL_SCALE).astype(np.int16).tobytes()
        write_recording(self._directory / f"{message_id}.wav", pcm, SAMPLE_RATE_HZ)
        duration_s = max(audio.size / SAMPLE_RATE_HZ, 0.1)
        status = MessageStatus(
            ts=now, message_id=message_id, sender=sender, transcript=transcript,
            duration_s=duration_s, state=MessageState.SENT, sent_ts=now,
        )
        self._blackboard.publish(topics.MESSAGE_STATUS, status, message_id=message_id)
        self.hand_over(status)
        LOGGER.info("message %s from %s kept (%.1f s) and handed to the room", message_id, sender, duration_s)
        return status

    def hand_over(self, status: MessageStatus | dict) -> None:
        """Ask the room to play a kept message: on arrival, or again (FR-95).

        :raises FileNotFoundError: its recording is no longer kept.
        """
        status = MessageStatus.model_validate(status)
        if self.recording(status.message_id) is None:
            raise FileNotFoundError("That message is no longer kept.")
        self._blackboard.publish(
            topics.MESSAGE_INCOMING,
            HomeMessage(
                ts=status.sent_ts, message_id=status.message_id, sender=status.sender,
                transcript=status.transcript, duration_s=status.duration_s, audio_file=f"{status.message_id}.wav",
            ),
        )

    def recording(self, message_id: str, reply: bool = False) -> Path | None:
        """A kept recording's file, or None if there is none by that id."""
        if not _MESSAGE_ID.match(message_id):
            return None
        path = self._directory / f"{message_id}{'-reply' if reply else ''}.wav"
        return path if path.is_file() else None

    def prune(self, known: list[dict]) -> list[str]:
        """Delete what is past keeping; returns the message ids withdrawn.

        :param known: every message's status as the board has it.
        """
        now = self._clock.now()
        oldest_s = now - self._config.keep_days * _SECONDS_PER_DAY
        newest_first = sorted(known, key=lambda status: status["sent_ts"], reverse=True)
        kept, withdrawn = set(), []
        for index, status in enumerate(newest_first):
            message_id = status["message_id"]
            if index < self._config.max_kept and status["sent_ts"] >= oldest_s:
                kept.add(message_id)
                continue
            for path in (self._directory / f"{message_id}.wav", self._directory / f"{message_id}-reply.wav"):
                path.unlink(missing_ok=True)
            self._blackboard.clear_retained(topics.MESSAGE_STATUS, message_id=message_id)
            withdrawn.append(message_id)
        # A file whose status the board no longer has, once it is old enough
        # that it cannot be one still on its way.
        if self._directory.is_dir():
            for path in self._directory.glob("msg_*.wav"):
                if path.stem.removesuffix("-reply") not in kept and path.stat().st_mtime < oldest_s:
                    path.unlink(missing_ok=True)
        if withdrawn:
            LOGGER.info("messages past keeping deleted: %s", ", ".join(withdrawn))
        return withdrawn


def write_recording(path: Path, pcm: bytes, rate_hz: int) -> None:
    """A WAV, written whole or not at all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    with wave.open(str(temporary), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(rate_hz)
        file.writeframes(pcm)
    temporary.replace(path)
