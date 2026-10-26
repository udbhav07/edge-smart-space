"""Messages home, played in the room in the sender's own voice (FR-95).

Someone away -- "my mom never picks up the phone" -- records a message on the
console's Talk page. The console keeps the recording as a file in the
messages directory and announces it on ``space/message/incoming``. Here it is
played at once: a chime, "Message from Ravi", and Ravi's own voice. Then the
room listens for an answer, without the wake word; whatever is said is kept
-- the one utterance whose audio is kept, because it was invited to be sent
back -- and the sender sees it on their page.

Every step is published as the message's retained status, so the sender's
ticks (sent, played, replied) and the room's inbox read the same record.
"""

from __future__ import annotations

import logging
import wave
from collections import OrderedDict
from pathlib import Path

from src.common import topics
from src.common.clock import Clock
from src.common.config import MessagesConfig
from src.common.mqtt_client import Blackboard
from src.common.schemas import HomeMessage, MessageState, MessageStatus
from src.speech.pipeline import MessageReply
from src.speech.voice import Voice

LOGGER = logging.getLogger(__name__)


class MessageDesk:
    """Plays each message home, keeps the reply, and tells everyone how it went."""

    def __init__(self, config: MessagesConfig, clock: Clock, blackboard: Blackboard, voice: Voice) -> None:
        self._config = config
        self._clock = clock
        self._blackboard = blackboard
        self._voice = voice
        self._directory = Path(config.directory)
        self._records: OrderedDict[str, MessageStatus] = OrderedDict()

    def subscribe(self) -> None:
        self._blackboard.subscribe(topics.MESSAGE_INCOMING, HomeMessage, self._on_incoming)

    def _on_incoming(self, _topic: str, message: HomeMessage) -> None:
        self._records[message.message_id] = MessageStatus(
            ts=self._clock.now(),
            message_id=message.message_id,
            sender=message.sender,
            transcript=message.transcript,
            duration_s=message.duration_s,
            state=MessageState.SENT,
            sent_ts=message.ts,
        )
        while len(self._records) > self._config.max_kept:
            self._records.popitem(last=False)
        try:
            pcm, rate_hz = read_recording(self._directory / message.audio_file)
        except (OSError, ValueError) as exc:
            LOGGER.warning("message %s could not be read: %s", message.message_id, exc)
            self._tell(message.message_id, MessageState.FAILED, detail="The recording could not be read.")
            return
        announce = f"Message from {message.sender}." if self._config.announce else ""
        queued = self._voice.play_message(
            message.message_id,
            announce,
            pcm,
            rate_hz,
            on_start=lambda: self._tell(message.message_id, MessageState.PLAYING),
            on_end=lambda: self._tell(message.message_id, MessageState.PLAYED, played_ts=self._clock.now()),
        )
        if not queued:
            self._tell(message.message_id, MessageState.FAILED, detail="This room has no speaker to play it on.")
        else:
            LOGGER.info("message %s from %s queued to play", message.message_id, message.sender)

    def reply(self, heard: MessageReply) -> None:
        """Keep what home said back, and let the sender hear it."""
        if not heard.pcm or not heard.transcript.strip():
            # A window that closed on silence or noise: the message stays "played".
            return
        name = f"{heard.message_id}-reply.wav"
        try:
            write_recording(self._directory / name, heard.pcm, heard.rate_hz)
        except OSError as exc:
            LOGGER.warning("the reply to %s could not be kept: %s", heard.message_id, exc)
            return
        self._tell(
            heard.message_id,
            MessageState.REPLIED,
            replied_ts=self._clock.now(),
            reply_transcript=heard.transcript.strip()[:1000],
            reply_duration_s=max(heard.duration_s, 0.1),
            reply_file=name,
        )
        LOGGER.info("reply to %s kept (%.1f s)", heard.message_id, heard.duration_s)

    def _tell(self, message_id: str, state: MessageState, **fields) -> None:
        record = self._records.get(message_id)
        if record is None:
            LOGGER.info("status for %s, a message this service never received, not published", message_id)
            return
        record = record.model_copy(update={"ts": self._clock.now(), "state": state, **fields})
        self._records[message_id] = record
        self._blackboard.publish(topics.MESSAGE_STATUS, record, message_id=message_id)


def read_recording(path: Path) -> tuple[bytes, int]:
    """16-bit mono PCM and its rate, from a WAV the console wrote.

    :raises ValueError: anything else -- the console writes only this.
    """
    with wave.open(str(path), "rb") as file:
        if file.getnchannels() != 1 or file.getsampwidth() != 2:
            raise ValueError("not 16-bit mono")
        return file.readframes(file.getnframes()), file.getframerate()


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
