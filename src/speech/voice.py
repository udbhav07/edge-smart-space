"""The room speaks its replies (FR-79, FR-82, FR-83).

A voice assistant that only answers on a screen sends the occupant to the
screen after every "Hey Jarvis". This says the reply aloud, here: Piper, a
neural voice whose model was fetched once by ``setup_models.py``, or the
system's own voices through pyttsx3 when Piper cannot load. Nothing is
fetched at runtime (NFR-06), and no audio is kept.

What is spoken, and when, is the occupant's choice, held in the retained
:class:`SpeechSettings`:

* a reply to a **spoken** request, unless replies are turned off;
* a reply to a **typed** one, only if asked for;
* a **change of mode**, in the same plain words the console's inbox uses
  (:mod:`src.common.plain`), unless alerts are off -- or it is quiet hours.
  Quiet hours silence only what nobody asked for (FR-83).

**The room must never hear itself.** It listens continuously (FR-76); a
reply played into a live microphone would come back as overheard speech, or
as a request. While speaking, and for ``speaking_tail_ms`` after, :attr:`busy`
is true and the pipeline drops every frame (``src/speech/__main__.py``).

**A reply is a turn in a conversation** (FR-93). Once a spoken reply has been
said, :meth:`take_follow_up` offers -- once -- to open a capture window
without the wake word, so "Sorry, could you say that again?" is answered
like any question. Whenever a window opens, :meth:`chime` plays two short
notes so the occupant knows the room is listening; while they play,
:attr:`chiming` is true and the microphone's frames are dropped, but the
window stays open.

Speaking runs on its own thread from a bounded queue: MQTT callbacks only
queue, as everywhere else, and a long reply never stalls the microphone loop.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time as day_time
from typing import Protocol

import numpy as np

from src.common.clock import Clock
from src.common.config import SpeechConfig
from src.common.localtime import local_time
from src.common.plain import mode_sentence
from src.common.schemas import AssistantReply, Mode, ModeState, SpeechSettings

LOGGER = logging.getLogger(__name__)

#: Replies waiting to be said. Bounded: a backlog is dropped, oldest first,
#: rather than a room still talking about something minutes old.
_QUEUE_LENGTH = 4

_INT16_FULL_SCALE = 32767

#: The chime: two rising notes, short enough not to talk over the occupant.
_CHIME_RATE_HZ = 22050
_CHIME_NOTES_HZ = (660.0, 990.0)
_CHIME_NOTE_S = 0.09
_CHIME_FADE_S = 0.012

#: Plays 16-bit mono PCM at a sample rate, and returns when it has played.
PcmPlayer = Callable[[bytes, int], None]

#: How long after a message has played its reply window may still open, in s:
#: a window offered while the microphone was muted is not opened minutes later.
_MESSAGE_WINDOW_FRESH_S = 30.0


@dataclass
class _Item:
    """One thing to say: a sentence, or a message home (FR-95)."""

    text: str
    reply: bool = False
    #: A recording to play after ``text``: a message from someone away.
    pcm: bytes = b""
    rate_hz: int = 0
    message_id: str = ""
    on_start: Callable[[], None] | None = None
    on_end: Callable[[], None] | None = None


def chime_pcm(volume: float) -> bytes:
    """The chime as 16-bit mono PCM: two notes, each faded in and out so it never clicks."""
    per_note = int(_CHIME_RATE_HZ * _CHIME_NOTE_S)
    fade = int(_CHIME_RATE_HZ * _CHIME_FADE_S)
    envelope = np.ones(per_note)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    t = np.arange(per_note) / _CHIME_RATE_HZ
    notes = [np.sin(2 * np.pi * frequency * t) * envelope for frequency in _CHIME_NOTES_HZ]
    samples = np.concatenate(notes) * max(0.0, min(1.0, volume))
    return (samples * _INT16_FULL_SCALE).astype(np.int16).tobytes()


def chime_seconds() -> float:
    return len(_CHIME_NOTES_HZ) * _CHIME_NOTE_S


def play_with_pyaudio(pcm: bytes, rate_hz: int) -> None:
    """The default output: the system's default device, through PyAudio."""
    import pyaudio

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=rate_hz, output=True)
        try:
            stream.write(pcm)
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        audio.terminate()


class Speaker(Protocol):
    """Says one sentence and returns when it has been said."""

    name: str

    def say(self, text: str, volume: float) -> None: ...


class PiperSpeaker:
    """A neural voice run on this node, played through PyAudio."""

    name = "piper"

    def __init__(self, model_path: str, length_scale: float = 1.0) -> None:
        from piper import PiperVoice
        from piper.config import SynthesisConfig

        self._voice = PiperVoice.load(model_path)
        self._config = SynthesisConfig(length_scale=length_scale)
        # The first synthesis loads the runtime; doing it now keeps it out of
        # the first reply's latency.
        for _ in self._voice.synthesize("ready", syn_config=self._config):
            pass

    def render(self, text: str, volume: float = 1.0) -> tuple[bytes, int]:
        """The sentence as 16-bit mono PCM and its sample rate, not played: for a
        reply that is heard on someone's phone rather than in the room (FR-94)."""
        chunks = list(self._voice.synthesize(text, syn_config=self._config))
        if not chunks:
            return b"", 22050
        samples = np.concatenate([chunk.audio_float_array for chunk in chunks]) * volume
        return (np.clip(samples, -1.0, 1.0) * _INT16_FULL_SCALE).astype(np.int16).tobytes(), chunks[0].sample_rate

    def say(self, text: str, volume: float) -> None:
        pcm, rate = self.render(text, volume)
        if pcm:
            play_with_pyaudio(pcm, rate)


class SystemSpeaker:
    """The operating system's own voices, through pyttsx3: always present."""

    name = "pyttsx3"

    def __init__(self) -> None:
        import pyttsx3

        self._engine = pyttsx3.init()

    def say(self, text: str, volume: float) -> None:
        self._engine.setProperty("volume", volume)
        self._engine.say(text)
        self._engine.runAndWait()


def build_speaker(config: SpeechConfig) -> Speaker | None:
    """The configured voice, falling back rather than falling silent (FR-79).

    :returns: None only when speaking is configured off, or no engine at all
        can start -- and then the room still works, it just does not talk.
    """
    if config.voice_engine == "none":
        return None
    if config.voice_engine == "piper":
        try:
            return PiperSpeaker(config.voice_model_path, config.voice_length_scale)
        except (OSError, RuntimeError, ValueError) as exc:
            # A missing model file or no audio output: the system's voices
            # are better than silence. Run setup_models.py to fetch the model.
            LOGGER.warning("Piper voice unavailable (%s); using the system's voices", exc)
    try:
        return SystemSpeaker()
    except (OSError, RuntimeError) as exc:
        # No speech engine on this machine (espeak on Linux, SAPI on Windows).
        LOGGER.warning("no voice available; replies will not be spoken: %s", exc)
        return None


class Voice:
    """Decides what is said, says it on its own thread, and guards the microphone."""

    def __init__(
        self,
        speaker: Speaker | None,
        clock: Clock,
        utc_offset_h: float,
        tail_s: float,
        on_speaking=None,
        chime: bool = True,
        chime_volume: float = 0.35,
        chime_tail_s: float = 0.15,
        follow_up: bool = True,
        player: PcmPlayer | None = play_with_pyaudio,
    ) -> None:
        """
        :param player: plays the chime; None plays nothing, which is what a
            test or a machine with no audio output needs.
        """
        self._speaker = speaker
        self._clock = clock
        self._utc_offset_h = utc_offset_h
        self._tail_s = tail_s
        #: Told True when speaking starts and False when it ends, for the status.
        self._on_speaking = on_speaking
        self._settings = SpeechSettings(ts=clock.now())
        self._mode: Mode | None = None
        self._queue: queue.Queue[_Item] = queue.Queue(maxsize=_QUEUE_LENGTH)
        self._speaking = threading.Event()
        self._quiet_after_s = 0.0
        self._thread: threading.Thread | None = None
        self._chime = chime
        self._chime_volume = chime_volume
        self._chime_tail_s = chime_tail_s
        self._player = player
        self._chiming_until_s = 0.0
        self._follow_up = follow_up
        #: A reply was just said; the next quiet moment opens a window for the answer.
        self._answer_awaited = threading.Event()
        #: A message home was just played: (message id, when), for its reply window.
        self._message_awaited: tuple[str, float] | None = None

    @property
    def engine(self) -> str:
        return self._speaker.name if self._speaker is not None else ""

    @property
    def settings(self) -> SpeechSettings:
        return self._settings

    @property
    def busy(self) -> bool:
        """Whether the microphone must be ignored: speaking, or just finished."""
        return self._speaking.is_set() or self._clock.monotonic() < self._quiet_after_s or not self._queue.empty()

    @property
    def chiming(self) -> bool:
        """Whether the chime is playing, or only just stopped: its sound is not speech."""
        return self._clock.monotonic() < self._chiming_until_s

    def take_follow_up(self) -> bool:
        """Whether to open a window for an answer now. True once per reply said (FR-93).

        Only when the room has finished talking, tail included: the window
        must open on the occupant's turn, not on the end of its own sentence.
        """
        if not self._answer_awaited.is_set() or self.busy:
            return False
        self._answer_awaited.clear()
        return self._follow_up

    def take_message_window(self) -> str | None:
        """The message to listen for an answer to, once it has finished playing (FR-95).

        None if there is none, if the room is still talking, or if the moment
        has passed: a window is not opened on a reply nobody is still giving.
        """
        awaited = self._message_awaited
        if awaited is None or self.busy:
            return None
        self._message_awaited = None
        message_id, played_at = awaited
        return message_id if self._clock.monotonic() - played_at <= _MESSAGE_WINDOW_FRESH_S else None

    def play_message(
        self,
        message_id: str,
        announce: str,
        pcm: bytes,
        rate_hz: int,
        on_start: Callable[[], None] | None = None,
        on_end: Callable[[], None] | None = None,
    ) -> bool:
        """Queue a message home: a chime, who it is from, then their own recording (FR-95).

        Played as one item, so nothing is said between the announcement and
        the voice it announces. Returns whether it was queued.
        """
        if self._player is None or not pcm:
            return False
        return self._enqueue(_Item(text=announce, pcm=pcm, rate_hz=rate_hz, message_id=message_id, on_start=on_start, on_end=on_end))

    def chime(self) -> None:
        """Say "I'm listening" without words: two short notes, on a thread of their own."""
        if not self._chime or self._player is None:
            return
        volume = self._chime_volume * self._settings.volume
        self._chiming_until_s = self._clock.monotonic() + chime_seconds() + self._chime_tail_s
        threading.Thread(target=self._play_chime, args=(volume,), name="chime", daemon=True).start()

    def _play_chime(self, volume: float) -> None:
        try:
            self._player(chime_pcm(volume), _CHIME_RATE_HZ)
        except (OSError, RuntimeError, ValueError) as exc:
            # A missing chime costs a cue, never the window it announces.
            LOGGER.warning("could not play the chime: %s", exc)

    def start(self) -> None:
        # A message home plays even with no voice to announce it (FR-95).
        if (self._speaker is None and self._player is None) or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()

    # --- what to say --------------------------------------------------

    def on_settings(self, _topic: str, settings: SpeechSettings) -> None:
        self._settings = settings

    def on_reply(self, _topic: str, reply: AssistantReply) -> None:
        if reply.remote:
            # Asked from a phone away from the room: answered there (FR-94).
            return
        settings = self._settings
        wanted = settings.speak_replies if reply.spoken else settings.speak_typed
        if wanted:
            self.say(reply.reply, reply=True)

    def on_mode(self, _topic: str, state: ModeState) -> None:
        """A change of mode, said in plain words -- not the first one heard,
        which is only the state the room was already in."""
        before, self._mode = self._mode, state.mode
        if before is None or before is state.mode:
            return
        if self._settings.speak_alerts and not self.quiet_now():
            self.say(mode_sentence(state.mode))

    def quiet_now(self) -> bool:
        """Whether it is within the occupant's quiet hours (FR-83)."""
        start, end = self._settings.quiet_from, self._settings.quiet_to
        if not start or not end:
            return False
        now = local_time(self._clock.now(), self._utc_offset_h).time()
        begin, finish = _hhmm(start), _hhmm(end)
        if begin <= finish:
            return begin <= now < finish
        return now >= begin or now < finish  # across midnight: 22:00-07:00

    def say(self, text: str, reply: bool = False) -> bool:
        """Queue a sentence. Returns whether it was queued.

        :param reply: an answer to something the occupant said, after which
            the room listens for theirs (FR-93); an alert is not.
        """
        if self._speaker is None or not text.strip():
            return False
        return self._enqueue(_Item(text=text.strip(), reply=reply))

    def _enqueue(self, item: _Item) -> bool:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            LOGGER.info("voice backlog full; dropping the oldest sentence")
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)
        return True

    # --- saying it ------------------------------------------------------

    def speak_next(self, timeout_s: float = 0.5) -> bool:
        """Say one queued sentence, if any. Returns whether one was said."""
        try:
            item = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return False
        self._speaking.set()
        self._tell(True)
        _call(item.on_start)
        try:
            if item.pcm:
                self._play_message(item)
            else:
                self._speaker.say(item.text, self._settings.volume)
        except (OSError, RuntimeError, ValueError) as exc:
            # A playback failure costs this sentence, never the service.
            LOGGER.warning("could not speak: %s", exc)
        finally:
            self._quiet_after_s = self._clock.monotonic() + self._tail_s
            if item.reply:
                self._answer_awaited.set()
            if item.message_id:
                self._message_awaited = (item.message_id, self._clock.monotonic())
            self._speaking.clear()
            self._tell(False)
            _call(item.on_end)
        return True

    def _play_message(self, item: _Item) -> None:
        """The chime, who it is from, and the recording, while the microphone is held."""
        volume = self._settings.volume
        if self._chime:
            self._player(chime_pcm(self._chime_volume * volume), _CHIME_RATE_HZ)
        if item.text and self._speaker is not None:
            self._speaker.say(item.text, volume)
        samples = np.frombuffer(item.pcm, dtype=np.int16).astype(np.float32) * volume
        self._player(np.clip(samples, -_INT16_FULL_SCALE, _INT16_FULL_SCALE).astype(np.int16).tobytes(), item.rate_hz)

    def _run(self) -> None:
        while True:
            self.speak_next()

    def _tell(self, speaking: bool) -> None:
        if self._on_speaking is None:
            return
        try:
            self._on_speaking(speaking)
        except (OSError, ValueError) as exc:  # showing it is a courtesy
            LOGGER.warning("could not report speaking: %s", exc)


def _hhmm(value: str) -> day_time:
    return datetime.strptime(value, "%H:%M").time()


def _call(callback: Callable[[], None] | None) -> None:
    """Tell whoever asked; a failure in telling never stops the playing."""
    if callback is None:
        return
    try:
        callback()
    except (OSError, ValueError, RuntimeError) as exc:
        LOGGER.warning("could not report a message's progress: %s", exc)
