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

Speaking runs on its own thread from a bounded queue: MQTT callbacks only
queue, as everywhere else, and a long reply never stalls the microphone loop.
"""

from __future__ import annotations

import logging
import queue
import threading
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

    def say(self, text: str, volume: float) -> None:
        import pyaudio

        chunks = list(self._voice.synthesize(text, syn_config=self._config))
        if not chunks:
            return
        samples = np.concatenate([chunk.audio_float_array for chunk in chunks]) * volume
        pcm = (np.clip(samples, -1.0, 1.0) * _INT16_FULL_SCALE).astype(np.int16).tobytes()
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=chunks[0].sample_rate, output=True)
            try:
                stream.write(pcm)
            finally:
                stream.stop_stream()
                stream.close()
        finally:
            audio.terminate()


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
    ) -> None:
        self._speaker = speaker
        self._clock = clock
        self._utc_offset_h = utc_offset_h
        self._tail_s = tail_s
        #: Told True when speaking starts and False when it ends, for the status.
        self._on_speaking = on_speaking
        self._settings = SpeechSettings(ts=clock.now())
        self._mode: Mode | None = None
        self._queue: queue.Queue[str] = queue.Queue(maxsize=_QUEUE_LENGTH)
        self._speaking = threading.Event()
        self._quiet_after_s = 0.0
        self._thread: threading.Thread | None = None

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

    def start(self) -> None:
        if self._speaker is None or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self._thread.start()

    # --- what to say --------------------------------------------------

    def on_settings(self, _topic: str, settings: SpeechSettings) -> None:
        self._settings = settings

    def on_reply(self, _topic: str, reply: AssistantReply) -> None:
        settings = self._settings
        wanted = settings.speak_replies if reply.spoken else settings.speak_typed
        if wanted:
            self.say(reply.reply)

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

    def say(self, text: str) -> bool:
        """Queue a sentence. Returns whether it was queued."""
        if self._speaker is None or not text.strip():
            return False
        try:
            self._queue.put_nowait(text.strip())
        except queue.Full:
            LOGGER.info("voice backlog full; dropping the oldest sentence")
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(text.strip())
        return True

    # --- saying it ------------------------------------------------------

    def speak_next(self, timeout_s: float = 0.5) -> bool:
        """Say one queued sentence, if any. Returns whether one was said."""
        try:
            text = self._queue.get(timeout=timeout_s)
        except queue.Empty:
            return False
        self._speaking.set()
        self._tell(True)
        try:
            self._speaker.say(text, self._settings.volume)
        except (OSError, RuntimeError, ValueError) as exc:
            # A playback failure costs this sentence, never the service.
            LOGGER.warning("could not speak: %s", exc)
        finally:
            self._quiet_after_s = self._clock.monotonic() + self._tail_s
            self._speaking.clear()
            self._tell(False)
        return True

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
