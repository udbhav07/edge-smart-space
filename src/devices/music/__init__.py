"""The music player on this machine, as a device service (FR-87, FR-96).

It plays files from the occupant's own music folder (``actuators.music``):
mp3, flac, wav and ogg, decoded and played by miniaudio. "Play some jazz"
reaches it as ``play(query="jazz")``, matched against file and folder names;
a file the occupant hands over in the console is saved to the uploads folder
and played by name, so only a name ever crosses the blackboard, never audio.

Asked for something the library does not have -- or asked for it "from
YouTube" -- it searches online and streams the song (:mod:`.youtube`,
:mod:`.stream`). The library always comes first, and "play some music",
which names nothing, never leaves it.

It behaves as the room's music should (FR-96):

- a song asked for aloud waits for Jarvis to finish saying so;
- the music is turned down while the room listens or speaks (:mod:`.room`);
- "stop" keeps the place, and "carry on" carries on from it;
- when a song ends it plays the next -- what YouTube would autoplay after it,
  or another from the library -- unless autoplay is off.

The decision of *what* to play lives in :class:`MusicDevice` and the sound
itself in a :class:`Player`, so the first is tested without a sound card.
"""

from __future__ import annotations

import logging
import random
import re
import threading
from collections import deque
from collections.abc import Callable, Generator, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from src.common import topics
from src.common.clock import Clock
from src.common.config import Config, MusicConfig
from src.common.devices import DeviceCapability, DeviceClass, DeviceKind, DeviceManifest
from src.common.mqtt_client import Blackboard
from src.common.schemas import AssistantReply, PreferenceHint, SpeechStatus
from src.common.tools import ArgumentValue, ParameterType, ProviderOutcome, ToolEffect, ToolParameter
from src.devices.music.room import RoomVoice
from src.devices.music.stream import StreamDecoder, pcm_source
from src.devices.music.track import Track, tidy
from src.devices.music.youtube import Candidate, CatalogueUnavailable, YouTubeCatalogue
from src.devices.runner import DeviceParts
from src.devices.service import DeviceService

__all__ = [
    "Candidate", "Catalogue", "CatalogueUnavailable", "Library", "MiniaudioPlayer", "MusicDevice", "Player", "RoomVoice", "Track",
    "build", "manifest",
]

LOGGER = logging.getLogger(__name__)

#: Words in a request that say nothing about which track: "play some music for me".
_FILLER = frozenset(
    "a an and any by for from me my music of on play please put some song songs something the track tracks to tune tunes".split()
)
#: Words that ask for the song to be found online, whatever the library has.
_ONLINE = frozenset("youtube online internet".split())
#: Songs remembered as played, so autoplay does not come back round to them.
_PLAYED_KEPT = 50
#: Songs held ready to play next.
_UP_NEXT_KEPT = 10


class Player(Protocol):
    """Makes the sound. One track at a time."""

    def play(self, track: Track, start_s: float = 0.0, paused: bool = False) -> None:
        """A file or a stream, from ``start_s`` in; opened but silent if ``paused``.

        :raises ValueError: it could not be opened, in words for the occupant.
        """

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def set_volume(self, volume: float) -> None:
        """0.0 to 1.0."""

    @property
    def finished(self) -> bool:
        """Whether the track that was playing has run out by itself."""

    @property
    def broken(self) -> bool:
        """Whether it ran out because a stream's connection dropped, not because the song ended."""

    @property
    def position_s(self) -> float:
        """How far into the track it has played."""


class Catalogue(Protocol):
    """Songs beyond the library, found by words, and what follows them (FR-96)."""

    def find(self, query: str) -> Track | None:
        """:raises CatalogueUnavailable: it could not be searched."""

    def resolve(self, candidate: Candidate) -> Track | None:
        """:raises CatalogueUnavailable: it could not be reached."""

    def up_next(self, video_id: str) -> list[Candidate]:
        """:raises CatalogueUnavailable: it could not be reached."""


def _words(text: str) -> list[str]:
    return [word for word in re.split(r"[^a-z0-9]+", text.lower()) if word]


class Library:
    """The tracks under the configured folders, found by words or by name."""

    def __init__(self, roots: Iterable[Path], extensions: Iterable[str], rng: random.Random | None = None) -> None:
        self._roots = [root.expanduser() for root in roots]
        self._extensions = {extension.lower() for extension in extensions}
        self._rng = rng or random.Random()

    def tracks(self) -> list[Path]:
        """Every playable file, in a stable order. Read afresh, so new files count at once."""
        found: list[Path] = []
        for root in self._roots:
            if root.is_dir():
                found.extend(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in self._extensions)
        return sorted(found, key=lambda path: str(path).lower())

    def find(self, query: str) -> Path | None:
        """The track whose name and folders share most words with the request.

        :returns: None if the request names something and nothing matches it;
            a random track if it names nothing at all ("play some music").
        """
        tracks = self.tracks()
        if not tracks:
            return None
        wanted = [word for word in _words(query) if word not in _FILLER]
        if not wanted:
            return self._rng.choice(tracks)
        scored = [(sum(word in self._words_of(track) for word in wanted), track) for track in tracks]
        best = max(score for score, _ in scored)
        return next(track for score, track in scored if score == best) if best else None

    def another(self, besides: Path | None) -> Path | None:
        """A track to follow ``besides``: any other, at random."""
        others = [track for track in self.tracks() if track != besides]
        return self._rng.choice(others) if others else None

    def named(self, name: str) -> Path | None:
        """A track by its file name, with or without the extension. Never a path: a
        name from the board must not reach outside the music folders."""
        wanted = Path(name).name.lower()
        for track in self.tracks():
            if wanted in (track.name.lower(), track.stem.lower()):
                return track
        return None

    def _words_of(self, track: Path) -> set[str]:
        root = next((root for root in self._roots if track.is_relative_to(root)), track.parent)
        return set(_words(" ".join(track.relative_to(root).with_suffix("").parts)))


class MusicDevice:
    """What to play, what is playing, and what comes next."""

    def __init__(
        self,
        library: Library,
        player: Player,
        config: MusicConfig,
        catalogue: Catalogue | None = None,
        room: RoomVoice | None = None,
        clock: Clock | None = None,
        background: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        """
        :param catalogue: where to look beyond the library; None keeps to it.
        :param room: what the room's voice is doing; None never waits or turns down.
        :param clock: needed with ``room``, to stop waiting for a reply that never comes.
        :param background: runs slow work -- fetching what plays next -- off
            the caller's thread; a daemon thread if not given.
        """
        self._library = library
        self._player = player
        self._config = config
        self._catalogue = catalogue
        self._room = room
        self._clock = clock
        self._background = background or (lambda job: threading.Thread(target=job, name="music-next", daemon=True).start())
        self._lock = threading.RLock()
        self._track: Track | None = None
        self._paused = False
        #: Held until Jarvis has said he is playing it: when to give up waiting.
        self._cued_until: float | None = None
        #: Changed off the tick -- a held song started as Jarvis finished -- and not yet published.
        self._unpublished = False
        self._volume = config.default_volume
        self._autoplay = config.autoplay
        self._up_next: deque[Candidate] = deque(maxlen=_UP_NEXT_KEPT)
        self._played: deque[str] = deque(maxlen=_PLAYED_KEPT)
        player.set_volume(self._volume / 100)

    # --- capabilities ------------------------------------------------------

    def play(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        name, query = arguments.get("file"), arguments.get("query")
        with self._lock:
            if name:
                path = self._library.named(str(name))
                if path is None:
                    raise ValueError(f"I couldn't find a file called {name} in your music.")
                track = Track.file(path)
            else:
                if self._paused and not query:
                    return self.resume(arguments)
                track = self._choose(str(query or ""))
            self._start(track, asked=True)
            where = " from YouTube" if track.origin == "youtube" else ""
            return ProviderOutcome(message=f"Playing {track.name}{where}.", detail=self.state())

    def pause(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            if self._track is None or self._paused:
                raise ValueError("Nothing is playing.")
            self._hold()
            return ProviderOutcome(message=f"Paused {self._track.name}.", detail=self.state())

    def resume(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            if self._track is None:
                raise ValueError("Nothing is paused.")
            if not self._paused:
                raise ValueError(f"{self._track.name} is already playing.")
            if self._player.broken:
                self._reopen()
            else:
                self._player.resume()
            self._paused = False
            return ProviderOutcome(message=f"Carrying on with {self._track.name}.", detail=self.state())

    def stop(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """Stops the sound but keeps the place: "stop", then "carry on", picks up where it was."""
        with self._lock:
            if self._track is None:
                raise ValueError("Nothing is playing.")
            if not self._paused:
                self._hold()
            return ProviderOutcome(message=f"Stopped {self._track.name}. Say carry on to pick it up again.", detail=self.state())

    def next(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            track = self._following()
            if track is None:
                raise ValueError("There's nothing lined up to play next yet.")
            self._start(track, asked=True)
            return ProviderOutcome(message=f"Next up, {track.name}.", detail=self.state())

    def set_autoplay(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            self._autoplay = bool(arguments["on"])
            if self._autoplay and self._track is not None and self._track.video_id and not self._up_next and self._catalogue:
                track = self._track
                self._background(lambda: self._fetch_up_next(track))
            said = "I'll keep the music going when a song ends." if self._autoplay else "I'll stop when this song ends."
            return ProviderOutcome(message=f"Autoplay is {'on' if self._autoplay else 'off'}. {said}", detail=self.state())

    def set_volume(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            self._volume = int(arguments["volume"])
            self._apply_volume()
            return ProviderOutcome(message=f"Music volume is {self._volume}%.", detail=self.state())

    def now_playing(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        with self._lock:
            if self._track is None:
                return ProviderOutcome(message="Nothing is playing.", detail=self.state())
            how = "Paused on" if self._paused else "Playing"
            return ProviderOutcome(message=f"{how} {self._track.name}, at {self._volume}% volume.", detail=self.state())

    # --- the passing of time ----------------------------------------------

    def check(self) -> bool:
        """Start a held song, follow the room's voice, notice a song ending.

        :returns: whether the state the board shows changed.
        """
        with self._lock:
            before = self._moment()
            self.follow_room()
            if self._track is not None and not self._paused and self._player.finished:
                self._after_the_end()
            changed, self._unpublished = self._moment() != before or self._unpublished, False
            return changed

    def follow_room(self) -> None:
        """The room's voice changed: start a song that was waiting for it, dip or restore the volume."""
        with self._lock:
            self._release_cue()
            self._apply_volume()

    def _moment(self) -> tuple:
        # The state without the library's count, which would read the folders on every tick.
        return (self._track, self._paused, self._cued_until is None, self._autoplay, self._up_next[0] if self._up_next else None)

    def state(self) -> dict[str, ArgumentValue]:
        with self._lock:
            return {
                "playing": self._playing(),
                "paused": self._paused,
                "starting": self._cued_until is not None,
                "track": self._track.name if self._track else "",
                "source": self._track.origin if self._track else "",
                "volume": self._volume,
                "autoplay": self._autoplay,
                "up_next": _named(self._up_next[0]) if self._up_next else "",
                "tracks": len(self._library.tracks()),
            }

    # --- inside ------------------------------------------------------------

    def _choose(self, query: str) -> Track:
        """The library first; online only for something named that it lacks, or asked for there."""
        words = [word for word in _words(query) if word not in _FILLER]
        online_asked = any(word in _ONLINE for word in words)
        wanted = " ".join(word for word in words if word not in _ONLINE)
        if not online_asked and self._library.tracks():
            path = self._library.find(wanted)
            if path is not None:
                return Track.file(path)
        if not wanted:
            raise ValueError(f"There's no music in {self._config.library_path} yet. Add some files there, or upload one on Home.")
        if self._catalogue is None:
            raise ValueError("Finding songs online is turned off here." if online_asked else f"Nothing in your music matches “{query}”.")
        track = self._catalogue.find(wanted)
        if track is None:
            raise ValueError(f"I couldn't find “{wanted}” on YouTube either." if not online_asked else f"I couldn't find “{wanted}” on YouTube.")
        return track

    def _start(self, track: Track, asked: bool) -> None:
        """Play a track -- or, asked for aloud, held until Jarvis has said he is playing it."""
        cue = asked and self._room is not None and self._clock is not None and self._room.awaiting_reply()
        self._apply_volume()
        track = self._open(track, paused=cue)
        self._track, self._paused = track, False
        self._cued_until = self._clock.now() + self._config.wait_for_reply_s if cue else None
        self._up_next.clear()
        if track.video_id:
            self._played.append(track.video_id)
            if self._autoplay and self._catalogue is not None:
                self._background(lambda: self._fetch_up_next(track))

    def _release_cue(self) -> None:
        if self._cued_until is None:
            return
        if self._room is not None and not self._room.reply_said() and self._clock.now() < self._cued_until:
            return
        self._cued_until = None
        self._unpublished = True
        if not self._paused:
            self._player.resume()

    def _after_the_end(self) -> None:
        """A song has run out: carry on if its stream only dropped, else play the next or stop."""
        if self._player.broken and self._track.origin == "youtube":
            try:
                self._reopen()
                return
            except ValueError as exc:
                LOGGER.warning("could not carry on with %s: %s", self._track.name, exc)
        following = self._following() if self._autoplay else None
        if following is not None:
            try:
                self._start(following, asked=False)
                return
            except ValueError as exc:
                LOGGER.warning("autoplay could not start %s: %s", following.name, exc)
        self._player.stop()
        self._track, self._paused = None, False

    def _hold(self) -> None:
        self._player.pause()
        self._paused, self._cued_until = True, None

    def _reopen(self) -> None:
        """The same song, from where it was, through a fresh connection.

        :raises ValueError: it cannot be found again.
        """
        track, position = self._track, self._player.position_s
        self._track = self._open(track, start_s=position)
        LOGGER.info("carrying on with %s from %.0f s", track.name, position)

    def _open(self, track: Track, start_s: float = 0.0, paused: bool = False) -> Track:
        """Play a track; a stream refused is looked up again once and tried afresh.

        YouTube's servers refuse some addresses outright (live: 403 Forbidden,
        and the same song opened a minute later from a fresh lookup), and an
        address expires after some hours.

        :returns: the track as played -- the fresh one, if it took a second look.
        :raises ValueError: it could not be played, in words for the occupant.
        """
        try:
            self._player.play(track, start_s=start_s, paused=paused)
            return track
        except ValueError:
            if self._catalogue is None or not track.video_id:
                raise
            LOGGER.info("%s was refused; looking it up again", track.name)
        fresh = self._catalogue.resolve(Candidate(track.video_id, track.title))
        if fresh is None:
            raise ValueError("That song couldn't be opened; try another.")
        self._player.play(fresh, start_s=start_s, paused=paused)
        return fresh

    def _following(self) -> Track | None:
        """What plays after the current song: YouTube's next, or another from the library."""
        while self._up_next:
            candidate = self._up_next.popleft()
            try:
                track = self._catalogue.resolve(candidate) if self._catalogue else None
            except CatalogueUnavailable as exc:
                LOGGER.warning("could not reach the next song: %s", exc)
                return None
            if track is not None:
                return track
        if self._track is not None and self._track.origin == "youtube":
            return None  # what follows a song from YouTube is not fetched yet
        another = self._library.another(self._track.path if self._track else None)
        return Track.file(another) if another is not None else None

    def _fetch_up_next(self, track: Track) -> None:
        try:
            following = self._catalogue.up_next(track.video_id)
        except CatalogueUnavailable as exc:
            LOGGER.warning("could not fetch what follows %s: %s", track.name, exc)
            return
        with self._lock:
            if self._track is None or self._track.video_id != track.video_id:
                return  # something else is playing by now
            self._up_next.extend(candidate for candidate in following if candidate.video_id not in self._played)
            LOGGER.info("%d songs lined up after %s", len(self._up_next), track.name)

    def _apply_volume(self) -> None:
        ducked = self._room is not None and self._room.ducked
        self._player.set_volume(self._volume / 100 * (self._config.duck_volume if ducked else 1.0))

    def _playing(self) -> bool:
        return self._track is not None and not self._paused and self._cued_until is None


def _named(candidate: Candidate) -> str:
    title, artist = tidy(candidate.title, candidate.channel)
    return f"{title} by {artist}" if artist else title


class MiniaudioPlayer:
    """Plays through the system's default output with miniaudio.

    One generator feeds the sound card for the whole track: it hands over
    silence while paused, applies the volume, and notes when the decoder runs
    dry. The device is opened per track and closed on stop, so the music holds
    no audio device while nothing plays. A file is decoded by miniaudio; a
    stream by PyAV, on a thread of its own, ahead of the sound card (FR-96).
    """

    SAMPLE_RATE = 44100
    CHANNELS = 2
    #: The sound card's buffer, in ms. Short, so a pause is heard at once; the
    #: card asks for several of these ahead of time.
    BUFFER_MS = 50
    #: Silence fed after the decoder runs dry before the track counts as over,
    #: in frames: what the card had queued must play out first, or every
    #: track loses its last moments when the device is closed.
    TAIL_FRAMES = SAMPLE_RATE // 2

    def __init__(self) -> None:
        import miniaudio

        self._miniaudio = miniaudio
        self._lock = threading.Lock()
        self._device = None
        self._source: Generator | None = None
        self._stream: StreamDecoder | None = None
        self._paused = False
        self._volume = 1.0
        self._finished = False
        self._drained = False
        self._tail = 0
        self._played_frames = 0

    def play(self, track: Track, start_s: float = 0.0, paused: bool = False) -> None:
        self.stop()
        if track.path is not None:
            source = self._miniaudio.stream_file(
                str(track.path), nchannels=self.CHANNELS, sample_rate=self.SAMPLE_RATE, seek_frame=int(start_s * self.SAMPLE_RATE)
            )
        else:
            self._stream = StreamDecoder(track.url, track.headers, sample_rate=self.SAMPLE_RATE, channels=self.CHANNELS, start_s=start_s)
            source = pcm_source(self._stream.buffer, self.CHANNELS)
            next(source)
        with self._lock:
            self._source, self._paused, self._finished, self._drained, self._tail = source, paused, False, False, 0
            self._played_frames = int(start_s * self.SAMPLE_RATE)
        device = self._miniaudio.PlaybackDevice(
            nchannels=self.CHANNELS, sample_rate=self.SAMPLE_RATE, buffersize_msec=self.BUFFER_MS, app_name="Smart Space music"
        )
        feed = self._feed()
        next(feed)
        device.start(feed)
        self._device = device

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def stop(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        with self._lock:
            self._source, self._paused = None, False

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(1.0, volume))

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

    @property
    def broken(self) -> bool:
        return self._stream is not None and self._stream.broken

    @property
    def position_s(self) -> float:
        with self._lock:
            return self._played_frames / self.SAMPLE_RATE

    def _feed(self) -> Generator[bytes, int, None]:
        import numpy

        frames = yield b""
        while True:
            silence = bytes(frames * self.CHANNELS * 2)
            with self._lock:
                source, paused, volume = self._source, self._paused, self._volume
                if self._drained:
                    self._tail += frames
                    self._finished = self._tail >= self.TAIL_FRAMES
            if source is None or paused:
                frames = yield silence
                continue
            try:
                samples = source.send(frames)
            except StopIteration:
                with self._lock:
                    self._source, self._drained = None, True
                frames = yield silence
                continue
            with self._lock:
                self._played_frames += frames
            scaled = (numpy.frombuffer(samples, dtype=numpy.int16) * volume).astype(numpy.int16).tobytes()
            frames = yield scaled


def manifest(config: MusicConfig, now: float) -> DeviceManifest:
    """What the music player is and what it can do."""

    def media(name: str, purpose: str, *parameters: ToolParameter, effect: ToolEffect = ToolEffect.WRITE) -> DeviceCapability:
        return DeviceCapability(name=name, purpose=purpose, effect=effect, device_class=DeviceClass.MEDIA, parameters=parameters)

    return DeviceManifest(
        ts=now,
        device_id=config.device_id,
        name=config.name,
        kind=DeviceKind.MEDIA,
        aliases=tuple(config.aliases),
        simulated=False,
        capabilities=(
            media(
                "play",
                "Play music: from the occupant's library, or found online when the library has nothing "
                "that matches. Give query for what they asked for (an artist, a song, a genre, a mood), "
                "or nothing to play anything from the library.",
                ToolParameter(name="query", type=ParameterType.STRING, description="What to play, in the occupant's words", required=False),
                ToolParameter(name="file", type=ParameterType.STRING, description="An exact file name, when the occupant gave one", required=False),
            ),
            media("pause", "Pause the music."),
            media("resume", "Carry on with music that was paused or stopped: continue, carry on, resume, unpause."),
            media("stop", "Stop the music. It keeps its place, so it can be carried on later."),
            media("next", "Skip to the next song."),
            media(
                "set_autoplay",
                "Turn autoplay on or off: whether another song follows when one ends.",
                ToolParameter(name="on", type=ParameterType.BOOLEAN, description="True to keep the music going, false to stop after this song"),
            ),
            media(
                "set_volume",
                "Set the music volume.",
                ToolParameter(name="volume", type=ParameterType.INTEGER, description="Volume in percent, 0 to 100", minimum=0, maximum=100),
            ),
            media("now_playing", "Say what is playing.", effect=ToolEffect.READ),
        ),
    )


def build(config: Config, clock: Clock, player: Player | None = None) -> DeviceParts:
    music = config.actuators.music
    library = Library([Path(music.library_path), Path(music.uploads_path)], music.extensions)
    catalogue = YouTubeCatalogue(music.online_max_minutes, music.online_timeout_s) if music.online else None
    room = RoomVoice(clock, music.request_window_s)
    device = MusicDevice(library, player or MiniaudioPlayer(), music, catalogue, room=room, clock=clock)
    # At once, not on the next tick: the music dips as Jarvis starts to listen.
    room.on_change = device.follow_room
    handlers = {
        "play": device.play,
        "pause": device.pause,
        "resume": device.resume,
        "stop": device.stop,
        "next": device.next,
        "set_autoplay": device.set_autoplay,
        "set_volume": device.set_volume,
        "now_playing": device.now_playing,
    }

    def tick(service: DeviceService) -> None:
        if device.check():
            service.publish_state()

    def close() -> None:
        state = device.state()
        if state["track"]:
            device.stop({})

    def subscribe(blackboard: Blackboard) -> None:
        # What the room's voice is doing, so a song waits for Jarvis and the music dips for him.
        blackboard.subscribe(topics.CONTEXT_PREFERENCE, PreferenceHint, room.on_request)
        blackboard.subscribe(topics.CONTEXT_REPLY, AssistantReply, room.on_reply)
        blackboard.subscribe(topics.SPEECH_STATUS, SpeechStatus, room.on_status)

    return DeviceParts(manifest=manifest(music, clock.now()), handlers=handlers, state=device.state, tick=tick, close=close, subscribe=subscribe)
