"""The music player on this machine, as a device service (FR-87).

It plays files from the occupant's own music folder (``actuators.music``):
mp3, flac, wav and ogg, decoded and played by miniaudio. "Play some jazz"
reaches it as ``play(query="jazz")``, matched against file and folder names;
a file the occupant hands over in the console is saved to the uploads folder
and played by name, so only a name ever crosses the blackboard, never audio.

The decision of *what* to play lives in :class:`MusicDevice` and the sound
itself in a :class:`Player`, so the first is tested without a sound card.
"""

from __future__ import annotations

import logging
import random
import re
import threading
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import Protocol

from src.common.clock import Clock
from src.common.config import Config, MusicConfig
from src.common.devices import DeviceCapability, DeviceClass, DeviceKind, DeviceManifest
from src.common.tools import ArgumentValue, ParameterType, ProviderOutcome, ToolEffect, ToolParameter
from src.devices.runner import DeviceParts
from src.devices.service import DeviceService

LOGGER = logging.getLogger(__name__)

#: Words in a request that say nothing about which track: "play some music for me".
_FILLER = frozenset(
    "a an and any by for me my music of on play please put some song songs something the track tracks to tune tunes".split()
)


class Player(Protocol):
    """Makes the sound. One track at a time."""

    def play(self, path: Path) -> None: ...

    def pause(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def set_volume(self, volume: float) -> None:
        """0.0 to 1.0."""

    @property
    def finished(self) -> bool:
        """Whether the track that was playing has run out by itself."""


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
    """What to play, and what is playing."""

    def __init__(self, library: Library, player: Player, config: MusicConfig) -> None:
        self._library = library
        self._player = player
        self._config = config
        self._track: Path | None = None
        self._paused = False
        self._volume = config.default_volume
        player.set_volume(self._volume / 100)

    def play(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        name, query = arguments.get("file"), arguments.get("query")
        if name:
            track = self._library.named(str(name))
            if track is None:
                raise ValueError(f"I couldn't find a file called {name} in your music.")
        else:
            if self._paused and not query:
                return self.resume(arguments)
            if not self._library.tracks():
                raise ValueError(f"There's no music in {self._config.library_path} yet. Add some files there, or upload one on Home.")
            track = self._library.find(str(query or ""))
            if track is None:
                raise ValueError(f"Nothing in your music matches “{query}”.")
        self._player.play(track)
        self._track, self._paused = track, False
        return ProviderOutcome(message=f"Playing {track.stem}.", detail=self.state())

    def pause(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if not self._playing():
            raise ValueError("Nothing is playing.")
        self._player.pause()
        self._paused = True
        return ProviderOutcome(message=f"Paused {self._track.stem}.", detail=self.state())

    def resume(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if self._track is None or not self._paused:
            raise ValueError("Nothing is paused.")
        self._player.resume()
        self._paused = False
        return ProviderOutcome(message=f"Playing {self._track.stem} again.", detail=self.state())

    def stop(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if self._track is None:
            raise ValueError("Nothing is playing.")
        self._player.stop()
        self._track, self._paused = None, False
        return ProviderOutcome(message="Stopped the music.", detail=self.state())

    def set_volume(self, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        self._volume = int(arguments["volume"])
        self._player.set_volume(self._volume / 100)
        return ProviderOutcome(message=f"Music volume is {self._volume}%.", detail=self.state())

    def now_playing(self, _arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        if self._track is None:
            return ProviderOutcome(message="Nothing is playing.", detail=self.state())
        how = "Paused on" if self._paused else "Playing"
        return ProviderOutcome(message=f"{how} {self._track.stem}, at {self._volume}% volume.", detail=self.state())

    def check(self) -> bool:
        """Notice a track that ran out by itself. Returns whether the state changed."""
        if self._track is not None and not self._paused and self._player.finished:
            self._player.stop()
            self._track = None
            return True
        return False

    def state(self) -> dict[str, ArgumentValue]:
        return {
            "playing": self._playing(),
            "paused": self._paused,
            "track": self._track.stem if self._track else "",
            "volume": self._volume,
            "tracks": len(self._library.tracks()),
        }

    def _playing(self) -> bool:
        return self._track is not None and not self._paused


class MiniaudioPlayer:
    """Plays through the system's default output with miniaudio.

    One generator feeds the sound card for the whole track: it hands over
    silence while paused, applies the volume, and notes when the decoder runs
    dry. The device is opened per track and closed on stop, so the music holds
    no audio device while nothing plays.
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
        self._paused = False
        self._volume = 1.0
        self._finished = False
        self._drained = False
        self._tail = 0

    def play(self, path: Path) -> None:
        self.stop()
        source = self._miniaudio.stream_file(str(path), nchannels=self.CHANNELS, sample_rate=self.SAMPLE_RATE)
        with self._lock:
            self._source, self._paused, self._finished, self._drained, self._tail = source, False, False, False, 0
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
        with self._lock:
            self._source, self._paused = None, False

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(1.0, volume))

    @property
    def finished(self) -> bool:
        with self._lock:
            return self._finished

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
                "Play music from the occupant's library. Give query for what they asked for "
                "(an artist, a song, a genre, a mood), or nothing to play anything.",
                ToolParameter(name="query", type=ParameterType.STRING, description="What to play, in the occupant's words", required=False),
                ToolParameter(name="file", type=ParameterType.STRING, description="An exact file name, when the occupant gave one", required=False),
            ),
            media("pause", "Pause the music."),
            media("resume", "Carry on with paused music."),
            media("stop", "Stop the music."),
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
    device = MusicDevice(library, player or MiniaudioPlayer(), music)
    handlers = {
        "play": device.play,
        "pause": device.pause,
        "resume": device.resume,
        "stop": device.stop,
        "set_volume": device.set_volume,
        "now_playing": device.now_playing,
    }

    def tick(service: DeviceService) -> None:
        if device.check():
            service.publish_state()

    def close() -> None:
        if device.state()["playing"] or device.state()["paused"]:
            device.stop({})

    return DeviceParts(manifest=manifest(music, clock.now()), handlers=handlers, state=device.state, tick=tick, close=close)
