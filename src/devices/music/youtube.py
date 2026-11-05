"""Finding a song online, with yt-dlp (FR-96).

Asked for something the occupant's own music does not have, the player
searches YouTube: a flat search, which is quick, ranked so that a song beats
a live take, a cover or a slowed edit nobody asked for; the best is resolved
to its audio stream. Nothing is downloaded; the stream is played as it
arrives (:mod:`.stream`).

What comes next is YouTube's own: the mix it would autoplay after a song
(``list=RD<id>``). Fetching it takes a while, so it is fetched while the song
plays, never before.

YouTube's terms restrict automated access. This is for a personal room, off
unless ``actuators.music.online`` is on, and yt-dlp must be kept current:
YouTube changes and an old yt-dlp stops finding anything.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from src.devices.music.track import Track, tidy

LOGGER = logging.getLogger(__name__)

#: How many results to look through for one that is a song, not a ten-hour mix.
_SEARCH_RESULTS = 8
#: Versions of a song nobody wants unless they say so.
_UNASKED = frozenset("live cover karaoke instrumental reaction slowed reverb sped 8d remix nightcore lofi mashup tutorial".split())


@dataclass(frozen=True)
class Candidate:
    """A video found but not yet resolved to a stream."""

    video_id: str
    title: str
    channel: str = ""
    duration_s: float = 0.0


class CatalogueUnavailable(ValueError):
    """Searching online did not work, in words for the occupant."""


class YouTubeCatalogue:
    """Songs on YouTube, found by the occupant's words, and what follows them."""

    def __init__(self, max_minutes: float, timeout_s: float, downloader: Callable[[dict], Any] | None = None) -> None:
        """
        :param downloader: makes a ``YoutubeDL`` from options; yt-dlp's own if not given.
        """
        self._max_s = max_minutes * 60.0
        self._timeout_s = timeout_s
        self._downloader = downloader

    def find(self, query: str) -> Track | None:
        """The best song-length result for the words, resolved to its stream.

        :returns: None if nothing found is short enough to be a song.
        :raises CatalogueUnavailable: YouTube could not be reached or searched.
        """
        found = self._extract(self._flat(), f"ytsearch{_SEARCH_RESULTS}:{query}")
        asked = set(query.lower().split())
        ranked = sorted(self.songs(_candidates(found)), key=lambda candidate: _penalty(candidate, asked))
        for candidate in ranked[:3]:
            track = self.resolve(candidate)
            if track is not None:
                LOGGER.info("found %r online for %r", track.name, query)
                return track
        return None

    def resolve(self, candidate: Candidate) -> Track | None:
        """A candidate's audio stream, and its name tidied for saying aloud.

        :raises CatalogueUnavailable: YouTube could not be reached.
        """
        info = self._extract({**self._base(), "format": "bestaudio/best"}, _watch(candidate.video_id))
        if not info.get("url"):
            return None
        title, artist = tidy(
            info.get("title") or candidate.title, info.get("channel") or candidate.channel, info.get("track") or "", info.get("artist") or ""
        )
        return Track(
            title=title,
            artist=artist,
            url=info["url"],
            headers=tuple((info.get("http_headers") or {}).items()),
            origin="youtube",
            page=info.get("webpage_url") or _watch(candidate.video_id),
            video_id=candidate.video_id,
        )

    def up_next(self, video_id: str) -> list[Candidate]:
        """What YouTube would autoplay after this song, songs only, itself left out.

        :raises CatalogueUnavailable: YouTube could not be reached.
        """
        mix = self._extract(self._flat(), f"{_watch(video_id)}&list=RD{video_id}")
        return [candidate for candidate in self.songs(_candidates(mix)) if candidate.video_id != video_id]

    def songs(self, candidates: Iterable[Candidate]) -> list[Candidate]:
        """Those long enough to be something and short enough to be a song."""
        return [candidate for candidate in candidates if 0 < candidate.duration_s <= self._max_s]

    def _base(self) -> dict:
        return {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True, "socket_timeout": self._timeout_s}

    def _flat(self) -> dict:
        return {**self._base(), "noplaylist": False, "extract_flat": "in_playlist", "playlistend": 25}

    def _extract(self, options: dict, target: str) -> dict:
        try:
            if self._downloader is None:
                import yt_dlp

                downloader = yt_dlp.YoutubeDL(options)
                failure: tuple[type[BaseException], ...] = (yt_dlp.utils.DownloadError, OSError)
            else:
                downloader = self._downloader(options)
                failure = (OSError, RuntimeError)
        except ImportError as exc:
            raise CatalogueUnavailable("Finding songs online isn't installed here (yt-dlp).") from exc
        try:
            with downloader:
                return downloader.extract_info(target, download=False) or {}
        except failure as exc:
            LOGGER.warning("searching online failed: %s", exc)
            raise CatalogueUnavailable("I couldn't reach YouTube just now.") from exc


def _watch(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def _candidates(listing: dict) -> list[Candidate]:
    return [
        Candidate(
            video_id=entry["id"],
            title=entry.get("title") or "",
            channel=entry.get("channel") or entry.get("uploader") or "",
            duration_s=float(entry.get("duration") or 0.0),
        )
        for entry in listing.get("entries") or ()
        if entry and entry.get("id")
    ]


def _penalty(candidate: Candidate, asked: set[str]) -> int:
    """Lower is better; ties keep YouTube's own order (the sort is stable)."""
    words = set(candidate.title.lower().replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").split())
    unasked = len((words & _UNASKED) - asked)
    # An auto-generated "Artist - Topic" upload is the record itself, and carries its own metadata.
    record = 0 if candidate.channel.endswith(" - Topic") else 1
    return unasked * 2 + record
