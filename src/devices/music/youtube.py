"""Finding a song online, with yt-dlp (FR-96).

Asked for something the occupant's own music does not have, the player
searches YouTube: a flat search first, which is quick, then the first result
short enough to be a song is resolved to its audio stream. Nothing is
downloaded; the stream is played as it arrives (:mod:`.stream`).

YouTube's terms restrict automated access. This is for a personal room, off
unless ``actuators.music.online`` is on, and yt-dlp must be kept current:
YouTube changes and an old yt-dlp stops finding anything.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.devices.music.track import Track

LOGGER = logging.getLogger(__name__)

#: How many results to look through for one that is a song, not a ten-hour mix.
_SEARCH_RESULTS = 5


class CatalogueUnavailable(ValueError):
    """Searching online did not work, in words for the occupant."""


class YouTubeCatalogue:
    """Songs on YouTube, found by the occupant's words."""

    def __init__(self, max_minutes: float, timeout_s: float, downloader: Callable[[dict], Any] | None = None) -> None:
        """
        :param downloader: makes a ``YoutubeDL`` from options; yt-dlp's own if not given.
        """
        self._max_s = max_minutes * 60.0
        self._timeout_s = timeout_s
        self._downloader = downloader

    def find(self, query: str) -> Track | None:
        """The first result that is a song, resolved to its audio stream.

        :returns: None if nothing found is short enough to be one.
        :raises CatalogueUnavailable: YouTube could not be reached or searched.
        """
        base = {"quiet": True, "no_warnings": True, "noplaylist": True, "skip_download": True, "socket_timeout": self._timeout_s}
        found = self._extract({**base, "extract_flat": "in_playlist"}, f"ytsearch{_SEARCH_RESULTS}:{query}")
        for entry in found.get("entries") or ():
            duration = (entry or {}).get("duration")
            if not duration or duration > self._max_s:
                continue  # a live stream, or something too long to be a song
            page = entry.get("url") or entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry['id']}"
            info = self._extract({**base, "format": "bestaudio/best"}, page)
            if not info.get("url"):
                continue
            LOGGER.info("found %r online for %r", info.get("title"), query)
            return Track(
                title=info.get("title") or entry.get("title") or query,
                url=info["url"],
                headers=tuple((info.get("http_headers") or {}).items()),
                origin="youtube",
                page=info.get("webpage_url") or page,
            )
        return None

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
