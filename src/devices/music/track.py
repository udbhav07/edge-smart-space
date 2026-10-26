"""Something to play, and the name it goes by when said aloud (FR-87, FR-96)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Bracketed asides a video title carries that are no part of the song's name.
_ASIDE = re.compile(
    r"\s*[\(\[][^\)\]]*\b(official|video|audio|lyric|lyrics|lyrical|visuali[sz]er|remaster(ed)?|hd|hq|4k|full song|"
    r"music video|explicit|clean|mv|from)\b[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
#: Trailing words of the same kind, outside brackets: "... Full Song With Lyrics".
_TRAILING = re.compile(r"\s+(official\s+(music\s+)?(video|audio)|full\s+(video\s+)?song.*|with\s+lyrics|lyrics?|lyrical)\s*$", re.IGNORECASE)
#: What a channel's name adds to an artist's: "Queen - Topic", "QueenVEVO", "Queen Official".
_CHANNEL_NOISE = re.compile(r"(\s*-\s*topic|vevo|\s+official|\s+music)$", re.IGNORECASE)
#: "Artist - Song", with a hyphen or a dash.
_SEPARATOR = re.compile(r"\s+[-–—]\s+")


@dataclass(frozen=True)
class Track:
    #: The song's own name, tidied: "Bohemian Rhapsody".
    title: str
    #: Who it is by, when known: "Queen".
    artist: str = ""
    path: Path | None = None
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    #: Where it came from: "library" or "youtube".
    origin: str = "library"
    #: The page it was found on, for the screen.
    page: str = ""
    #: YouTube's id for it, so it can be found again and its mix fetched.
    video_id: str = ""

    @classmethod
    def file(cls, path: Path) -> Track:
        title, artist = tidy(path.stem)
        return cls(title=title, artist=artist, path=path)

    @property
    def name(self) -> str:
        """As it is said aloud: "Bohemian Rhapsody by Queen"."""
        return f"{self.title} by {self.artist}" if self.artist else self.title


def tidy(title: str, channel: str = "", track: str = "", artist: str = "") -> tuple[str, str]:
    """A song's name and artist from what a video or a file is called.

    Metadata wins when there is some (``track``, ``artist``). Otherwise the
    asides are dropped -- "(Official Video Remastered)", "| Aditya Roy Kapur"
    -- a quoted name is taken as the song's, and "Queen - Bohemian Rhapsody"
    is split. The channel is the artist only when it plainly is one.

    :returns: (title, artist); the artist may be empty.
    """
    if track:
        return _strip(track), artist.split(",")[0].strip()
    text = title.split("|")[0].strip()
    quoted = re.search(r"[\"“]([^\"”]+)[\"”]", text)
    if quoted:
        return _strip(quoted.group(1)), artist
    parts = _SEPARATOR.split(text, maxsplit=1)
    if len(parts) == 2 and parts[0] and parts[1]:
        left, right = parts[0].strip(), parts[1]
        by_channel = channel and _same(left, _CHANNEL_NOISE.sub("", channel.strip()))
        if "|" in title and not by_channel:
            # "Kesariya - Brahmastra | Ranbir Kapoor": a film song is "Song - Film | cast".
            return _strip(left), artist
        return _strip(right), artist or left
    named = _CHANNEL_NOISE.sub("", channel.strip()) if _CHANNEL_NOISE.search(channel.strip()) else ""
    return _strip(text), artist or named


def _same(a: str, b: str) -> bool:
    letters = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())  # noqa: E731
    return bool(letters(a)) and letters(a) == letters(b)


def _strip(text: str) -> str:
    original, previous = text, None
    while previous != text:
        previous = text
        text = _TRAILING.sub("", _ASIDE.sub("", text)).strip(" -–—\"'")
    return text or original
