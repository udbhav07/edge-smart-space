"""Something to play: a file in the library, or a stream found online (FR-87, FR-96)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Track:
    title: str
    path: Path | None = None
    url: str = ""
    headers: tuple[tuple[str, str], ...] = ()
    #: Where it came from: "library" or "youtube".
    origin: str = "library"
    #: The page it was found on, for the screen.
    page: str = ""

    @classmethod
    def file(cls, path: Path) -> Track:
        return cls(title=path.stem, path=path)
