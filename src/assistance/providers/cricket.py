"""Live cricket scores, from Cricinfo's live-scores feed (FR-103).

Not a web search. A search snippet is a snapshot, and a score page fills its
numbers in with JavaScript, so neither says what the score is now. Cricinfo's
RSS feed does: one item per live, recent or coming match, its title the score
line -- ``Bangladesh 88/3 * v West Indies 356/10`` -- with ``*`` on the side
batting, and the feed's ``pubDate`` saying when. That is all it carries: no
overs, no ball by ball, and nothing per item about whether a match is over.

:class:`ScoreSource` is the seam; :class:`CricinfoFeed` is the one built, and
CricketData.org would be another. The feed is read at most once per
``cache_s``, so a question asked twice in a row reaches Cricinfo once.

Parsed with the standard library: the feed is small, its XML is simple, and
Python's expat refuses the entity expansion a hostile feed could try.
"""

from __future__ import annotations

import logging
import re
import urllib.request
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Protocol

from src.common.clock import Clock
from src.common.localtime import local_time
from src.common.tools import CRICKET_SCORE, ArgumentValue, ProviderOutcome

LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "cricinfo"

_USER_AGENT = "edge-smart-space/0.1 (+live scores for a personal room)"

#: A side of a score line: its name, then its score if play has started.
_SIDE = re.compile(r"^(?P<name>.*?)\s+(?P<score>\d[\d/&\s]*?)\s*(?P<batting>\*)?\s*$")

Fetch = Callable[[str, float], bytes]


class ScoresUnavailableError(ValueError):
    """The score feed did not answer usefully. The text is said to the occupant."""


@dataclass(frozen=True)
class Side:
    name: str
    #: The score as the feed writes it, '151/8 & 111/10'; empty before play.
    score: str
    batting: bool

    def said(self) -> str:
        if not self.score:
            return self.name
        return f"{self.name} {self.score}" + (" (batting)" if self.batting else "")


@dataclass(frozen=True)
class Match:
    home: Side
    away: Side
    url: str

    @property
    def started(self) -> bool:
        return bool(self.home.score or self.away.score)

    def said(self) -> str:
        line = f"{self.home.said()} v {self.away.said()}"
        return line if self.started else f"{line}: no score yet"


@dataclass(frozen=True)
class Scores:
    matches: tuple[Match, ...]
    #: When the feed says it was written, in epoch seconds; None if it does not say.
    as_of: float | None


class ScoreSource(Protocol):
    """Whatever says what the live matches are. The seam another feed plugs into."""

    @property
    def name(self) -> str:
        """Short identifier, and who the scores are said to be from."""

    def scores(self) -> Scores:
        """Every match the source lists now.

        :raises ScoresUnavailableError: when it cannot be read.
        """


def fetch_bytes(url: str, timeout_s: float) -> bytes:
    """GET a URL's body. :raises OSError: when nothing answers in time."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


class CricinfoFeed:
    """Cricinfo's live-scores RSS, read at most once per ``cache_s`` (FR-103)."""

    def __init__(self, url: str, timeout_s: float, clock: Clock, cache_s: float, fetch: Fetch = fetch_bytes) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._clock = clock
        self._cache_s = cache_s
        self._fetch = fetch
        self._cached: tuple[float, Scores] | None = None

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def scores(self) -> Scores:
        now = self._clock.now()
        if self._cached is not None and now - self._cached[0] < self._cache_s:
            return self._cached[1]
        try:
            body = self._fetch(self._url, self._timeout_s)
            scores = _parse(body)
        except (OSError, ValueError, ElementTree.ParseError) as exc:
            LOGGER.warning("cricinfo feed at %s could not be read: %s", self._url, exc)
            raise ScoresUnavailableError("Cricinfo isn't answering right now, so I can't get the score. Try again in a moment.") from exc
        self._cached = (now, scores)
        return scores


def _parse(body: bytes) -> Scores:
    channel = ElementTree.fromstring(body).find("channel")
    if channel is None:
        raise ValueError("no channel in the feed")
    matches = []
    for item in channel.iter("item"):
        match = _match(item.findtext("title") or "", (item.findtext("guid") or item.findtext("link") or "").strip())
        if match is not None:
            matches.append(match)
    return Scores(matches=tuple(matches), as_of=_feed_time(channel.findtext("pubDate")))


def _match(title: str, url: str) -> Match | None:
    home, separator, away = " ".join(title.split()).partition(" v ")
    if not separator:
        return None
    return Match(home=_side(home), away=_side(away), url=url)


def _side(text: str) -> Side:
    found = _SIDE.match(text)
    if found is None:
        return Side(name=text.strip(" *"), score="", batting=text.rstrip().endswith("*"))
    score = " ".join(found.group("score").split())
    return Side(name=found.group("name").strip(), score=score, batting=bool(found.group("batting")))


def _feed_time(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return parsedate_to_datetime(text).timestamp()
    except (TypeError, ValueError):
        return None


class CricketScores:
    """``cricket_score`` over a score source (FR-103)."""

    def __init__(
        self,
        source: ScoreSource,
        clock: Clock,
        *,
        utc_offset_h: float,
        aliases: Mapping[str, str],
        max_matches: int,
    ) -> None:
        self._source = source
        self._clock = clock
        self._utc_offset_h = utc_offset_h
        self._aliases = {short.lower(): full for short, full in aliases.items()}
        self._max_matches = max_matches

    @property
    def name(self) -> str:
        return self._source.name

    @property
    def simulated(self) -> bool:
        return False

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """The live score lines, for a team or every match, said to be Cricinfo's.

        :raises ValueError: a sentence for the occupant -- no live match for
            the team, none at all, or the feed not answering -- or a tool not
            served here.
        """
        if tool != CRICKET_SCORE.name:
            raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")
        scores = self._source.scores()
        asked = str(arguments.get("team") or "").strip()
        team = self._aliases.get(asked.lower(), asked)
        matches = _for_team(scores.matches, team) if team else _started_first(scores.matches)
        source = self._source.name.capitalize()
        if not matches:
            if team:
                raise ValueError(f"No live match for {asked} right now, according to {source}.")
            raise ValueError(f"There are no live matches right now, according to {source}.")
        as_of = ""
        if scores.as_of is not None:
            moment = local_time(scores.as_of, self._utc_offset_h)
            as_of = f", as of {moment.day} {moment:%b %Y, %H:%M}"
        shown = matches[: self._max_matches]
        heading = f"Live cricket{' for ' + asked if asked else ''} from {source}{as_of}:"
        lines = [heading] + [f"{number}. {match.said()} ({match.url})" for number, match in enumerate(shown, start=1)]
        return ProviderOutcome(message="\n".join(lines), detail={"matches": len(shown), "first_url": shown[0].url})


def _for_team(matches: tuple[Match, ...], team: str) -> list[Match]:
    """Matches the team plays in: its own name first, then its A side, under-19s and the like."""
    wanted = team.lower()
    words = re.compile(rf"\b{re.escape(wanted)}\b")
    exact = [match for match in matches if wanted in (match.home.name.lower(), match.away.name.lower())]
    related = [
        match
        for match in matches
        if match not in exact and (words.search(match.home.name.lower()) or words.search(match.away.name.lower()))
    ]
    return _started_first(tuple(exact)) + _started_first(tuple(related))


def _started_first(matches: tuple[Match, ...]) -> list[Match]:
    return [match for match in matches if match.started] + [match for match in matches if not match.started]
