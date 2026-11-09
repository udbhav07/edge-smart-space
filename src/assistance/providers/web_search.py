"""Web and news search, through SearXNG on this machine (FR-101, FR-102).

Two parts, split on purpose. :class:`SearchBackend` is the seam: it turns a
query into hits, and :class:`SearxngBackend` is the one built. A hosted
provider -- Tavily, Brave -- would be one more class here and a line in
``build_registry``. :class:`WebSearch` is the tool provider: it asks a
backend, keeps what falls in the span asked for, and writes what the model
reads -- numbered results with their source and age, and when the search was
made, so the reply can say how fresh it is (FR-101).

Ages are read, not assumed. Measured on the pinned image, a result carries its
date as an ISO ``publishedDate`` or ``pubdate``, a relative ``pubdate`` ("11
hours ago"), the head of ``metadata`` ("48 minutes ago | News18"), or not at
all. SearXNG's own ``time_range`` is not used: on the pinned image it drops
Brave News and DuckDuckGo News and leaves results with no dates, so news is
narrowed here instead, on the room's local day (FR-102).

Every failure is a sentence for the occupant, raised as ``ValueError`` so the
registry passes it on (U-19): SearXNG not answering, nothing found, nothing in
the span asked for. None of it is filled in from anywhere else.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from src.common.clock import Clock
from src.common.localtime import epoch_of, local_time
from src.common.tools import NEWS_SEARCH, WEB_SEARCH, ArgumentValue, ProviderOutcome

LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "searxng"

#: Sent with every request; SearXNG and the engines behind it see this, not a browser.
_USER_AGENT = "edge-smart-space/0.1 (+web search for a personal room)"

#: What news is searched for when no topic was named.
_HEADLINES_QUERY = "latest news"

_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 7 * 86400, "month": 30 * 86400, "year": 365 * 86400}
_RELATIVE = re.compile(r"^\s*(\d+|an?|one)\s+(second|minute|hour|day|week|month|year)s?\s+ago\b", re.IGNORECASE)

Fetch = Callable[[str, float], object]


class SearchUnavailableError(ValueError):
    """The search service did not answer usefully. The text is said to the occupant."""


@dataclass(frozen=True)
class Hit:
    """One search result, as the backend understood it."""

    title: str
    url: str
    source: str
    snippet: str
    #: Epoch seconds it was published, or None when nothing says.
    published_ts: float | None


class SearchBackend(Protocol):
    """Whatever turns a query into hits. The seam a hosted search would plug into."""

    @property
    def name(self) -> str:
        """Short identifier for the log and for ``ToolResult.provider``."""

    def search(self, query: str, *, news: bool) -> list[Hit]:
        """Results for the query, best first.

        :raises SearchUnavailableError: when the service cannot be asked.
        """


def fetch_json(url: str, timeout_s: float) -> object:
    """GET a URL and parse its body as JSON.

    :raises OSError: when nothing answers in time.
    :raises ValueError: when the answer is not JSON.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


class SearxngBackend:
    """SearXNG's JSON API (FR-101)."""

    def __init__(self, url: str, region: str, timeout_s: float, clock: Clock, fetch: Fetch = fetch_json) -> None:
        self._url = url.rstrip("/")
        self._region = region
        self._timeout_s = timeout_s
        self._clock = clock
        self._fetch = fetch

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def search(self, query: str, *, news: bool) -> list[Hit]:
        parameters = {"q": query, "format": "json", "language": self._region}
        if news:
            parameters["categories"] = "news"
        url = f"{self._url}/search?{urllib.parse.urlencode(parameters)}"
        asked_at = self._clock.now()
        try:
            answer = self._fetch(url, self._timeout_s)
        except (OSError, ValueError) as exc:
            LOGGER.warning("searxng at %s did not answer: %s", self._url, exc)
            raise SearchUnavailableError("Web search isn't answering right now, so I can't look that up. Try again in a moment.") from exc
        if not isinstance(answer, Mapping) or not isinstance(answer.get("results"), list):
            LOGGER.warning("searxng at %s answered something that is not a search: %.200r", self._url, answer)
            raise SearchUnavailableError("Web search isn't answering properly right now, so I can't look that up.")
        unresponsive = answer.get("unresponsive_engines") or []
        if unresponsive:
            LOGGER.info("engines that did not answer: %s", unresponsive)
        return [hit for hit in (_hit(item, asked_at) for item in answer["results"]) if hit is not None]


def _hit(item: object, asked_at: float) -> Hit | None:
    if not isinstance(item, Mapping):
        return None
    url = str(item.get("url") or "").strip()
    title = " ".join(str(item.get("title") or "").split())
    if not url or not title:
        return None
    metadata = str(item.get("metadata") or "")
    published = (
        _absolute(item.get("publishedDate"))
        or _absolute(item.get("pubdate"))
        or _relative(item.get("pubdate"), asked_at)
        or _relative(metadata, asked_at)
    )
    return Hit(
        title=title,
        url=url,
        source=_source(metadata, url),
        snippet=" ".join(str(item.get("content") or "").split()),
        published_ts=published,
    )


def _absolute(value: object) -> float | None:
    """An ISO date SearXNG wrote, read as UTC when it names no zone."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        moment = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def _relative(value: object, asked_at: float) -> float | None:
    """'11 hours ago' at the head of a string, as of when the search was made."""
    if not isinstance(value, str):
        return None
    match = _RELATIVE.match(value)
    if match is None:
        return None
    count = 1 if match.group(1).lower() in {"a", "an", "one"} else int(match.group(1))
    return asked_at - count * _SECONDS[match.group(2).lower()]


def _source(metadata: str, url: str) -> str:
    """Who published it: named after the age in metadata, or else the site's address."""
    _, separator, named = metadata.partition("|")
    if separator and named.strip():
        return named.strip()
    host = urllib.parse.urlparse(url).hostname or url
    return host.removeprefix("www.")


@dataclass(frozen=True)
class _Span:
    """A stretch of the room's local time that news was asked for, in epoch seconds."""

    words: str
    start: float
    end: float


class WebSearch:
    """``web_search`` and ``news_search`` over a backend (FR-101, FR-102)."""

    def __init__(
        self,
        backend: SearchBackend,
        clock: Clock,
        *,
        max_results: int,
        snippet_chars: int,
        utc_offset_h: float,
        morning_from_h: int,
        remembered_urls: int,
    ) -> None:
        self._backend = backend
        self._clock = clock
        self._max_results = max_results
        self._snippet_chars = snippet_chars
        self._utc_offset_h = utc_offset_h
        self._morning_from_h = morning_from_h
        self._returned: deque[str] = deque(maxlen=remembered_urls)

    @property
    def name(self) -> str:
        return self._backend.name

    def returned(self, url: str) -> bool:
        """Whether a recent search gave this address: the only pages that may be read (FR-104)."""
        return url in self._returned

    @property
    def simulated(self) -> bool:
        return False

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """Search, and say what was found with where it came from and how old it is.

        :raises ValueError: a sentence for the occupant -- nothing found, or
            the search service not answering -- or a tool not served here.
        """
        now = self._clock.now()
        if tool == WEB_SEARCH.name:
            query = str(arguments["query"]).strip()
            hits = self._backend.search(query, news=False)
            if not hits:
                raise ValueError(f"I searched the web for “{query}” and found nothing.")
            return self._outcome(f"Web results for “{query}”", hits, now)
        if tool == NEWS_SEARCH.name:
            topic = str(arguments.get("topic") or "").strip()
            when = str(arguments.get("when") or "latest")
            span = self._span(when, now)
            hits = self._backend.search(topic or _HEADLINES_QUERY, news=True)
            about = f" about {topic}" if topic else ""
            if span is not None:
                hits = [hit for hit in hits if hit.published_ts is not None and span.start <= hit.published_ts < span.end]
            if not hits:
                raise ValueError(f"I found no news{about} from {span.words if span else 'just now'}.")
            hits.sort(key=lambda hit: -hit.published_ts if hit.published_ts is not None else float("inf"))
            heading = f"News{about}" + (f" from {span.words}" if span else ", newest first")
            return self._outcome(heading, hits, now)
        raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")

    def _span(self, when: str, now: float) -> _Span | None:
        """The local stretch of time a span names; None for the latest, unfiltered."""
        local = local_time(now, self._utc_offset_h)
        midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
        start_of = lambda moment: epoch_of(moment, self._utc_offset_h)  # noqa: E731
        if when == "today":
            return _Span("today", start_of(midnight), now + 1.0)
        if when == "this_morning":
            morning = midnight.replace(hour=self._morning_from_h)
            if local < morning:
                raise ValueError(f"It isn't morning yet, so there is no news from this morning; it starts at {self._morning_from_h}:00.")
            return _Span("this morning", start_of(morning), min(now + 1.0, start_of(midnight.replace(hour=12))))
        if when == "yesterday":
            return _Span("yesterday", start_of(midnight - timedelta(days=1)), start_of(midnight))
        if when == "this_week":
            return _Span("this week", now - 7 * 86400.0, now + 1.0)
        return None

    def _outcome(self, heading: str, hits: list[Hit], now: float) -> ProviderOutcome:
        searched = _day(local_time(now, self._utc_offset_h)) + local_time(now, self._utc_offset_h).strftime(", %H:%M")
        shown = hits[: self._max_results]
        lines = [f"{heading} (searched {searched}):"]
        for number, hit in enumerate(shown, start=1):
            age = _age(hit.published_ts, now, self._utc_offset_h)
            snippet = _cut(hit.snippet, self._snippet_chars)
            lines.append(f"{number}. {hit.title} — {hit.source}, {age}: {snippet} ({hit.url})")
        first = shown[0]
        self._returned.extend(hit.url for hit in shown)
        return ProviderOutcome(
            message="\n".join(lines),
            detail={
                **{f"url_{number}": hit.url for number, hit in enumerate(shown, start=1)},
                "results": len(shown),
                "first_url": first.url,
                "first_source": first.source,
                # Said as it is when no model can phrase the answer: no links, one result.
                "spoken": f"The top result, from {first.source}, {_age(first.published_ts, now, self._utc_offset_h)}: {first.title}.",
            },
        )


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "…"


def _age(published_ts: float | None, now: float, utc_offset_h: float) -> str:
    """How old a result is, as a person would say it."""
    if published_ts is None:
        return "undated"
    seconds = max(0.0, now - published_ts)
    if seconds < 120:
        return "just now"
    for unit, size in (("minute", 60), ("hour", 3600)):
        count = int(seconds // size)
        if seconds < (3600 if unit == "minute" else 86400):
            return f"{count} {unit}{'s' if count != 1 else ''} ago"
    days = int(seconds // 86400)
    if days < 7:
        return f"{days} day{'s' if days != 1 else ''} ago"
    return "on " + _day(local_time(published_ts, utc_offset_h))


def _day(moment: datetime) -> str:
    """'6 Nov 2026'. Built by hand: strftime's %-d is not portable to Windows."""
    return f"{moment.day} {moment:%b %Y}"
