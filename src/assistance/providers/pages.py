"""Reading the pages a search returned, for a deep answer (FR-104).

When a search's snippets do not hold the answer, the top pages are fetched,
their main text pulled out, and what bears on the question kept, so a 7B
model reads a page's worth of each rather than its menus and footers.

Three rules make fetching a page the model never saw safe to run:

* **Only what a search returned.** ``returned`` says whether a search just
  gave this address; anything else is refused before a byte is sent. The
  model never names a page here -- the assistant passes the search's own
  results -- and this holds even if that changes.
* **Only public addresses.** A name that resolves to this machine or its
  network (loopback, private, link-local, reserved) is refused, and so is
  every redirect leading to one. The console, the broker and the router are
  on that network.
* **Bounded.** Pages are read at once, each waited for ``timeout_s`` and cut
  at ``max_bytes``; one that fails is named, and the rest are still used.

The text is pulled out with the standard library's HTML parser: what an
``<article>`` or ``<main>`` holds when there is one, every paragraph when
not, never a script, a menu, a footer or an aside. A library such as
trafilatura does this better on awkward pages, and would replace
:func:`main_text` without touching anything else.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from html.parser import HTMLParser

from src.common.tools import READ_PAGES, ArgumentValue, ProviderOutcome

LOGGER = logging.getLogger(__name__)

PROVIDER_NAME = "pages"

_USER_AGENT = "Mozilla/5.0 (compatible; edge-smart-space/0.1; +reading a page a search returned)"

#: What a page may be served as and still be read.
_READABLE = frozenset({"text/html", "application/xhtml+xml", "text/plain"})

#: Elements whose text is never the page's content.
_SKIPPED = frozenset({"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "button", "iframe", "template", "select"})

#: Elements whose text is a paragraph of it.
_BLOCKS = frozenset({"p", "h1", "h2", "h3", "li", "blockquote"})

#: Elements that hold the main content, when a page marks it.
_MAIN = frozenset({"article", "main"})

#: A paragraph shorter than this is a menu item or a caption, not content.
_MIN_PARAGRAPH_CHARS = 30

#: Main-content text below this means the page did not really mark it.
_MIN_MAIN_CHARS = 200

#: Opening paragraphs kept whatever the question, because they say what the page is.
_LEAD_PARAGRAPHS = 2

_STOPWORDS = frozenset(
    "the and for are was were what why how who when where which that this with from have has had did does "
    "about into than then them they their there will would could should just your you can its it's not".split()
)

Resolve = Callable[..., list]


class AddressRefusedError(ValueError):
    """An address this machine must not fetch: not the web, or inside the network."""


@dataclass(frozen=True)
class Page:
    """A fetched page: where it ended up, what it was served as, and its text."""

    url: str
    content_type: str
    text: str


Fetch = Callable[[str, float, int], Page]


def public_address(url: str, resolve: Resolve = socket.getaddrinfo) -> None:
    """Refuse anything but an http(s) address whose every resolved address is public.

    :raises AddressRefusedError: naming why.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise AddressRefusedError(f"not a web address: {url[:80]}")
    try:
        found = resolve(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError) as exc:
        raise AddressRefusedError(f"{parts.hostname} does not resolve: {exc}") from exc
    addresses = {entry[4][0] for entry in found}
    if not addresses:
        raise AddressRefusedError(f"{parts.hostname} does not resolve")
    for address in addresses:
        if not ipaddress.ip_address(address.split("%", 1)[0]).is_global:
            raise AddressRefusedError(f"{parts.hostname} is inside this network ({address})")


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to another public address."""

    def __init__(self, resolve: Resolve = socket.getaddrinfo) -> None:
        super().__init__()
        self._resolve = resolve

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 -- the stdlib's hook
        public_address(newurl, self._resolve)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_page(url: str, timeout_s: float, max_bytes: int, resolve: Resolve = socket.getaddrinfo) -> Page:
    """GET a public page, at most ``max_bytes`` of it.

    :raises AddressRefusedError: before anything is sent, for an address
        inside the network or not on the web, or on a redirect to one.
    :raises OSError: when it does not arrive in time.
    """
    public_address(url, resolve)
    opener = urllib.request.build_opener(_GuardedRedirects(resolve))
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9"})
    with opener.open(request, timeout=timeout_s) as response:
        content_type = response.headers.get_content_type()
        body = response.read(max_bytes) if content_type in _READABLE else b""
        charset = response.headers.get_content_charset() or "utf-8"
        final = response.geturl()
    return Page(url=final, content_type=content_type, text=body.decode(charset, errors="replace"))


class _MainText(HTMLParser):
    """Collects a page's title and its paragraphs, noting which were in its main content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.paragraphs: list[tuple[str, bool]] = []
        self._skipping = 0
        self._main = 0
        self._in_title = False
        self._block: list[str] | None = None
        self._block_in_main = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _SKIPPED:
            self._flush()
            self._skipping += 1
        elif tag in _MAIN:
            self._main += 1
        elif tag in _BLOCKS and not self._skipping:
            self._flush()
            self._block, self._block_in_main = [], self._main > 0
        elif tag == "title":
            self._in_title = True
        elif tag == "br" and self._block is not None:
            self._block.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED:
            self._skipping = max(0, self._skipping - 1)
        elif tag in _MAIN:
            self._flush()
            self._main = max(0, self._main - 1)
        elif tag in _BLOCKS:
            self._flush()
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._block is not None and not self._skipping:
            self._block.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._block is None:
            return
        text = " ".join("".join(self._block).split())
        if len(text) >= _MIN_PARAGRAPH_CHARS:
            self.paragraphs.append((text, self._block_in_main))
        self._block = None


def main_text(text: str, content_type: str) -> tuple[str, list[str]]:
    """A page's title and its content paragraphs, in order."""
    if content_type == "text/plain":
        paragraphs = [" ".join(chunk.split()) for chunk in re.split(r"\n\s*\n", text)]
        return "", [paragraph for paragraph in paragraphs if len(paragraph) >= _MIN_PARAGRAPH_CHARS]
    parser = _MainText()
    parser.feed(text)
    parser.close()
    main = [paragraph for paragraph, in_main in parser.paragraphs if in_main]
    chosen = main if sum(map(len, main)) >= _MIN_MAIN_CHARS else [paragraph for paragraph, _ in parser.paragraphs]
    return " ".join(parser.title.split()), chosen


def _trimmed(paragraphs: list[str], question: str, limit: int) -> str:
    """The opening paragraphs and those that bear on the question, in the page's order, within ``limit``."""
    words = {word for word in re.findall(r"[a-z0-9]+", question.lower()) if len(word) > 2 and word not in _STOPWORDS}

    def bearing(paragraph: str) -> int:
        return sum(1 for word in re.findall(r"[a-z0-9]+", paragraph.lower()) if word in words)

    ranked = list(range(min(_LEAD_PARAGRAPHS, len(paragraphs))))
    ranked += sorted(
        (index for index in range(len(paragraphs)) if index not in ranked and bearing(paragraphs[index])),
        key=lambda index: -bearing(paragraphs[index]),
    )
    ranked += [index for index in range(len(paragraphs)) if index not in ranked]
    kept: list[int] = []
    used = 0
    for index in ranked:
        if used + len(paragraphs[index]) > limit:
            continue
        kept.append(index)
        used += len(paragraphs[index]) + 1
    if not kept and paragraphs:
        return paragraphs[0][: max(0, limit - 1)].rsplit(" ", 1)[0] + "…"
    return "\n".join(paragraphs[index] for index in sorted(kept))


class PageReader:
    """``read_pages``: the main text of pages a search returned, trimmed to the question (FR-104)."""

    def __init__(
        self,
        *,
        returned: Callable[[str], bool],
        pages: int,
        page_chars: int,
        timeout_s: float,
        max_bytes: int,
        fetch: Fetch = fetch_page,
    ) -> None:
        self._returned = returned
        self._pages = pages
        self._page_chars = page_chars
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes
        self._fetch = fetch

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def simulated(self) -> bool:
        return False

    def invoke(self, tool: str, arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
        """Read the pages, at once, and say what each held.

        :raises ValueError: a sentence for the occupant -- nothing a search
            returned was asked for, or none of it could be opened -- or a
            tool not served here.
        """
        if tool != READ_PAGES.name:
            raise ValueError(f"{PROVIDER_NAME} does not serve {tool!r}")
        question = str(arguments.get("question") or "")
        asked = list(dict.fromkeys(str(arguments["urls"]).split()))
        chosen = [url for url in asked if self._returned(url)][: self._pages]
        refused = [url for url in asked if url not in chosen and not self._returned(url)]
        if refused:
            LOGGER.warning("not reading pages no search returned: %s", refused)
        if not chosen:
            raise ValueError("I can only read pages a search just found, and none of those were.")
        read = self._read_all(chosen)
        lines = [f"Pages read for “{question}”:" if question else "Pages read:"]
        opened = 0
        for number, url in enumerate(chosen, start=1):
            outcome = read.get(url)
            if isinstance(outcome, str):
                lines.append(f"[{number}] could not be read: {outcome} ({url})")
                continue
            title, paragraphs = outcome
            host = (urllib.parse.urlsplit(url).hostname or url).removeprefix("www.")
            lines.append(f"[{number}] {title or host} — {host} ({url})\n{_trimmed(paragraphs, question, self._page_chars)}")
            opened += 1
        if not opened:
            raise ValueError("I couldn't open any of the pages the search found.")
        return ProviderOutcome(
            message="\n\n".join(lines),
            detail={"pages_read": opened, "pages_failed": len(chosen) - opened, "spoken": f"I also read {opened} of the pages it found."},
        )

    def _read_all(self, urls: list[str]) -> dict[str, tuple[str, list[str]] | str]:
        """Each page's title and paragraphs, or why it could not be read. Never longer than one timeout."""
        pool = ThreadPoolExecutor(max_workers=len(urls), thread_name_prefix="read-page")
        futures = {pool.submit(self._read_one, url): url for url in urls}
        done, _ = wait(futures, timeout=self._timeout_s + 1.0)
        # A page still arriving is given up on, not waited for: its socket
        # times out on its own, and the answer must not.
        pool.shutdown(wait=False, cancel_futures=True)
        read: dict[str, tuple[str, list[str]] | str] = {url: "it took too long" for url in urls}
        for future in done:
            read[futures[future]] = future.result()
        return read

    def _read_one(self, url: str) -> tuple[str, list[str]] | str:
        try:
            page = self._fetch(url, self._timeout_s, self._max_bytes)
        except AddressRefusedError as exc:
            LOGGER.warning("refused to read %s: %s", url, exc)
            return "it is not a public page"
        except (OSError, ValueError) as exc:
            LOGGER.info("could not read %s: %s", url, exc)
            return "it did not open"
        if page.content_type not in _READABLE:
            return f"it is not a web page ({page.content_type})"
        title, paragraphs = main_text(page.text, page.content_type)
        if sum(map(len, paragraphs)) < _MIN_MAIN_CHARS:
            # A block page ("Access Denied", "Just a moment...") is served as
            # 200 with a line or two; it is not the page, so it is not read.
            return "it had too little text to read"
        return title, paragraphs
