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
  on that network. The name is looked up once per connection, and the
  connection is made to the very address that was checked, so a name that
  answers public when checked and private when connected to (DNS
  rebinding) cannot reach inside.
* **Bounded.** Pages are read at once, and the answer waits for them
  ``timeout_s`` plus ``wait_grace_s`` at most; each is cut at ``max_bytes``.
  A page still arriving then is told to stop and is no longer read, though
  a lookup or a single read already under way ends only at its own
  timeout. One that fails is named, and the rest are still used.

The text is pulled out with the standard library's HTML parser: what an
``<article>`` or ``<main>`` holds when there is one, every paragraph when
not, never a script, a menu, a footer or an aside. A library such as
trafilatura does this better on awkward pages, and would replace
:func:`main_text` without touching anything else.
"""

from __future__ import annotations

import codecs
import http.client
import ipaddress
import logging
import re
import socket
import threading
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

#: Elements that begin and end a paragraph of text; inline text between them joins up.
_BLOCKS = frozenset(
    {
        "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div", "section", "td", "th", "tr",
        "table", "ul", "ol", "dd", "dt", "pre", "figcaption", "body", "article", "main",
    }
)  # fmt: skip

#: Elements that hold the main content, when a page marks it.
_MAIN = frozenset({"article", "main"})

#: Elements that are never closed, so are never open.
_VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"})

#: Deepest nesting followed. A page nested deeper is read as if it were not; it bounds the parser's memory.
_MAX_DEPTH = 512

#: How much of a page is read at a time, between looks at whether to stop.
_CHUNK_BYTES = 64 * 1024

_STOPWORDS = frozenset(
    "the and for are was were what why how who when where which that this with from have has had did does "
    "about into than then them they their there will would could should just your you can its it's not".split()
)

#: socket.getaddrinfo, or a test's stand-in for it.
Resolve = Callable[..., list]

#: socket.create_connection, or a test's stand-in for it.
Connect = Callable[..., socket.socket]


class AddressRefusedError(ValueError):
    """An address this machine must not fetch: not the web, or inside the network."""


@dataclass(frozen=True)
class Page:
    """A fetched page: where it ended up, what it was served as, and its text."""

    url: str
    content_type: str
    text: str


#: (url, timeout_s, max_bytes, stop): the page, read until done or ``stop`` is set.
Fetch = Callable[[str, float, int, threading.Event], Page]


def public_address(url: str, resolve: Resolve = socket.getaddrinfo) -> None:
    """Refuse anything but an http(s) address whose every resolved address is public.

    :raises AddressRefusedError: naming why.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise AddressRefusedError(f"not a web address: {url[:80]}")
    _public_addresses(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80), resolve)


def _public_addresses(host: str, port: int, resolve: Resolve) -> list[str]:
    """Every address the name has, in the order given, if all of them are public.

    :raises AddressRefusedError: when it does not resolve, or any address is inside the network.
    """
    try:
        found = resolve(host, port, proto=socket.IPPROTO_TCP)
    except (OSError, UnicodeError) as exc:
        raise AddressRefusedError(f"{host} does not resolve: {exc}") from exc
    addresses = list(dict.fromkeys(entry[4][0] for entry in found))
    if not addresses:
        raise AddressRefusedError(f"{host} does not resolve")
    for address in addresses:
        if not ipaddress.ip_address(address.split("%", 1)[0]).is_global:
            raise AddressRefusedError(f"{host} is inside this network ({address})")
    return addresses


def _pinned_socket(host: str, port: int, timeout, source_address, resolve: Resolve, connect: Connect) -> socket.socket:
    """A connection to the address the check passed, never to a second lookup's answer."""
    return connect((_public_addresses(host, port, resolve)[0], port), timeout, source_address)


class _PublicHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, resolve: Resolve, connect: Connect, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._resolve, self._connect = resolve, connect

    def connect(self) -> None:
        self.sock = _pinned_socket(self.host, self.port, self.timeout, self.source_address, self._resolve, self._connect)


class _PublicHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, resolve: Resolve, connect: Connect, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._resolve, self._connect = resolve, connect

    def connect(self) -> None:
        sock = _pinned_socket(self.host, self.port, self.timeout, self.source_address, self._resolve, self._connect)
        # The certificate is still checked against the name, not the address.
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PublicHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, resolve: Resolve, connect: Connect) -> None:
        super().__init__()
        self._resolve, self._connect = resolve, connect

    def http_open(self, req):  # noqa: D102 -- the stdlib's hook
        return self.do_open(lambda *a, **k: _PublicHTTPConnection(*a, resolve=self._resolve, connect=self._connect, **k), req)


class _PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, resolve: Resolve, connect: Connect) -> None:
        super().__init__()
        self._resolve, self._connect = resolve, connect

    def https_open(self, req):  # noqa: D102 -- the stdlib's hook
        return self.do_open(
            lambda *a, **k: _PublicHTTPSConnection(*a, resolve=self._resolve, connect=self._connect, **k),
            req,
            context=self._context,
        )


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect only to another public address."""

    def __init__(self, resolve: Resolve = socket.getaddrinfo) -> None:
        super().__init__()
        self._resolve = resolve

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102 -- the stdlib's hook
        public_address(newurl, self._resolve)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_page(
    url: str,
    timeout_s: float,
    max_bytes: int,
    stop: threading.Event,
    *,
    resolve: Resolve = socket.getaddrinfo,
    connect: Connect = socket.create_connection,
) -> Page:
    """GET a public page, at most ``max_bytes`` of it, until ``stop`` is set.

    No proxy is used: a proxy would be the address connected to, and the
    check is of the page's own.

    :raises AddressRefusedError: before anything is sent, for an address
        inside the network or not on the web, or on a redirect to one.
    :raises OSError: when it does not arrive in time.
    :raises http.client.HTTPException: when what arrives is not HTTP, or is cut short.
    """
    public_address(url, resolve)
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler({}),
        urllib.request.UnknownHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        _GuardedRedirects(resolve),
        urllib.request.HTTPErrorProcessor(),
        _PublicHTTPHandler(resolve, connect),
        _PublicHTTPSHandler(resolve, connect),
    ):
        opener.add_handler(handler)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html,text/plain;q=0.9"})
    with opener.open(request, timeout=timeout_s) as response:
        content_type = response.headers.get_content_type()
        body = bytearray()
        while content_type in _READABLE and len(body) < max_bytes and not stop.is_set():
            chunk = response.read1(min(_CHUNK_BYTES, max_bytes - len(body)))
            if not chunk:
                break
            body += chunk
        charset = _known_charset(response.headers.get_content_charset())
        final = response.geturl()
    return Page(url=final, content_type=content_type, text=bytes(body).decode(charset, errors="replace"))


def _known_charset(name: str | None) -> str:
    """The page's own charset if Python knows it; UTF-8 when it names none, or one that does not exist."""
    if name:
        try:
            return codecs.lookup(name).name
        except LookupError:
            LOGGER.info("page names an unknown charset %r; reading it as UTF-8", name)
    return "utf-8"


class _MainText(HTMLParser):
    """Collects a page's title and its paragraphs, noting which were in its main content.

    Follows which elements are open, as a browser does, so an element left
    unclosed ends with whatever holds it: a menu missing its ``</nav>`` does
    not swallow the article after it.
    """

    def __init__(self, min_paragraph_chars: int) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.paragraphs: list[tuple[str, bool]] = []
        self._min_paragraph_chars = min_paragraph_chars
        self._open: list[str] = []
        self._skipping = 0
        self._main = 0
        self._in_title = False
        self._block: list[str] = []
        self._block_in_main = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "br":
            self._block.append(" ")
            return
        if tag in _VOID:
            return
        if tag in _BLOCKS or tag in _SKIPPED:
            self._flush()
        if len(self._open) < _MAX_DEPTH:
            self._open.append(tag)
            self._skipping += tag in _SKIPPED
            self._main += tag in _MAIN
        if tag == "title" and not self._skipping:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag not in self._open:
            return
        self._flush()
        # Everything opened inside it, and left open, closes with it.
        while self._open:
            closed = self._open.pop()
            self._skipping -= closed in _SKIPPED
            self._main -= closed in _MAIN
            if closed == "title":
                self._in_title = False
            if closed == tag:
                break

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skipping:
            if not self._block:
                self._block_in_main = self._main > 0
            self._block.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = " ".join("".join(self._block).split())
        if len(text) >= self._min_paragraph_chars:
            self.paragraphs.append((text, self._block_in_main))
        self._block = []


def main_text(text: str, content_type: str, *, min_paragraph_chars: int, min_main_chars: int) -> tuple[str, list[str]]:
    """A page's title and its content paragraphs, in order.

    :param min_paragraph_chars: shorter text is a menu item or a caption, not content.
    :param min_main_chars: less than this in ``<article>``/``<main>`` means the
        page did not really mark its content, and every paragraph is used.
    """
    if content_type == "text/plain":
        paragraphs = [" ".join(chunk.split()) for chunk in re.split(r"\n\s*\n", text)]
        return "", [paragraph for paragraph in paragraphs if len(paragraph) >= min_paragraph_chars]
    parser = _MainText(min_paragraph_chars)
    parser.feed(text)
    parser.close()
    main = [paragraph for paragraph, in_main in parser.paragraphs if in_main]
    chosen = main if sum(map(len, main)) >= min_main_chars else [paragraph for paragraph, _ in parser.paragraphs]
    return " ".join(parser.title.split()), chosen


def _trimmed(paragraphs: list[str], question: str, limit: int, lead: int) -> str:
    """The ``lead`` opening paragraphs and those that bear on the question, in the page's order, within ``limit``.

    Nothing else fills what room is left: the 7B model reads every character,
    and a paragraph sharing no word with the question rarely answers it.
    """
    words = {word for word in re.findall(r"[a-z0-9]+", question.lower()) if len(word) > 2 and word not in _STOPWORDS}

    def bearing(paragraph: str) -> int:
        return sum(1 for word in re.findall(r"[a-z0-9]+", paragraph.lower()) if word in words)

    ranked = list(range(min(lead, len(paragraphs))))
    ranked += sorted(
        (index for index in range(len(paragraphs)) if index not in ranked and bearing(paragraphs[index])),
        key=lambda index: -bearing(paragraphs[index]),
    )
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
        wait_grace_s: float,
        min_page_chars: int,
        min_paragraph_chars: int,
        lead_paragraphs: int,
        fetch: Fetch = fetch_page,
    ) -> None:
        self._returned = returned
        self._pages = pages
        self._page_chars = page_chars
        self._timeout_s = timeout_s
        self._max_bytes = max_bytes
        self._wait_grace_s = wait_grace_s
        self._min_page_chars = min_page_chars
        self._min_paragraph_chars = min_paragraph_chars
        self._lead_paragraphs = lead_paragraphs
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
        returned = [url for url in asked if self._returned(url)]
        chosen = returned[: self._pages]
        refused = [url for url in asked if url not in returned]
        if refused:
            LOGGER.warning("not reading pages no search returned: %s", refused)
        if len(returned) > len(chosen):
            LOGGER.info("reading the first %d pages, not %s", self._pages, returned[self._pages :])
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
            lines.append(f"[{number}] {title or host} — {host} ({url})\n{_trimmed(paragraphs, question, self._page_chars, self._lead_paragraphs)}")
            opened += 1
        if not opened:
            raise ValueError("I couldn't open any of the pages the search found.")
        return ProviderOutcome(
            message="\n\n".join(lines),
            detail={"pages_read": opened, "pages_failed": len(chosen) - opened, "spoken": f"I also read {opened} of the pages it found."},
        )

    def _read_all(self, urls: list[str]) -> dict[str, tuple[str, list[str]] | str]:
        """Each page's title and paragraphs, or why it could not be read. Never longer than one timeout and its grace."""
        stop = threading.Event()
        pool = ThreadPoolExecutor(max_workers=len(urls), thread_name_prefix="read-page")
        futures = {pool.submit(self._read_one, url, stop): url for url in urls}
        done, _ = wait(futures, timeout=self._timeout_s + self._wait_grace_s)
        # A page still arriving is given up on, not waited for: it is told to
        # stop between reads, and the answer does not wait for it to.
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        read: dict[str, tuple[str, list[str]] | str] = {url: "it took too long" for url in urls}
        for future in done:
            read[futures[future]] = future.result()
        return read

    def _read_one(self, url: str, stop: threading.Event) -> tuple[str, list[str]] | str:
        try:
            page = self._fetch(url, self._timeout_s, self._max_bytes, stop)
        except AddressRefusedError as exc:
            LOGGER.warning("refused to read %s: %s", url, exc)
            return "it is not a public page"
        except (OSError, ValueError, LookupError, http.client.HTTPException) as exc:
            # LookupError: a charset Python does not know; HTTPException: an
            # answer cut short, or not HTTP at all. One page, not the read.
            LOGGER.info("could not read %s: %s: %s", url, type(exc).__name__, exc)
            return "it did not open"
        if page.content_type not in _READABLE:
            return f"it is not a web page ({page.content_type})"
        title, paragraphs = main_text(
            page.text, page.content_type, min_paragraph_chars=self._min_paragraph_chars, min_main_chars=self._min_page_chars
        )
        if sum(map(len, paragraphs)) < self._min_page_chars:
            # A block page ("Access Denied", "Just a moment...") is served as
            # 200 with a line or two; it is not the page, so it is not read.
            return "it had too little text to read"
        return title, paragraphs
