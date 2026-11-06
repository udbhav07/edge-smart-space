"""Reading the pages a search returned (FR-104).

Nothing here reaches the internet. Pages are handed over by a fake fetch, and
addresses are resolved by a fake resolver; the one real fetch is aimed at a
server on this machine, and must be refused before it is asked anything.
"""

import http.server
import socket
import threading

import pytest

from src.assistance.providers.pages import (
    AddressRefusedError,
    Page,
    PageReader,
    fetch_page,
    main_text,
    public_address,
)
from src.common.tools import READ_PAGES

ARTICLE = """
<html><head><title>Sensex falls 800 points | The Hindu</title>
<script>var tracking = "Sensex Sensex Sensex";</script><style>p { color: red }</style></head>
<body>
<header><nav><li>Home</li><li>Markets and the Sensex today, all the links</li></nav></header>
<article>
<h1>Sensex falls 800 points as banks drag</h1>
<p>The BSE Sensex fell 800 points on Friday, its worst day in a month, as bank shares slid.</p>
<p>Traders said foreign investors sold heavily after the US Federal Reserve held rates.</p>
<p>The rupee was steady against the dollar through the afternoon session in Mumbai.</p>
<p>Analysts expect the index to stay volatile until the RBI policy review next week.</p>
</article>
<aside><p>Subscribe to our newsletter for daily market updates and more stories.</p></aside>
<footer><p>Copyright The Hindu Group, all rights reserved, privacy policy and terms.</p></footer>
</body></html>
"""

PUBLIC = "93.184.216.34"


def _resolver(addresses):
    """getaddrinfo, answering every name with these addresses."""

    def resolve(host, port, *args, **kwargs):
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addresses]

    return resolve


class FakeFetch:
    def __init__(self, pages=None, failing=()):
        self.pages = pages or {}
        self.failing = set(failing)
        self.asked = []

    def __call__(self, url, timeout_s, max_bytes):
        self.asked.append(url)
        if url in self.failing:
            raise OSError("timed out")
        return self.pages.get(url) or Page(url=url, content_type="text/html", text=ARTICLE)


def _reader(fetch, returned=lambda url: True, pages=3, page_chars=1500):
    return PageReader(fetch=fetch, returned=returned, pages=pages, page_chars=page_chars, timeout_s=5.0, max_bytes=2_000_000)


def _read(reader, urls, question="why did the Sensex fall"):
    return reader.invoke(READ_PAGES.name, {"urls": " ".join(urls), "question": question})


class TestMainText:
    def test_the_title_is_the_pages_own(self):
        title, _ = main_text(ARTICLE, "text/html")
        assert title == "Sensex falls 800 points | The Hindu"

    def test_the_article_is_kept(self):
        _, paragraphs = main_text(ARTICLE, "text/html")
        assert any("fell 800 points on Friday" in paragraph for paragraph in paragraphs)

    def test_menus_footers_asides_and_scripts_are_not(self):
        _, paragraphs = main_text(ARTICLE, "text/html")
        text = " ".join(paragraphs)
        assert "all the links" not in text and "Copyright" not in text
        assert "newsletter" not in text and "tracking" not in text

    def test_a_page_without_an_article_keeps_its_paragraphs(self):
        _, paragraphs = main_text("<html><body><p>The monsoon reached Kerala three days early this year.</p></body></html>", "text/html")
        assert paragraphs == ["The monsoon reached Kerala three days early this year."]

    def test_plain_text_is_read_by_paragraph(self):
        _, paragraphs = main_text("First paragraph that is long enough to keep.\n\nSecond paragraph, also long enough to keep.", "text/plain")
        assert len(paragraphs) == 2


class TestTrimming:
    def test_what_bears_on_the_question_is_kept_within_the_limit(self):
        message = _read(_reader(FakeFetch(), page_chars=260), ["https://thehindu.com/1"], question="why did foreign investors sell")
        assert "foreign investors sold heavily" in message.message
        assert "The rupee was steady" not in message.message

    def test_the_pages_text_is_kept_in_its_own_order(self):
        message = _read(_reader(FakeFetch()), ["https://thehindu.com/1"]).message
        assert message.index("fell 800 points") < message.index("Analysts expect")


class TestWhichPagesAreRead:
    def test_a_page_no_search_returned_is_never_fetched(self):
        fetch = FakeFetch()
        reader = _reader(fetch, returned=lambda url: url == "https://thehindu.com/1")
        _read(reader, ["https://thehindu.com/1", "http://192.168.1.1/admin"])
        assert fetch.asked == ["https://thehindu.com/1"]

    def test_with_nothing_a_search_returned_nothing_is_read(self):
        fetch = FakeFetch()
        with pytest.raises(ValueError, match="pages a search just found"):
            _read(_reader(fetch, returned=lambda url: False), ["http://127.0.0.1:8077/api"])
        assert fetch.asked == []

    def test_no_more_than_the_configured_number(self):
        fetch = FakeFetch()
        _read(_reader(fetch, pages=2), [f"https://a.in/{n}" for n in range(5)])
        assert sorted(fetch.asked) == ["https://a.in/0", "https://a.in/1"]

    def test_a_page_that_fails_is_named_and_the_rest_are_still_read(self):
        fetch = FakeFetch(failing={"https://a.in/1"})
        message = _read(_reader(fetch), ["https://a.in/1", "https://a.in/2"]).message
        assert "could not be read" in message and "fell 800 points" in message

    def test_when_none_can_be_read_it_is_said_plainly(self):
        fetch = FakeFetch(failing={"https://a.in/1", "https://a.in/2"})
        with pytest.raises(ValueError, match="couldn't open"):
            _read(_reader(fetch), ["https://a.in/1", "https://a.in/2"])

    def test_something_that_is_not_a_page_is_not_read(self):
        fetch = FakeFetch(pages={"https://a.in/f.pdf": Page(url="https://a.in/f.pdf", content_type="application/pdf", text="%PDF")})
        with pytest.raises(ValueError, match="couldn't open"):
            _read(_reader(fetch), ["https://a.in/f.pdf"])

    def test_a_block_page_with_almost_no_text_is_not_counted_as_read(self):
        # Live: theweek.in answered 200 with "Access Denied" and a reference number.
        denied = Page(
            url="https://a.in/blocked",
            content_type="text/html",
            text="<html><title>Access Denied</title><body><p>Reference #18.53cd717.1793979496.2c948c7a</p>"
            "<p>https://errors.edgesuite.net/18.53cd717.1793979496.2c948c7a</p></body></html>",
        )
        message = _read(_reader(FakeFetch(pages={"https://a.in/blocked": denied})), ["https://a.in/blocked", "https://a.in/2"]).message
        assert "[1] could not be read: it had too little text" in message

    def test_what_was_read_has_a_sentence_to_say_without_a_model(self):
        outcome = _read(_reader(FakeFetch()), ["https://a.in/1", "https://a.in/2"])
        assert outcome.detail["spoken"] == "I also read 2 of the pages it found."


class TestOnlyPublicAddresses:
    @pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.5", "192.168.1.10", "172.16.0.1", "169.254.169.254", "::1", "fd00::1", "0.0.0.0"])
    def test_a_name_resolving_inside_the_network_is_refused(self, address):
        with pytest.raises(AddressRefusedError):
            public_address("https://looks-public.example/x", resolve=_resolver([address]))

    def test_one_private_address_among_public_ones_is_enough_to_refuse(self):
        with pytest.raises(AddressRefusedError):
            public_address("https://a.example/x", resolve=_resolver([PUBLIC, "127.0.0.1"]))

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://a.example/x", "javascript:alert(1)", "https:///nohost"])
    def test_anything_but_a_web_address_is_refused(self, url):
        with pytest.raises(AddressRefusedError):
            public_address(url, resolve=_resolver([PUBLIC]))

    def test_a_public_address_is_allowed(self):
        public_address("https://www.thehindu.com/news", resolve=_resolver([PUBLIC]))

    def test_a_redirect_into_the_network_is_refused(self):
        from src.assistance.providers.pages import _GuardedRedirects

        handler = _GuardedRedirects(_resolver(["127.0.0.1"]))
        with pytest.raises(AddressRefusedError):
            handler.redirect_request(None, None, 302, "Found", {}, "http://127.0.0.1:8077/api/settings")

    def test_the_real_fetch_refuses_this_machine_before_asking_it_anything(self):
        asked = []

        class Recorder(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802 -- the stdlib's name
                asked.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), Recorder)
        threading.Thread(target=server.handle_request, daemon=True).start()
        try:
            with pytest.raises(AddressRefusedError):
                fetch_page(f"http://127.0.0.1:{server.server_port}/", 2.0, 1000)
        finally:
            server.server_close()
        assert asked == []


class TestAsAProvider:
    def test_it_is_real_and_says_what_it_is(self):
        reader = _reader(FakeFetch())
        assert reader.name == "pages" and reader.simulated is False

    def test_it_refuses_a_tool_it_does_not_serve(self):
        with pytest.raises(ValueError):
            _reader(FakeFetch()).invoke("web_search", {"query": "x"})
