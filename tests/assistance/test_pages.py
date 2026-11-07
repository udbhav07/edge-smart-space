"""Reading the pages a search returned (FR-104).

Nothing here reaches the internet. Pages are handed over by a fake fetch, and
addresses are resolved by a fake resolver; the one real fetch is aimed at a
server on this machine, and must be refused before it is asked anything.
"""

import http.client
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
RULES = {"min_paragraph_chars": 30, "min_main_chars": 200}
MONSOON = "The monsoon reached Kerala three days early this year, the IMD said."


def _resolver(addresses):
    """getaddrinfo, answering every name with these addresses."""

    def resolve(host, port, *args, **kwargs):
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in addresses]

    return resolve


class FakeFetch:
    def __init__(self, pages=None, failing=()):
        self.pages = pages or {}
        self.failing = failing if isinstance(failing, dict) else set(failing)
        self.asked = []

    def __call__(self, url, timeout_s, max_bytes, stop):
        self.asked.append(url)
        if url in self.failing:
            raise self.failing[url] if isinstance(self.failing, dict) else OSError("timed out")
        return self.pages.get(url) or Page(url=url, content_type="text/html", text=ARTICLE)


def _reader(fetch, returned=lambda url: True, pages=3, page_chars=1500, timeout_s=5.0):
    return PageReader(
        fetch=fetch,
        returned=returned,
        pages=pages,
        page_chars=page_chars,
        timeout_s=timeout_s,
        max_bytes=2_000_000,
        wait_grace_s=0.2,
        min_page_chars=200,
        min_paragraph_chars=30,
        lead_paragraphs=2,
    )


def _read(reader, urls, question="why did the Sensex fall"):
    return reader.invoke(READ_PAGES.name, {"urls": " ".join(urls), "question": question})


class TestMainText:
    def test_the_title_is_the_pages_own(self):
        title, _ = main_text(ARTICLE, "text/html", **RULES)
        assert title == "Sensex falls 800 points | The Hindu"

    def test_the_article_is_kept(self):
        _, paragraphs = main_text(ARTICLE, "text/html", **RULES)
        assert any("fell 800 points on Friday" in paragraph for paragraph in paragraphs)

    def test_menus_footers_asides_and_scripts_are_not(self):
        _, paragraphs = main_text(ARTICLE, "text/html", **RULES)
        text = " ".join(paragraphs)
        assert "all the links" not in text and "Copyright" not in text
        assert "newsletter" not in text and "tracking" not in text

    def test_a_page_without_an_article_keeps_its_paragraphs(self):
        _, paragraphs = main_text("<html><body><p>The monsoon reached Kerala three days early this year.</p></body></html>", "text/html", **RULES)
        assert paragraphs == ["The monsoon reached Kerala three days early this year."]

    def test_plain_text_is_read_by_paragraph(self):
        _, paragraphs = main_text("First paragraph that is long enough to keep.\n\nSecond paragraph, also long enough to keep.", "text/plain", **RULES)
        assert len(paragraphs) == 2


    def test_an_icons_title_is_not_part_of_the_pages(self):
        page = "<html><head><title>Real title</title></head><body><svg><title>search icon</title></svg></body></html>"
        assert main_text(page, "text/html", **RULES)[0] == "Real title"

    def test_a_menu_left_open_ends_with_what_holds_it(self):
        page = f"<body><div><nav><a>Home</a><a>World</a></div><p>{MONSOON}</p></body>"
        assert main_text(page, "text/html", **RULES)[1] == [MONSOON]

    def test_text_in_a_div_or_a_table_cell_is_read(self):
        page = f"<body><div>{MONSOON}</div><table><tr><td>Rainfall was forty percent above normal across the state.</td></tr></table></body>"
        assert main_text(page, "text/html", **RULES)[1] == [MONSOON, "Rainfall was forty percent above normal across the state."]

    def test_text_after_a_nested_block_in_the_same_div_is_kept(self):
        page = f"<body><div><div>{MONSOON}</div>Officials expect the rains to cover the state within a week.</div></body>"
        assert main_text(page, "text/html", **RULES)[1][-1] == "Officials expect the rains to cover the state within a week."


class TestTrimming:
    def test_what_bears_on_the_question_is_kept_within_the_limit(self):
        message = _read(_reader(FakeFetch(), page_chars=260), ["https://thehindu.com/1"], question="why did foreign investors sell")
        assert "foreign investors sold heavily" in message.message
        assert "The rupee was steady" not in message.message

    def test_what_does_not_bear_on_the_question_is_left_out_even_with_room(self):
        message = _read(_reader(FakeFetch(), page_chars=6000), ["https://thehindu.com/1"], question="why did foreign investors sell").message
        assert "foreign investors sold heavily" in message
        assert "The rupee was steady" not in message and "RBI policy review" not in message

    def test_the_pages_text_is_kept_in_its_own_order(self):
        message = _read(_reader(FakeFetch()), ["https://thehindu.com/1"], question="what did traders and analysts say").message
        assert message.index("fell 800 points") < message.index("Traders said") < message.index("Analysts expect")


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

    @pytest.mark.parametrize(
        "error",
        [LookupError("unknown encoding: x-foo"), http.client.IncompleteRead(b"half"), http.client.BadStatusLine("garbage")],
        ids=["bad charset", "cut short", "not http"],
    )
    def test_a_page_failing_in_any_way_a_fetch_can_still_leaves_the_rest(self, error):
        fetch = FakeFetch(failing={"https://a.in/1": error})
        message = _read(_reader(fetch), ["https://a.in/1", "https://a.in/2"]).message
        assert "[1] could not be read" in message and "fell 800 points" in message

    def test_a_page_still_arriving_is_told_to_stop(self):
        stops = []

        def slow(url, timeout_s, max_bytes, stop):
            stops.append(stop)
            stop.wait(5.0)
            raise OSError("given up")

        message = _read(_reader(lambda *a: slow(*a) if a[0].endswith("/slow") else FakeFetch()(*a), timeout_s=0.1), ["https://a.in/slow", "https://a.in/2"]).message
        assert "[1] could not be read: it took too long" in message
        assert stops and stops[0].is_set()

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

    def test_the_connection_goes_to_the_address_that_was_checked(self):
        # DNS rebinding: the name answers public when checked, then this
        # machine when connected to. Only one answer may be used for both.
        answers = iter([PUBLIC, "127.0.0.1", "127.0.0.1"])
        connected = []

        def resolve(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port))]

        def connect(address, timeout, source_address=None):
            connected.append(address[0])
            raise OSError("not really connecting")

        with pytest.raises((OSError, AddressRefusedError)):
            fetch_page("http://rebind.example/x", 1.0, 1000, threading.Event(), resolve=resolve, connect=connect)
        assert "127.0.0.1" not in connected

    def test_a_page_dripping_in_stops_being_read_when_told(self):
        ours, theirs = socket.socketpair()

        class StopOnceSomethingIsRead(threading.Event):
            looks = 0

            def is_set(self):
                self.looks += 1
                return self.looks > 1

        stop = StopOnceSomethingIsRead()

        def connect(address, timeout, source_address=None):
            ours.settimeout(timeout)
            return ours

        def drip():
            theirs.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n" + MONSOON.encode())  # and never the end

        threading.Thread(target=drip, daemon=True).start()
        try:
            page = fetch_page("http://news.example/x", 3.0, 100_000, stop, resolve=_resolver([PUBLIC]), connect=connect)
        finally:
            theirs.close()
        assert page.text.startswith(MONSOON)

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
                fetch_page(f"http://127.0.0.1:{server.server_port}/", 2.0, 1000, threading.Event())
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
