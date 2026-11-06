"""Web and news search through SearXNG (FR-101, FR-102).

SearXNG is never reached: the backend is handed a fetch function, and these
answer with the shapes measured on the pinned image -- an ISO
``publishedDate``, a relative ``pubdate`` from Brave News, an age at the head
of ``metadata`` from Bing and DuckDuckGo News, and results with no date at all.
"""

from datetime import datetime
from urllib.parse import parse_qs, urlparse

import pytest

from src.assistance.providers.web_search import SearxngBackend, WebSearch
from src.common.clock import SimClock
from src.common.localtime import epoch_of
from src.common.tools import NEWS_SEARCH, WEB_SEARCH

IST_H = 5.5
#: Friday 6 November 2026, 07:00 in the room.
NOW = epoch_of(datetime(2026, 11, 6, 7, 0), IST_H)


def _clock(local=datetime(2026, 11, 6, 7, 0)) -> SimClock:
    return SimClock(start_epoch_s=epoch_of(local, IST_H))


def _result(title, url, content="", **dates):
    return {"title": title, "url": url, "content": content, "engines": ["brave"], **dates}


class FakeSearxng:
    """Answers one search the way SearXNG's JSON does, and remembers the ask."""

    def __init__(self, results=(), error=None):
        self.results = list(results)
        self.error = error
        self.asked = []

    def __call__(self, url, timeout_s):
        self.asked.append((url, timeout_s))
        if self.error is not None:
            raise self.error
        return {"query": "q", "results": self.results, "answers": [], "infoboxes": [], "unresponsive_engines": []}

    def parameters(self):
        return {key: values[0] for key, values in parse_qs(urlparse(self.asked[-1][0]).query).items()}


def _provider(fake, clock=None, **settings):
    clock = clock or _clock()
    backend = SearxngBackend("http://127.0.0.1:8888", "en-IN", 6.0, clock, fetch=fake)
    options = {"max_results": 5, "snippet_chars": 300, "utc_offset_h": IST_H, "morning_from_h": 5} | settings
    return WebSearch(backend, clock, **options)


def _search(fake, query="who won the IPL", **settings):
    return _provider(fake, **settings).invoke(WEB_SEARCH.name, {"query": query})


def _news(fake, clock=None, **arguments):
    return _provider(fake, clock).invoke(NEWS_SEARCH.name, arguments)


class TestAskingSearxng:
    def test_it_asks_for_json(self):
        fake = FakeSearxng([_result("A", "https://a.in/x")])
        _search(fake)
        assert fake.parameters()["format"] == "json"

    def test_it_asks_in_the_configured_region(self):
        fake = FakeSearxng([_result("A", "https://a.in/x")])
        _search(fake)
        assert fake.parameters()["language"] == "en-IN"

    def test_it_sends_the_words_asked_for(self):
        fake = FakeSearxng([_result("A", "https://a.in/x")])
        _search(fake, query="who won the IPL")
        assert fake.parameters()["q"] == "who won the IPL"

    def test_it_waits_no_longer_than_configured(self):
        fake = FakeSearxng([_result("A", "https://a.in/x")])
        _search(fake)
        assert fake.asked[-1][1] == 6.0

    def test_news_is_asked_of_the_news_engines(self):
        fake = FakeSearxng([_result("A", "https://a.in/x", pubdate="1 hour ago")])
        _news(fake)
        assert fake.parameters()["categories"] == "news"

    def test_news_is_not_narrowed_by_searxngs_own_time_range(self):
        # Measured: time_range=day drops Brave and DuckDuckGo News and every date.
        fake = FakeSearxng([_result("A", "https://a.in/x", pubdate="1 hour ago")])
        _news(fake, when="today")
        assert "time_range" not in fake.parameters()


class TestWhatTheModelReads:
    def test_each_result_names_its_source(self):
        fake = FakeSearxng([_result("IPL final", "https://www.espncricinfo.com/story/1", "RCB won.")])
        assert "IPL final — espncricinfo.com, " in _search(fake).message

    def test_a_source_named_in_the_metadata_is_used_over_the_address(self):
        fake = FakeSearxng([_result("Story", "https://www.msn.com/en-in/x", metadata="48 minutes ago | The Sunday Guardian")])
        assert "The Sunday Guardian" in _search(fake).message

    def test_an_iso_date_becomes_an_age(self):
        fake = FakeSearxng([_result("Story", "https://a.in/x", publishedDate="2026-11-06T00:30:00")])
        # 00:30 UTC is 06:00 in the room, an hour before now.
        assert "1 hour ago" in _search(fake).message

    def test_a_relative_pubdate_becomes_an_age(self):
        fake = FakeSearxng([_result("Story", "https://a.in/x", pubdate="11 hours ago")])
        assert "11 hours ago" in _search(fake).message

    def test_an_age_at_the_head_of_the_metadata_becomes_an_age(self):
        fake = FakeSearxng([_result("Story", "https://a.in/x", metadata="48 minutes ago | News18")])
        assert "48 minutes ago" in _search(fake).message

    def test_a_result_with_no_date_is_said_to_have_none(self):
        fake = FakeSearxng([_result("2026 Indian Premier League", "https://en.wikipedia.org/wiki/2026_IPL")])
        assert "undated" in _search(fake).message

    def test_a_long_snippet_is_cut_to_the_configured_length(self):
        fake = FakeSearxng([_result("Story", "https://a.in/x", "word " * 200)])
        message = _search(fake, snippet_chars=60).message
        assert "word " * 20 not in message

    def test_no_more_than_the_configured_number_of_results(self):
        fake = FakeSearxng([_result(f"Story {n}", f"https://a.in/{n}") for n in range(9)])
        message = _search(fake, max_results=3).message
        assert "Story 2" in message and "Story 3" not in message

    def test_the_search_is_dated_so_the_model_can_say_how_fresh_it_is(self):
        fake = FakeSearxng([_result("A", "https://a.in/x")])
        assert "6 Nov 2026, 07:00" in _search(fake).message

    def test_the_first_link_is_kept_in_the_detail_for_the_log(self):
        fake = FakeSearxng([_result("A", "https://a.in/x")])
        assert _search(fake).detail["first_url"] == "https://a.in/x"


class TestWhenNothingComesBack:
    def test_no_results_is_said_plainly(self):
        with pytest.raises(ValueError, match="found nothing"):
            _search(FakeSearxng([]))

    def test_searxng_not_answering_is_said_plainly(self):
        with pytest.raises(ValueError, match="isn't answering"):
            _search(FakeSearxng(error=OSError("connection refused")))

    def test_a_reply_that_is_not_searxngs_is_said_plainly(self):
        class Garbled(FakeSearxng):
            def __call__(self, url, timeout_s):
                return ["not", "a", "search"]

        with pytest.raises(ValueError, match="isn't answering"):
            _search(Garbled())


class TestNewsBySpan:
    """FR-102, on the room's own clock: Friday 6 November, and the hour set per test."""

    def _titles(self, message):
        return {line.split(". ", 1)[1].split(" — ")[0] for line in message.splitlines() if line[:1].isdigit()}

    def _feed(self):
        return FakeSearxng([
            _result("Overnight", "https://a.in/1", publishedDate="2026-11-05T20:00:00"),  # 01:30 Fri, local
            _result("Breakfast", "https://a.in/2", publishedDate="2026-11-06T01:00:00"),  # 06:30 Fri
            _result("Thursday lunch", "https://a.in/3", publishedDate="2026-11-05T07:00:00"),  # 12:30 Thu
            _result("Monday", "https://a.in/4", publishedDate="2026-11-02T07:00:00"),  # 12:30 Mon
            _result("Last month", "https://a.in/5", publishedDate="2026-10-01T07:00:00"),
            _result("No date", "https://a.in/6"),
        ])

    def test_today_runs_from_local_midnight(self):
        message = _news(self._feed(), when="today").message
        assert self._titles(message) == {"Overnight", "Breakfast"}

    def test_this_morning_starts_at_the_configured_hour(self):
        message = _news(self._feed(), when="this_morning").message
        assert self._titles(message) == {"Breakfast"}

    def test_yesterday_is_the_whole_local_day_before(self):
        message = _news(self._feed(), when="yesterday").message
        assert self._titles(message) == {"Thursday lunch"}

    def test_this_week_is_the_last_seven_days(self):
        message = _news(self._feed(), when="this_week").message
        assert self._titles(message) == {"Overnight", "Breakfast", "Thursday lunch", "Monday"}

    def test_latest_puts_the_newest_first_and_the_undated_last(self):
        fake = FakeSearxng([
            _result("No date", "https://a.in/6"),
            _result("Thursday lunch", "https://a.in/3", publishedDate="2026-11-05T07:00:00"),
            _result("Breakfast", "https://a.in/2", publishedDate="2026-11-06T01:00:00"),
        ])
        message = _news(fake).message
        lines = [line for line in message.splitlines() if line[:1].isdigit()]
        assert lines[0].startswith("1. Breakfast") and "No date" in lines[-1]

    def test_before_the_morning_has_started_there_is_no_morning_news_yet(self):
        with pytest.raises(ValueError, match="this morning"):
            _news(self._feed(), clock=_clock(datetime(2026, 11, 6, 4, 0)), when="this_morning")

    def test_a_span_with_nothing_in_it_is_said_plainly(self):
        fake = FakeSearxng([_result("Last month", "https://a.in/5", publishedDate="2026-10-01T07:00:00")])
        with pytest.raises(ValueError, match="yesterday"):
            _news(fake, when="yesterday")

    def test_with_no_topic_the_headlines_are_searched(self):
        fake = FakeSearxng([_result("A", "https://a.in/x", pubdate="1 hour ago")])
        _news(fake)
        assert fake.parameters()["q"]

    def test_a_topic_is_what_is_searched(self):
        fake = FakeSearxng([_result("A", "https://a.in/x", pubdate="1 hour ago")])
        _news(fake, topic="ISRO")
        assert fake.parameters()["q"] == "ISRO"


class TestAsAProvider:
    def test_it_is_real_not_simulated(self):
        assert _provider(FakeSearxng()).simulated is False

    def test_it_names_searxng(self):
        assert _provider(FakeSearxng()).name == "searxng"

    def test_it_refuses_a_tool_it_does_not_serve(self):
        with pytest.raises(ValueError):
            _provider(FakeSearxng()).invoke("schedule_event", {})
