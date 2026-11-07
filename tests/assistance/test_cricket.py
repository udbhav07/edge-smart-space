"""Live cricket scores from Cricinfo's feed (FR-103).

The feed is never reached. ``feeds/cricinfo_livescores.xml`` is the real feed
as it was on 6 November 2026, and the fetch function hands it over.
"""

from datetime import datetime
from pathlib import Path

import pytest

from src.assistance.providers.cricket import CricinfoFeed, CricketScores
from src.common.clock import SimClock
from src.common.localtime import epoch_of
from src.common.tools import CRICKET_SCORE

FEED = (Path(__file__).parent / "feeds" / "cricinfo_livescores.xml").read_bytes()
IST_H = 5.5
ALIASES = {"rcb": "Royal Challengers Bengaluru"}


class FakeFeed:
    def __init__(self, body=FEED, error=None):
        self.body = body
        self.error = error
        self.reads = 0

    def __call__(self, url, timeout_s):
        self.reads += 1
        if self.error is not None:
            raise self.error
        return self.body


def _clock():
    return SimClock(start_epoch_s=epoch_of(datetime(2026, 11, 6, 20, 35), IST_H))


def _scores(fake=None, clock=None, cache_s=30.0, max_matches=6):
    clock = clock or _clock()
    feed = CricinfoFeed("https://static.cricinfo.com/rss/livescores.xml", 5.0, clock, cache_s, fetch=fake or FakeFeed())
    return CricketScores(feed, clock, utc_offset_h=IST_H, aliases=ALIASES, max_matches=max_matches)


def _ask(team=None, **kwargs):
    arguments = {"team": team} if team else {}
    return _scores(**kwargs).invoke(CRICKET_SCORE.name, arguments)


def _lines(message):
    return [line for line in message.splitlines() if line[:1].isdigit()]


class TestReadingTheFeed:
    def test_the_side_batting_is_named_as_batting(self):
        assert "Bangladesh 88/3 (batting) v West Indies 356/10" in _ask("Bangladesh").message

    def test_a_match_with_no_score_yet_says_so(self):
        assert "New Zealand v India: no score yet" in _ask("New Zealand").message

    def test_a_second_innings_score_is_kept_whole(self):
        assert "111/10 (batting)" in _ask("Sri Lanka").message

    def test_it_says_the_scores_are_cricinfos_and_when_they_were(self):
        # The feed's own pubDate, 15:00:04 UTC, is 20:30 in the room.
        assert "from Cricinfo, as of 6 Nov 2026, 20:30" in _ask().message

    def test_a_feed_that_does_not_say_when_is_said_to_be_as_read(self):
        # The clock reads 20:35; the feed is read then, and says nothing of its own time.
        undated = b"\n".join(line for line in FEED.splitlines() if b"<pubDate>" not in line)
        outcome = _ask("Bangladesh", fake=FakeFeed(undated))
        assert "from Cricinfo, read at 6 Nov 2026, 20:35:" in outcome.message
        assert outcome.detail["spoken"].startswith("According to Cricinfo, read at 6 Nov 2026, 20:35:")

    def test_a_cached_feed_says_when_it_was_read_not_when_asked(self):
        undated = b"\n".join(line for line in FEED.splitlines() if b"<pubDate>" not in line)
        clock = _clock()
        scores = _scores(FakeFeed(undated), clock)
        scores.invoke(CRICKET_SCORE.name, {})
        clock.advance(20.0)
        assert "read at 6 Nov 2026, 20:35:" in scores.invoke(CRICKET_SCORE.name, {}).message


class TestFindingTheTeam:
    def test_the_team_itself_comes_before_its_a_side_and_under_19s(self):
        lines = _lines(_ask("India").message)
        assert lines[0].startswith("1. New Zealand v India")
        assert any("India A" in line for line in lines) and any("India Under-19s" in line for line in lines)

    def test_a_team_is_found_whatever_its_case(self):
        assert "Bangladesh 88/3" in _ask("bangladesh").message

    def test_a_short_name_is_read_through_the_configured_aliases(self):
        body = FEED.replace(b"<title>Easterns v Dolphins</title>", b"<title>Royal Challengers Bengaluru 161/5 * v Gujarat Titans 155/8 </title>")
        assert "Royal Challengers Bengaluru 161/5 (batting)" in _ask("RCB", fake=FakeFeed(body)).message

    def test_a_team_not_playing_is_said_plainly(self):
        with pytest.raises(ValueError, match="No live match for Australia"):
            _ask("Australia")

    def test_with_no_team_every_live_match_up_to_the_limit(self):
        assert len(_lines(_ask(max_matches=4).message)) == 4

    def test_matches_being_played_come_before_those_not_started(self):
        lines = _lines(_ask(max_matches=20).message)
        first_unstarted = next(n for n, line in enumerate(lines) if "no score yet" in line)
        assert all("no score yet" in line for line in lines[first_unstarted:])


class TestTheCache:
    def test_a_question_repeated_soon_reads_the_feed_once(self):
        fake = FakeFeed()
        clock = _clock()
        scores = _scores(fake, clock)
        scores.invoke(CRICKET_SCORE.name, {})
        clock.advance(10.0)
        scores.invoke(CRICKET_SCORE.name, {})
        assert fake.reads == 1

    def test_after_the_cache_runs_out_the_feed_is_read_again(self):
        fake = FakeFeed()
        clock = _clock()
        scores = _scores(fake, clock)
        scores.invoke(CRICKET_SCORE.name, {})
        clock.advance(31.0)
        scores.invoke(CRICKET_SCORE.name, {})
        assert fake.reads == 2


class TestWhenTheFeedFails:
    def test_a_feed_not_answering_is_said_plainly(self):
        with pytest.raises(ValueError, match="Cricinfo isn't answering"):
            _ask(fake=FakeFeed(error=OSError("timed out")))

    def test_a_feed_that_is_not_rss_is_said_plainly(self):
        with pytest.raises(ValueError, match="Cricinfo isn't answering"):
            _ask(fake=FakeFeed(b"<html>maintenance</html"))

    def test_an_empty_feed_means_no_live_matches(self):
        empty = b"<?xml version='1.0'?><rss><channel><title>Cricinfo Live Scores</title></channel></rss>"
        with pytest.raises(ValueError, match="no live matches"):
            _ask(fake=FakeFeed(empty))


class TestAsAProvider:
    def test_it_is_real_and_names_cricinfo(self):
        scores = _scores()
        assert scores.name == "cricinfo" and scores.simulated is False

    def test_it_refuses_a_tool_it_does_not_serve(self):
        with pytest.raises(ValueError):
            _scores().invoke("web_search", {"query": "x"})
