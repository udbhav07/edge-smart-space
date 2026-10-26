"""Songs beyond the library (FR-96): found online, streamed as they arrive.

What matters: the library always comes first and "play some music" never
leaves it; something named that the library lacks is searched for online,
as is anything asked for "from YouTube"; a result too long to be a song is
passed over; a network failure is said plainly; and the sound card is never
kept waiting on the network.
"""

import random
import threading
import wave

import numpy as np
import pytest

from src.common.config import MusicConfig
from src.common.schemas import AssistantReply, Comfort, Intent, PreferenceHint, SpeechActivity, SpeechStatus
from src.devices.music import CatalogueUnavailable, Library, MusicDevice, RoomVoice, Track
from src.devices.music.stream import PcmBuffer, StreamDecoder, StreamUnavailable, pcm_source
from src.devices.music.track import tidy
from src.devices.music.youtube import Candidate, YouTubeCatalogue


class _Player:
    def __init__(self):
        self.played = []
        self.finished = False
        self.broken = False
        self.position_s = 0.0

    def play(self, track, start_s=0.0, paused=False):
        self.played.append(track)

    def pause(self):
        pass

    def resume(self):
        pass

    def stop(self):
        pass

    def set_volume(self, volume):
        pass


class _Catalogue:
    def __init__(self, track=None, error=None):
        self.track, self.error, self.asked = track, error, []

    def find(self, query):
        self.asked.append(query)
        if self.error:
            raise self.error
        return self.track


ONLINE = Track(title="Queen - Bohemian Rhapsody", url="https://example.invalid/a", origin="youtube", page="https://youtube.com/watch?v=x")


def _device(tmp_path, *names, catalogue=None):
    for name in names:
        path = tmp_path / "music" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    library = Library([tmp_path / "music"], [".mp3"], random.Random(1))
    player = _Player()
    return MusicDevice(library, player, MusicConfig(library_path=str(tmp_path / "music")), catalogue), player


class TestTheLibraryComesFirst:
    def test_a_local_match_never_goes_online(self, tmp_path):
        catalogue = _Catalogue(ONLINE)
        device, player = _device(tmp_path, "Queen - Bohemian Rhapsody.mp3", catalogue=catalogue)
        device.play({"query": "bohemian rhapsody"})
        assert player.played[-1].origin == "library" and catalogue.asked == []

    def test_play_some_music_stays_local(self, tmp_path):
        catalogue = _Catalogue(ONLINE)
        device, player = _device(tmp_path, "a.mp3", catalogue=catalogue)
        device.play({"query": "play some music"})
        assert player.played[-1].origin == "library" and catalogue.asked == []

    def test_play_some_music_with_an_empty_library_says_where_to_put_it(self, tmp_path):
        device, _ = _device(tmp_path, catalogue=_Catalogue(ONLINE))
        with pytest.raises(ValueError, match="no music in"):
            device.play({})


class TestOnline:
    def test_something_the_library_lacks_is_found_online(self, tmp_path):
        catalogue = _Catalogue(ONLINE)
        device, player = _device(tmp_path, "Miles Davis - So What.mp3", catalogue=catalogue)
        outcome = device.play({"query": "play bohemian rhapsody by queen"})
        assert catalogue.asked == ["bohemian rhapsody queen"] and player.played[-1] is ONLINE
        assert outcome.message == "Playing Queen - Bohemian Rhapsody from YouTube."
        assert outcome.detail["source"] == "youtube" and outcome.detail["track"] == "Queen - Bohemian Rhapsody"

    def test_from_youtube_goes_online_even_with_a_local_match(self, tmp_path):
        catalogue = _Catalogue(ONLINE)
        device, player = _device(tmp_path, "Queen - Bohemian Rhapsody.mp3", catalogue=catalogue)
        device.play({"query": "bohemian rhapsody from youtube"})
        assert catalogue.asked == ["bohemian rhapsody"] and player.played[-1].origin == "youtube"

    def test_nothing_found_online_is_said(self, tmp_path):
        device, _ = _device(tmp_path, "a.mp3", catalogue=_Catalogue(None))
        with pytest.raises(ValueError, match="couldn't find “zzz” on YouTube either"):
            device.play({"query": "zzz"})

    def test_a_network_failure_is_said_plainly(self, tmp_path):
        device, _ = _device(tmp_path, catalogue=_Catalogue(error=CatalogueUnavailable("I couldn't reach YouTube just now.")))
        with pytest.raises(ValueError, match="couldn't reach YouTube"):
            device.play({"query": "bohemian rhapsody"})

    def test_turned_off_it_keeps_to_the_library(self, tmp_path):
        device, _ = _device(tmp_path, "a.mp3")
        with pytest.raises(ValueError, match="Nothing in your music matches"):
            device.play({"query": "bohemian rhapsody"})

    def test_asking_for_youtube_when_it_is_off_says_so(self, tmp_path):
        device, _ = _device(tmp_path, "a.mp3")
        with pytest.raises(ValueError, match="turned off"):
            device.play({"query": "bohemian rhapsody on youtube"})

    def test_a_streamed_song_pauses_and_says_its_title(self, tmp_path):
        device, _ = _device(tmp_path, catalogue=_Catalogue(ONLINE))
        device.play({"query": "bohemian rhapsody"})
        assert device.pause({}).message == "Paused Queen - Bohemian Rhapsody."


class _YoutubeDL:
    """yt-dlp's interface, answering from a script of results."""

    calls: list = []

    def __init__(self, results, fail=False):
        self.results, self.fail = results, fail

    def __call__(self, options):
        self.options = options
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, target, download):
        _YoutubeDL.calls.append((target, dict(self.options)))
        if self.fail:
            raise OSError("network down")
        return self.results[target]


SEARCH = "ytsearch8:bohemian rhapsody"


def _watch(id_):
    return f"https://www.youtube.com/watch?v={id_}"


class TestTheCatalogue:
    def setup_method(self):
        _YoutubeDL.calls = []

    def _results(self, *entries, search=SEARCH, metadata=None):
        resolved = {
            _watch(entry["id"]): {
                "title": entry["title"], "channel": entry.get("channel", ""), "url": f"https://cdn.invalid/{entry['id']}",
                "webpage_url": _watch(entry["id"]), "http_headers": {"User-Agent": "x"}, **(metadata or {}).get(entry["id"], {}),
            }
            for entry in entries
        }
        return {search: {"entries": list(entries)}, **resolved}

    def _entry(self, id_, duration, title=None, channel="Queen Official"):
        return {"id": id_, "title": title or f"Queen - Song {id_} (Official Video)", "duration": duration, "channel": channel}

    def test_the_first_song_length_result_is_resolved_and_its_name_tidied(self):
        downloader = _YoutubeDL(self._results(self._entry("live", None), self._entry("mix", 36000), self._entry("song", 354)))
        track = YouTubeCatalogue(15, 10, downloader=downloader).find("bohemian rhapsody")
        assert (track.title, track.artist, track.name) == ("Song song", "Queen", "Song song by Queen")
        assert track.url == "https://cdn.invalid/song" and track.origin == "youtube" and track.video_id == "song"
        assert track.headers == (("User-Agent", "x"),) and track.page.endswith("v=song")

    def test_the_records_own_name_wins_when_youtube_has_it(self):
        metadata = {"a": {"track": "Hotel California (2013 Remaster)", "artist": "Eagles"}}
        downloader = _YoutubeDL(self._results(self._entry("a", 392, title="Hotel California (2013 Remaster)", channel="Eagles"), metadata=metadata))
        assert YouTubeCatalogue(15, 10, downloader=downloader).find("bohemian rhapsody").name == "Hotel California by Eagles"

    def test_a_live_take_or_cover_loses_to_the_song_unless_asked_for(self):
        entries = (self._entry("live", 300, title="Queen - Bohemian Rhapsody (Live Aid 1985)"), self._entry("cover", 300, title="Bohemian Rhapsody cover"), self._entry("song", 354))
        assert YouTubeCatalogue(15, 10, downloader=_YoutubeDL(self._results(*entries))).find("bohemian rhapsody").video_id == "song"
        live = _YoutubeDL(self._results(*entries, search="ytsearch8:bohemian rhapsody live"))
        assert YouTubeCatalogue(15, 10, downloader=live).find("bohemian rhapsody live").video_id == "live"

    def test_the_record_itself_is_preferred(self):
        entries = (self._entry("video", 360), self._entry("record", 354, title="Bohemian Rhapsody", channel="Queen - Topic"))
        assert YouTubeCatalogue(15, 10, downloader=_YoutubeDL(self._results(*entries))).find("bohemian rhapsody").video_id == "record"

    def test_the_search_is_flat_and_only_one_is_resolved(self):
        downloader = _YoutubeDL(self._results(self._entry("a", 200), self._entry("b", 200)))
        YouTubeCatalogue(15, 10, downloader=downloader).find("bohemian rhapsody")
        (search, search_options), (resolve, resolve_options) = _YoutubeDL.calls
        assert search == SEARCH and search_options["extract_flat"] and resolve.endswith("v=a")
        assert resolve_options["format"] == "bestaudio/best" and resolve_options["socket_timeout"] == 10

    def test_nothing_song_length_is_none(self):
        downloader = _YoutubeDL(self._results(self._entry("mix", 36000)))
        assert YouTubeCatalogue(15, 10, downloader=downloader).find("bohemian rhapsody") is None

    def test_a_network_failure_is_unavailable(self):
        with pytest.raises(CatalogueUnavailable, match="couldn't reach YouTube"):
            YouTubeCatalogue(15, 10, downloader=_YoutubeDL({}, fail=True)).find("bohemian rhapsody")

    def test_what_follows_is_youtubes_mix_songs_only_itself_left_out(self):
        mix = {"entries": [self._entry("now", 354), self._entry("next", 210), self._entry("hours", 36000), self._entry("after", 260)]}
        downloader = _YoutubeDL({f"{_watch('now')}&list=RDnow": mix})
        following = YouTubeCatalogue(15, 10, downloader=downloader).up_next("now")
        assert [candidate.video_id for candidate in following] == ["next", "after"]
        assert _YoutubeDL.calls[0][1]["extract_flat"]


class TestTitles:
    @pytest.mark.parametrize(
        ("title", "channel", "expected"),
        [
            ("Queen \u2013 Bohemian Rhapsody (Official Video Remastered)", "Queen Official", ("Bohemian Rhapsody", "Queen")),
            ('"Tum Hi Ho" Aashiqui 2 Full Song With Lyrics | Aditya Roy Kapur, Shraddha Kapoor', "T-Series", ("Tum Hi Ho", "")),
            ("Kesariya - Brahmastra | Ranbir Kapoor | Alia Bhatt | Pritam | Arijit Singh", "Sony Music India", ("Kesariya", "")),
            ("The Stranglers - Golden Brown (Official Lyric Video)", "Classic Rockers", ("Golden Brown", "The Stranglers")),
            ("Shape of You", "Ed Sheeran - Topic", ("Shape of You", "Ed Sheeran")),
            ("Billie Jean", "Michael Jackson", ("Billie Jean", "")),
            ("Miles Davis - So What", "", ("So What", "Miles Davis")),
        ],
    )
    def test_a_video_title_becomes_a_song_name(self, title, channel, expected):
        assert tidy(title, channel) == expected

    def test_metadata_wins(self):
        assert tidy("whatever", "x", track="Hotel California (2013 Remaster)", artist="Eagles, Don Henley") == ("Hotel California", "Eagles")

    def test_a_name_that_is_all_aside_is_kept(self):
        assert tidy("(Official Video)") == ("(Official Video)", "")


class TestTheBuffer:
    def test_short_reads_are_padded_with_silence_not_waited_for(self):
        buffer = PcmBuffer(1000)
        buffer.put(b"\x01\x02" * 4)
        assert buffer.take(16) == b"\x01\x02" * 4 + bytes(8)

    def test_it_ends_only_when_finished_and_empty(self):
        buffer = PcmBuffer(1000)
        buffer.put(b"\x01" * 8)
        buffer.finish()
        assert buffer.take(4) == b"\x01" * 4 and buffer.take(4) == b"\x01" * 4 and buffer.take(4) is None

    def test_a_full_buffer_holds_the_decoder_back(self):
        buffer = PcmBuffer(8)
        buffer.put(b"\x00" * 8)
        done = threading.Event()
        threading.Thread(target=lambda: (buffer.put(b"\x00" * 8), done.set()), daemon=True).start()
        assert not done.wait(0.2)
        buffer.take(8)
        assert done.wait(2.0)

    def test_closing_releases_a_waiting_decoder(self):
        buffer = PcmBuffer(8)
        buffer.put(b"\x00" * 8)
        result = []
        thread = threading.Thread(target=lambda: result.append(buffer.put(b"\x00")), daemon=True)
        thread.start()
        buffer.close()
        thread.join(2.0)
        assert result == [False] and buffer.take(4) is None

    def test_the_source_hands_over_the_frames_asked_for(self):
        buffer = PcmBuffer(1000)
        buffer.put(b"\x05" * 40)
        buffer.finish()
        source = pcm_source(buffer, channels=2)
        next(source)
        assert len(source.send(5)) == 20 and len(source.send(5)) == 20
        with pytest.raises(StopIteration):
            source.send(5)


class TestDecoding:
    def _wav(self, path, seconds=1.0, rate=22050):
        samples = (np.sin(np.arange(int(rate * seconds)) / 5.0) * 8000).astype(np.int16)
        with wave.open(str(path), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(rate)
            file.writeframes(samples.tobytes())
        return path

    def test_a_stream_is_decoded_to_the_cards_format(self, tmp_path):
        decoder = StreamDecoder(str(self._wav(tmp_path / "a.wav")), sample_rate=44100, channels=2)
        assert decoder.wait(10.0)
        source = pcm_source(decoder.buffer, 2)
        next(source)
        total = 0
        try:
            for _ in range(10_000):
                total += len(source.send(4410))
        except StopIteration:
            pass
        assert abs(total / 4 - 44100) <= 4410  # about a second of stereo at 44.1 kHz, give or take one read

    def test_what_cannot_be_opened_is_said_plainly(self, tmp_path):
        (tmp_path / "x.wav").write_bytes(b"not audio")
        with pytest.raises(StreamUnavailable, match="couldn't be opened"):
            StreamDecoder(str(tmp_path / "x.wav"))


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t


def _hint(typed=False, remote=False):
    return PreferenceHint(ts=1.0, comfort=Comfort.UNCHANGED, rationale="music", transcript="play bohemian rhapsody", intent=Intent.SERVICE, typed=typed, remote=remote)


def _reply(spoken=True):
    return AssistantReply(ts=2.0, transcript="play bohemian rhapsody", intent=Intent.SERVICE, reply="Playing it.", spoken=spoken)


def _status(activity):
    return SpeechStatus(ts=3.0, activity=activity, heartbeat_s=5.0)


class _HeldPlayer(_Player):
    def __init__(self):
        super().__init__()
        self.paused_at_start, self.resumed, self.volumes = [], 0, []

    def play(self, track, start_s=0.0, paused=False):
        super().play(track)
        self.paused_at_start.append(paused)
        self.start_s = start_s

    def resume(self):
        self.resumed += 1

    def set_volume(self, volume):
        self.volumes.append(round(volume, 3))


def _room_device(tmp_path, catalogue=None, *names):
    clock = _Clock()
    room = RoomVoice(clock, request_window_s=45)
    for name in names:
        (tmp_path / "music").mkdir(exist_ok=True)
        (tmp_path / "music" / name).write_bytes(b"")
    player = _HeldPlayer()
    jobs = []
    device = MusicDevice(
        Library([tmp_path / "music"], [".mp3"], random.Random(1)), player, MusicConfig(library_path=str(tmp_path / "music")),
        catalogue, room=room, clock=clock, background=jobs.append,
    )
    room.on_change = device.follow_room
    return device, player, room, clock, jobs


class TestJarvisSpeaksFirst:
    def test_a_song_asked_for_aloud_waits_for_the_reply_to_be_said(self, tmp_path):
        device, player, room, _, _ = _room_device(tmp_path, _Catalogue(ONLINE))
        room.on_request("t", _hint())
        device.play({"query": "bohemian rhapsody"})
        assert player.paused_at_start == [True] and device.state()["starting"] and not device.state()["playing"]
        room.on_reply("t", _reply())
        room.on_status("t", _status(SpeechActivity.SPEAKING))
        assert player.resumed == 0  # still saying it
        room.on_status("t", _status(SpeechActivity.LISTENING))
        assert player.resumed == 1 and device.state()["playing"]
        assert device.check() is True  # started off the tick, so the next tick publishes it
        assert device.check() is False

    def test_a_reply_not_said_aloud_lets_it_start_at_once(self, tmp_path):
        device, player, room, _, _ = _room_device(tmp_path, _Catalogue(ONLINE))
        room.on_request("t", _hint())
        device.play({"query": "bohemian rhapsody"})
        room.on_reply("t", _reply(spoken=False))
        assert player.resumed == 1

    def test_it_stops_waiting_after_a_while(self, tmp_path):
        device, player, room, clock, _ = _room_device(tmp_path, _Catalogue(ONLINE))
        room.on_request("t", _hint())
        device.play({"query": "bohemian rhapsody"})
        clock.t += 21
        device.check()
        assert player.resumed == 1

    def test_a_tap_or_a_typed_request_starts_at_once(self, tmp_path):
        device, player, room, _, _ = _room_device(tmp_path, _Catalogue(ONLINE))
        room.on_request("t", _hint(typed=True))
        device.play({"query": "bohemian rhapsody"})
        assert player.paused_at_start == [False] and device.state()["playing"]

    def test_an_old_request_does_not_hold_a_song(self, tmp_path):
        device, player, room, clock, _ = _room_device(tmp_path, _Catalogue(ONLINE))
        room.on_request("t", _hint())
        clock.t += 60
        device.play({"query": "bohemian rhapsody"})
        assert player.paused_at_start == [False]


class TestTheMusicDips:
    def test_down_while_the_room_listens_or_speaks_and_back_after(self, tmp_path):
        device, player, room, _, _ = _room_device(tmp_path, None, "a.mp3")
        device.play({})
        room.on_status("t", _status(SpeechActivity.LISTENING))
        assert player.volumes[-1] == 0.15  # a quarter of 60 %
        room.on_status("t", _status(SpeechActivity.SPEAKING))
        assert player.volumes[-1] == 0.15
        room.on_status("t", _status(SpeechActivity.WAITING))
        assert player.volumes[-1] == 0.6

    def test_overheard_talk_does_not_dip_it(self, tmp_path):
        device, player, room, _, _ = _room_device(tmp_path, None, "a.mp3")
        device.play({})
        room.on_status("t", _status(SpeechActivity.OVERHEARING))
        assert player.volumes[-1] == 0.6


class _Following(_Catalogue):
    def __init__(self, following):
        super().__init__(ONLINE_WITH_ID)
        self.following, self.resolved = following, []

    def up_next(self, video_id):
        return self.following

    def resolve(self, candidate):
        self.resolved.append(candidate.video_id)
        return Track(title=candidate.title, url=f"https://cdn.invalid/{candidate.video_id}", origin="youtube", video_id=candidate.video_id)


ONLINE_WITH_ID = Track(title="Bohemian Rhapsody", artist="Queen", url="https://cdn.invalid/now", origin="youtube", video_id="now")


class TestAutoplay:
    def _playing(self, tmp_path, following):
        catalogue = _Following(following)
        device, player, room, clock, jobs = _room_device(tmp_path, catalogue)
        device.play({"query": "bohemian rhapsody"})
        for job in jobs:
            job()
        return device, player, catalogue

    def test_what_comes_next_is_lined_up_while_it_plays(self, tmp_path):
        device, _, _ = self._playing(tmp_path, [Candidate("next", "Scorpions - Still Loving You (Official Video)", "Scorpions", 386)])
        assert device.state()["up_next"] == "Still Loving You by Scorpions"

    def test_when_a_song_ends_the_next_plays(self, tmp_path):
        device, player, catalogue = self._playing(tmp_path, [Candidate("next", "Billie Jean", "", 294)])
        player.finished = True
        assert device.check() is True and player.played[-1].video_id == "next" and device.state()["playing"]

    def test_a_song_already_played_is_not_lined_up_again(self, tmp_path):
        device, _, _ = self._playing(tmp_path, [Candidate("now", "Bohemian Rhapsody", "", 354), Candidate("next", "Billie Jean", "", 294)])
        assert device.state()["up_next"] == "Billie Jean"

    def test_off_it_stops_at_the_end(self, tmp_path):
        device, player, _ = self._playing(tmp_path, [Candidate("next", "Billie Jean", "", 294)])
        device.set_autoplay({"on": False})
        player.finished = True
        device.check()
        assert device.state()["track"] == "" and player.played[-1].video_id == "now"

    def test_next_skips_now(self, tmp_path):
        device, player, _ = self._playing(tmp_path, [Candidate("next", "Billie Jean", "", 294)])
        assert device.next({}).message == "Next up, Billie Jean." and player.played[-1].video_id == "next"

    def test_next_with_nothing_lined_up_is_said(self, tmp_path):
        device, _, _ = self._playing(tmp_path, [])
        with pytest.raises(ValueError, match="nothing lined up"):
            device.next({})

    def test_the_library_follows_itself(self, tmp_path):
        device, player, _, _, _ = _room_device(tmp_path, None, "a.mp3", "b.mp3")
        device.play({"file": "a.mp3"})
        player.finished = True
        device.check()
        assert player.played[-1].path.name == "b.mp3"


class TestCarryingOn:
    def test_a_dropped_stream_carries_on_from_where_it_was(self, tmp_path):
        device, player, _, _, _ = _room_device(tmp_path, _Catalogue(ONLINE_WITH_ID))
        device.play({"query": "bohemian rhapsody"})
        player.position_s, player.broken, player.finished = 95.0, True, True
        device.check()
        assert player.played[-1] is ONLINE_WITH_ID and player.start_s == 95.0

    def test_resuming_a_stream_that_dropped_while_paused_reopens_it(self, tmp_path):
        device, player, _, _, _ = _room_device(tmp_path, _Catalogue(ONLINE_WITH_ID))
        device.play({"query": "bohemian rhapsody"})
        device.stop({})
        player.position_s, player.broken = 40.0, True
        device.resume({})
        assert len(player.played) == 2 and player.start_s == 40.0 and device.state()["playing"]
