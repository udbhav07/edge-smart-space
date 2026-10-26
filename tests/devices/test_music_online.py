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
from src.devices.music import CatalogueUnavailable, Library, MusicDevice, Track
from src.devices.music.stream import PcmBuffer, StreamDecoder, StreamUnavailable, pcm_source
from src.devices.music.youtube import YouTubeCatalogue


class _Player:
    def __init__(self):
        self.played = []
        self.finished = False

    def play(self, track):
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


SEARCH = "ytsearch5:bohemian rhapsody"


class TestTheCatalogue:
    def setup_method(self):
        _YoutubeDL.calls = []

    def _results(self, *entries):
        resolved = {
            entry["url"]: {"title": entry["title"], "url": f"https://cdn.invalid/{entry['id']}", "webpage_url": entry["url"], "http_headers": {"User-Agent": "x"}}
            for entry in entries
        }
        return {SEARCH: {"entries": list(entries)}, **resolved}

    def _entry(self, id_, duration):
        return {"id": id_, "title": f"song {id_}", "duration": duration, "url": f"https://www.youtube.com/watch?v={id_}"}

    def test_the_first_song_length_result_is_resolved_to_its_stream(self):
        downloader = _YoutubeDL(self._results(self._entry("live", None), self._entry("mix", 36000), self._entry("song", 354)))
        track = YouTubeCatalogue(15, 10, downloader=downloader).find("bohemian rhapsody")
        assert track.title == "song song" and track.url == "https://cdn.invalid/song" and track.origin == "youtube"
        assert track.headers == (("User-Agent", "x"),) and track.page.endswith("v=song")

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
