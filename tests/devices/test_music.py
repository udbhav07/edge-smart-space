"""The music player (FR-87): finding a track, playing it, and the sound itself.

No sound card is touched: the device logic runs against a recording player,
and the generator that feeds the sound card is driven by hand.
"""

import random
from array import array
from pathlib import Path

import pytest

from src.common.config import MusicConfig, load_config
from src.common.devices import DeviceClass
from src.common.tools import ToolEffect
from src.devices.music import Library, MiniaudioPlayer, MusicDevice, build, manifest


class RecordingPlayer:
    def __init__(self):
        self.calls = []
        self.finished = False

    def play(self, path):
        self.calls.append(("play", path.name))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def stop(self):
        self.calls.append(("stop",))

    def set_volume(self, volume):
        self.calls.append(("volume", volume))


def _library(tmp_path, *names, rng=None):
    for name in names:
        path = tmp_path / "music" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    return Library([tmp_path / "music", tmp_path / "uploads"], [".mp3", ".flac", ".wav"], rng or random.Random(1))


@pytest.fixture(name="music")
def _music(tmp_path):
    library = _library(tmp_path, "Jazz/Miles Davis - So What.mp3", "Lofi/rainy evening.flac", "Bach - Cello Suite 1.wav", "notes.txt")
    player = RecordingPlayer()
    return MusicDevice(library, player, MusicConfig(library_path=str(tmp_path / "music"))), player


class TestLibrary:
    def test_only_playable_files_count(self, tmp_path):
        assert [path.name for path in _library(tmp_path, "a.mp3", "b.txt", "c.WAV").tracks()] == ["a.mp3", "c.WAV"]

    def test_a_folder_name_matches(self, tmp_path):
        assert _library(tmp_path, "Jazz/one.mp3", "Rock/two.mp3").find("play some jazz").name == "one.mp3"

    def test_the_best_match_wins(self, tmp_path):
        library = _library(tmp_path, "Miles Davis - Blue.mp3", "Miles Davis - So What.mp3")
        assert library.find("so what by miles davis").name == "Miles Davis - So What.mp3"

    def test_a_request_naming_nothing_plays_anything(self, tmp_path):
        assert _library(tmp_path, "a.mp3", "b.mp3").find("play some music") is not None

    def test_a_request_naming_something_absent_is_none(self, tmp_path):
        assert _library(tmp_path, "a.mp3").find("play polka") is None

    def test_an_empty_library_finds_nothing(self, tmp_path):
        assert _library(tmp_path).find("anything") is None

    def test_a_named_file_is_found_with_or_without_extension(self, tmp_path):
        library = _library(tmp_path, "Lofi/rainy evening.flac")
        assert library.named("rainy evening.flac") == library.named("Rainy Evening") is not None

    def test_a_name_cannot_reach_outside_the_folders(self, tmp_path):
        (tmp_path / "secret.mp3").write_bytes(b"")
        library = _library(tmp_path, "a.mp3")
        assert library.named("../secret.mp3") is None

    def test_a_new_file_counts_at_once(self, tmp_path):
        library = _library(tmp_path, "a.mp3")
        (tmp_path / "uploads").mkdir()
        (tmp_path / "uploads" / "mine.mp3").write_bytes(b"")
        assert library.named("mine.mp3") is not None


class TestDevice:
    def test_play_by_words(self, music):
        device, player = music
        assert device.play({"query": "jazz"}).message == "Playing Miles Davis - So What."
        assert player.calls[-1] == ("play", "Miles Davis - So What.mp3")

    def test_play_by_file(self, music):
        device, player = music
        device.play({"file": "rainy evening.flac"})
        assert device.state()["track"] == "rainy evening" and device.state()["playing"] is True

    def test_an_unknown_file_is_said_plainly(self, music):
        with pytest.raises(ValueError, match="couldn't find a file called nope.mp3"):
            music[0].play({"file": "nope.mp3"})

    def test_nothing_matching_is_said_plainly(self, music):
        with pytest.raises(ValueError, match="Nothing in your music matches"):
            music[0].play({"query": "polka"})

    def test_an_empty_library_says_where_to_put_music(self, tmp_path):
        device = MusicDevice(_library(tmp_path), RecordingPlayer(), MusicConfig(library_path="~/Music"))
        with pytest.raises(ValueError, match="no music in ~/Music"):
            device.play({})

    def test_pause_and_resume(self, music):
        device, player = music
        device.play({"query": "bach"})
        device.pause({})
        assert device.state()["paused"] is True and device.state()["playing"] is False
        device.resume({})
        assert player.calls[-1] == ("resume",) and device.state()["playing"] is True

    def test_play_with_nothing_said_resumes_what_was_paused(self, music):
        device, player = music
        device.play({"query": "bach"})
        device.pause({})
        device.play({})
        assert player.calls[-1] == ("resume",)

    def test_pause_with_nothing_playing_is_said(self, music):
        with pytest.raises(ValueError, match="Nothing is playing"):
            music[0].pause({})

    def test_resume_with_nothing_paused_is_said(self, music):
        with pytest.raises(ValueError, match="Nothing is paused"):
            music[0].resume({})

    def test_stop(self, music):
        device, player = music
        device.play({"query": "bach"})
        device.stop({})
        assert player.calls[-1] == ("stop",) and device.state()["track"] == ""

    def test_volume_reaches_the_player_as_a_fraction(self, music):
        device, player = music
        device.set_volume({"volume": 25})
        assert player.calls[-1] == ("volume", 0.25) and device.state()["volume"] == 25

    def test_the_configured_volume_is_set_at_start(self, tmp_path):
        player = RecordingPlayer()
        MusicDevice(_library(tmp_path), player, MusicConfig(default_volume=40))
        assert player.calls == [("volume", 0.4)]

    def test_a_track_ending_by_itself_is_noticed_once(self, music):
        device, player = music
        device.play({"query": "bach"})
        player.finished = True
        assert device.check() is True and device.state()["playing"] is False
        assert device.check() is False

    def test_now_playing(self, music):
        device, _ = music
        device.play({"query": "bach"})
        assert device.now_playing({}).message == "Playing Bach - Cello Suite 1, at 60% volume."


class TestManifest:
    def test_everything_is_media_and_real(self):
        built = manifest(MusicConfig(), 1.0)
        assert built.simulated is False and {c.device_class for c in built.capabilities} == {DeviceClass.MEDIA}

    def test_now_playing_is_a_read(self):
        assert manifest(MusicConfig(), 1.0).capability("now_playing").effect is ToolEffect.READ

    def test_volume_is_bounded(self):
        volume = manifest(MusicConfig(), 1.0).capability("set_volume").parameters[0]
        assert (volume.minimum, volume.maximum) == (0, 100)

    def test_every_capability_has_a_handler(self):
        parts = build(load_config(Path("config/default.yaml")), _Clock(), player=RecordingPlayer())
        assert set(parts.handlers) == {c.name for c in parts.manifest.capabilities}


class _Clock:
    def now(self):
        return 1.0


def _decoder(chunks):
    """Stands in for miniaudio's decoder: already primed, answers send() with samples."""

    def generate():
        yield array("h")
        for chunk in chunks:
            yield array("h", chunk)

    source = generate()
    next(source)
    return source


class TestTheSoundFeed:
    def _player(self, chunks, volume=1.0):
        player = MiniaudioPlayer()
        player._source = _decoder(chunks)
        player.set_volume(volume)
        feed = player._feed()
        next(feed)
        return player, feed

    def test_samples_pass_through_at_full_volume(self):
        _, feed = self._player([[1000, -1000]])
        assert array("h", feed.send(1)).tolist() == [1000, -1000]

    def test_volume_scales_the_samples(self):
        _, feed = self._player([[1000, -1000]], volume=0.5)
        assert array("h", feed.send(1)).tolist() == [500, -500]

    def test_paused_is_silence_and_keeps_its_place(self):
        player, feed = self._player([[1000, 1000], [2000, 2000]])
        player.pause()
        assert array("h", feed.send(1)).tolist() == [0, 0]
        player.resume()
        assert array("h", feed.send(1)).tolist() == [1000, 1000]

    def test_the_end_counts_only_once_the_queued_sound_has_played(self):
        """The card asks ahead; closing it when the decoder runs dry would cut
        off what it still had queued."""
        player, feed = self._player([[5, 5]])
        feed.send(1)
        assert array("h", feed.send(1)).tolist() == [0, 0] and player.finished is False
        feed.send(MiniaudioPlayer.TAIL_FRAMES - 1)
        assert player.finished is False
        feed.send(1)
        assert player.finished is True

    def test_nothing_loaded_is_silence(self):
        player = MiniaudioPlayer()
        feed = player._feed()
        next(feed)
        assert feed.send(3) == bytes(3 * 2 * 2)
