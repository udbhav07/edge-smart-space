"""Hearing the wake word over the room's own music (FR-98).

What matters: the reference is what the speakers played, as 16 kHz mono,
kept in step with the microphone and never piling up; the canceller returns
exactly a frame for a frame; the second detector hears the cancelled signal
while the first hears the microphone untouched; and with nothing to cancel
with, the room still hears.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.speech import echo, wakeword
from src.speech.echo import EchoCanceller, EchoUnavailable, ListeningTwice, LoopbackReference, to_16k

CONFIG = load_config(Path("config/default.yaml")).speech


class _Apm:
    """WebRTC's interface; 'cancels' by subtracting the last reference block."""

    def __init__(self):
        self.far, self.delays = np.zeros(160, dtype=np.int16), []

    def process_reverse_stream(self, frame):
        self.far = np.frombuffer(bytes(frame.data), dtype=np.int16).copy()

    def set_stream_delay_ms(self, ms):
        self.delays.append(ms)

    def process_stream(self, frame):
        near = np.frombuffer(bytes(frame.data), dtype=np.int16)
        frame.data[:] = (near - self.far).astype(np.int16).tobytes()


class TestTheCanceller:
    def test_a_frame_in_is_a_frame_out_whatever_its_size(self):
        canceller = EchoCanceller(40, apm=_Apm())
        for size in (512, 512, 1280, 300):
            assert canceller.process(np.ones(size, dtype=np.int16), np.zeros(size, dtype=np.int16)).size == size

    def test_the_reference_is_taken_out(self):
        canceller = EchoCanceller(40, apm=_Apm())
        music = np.full(1600, 1000, dtype=np.int16)
        out = np.concatenate([canceller.process(music[i: i + 320], music[i: i + 320]) for i in range(0, 1600, 320)])
        assert np.all(out[160:] == 0)  # after the one block of lead

    def test_the_delay_guess_is_given(self):
        apm = _Apm()
        EchoCanceller(55, apm=apm).process(np.zeros(160, dtype=np.int16), np.zeros(160, dtype=np.int16))
        assert apm.delays == [55]

    def test_not_installed_is_said(self, monkeypatch):
        import builtins

        real = builtins.__import__
        monkeypatch.setattr(builtins, "__import__", lambda name, *a, **k: (_ for _ in ()).throw(ImportError()) if name == "livekit" else real(name, *a, **k))
        with pytest.raises(EchoUnavailable):
            EchoCanceller(40)


class TestTheReference:
    def test_speaker_audio_becomes_16k_mono(self):
        reference = LoopbackReference(source=object())
        stereo = np.repeat(np.full(4800, 3000, dtype=np.int16), 2)  # 0.1 s at 48 kHz, both channels
        reference.feed(stereo, 48000, 2)
        taken = reference.take(1600)
        assert taken.size == 1600 and abs(int(np.median(taken)) - 3000) < 60

    def test_nothing_played_is_silence(self):
        assert not LoopbackReference(source=object()).take(512).any()

    def test_it_never_piles_up(self):
        reference = LoopbackReference(source=object())
        for _ in range(100):
            reference.feed(np.zeros(4800 * 2, dtype=np.int16), 48000, 2)
        assert reference._samples <= 0.5 * 16000

    def test_taken_in_order_across_feeds(self):
        reference = LoopbackReference(source=object())
        reference.feed(np.arange(0, 300, dtype=np.int16), 16000, 1)
        reference.feed(np.arange(300, 500, dtype=np.int16), 16000, 1)
        assert list(reference.take(500)) == list(range(500))

    def test_what_was_played_long_ago_is_not_what_the_frame_heard(self):
        reference = LoopbackReference(source=object())
        reference.feed(np.arange(0, 8000, dtype=np.int16), 16000, 1)  # half a second backlog
        taken = reference.take(512)
        # Only the cushion (30 ms, 480 samples) is left waiting: the frame gets what came just before it.
        assert taken[0] == 8000 - 512 - 480 and reference._samples == 480

    def test_low_passed_before_it_is_thinned(self):
        tone = np.sin(2 * np.pi * 15000 * np.arange(48000) / 48000)  # above what 16 kHz can hold
        assert np.sqrt(np.mean(to_16k(tone, 48000) ** 2)) < 0.05


class _Detector:
    def __init__(self, hears=lambda frame: False):
        self.frames, self.resets, self.hears = [], 0, hears

    def heard(self, frame):
        self.frames.append(frame.copy())
        return self.hears(frame)

    def reset(self):
        self.resets += 1


class _Reference:
    def take(self, count):
        return np.full(count, 7, dtype=np.int16)


class TestListeningTwice:
    def test_the_microphone_reaches_the_first_untouched_and_the_second_cleaned(self):
        raw, cleaned = _Detector(), _Detector()
        twice = ListeningTwice(raw, cleaned, EchoCanceller(40, apm=_Apm()), _Reference())
        frame = np.full(512, 100, dtype=np.int16)
        twice.heard(frame)
        twice.heard(frame)
        assert np.all(raw.frames[1] == 100) and np.all(cleaned.frames[1] == 93)

    def test_either_hearing_wakes_it_and_both_start_again(self):
        raw, cleaned = _Detector(), _Detector(hears=lambda frame: True)
        twice = ListeningTwice(raw, cleaned, EchoCanceller(40, apm=_Apm()), _Reference())
        assert twice.heard(np.zeros(512, dtype=np.int16)) and raw.resets == cleaned.resets == 1


class TestWiredIn:
    def test_on_by_default(self):
        assert CONFIG.wake_echo_cancel is True

    def test_built_around_the_chosen_engine(self, monkeypatch):
        monkeypatch.setattr(wakeword, "_engine", lambda config: _Detector())
        monkeypatch.setattr(echo, "LoopbackReference", lambda: _Reference())
        monkeypatch.setattr(echo, "EchoCanceller", lambda delay: EchoCanceller(delay, apm=_Apm()))
        assert isinstance(wakeword.build_wake_detector(CONFIG), ListeningTwice)

    def test_without_a_loopback_the_room_still_hears(self, monkeypatch):
        plain = _Detector()
        monkeypatch.setattr(wakeword, "_engine", lambda config: plain)
        monkeypatch.setattr(echo, "EchoCanceller", lambda delay: EchoCanceller(delay, apm=_Apm()))

        def no_loopback():
            raise EchoUnavailable("not here")

        monkeypatch.setattr(echo, "LoopbackReference", no_loopback)
        assert wakeword.build_wake_detector(CONFIG) is plain
