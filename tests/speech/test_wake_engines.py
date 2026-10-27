"""The wake word detector as a seam (FR-98).

What matters: any detector with heard() and reset() will do; the keyword
spotter listens for the configured phrases as words; "both" wakes only when
the two agree in time, "either" when one hears it; every detector hears every
frame, so none loses its context; and configuration alone chooses.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.config import load_config
from src.speech import wakeword
from src.speech.wakeword import Agreeing, Either, KeywordSpotter, build_wake_detector

CONFIG = load_config(Path("config/default.yaml")).speech
FRAME = np.zeros(1280, dtype=np.int16)


class _Scripted:
    """A detector that hears on the frames it is told to."""

    def __init__(self, *hits):
        self.hits, self.frame, self.heard_frames, self.resets = set(hits), 0, [], 0

    def heard(self, frame):
        self.frame += 1
        self.heard_frames.append(self.frame)
        return self.frame in self.hits

    def reset(self):
        self.resets += 1


class _Spotter:
    """sherpa-onnx's interface: results come once a stream has enough audio."""

    def __init__(self, fire_after=None):
        self.fire_after, self.samples, self.streams, self.resets = fire_after, 0, 0, 0
        self.fired = False

    def create_stream(self):
        self.streams += 1
        self.samples = 0
        return self

    def accept_waveform(self, rate, samples):
        assert rate == 16000 and samples.dtype == np.float32 and np.max(np.abs(samples)) <= 1.0
        self.samples += samples.size

    def is_ready(self, _stream):
        return self.samples >= 1280 and not getattr(self, "_decoded", False)

    def decode_stream(self, _stream):
        self._decoded = True

    def get_result(self, _stream):
        self._decoded = False
        self.samples = 0
        if self.fire_after is not None and not self.fired:
            self.fire_after -= 1
            if self.fire_after <= 0:
                self.fired = True
                return "HEY_JARVIS"
        return ""

    def reset_stream(self, _stream):
        self.resets += 1


class TestTheKeywordSpotter:
    def test_it_hears_when_the_spotter_reports_the_phrase(self):
        spotter = KeywordSpotter(CONFIG, spotter=_Spotter(fire_after=2))
        assert [spotter.heard(FRAME) for _ in range(3)] == [False, True, False]

    def test_audio_reaches_it_as_float_in_range(self):
        spotter = KeywordSpotter(CONFIG, spotter=_Spotter())
        spotter.heard(np.full(1280, -32768, dtype=np.int16))  # the fake asserts the range

    def test_reset_starts_a_fresh_stream(self):
        fake = _Spotter()
        spotter = KeywordSpotter(CONFIG, spotter=fake)
        spotter.reset()
        assert fake.streams == 2


class TestAgreeing:
    def test_wakes_when_both_hear_it_close_together(self):
        a, b = _Scripted(3), _Scripted(5)
        both = Agreeing([a, b], within_frames=4)
        assert [both.heard(FRAME) for _ in range(6)] == [False, False, False, False, True, False]

    def test_not_when_too_far_apart(self):
        both = Agreeing([_Scripted(1), _Scripted(9)], within_frames=4)
        assert not any(both.heard(FRAME) for _ in range(10))

    def test_one_alone_never_wakes_it(self):
        both = Agreeing([_Scripted(2, 4, 6), _Scripted()], within_frames=4)
        assert not any(both.heard(FRAME) for _ in range(8))

    def test_waking_resets_both(self):
        a, b = _Scripted(1), _Scripted(1)
        Agreeing([a, b], within_frames=2).heard(FRAME)
        assert a.resets == b.resets == 1


class TestEither:
    def test_either_one_wakes_it(self):
        either = Either([_Scripted(), _Scripted(2)])
        assert [either.heard(FRAME) for _ in range(3)] == [False, True, False]

    def test_every_detector_hears_every_frame(self):
        a, b = _Scripted(1), _Scripted()
        either = Either([a, b])
        for _ in range(3):
            either.heard(FRAME)
        assert a.heard_frames == b.heard_frames == [1, 2, 3]


class TestChosenByConfiguration:
    @pytest.fixture(autouse=True)
    def _no_models(self, monkeypatch):
        monkeypatch.setattr(wakeword, "WakeWordDetector", lambda config: _Scripted())
        monkeypatch.setattr(wakeword, "KeywordSpotter", type("Spotter", (_Scripted,), {"SAMPLE_RATE_HZ": 16000, "__init__": lambda self, config: _Scripted.__init__(self)}))

    @pytest.mark.parametrize(("engine", "kind"), [("openwakeword", _Scripted), ("both", Agreeing), ("either", Either)])
    def test_the_engine_named_is_built(self, engine, kind):
        assert isinstance(build_wake_detector(CONFIG.model_copy(update={"wake_engine": engine})), kind)

    def test_both_agree_within_the_configured_span(self):
        detector = build_wake_detector(CONFIG.model_copy(update={"wake_engine": "both", "wake_agree_s": 1.6}))
        assert detector._within == round(1.6 * 16000 / CONFIG.chunk_samples)

    def test_the_shipped_default_is_either(self):
        assert CONFIG.wake_engine == "either" and CONFIG.kws_phrases == ["HEY JARVIS", "JARVIS"]

    def test_an_unknown_engine_is_refused(self):
        with pytest.raises(ValueError):
            CONFIG.model_validate({**CONFIG.model_dump(), "wake_engine": "porcupine"})
