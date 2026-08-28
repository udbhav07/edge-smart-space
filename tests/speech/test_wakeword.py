"""Unit tests for the wake-word gate.

The detector model is injected, so what is tested is the threshold decision
itself: the thing that determines whether the capture window opens, and
therefore whether the claim in section 5.8 about not always-listening holds.
"""

from pathlib import Path

import numpy as np
import pytest

from src.common.config import SpeechConfig, load_config
from src.speech.wakeword import WakeWordDetector

FRAME = np.zeros(512, dtype=np.int16)


@pytest.fixture(name="config")
def _config() -> SpeechConfig:
    return load_config(Path("config/default.yaml")).speech


class FakeModel:
    """Returns a scripted score for the configured wake word."""

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores if scores is not None else {}
        self.reset_count = 0
        self.predictions = 0

    def predict(self, frame) -> dict[str, float]:
        self.predictions += 1
        return self.scores

    def reset(self) -> None:
        self.reset_count += 1


def _detector(config, scores=None) -> tuple[WakeWordDetector, FakeModel]:
    model = FakeModel(scores)
    return WakeWordDetector(config, model=model), model


class TestScoring:
    def test_reports_the_score_for_the_configured_word(self, config):
        detector, _ = _detector(config, {config.wake_word: 0.83})
        assert detector.score(FRAME) == 0.83

    def test_an_absent_key_scores_as_no_detection(self, config):
        """A model that reports nothing must not read as a detection."""
        detector, _ = _detector(config, {})
        assert detector.score(FRAME) == 0.0

    def test_a_score_for_another_word_is_ignored(self, config):
        detector, _ = _detector(config, {"some_other_phrase": 0.99})
        assert detector.heard(FRAME) is False

    def test_the_configured_word_is_reported(self, config):
        detector, _ = _detector(config)
        assert detector.wake_word == config.wake_word


class TestThreshold:
    def test_a_score_above_the_threshold_opens_the_window(self, config):
        detector, _ = _detector(config, {config.wake_word: config.wake_word_threshold + 0.1})
        assert detector.heard(FRAME) is True

    def test_a_score_below_the_threshold_does_not(self, config):
        detector, _ = _detector(config, {config.wake_word: config.wake_word_threshold - 0.1})
        assert detector.heard(FRAME) is False

    def test_the_threshold_itself_does_not_trigger(self, config):
        detector, _ = _detector(config, {config.wake_word: config.wake_word_threshold})
        assert detector.heard(FRAME) is False

    def test_silence_does_not_trigger(self, config):
        detector, _ = _detector(config, {config.wake_word: 0.0})
        assert detector.heard(FRAME) is False

    def test_the_shipped_threshold_still_requires_a_detection(self, config):
        """The threshold is 0.1, low, and measured rather than chosen: the
        hey_jarvis model is trained on American-accented speech and does not
        fire reliably for this team at 0.5.

        What must stay true is that a detection is still required. At zero
        every frame would wake the system and FR-50's claim that capture
        follows detection would be empty. The cost of a low threshold is a
        higher false-wake rate, which is recorded in the config rather than
        hidden here.
        """
        assert config.wake_word_threshold > 0.0


class TestReset:
    def test_reset_clears_the_model_state(self, config):
        detector, model = _detector(config)
        detector.reset()
        assert model.reset_count == 1

    def test_scoring_consults_the_model_every_frame(self, config):
        detector, model = _detector(config, {config.wake_word: 0.1})
        for _ in range(4):
            detector.heard(FRAME)
        assert model.predictions == 4
