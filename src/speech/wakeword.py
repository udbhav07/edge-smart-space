"""Wake-word detection (FR-50).

The only always-resident model in the speech path. Nothing is recorded for
transcription until this fires: the capture window opens on detection and
closes at end of utterance or a hard timeout, which is what lets the system
say it is not always-listening in the sense that matters (DESIGN.md
section 5.8).

The detection threshold is configuration, not a constant, and it is worth
understanding what it trades. Set it too low and the window opens on
background noise, so the microphone is effectively always recording and the
privacy claim above stops being true.
"""

from __future__ import annotations

import logging

import numpy as np
from openwakeword.model import Model

from src.common.config import SpeechConfig

LOGGER = logging.getLogger(__name__)

#: openWakeWord scores absent keys as no detection.
_NO_DETECTION = 0.0

#: ONNX rather than TFLite: it is the runtime available on both the
#: development laptop and the Jetson.
_INFERENCE_FRAMEWORK = "onnx"


class WakeWordDetector:
    """Wraps openWakeWord and answers one question: was the phrase spoken?

    Model weights are expected to be present already. Downloading them at
    startup would make the node depend on an internet connection it is
    specified not to have (NFR-06); ``setup_models.py`` fetches them once.
    """

    def __init__(self, config: SpeechConfig, model: Model | None = None) -> None:
        self._config = config
        self._model = model if model is not None else Model(
            wakeword_models=[config.wake_word],
            inference_framework=_INFERENCE_FRAMEWORK,
        )

    @property
    def wake_word(self) -> str:
        return self._config.wake_word

    def score(self, frame: np.ndarray) -> float:
        """Detection confidence for this frame, in [0, 1]."""
        scores = self._model.predict(frame)
        return scores.get(self._config.wake_word, _NO_DETECTION)

    def heard(self, frame: np.ndarray) -> bool:
        """Whether this frame crosses the configured threshold."""
        return self.score(frame) > self._config.wake_word_threshold

    def reset(self) -> None:
        """Clear internal state so the next utterance starts clean."""
        self._model.reset()
