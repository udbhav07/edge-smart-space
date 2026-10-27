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

The detector is a seam (FR-98): anything with ``heard(frame)`` and
``reset()`` will do, and ``speech.wake_engine`` chooses which --

- ``openwakeword``: a model trained for the one phrase (the default);
- ``sherpa``: sherpa-onnx's keyword spotter, which listens for the phrase as
  words, so another way of saying it is a line of configuration, not a
  retraining;
- ``both``: wakes only when the two agree, for fewer false wakes;
- ``either``: wakes when either hears it, for fewer missed ones.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from src.common.config import SpeechConfig

LOGGER = logging.getLogger(__name__)

#: openWakeWord scores absent keys as no detection.
_NO_DETECTION = 0.0

#: ONNX rather than TFLite: it is the runtime available on both the
#: development laptop and the Jetson.
_INFERENCE_FRAMEWORK = "onnx"


class WakeDetector(Protocol):
    """What the pipeline asks of a wake word detector, frame by frame."""

    def heard(self, frame: np.ndarray) -> bool:
        """Whether the phrase has just been spoken, given the next 16 kHz int16 frame."""

    def reset(self) -> None:
        """Forget what has been heard, so the next utterance starts clean."""


class WakeWordDetector:
    """Wraps openWakeWord and answers one question: was the phrase spoken?

    Model weights are expected to be present already. Downloading them at
    startup would make the node depend on an internet connection it is
    specified not to have (NFR-06); ``setup_models.py`` fetches them once.
    """

    def __init__(self, config: SpeechConfig, model=None) -> None:
        self._config = config
        self._model = model if model is not None else self._load(config)

    @staticmethod
    def _load(config: SpeechConfig):
        """Build the default detector.

        openWakeWord is imported here rather than at module scope so the
        threshold logic stays importable on a machine without it (NFR-09).
        """
        from openwakeword.model import Model

        LOGGER.info("loading wake word model %r", config.wake_word)
        return Model(
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


class KeywordSpotter:
    """sherpa-onnx's streaming keyword spotter, listening for phrases as words (FR-98).

    The phrases are plain words -- "HEY JARVIS" -- turned into the model's
    word pieces when it loads, so a way of saying it that is missed is added
    in configuration, not by retraining. It runs on the CPU.
    """

    #: The spotter's own sample rate: the microphone's.
    SAMPLE_RATE_HZ = 16000

    def __init__(self, config: SpeechConfig, spotter=None) -> None:
        self._config = config
        self._spotter = spotter if spotter is not None else self._load(config)
        self._stream = self._spotter.create_stream()

    @staticmethod
    def _load(config: SpeechConfig):
        """Build the spotter. Imported here, so the rest loads without sherpa-onnx (NFR-09)."""
        import sentencepiece
        import sherpa_onnx

        folder = Path(config.kws_model_dir)
        pieces = sentencepiece.SentencePieceProcessor(model_file=str(folder / "bpe.model"))
        keywords = folder / "keywords.local.txt"
        lines = [" ".join(pieces.encode_as_pieces(phrase.upper())) + " @" + phrase.replace(" ", "_") for phrase in config.kws_phrases]
        keywords.write_text("\n".join(lines) + "\n", encoding="utf-8")
        found = {part: next(folder.glob(f"{part}-*chunk-16*.int8.onnx")) for part in ("encoder", "decoder", "joiner")}
        LOGGER.info("loading sherpa-onnx keyword spotter for %s", ", ".join(config.kws_phrases))
        return sherpa_onnx.KeywordSpotter(
            tokens=str(folder / "tokens.txt"),
            encoder=str(found["encoder"]),
            decoder=str(found["decoder"]),
            joiner=str(found["joiner"]),
            keywords_file=str(keywords),
            keywords_score=config.kws_boost,
            keywords_threshold=config.kws_threshold,
            num_trailing_blanks=1,
            num_threads=1,
            provider="cpu",
        )

    def heard(self, frame: np.ndarray) -> bool:
        self._stream.accept_waveform(self.SAMPLE_RATE_HZ, np.asarray(frame, dtype=np.float32) / 32768.0)
        heard = False
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            if self._spotter.get_result(self._stream):
                heard = True
                self._spotter.reset_stream(self._stream)
        return heard

    def reset(self) -> None:
        self._stream = self._spotter.create_stream()


class Agreeing:
    """Wakes only when every detector has heard the phrase within a short span (FR-98).

    Two detectors that fail differently -- one trained on the phrase, one
    listening for its words -- rarely both mistake the same music for it.
    """

    def __init__(self, detectors: Sequence[WakeDetector], within_frames: int) -> None:
        self._detectors = list(detectors)
        self._within = within_frames
        self._since: list[int | None] = [None] * len(self._detectors)

    def heard(self, frame: np.ndarray) -> bool:
        for index, detector in enumerate(self._detectors):
            if detector.heard(frame):
                self._since[index] = 0
            elif self._since[index] is not None:
                self._since[index] += 1
                if self._since[index] > self._within:
                    self._since[index] = None
        if all(since is not None for since in self._since):
            self.reset()
            return True
        return False

    def reset(self) -> None:
        for detector in self._detectors:
            detector.reset()
        self._since = [None] * len(self._detectors)


class Either:
    """Wakes when any detector hears the phrase (FR-98): fewer missed wakes, more false ones."""

    def __init__(self, detectors: Sequence[WakeDetector]) -> None:
        self._detectors = list(detectors)

    def heard(self, frame: np.ndarray) -> bool:
        # Every detector hears every frame, so each keeps its own context whole.
        results = [detector.heard(frame) for detector in self._detectors]
        if any(results):
            self.reset()
            return True
        return False

    def reset(self) -> None:
        for detector in self._detectors:
            detector.reset()


def build_wake_detector(config: SpeechConfig) -> WakeDetector:
    """The detector ``speech.wake_engine`` names (FR-98) -- listening twice, with
    the room's own sound taken out of the second hearing, if ``wake_echo_cancel``."""
    detector = _engine(config)
    if not config.wake_echo_cancel:
        return detector
    from src.speech.echo import EchoCanceller, EchoUnavailable, ListeningTwice, LoopbackReference

    try:
        canceller = EchoCanceller(config.echo_delay_ms)
        reference = LoopbackReference()
    except (EchoUnavailable, OSError) as exc:
        # Hearing over the music is better with it, but the room hears without it.
        LOGGER.warning("wake word without echo cancelling: %s", exc)
        return detector
    LOGGER.info("wake word heard twice: as the microphone hears it, and with the room's own sound taken out")
    return ListeningTwice(detector, _engine(config), canceller, reference)


def _engine(config: SpeechConfig) -> WakeDetector:
    engine = config.wake_engine
    if engine == "openwakeword":
        return WakeWordDetector(config)
    if engine == "sherpa":
        return KeywordSpotter(config)
    pair = [WakeWordDetector(config), KeywordSpotter(config)]
    if engine == "both":
        return Agreeing(pair, within_frames=max(1, round(config.wake_agree_s * KeywordSpotter.SAMPLE_RATE_HZ / config.chunk_samples)))
    return Either(pair)
