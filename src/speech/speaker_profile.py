"""Who is speaking, for personalisation only (FR-52).

Speaker recognition here selects whose profile -- whose memory and
preferences -- the assistant answers with. It is not a security control and
grants nothing: no tool, setpoint or booking is gated on it, and the reply
says so (FR-52). That is what makes a modest recogniser acceptable: a wrong
match costs a reply addressed to the wrong name, never an action.

**The features** are the voice's cepstral shape and its pitch: liftered
mel-frequency cepstral coefficients over the voiced frames, summarised by
their mean and spread, and the median and range of the fundamental
frequency. numpy only, which matters on a node where torch is optional, and
deterministic. Weaker than a neural speaker embedding (an ECAPA-TDNN, say);
:class:`SpeakerEmbedder` is the seam for one.

**The match** is a distance in each enrolled voice's own spread: every
feature of an utterance is compared with that voice's enrolment mean in units
of its enrolment standard deviation, and the score is ``exp(-mean(z^2)/2)``.
Plain cosine similarity was measured first and could not separate voices at
all -- every cepstral vector shares one large common shape -- so the
spread-scaled distance is what ships. A voice needs
``speech.speaker_min_enrolment_utterances`` utterances before it can be
matched, because one utterance has no spread.

Measured on synthetic voices only (``tests/speech/voices.py``: pitch and
vocal-tract length varied), at the shipped threshold of 0.15: the nearest
voice was the right one 36 of 36 times; a household voice was recognised
88 % of the time and otherwise taken for a guest; a stranger was taken for
someone 5 % of the time; nobody scored above the threshold as the wrong
member. Being a guest is the safe error, and the threshold is set for it.
Real voices are a bring-up measurement: ``tools.enroll`` reports how close a
new voice is to those already enrolled.

**What is stored** is statistics of feature vectors, never audio (FR-51):
the utterance is analysed in memory while it is still held for
transcription, and dropped with it. Deleting a profile deletes its print.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from src.common.config import SpeechConfig
from src.common.schemas import PROFILE_ID_PATTERN

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_ENCODING = "utf-8"
_TEMPORARY_SUFFIX = ".tmp"

#: Analysis frames: the usual 25 ms windows every 10 ms. Part of the
#: features' definition, not policy: changing any of these invalidates every
#: enrolled voiceprint, which is why they are not configuration.
_FRAME_S = 0.025
_HOP_S = 0.010
_MEL_BANDS = 40
_MEL_LOW_HZ = 60.0
_MEL_HIGH_HZ = 7600.0
#: Cepstral coefficients kept. c0 is loudness, which says how close the
#: speaker stood, not who they are, so it is dropped.
_CEPSTRA = 20
#: Sinusoidal liftering: raises the higher coefficients, which carry the
#: vocal tract's detail, above the common spectral tilt every voice shares.
_LIFTER = 22
#: Frames quieter than this below the loudest are silence, not voice; and
#: any frame below the absolute floor is silence whatever its neighbours,
#: or a silent clip would count every frame as voiced.
_VOICED_WITHIN_DB = 35.0
_SILENCE_DBFS = -60.0
_MIN_VOICED_FRAMES = 20
#: Speaking pitch searched for, and how periodic a frame must be to count.
_PITCH_LOW_HZ = 70.0
_PITCH_HIGH_HZ = 400.0
_PERIODICITY = 0.3
#: Log-pitch scaled to sit on the cepstra's scale, so the variance floor
#: below treats both alike.
_PITCH_SCALE = 20.0
#: No feature's spread is taken as smaller than this fraction of the mean
#: magnitude: three enrolment utterances can agree by chance on one feature,
#: and a near-zero spread would then make that feature decide everything.
_SPREAD_FLOOR_FRACTION = 0.1
_EPSILON = 1e-10


class SpeakerEmbedder(Protocol):
    """Turns an utterance into a vector that two of the same voice share."""

    def embed(self, audio: np.ndarray, sample_rate_hz: int) -> np.ndarray | None:
        """A feature vector, or None when there is too little voice to judge."""


class SpectralEmbedder:
    """Liftered cepstral mean and spread, and pitch, over the voiced frames."""

    def embed(self, audio: np.ndarray, sample_rate_hz: int) -> np.ndarray | None:
        frames = _frames(np.asarray(audio, dtype=np.float64), sample_rate_hz)
        if frames is None:
            return None
        n_fft = 1 << (frames.shape[1] - 1).bit_length()
        spectrum = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), n=n_fft)) ** 2
        energy_db = 10.0 * np.log10(spectrum.sum(axis=1) + _EPSILON)
        level_dbfs = 10.0 * np.log10(np.mean(frames * frames, axis=1) + _EPSILON)
        voiced = (energy_db >= energy_db.max() - _VOICED_WITHIN_DB) & (level_dbfs >= _SILENCE_DBFS)
        if int(voiced.sum()) < _MIN_VOICED_FRAMES:
            return None
        bands = np.log(spectrum[voiced] @ _mel_filters(n_fft, sample_rate_hz).T + _EPSILON)
        cepstra = (bands @ _dct_matrix(_MEL_BANDS, _CEPSTRA).T)[:, 1:]
        order = np.arange(1, _CEPSTRA)
        cepstra = cepstra * (1.0 + (_LIFTER / 2.0) * np.sin(np.pi * order / _LIFTER))
        pitch = _log_pitch(frames[voiced], sample_rate_hz)
        return np.concatenate([cepstra.mean(axis=0), cepstra.std(axis=0), pitch])


def _log_pitch(frames: np.ndarray, sample_rate_hz: int) -> np.ndarray:
    """Median and interquartile range of log pitch, over the periodic frames."""
    shortest = int(sample_rate_hz / _PITCH_HIGH_HZ)
    longest = min(int(sample_rate_hz / _PITCH_LOW_HZ), frames.shape[1] - 1)
    pitches = []
    for frame in frames:
        centred = frame - frame.mean()
        correlation = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
        if correlation[0] <= 0.0:
            continue
        lag = shortest + int(np.argmax(correlation[shortest:longest]))
        if correlation[lag] / correlation[0] > _PERIODICITY:
            pitches.append(sample_rate_hz / lag)
    if not pitches:
        return np.zeros(2)
    log_pitch = np.log(np.asarray(pitches))
    upper, lower = np.percentile(log_pitch, [75, 25])
    return np.array([np.median(log_pitch), upper - lower]) * _PITCH_SCALE


def _frames(audio: np.ndarray, sample_rate_hz: int) -> np.ndarray | None:
    frame = int(round(_FRAME_S * sample_rate_hz))
    hop = int(round(_HOP_S * sample_rate_hz))
    if audio.ndim != 1 or len(audio) < frame:
        return None
    count = 1 + (len(audio) - frame) // hop
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    return audio[index]


def _mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filters(n_fft: int, sample_rate_hz: int) -> np.ndarray:
    high = min(_MEL_HIGH_HZ, sample_rate_hz / 2.0)
    edges = _hz(np.linspace(_mel(_MEL_LOW_HZ), _mel(high), _MEL_BANDS + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate_hz)
    filters = np.zeros((_MEL_BANDS, len(bins)))
    for band in range(_MEL_BANDS):
        low, centre, top = edges[band], edges[band + 1], edges[band + 2]
        rising = (bins - low) / (centre - low)
        falling = (top - bins) / (top - centre)
        filters[band] = np.clip(np.minimum(rising, falling), 0.0, None)
    return filters


def _dct_matrix(inputs: int, outputs: int) -> np.ndarray:
    """Orthonormal DCT-II, the step from log-mel energies to cepstra."""
    n = np.arange(inputs)
    k = np.arange(outputs)[:, None]
    matrix = np.cos(math.pi * k * (2 * n + 1) / (2 * inputs)) * math.sqrt(2.0 / inputs)
    matrix[0] /= math.sqrt(2.0)
    return matrix


@dataclass(frozen=True)
class Identification:
    """Whose voice it was, if anyone's, and how close."""

    profile_id: str  #: empty for a guest
    score: float  #: to the closest matchable voice; 0.0 when none was judged
    judged: bool  #: False when there was too little voice, or nobody to match

    @property
    def recognised(self) -> bool:
        return bool(self.profile_id)


GUEST = Identification(profile_id="", score=0.0, judged=False)


class VoiceprintError(ValueError):
    """An enrolment that cannot be made, and why."""


@dataclass(frozen=True)
class Voiceprint:
    """Running statistics of one voice's feature vectors (Welford's method)."""

    utterances: int
    mean: np.ndarray
    squares: np.ndarray  #: sum of squared deviations from the mean

    def added(self, vector: np.ndarray) -> Voiceprint:
        count = self.utterances + 1
        delta = vector - self.mean
        mean = self.mean + delta / count
        return Voiceprint(count, mean, self.squares + delta * (vector - mean))

    @property
    def spread(self) -> np.ndarray:
        spread = np.sqrt(self.squares / max(self.utterances, 1))
        floor = _SPREAD_FLOOR_FRACTION * float(np.abs(self.mean).mean()) + _EPSILON
        return np.maximum(spread, floor)

    def score(self, vector: np.ndarray) -> float:
        """exp(-mean(z^2)/2): 1 at this voice's mean, falling with distance."""
        z = (vector - self.mean) / self.spread
        return float(np.exp(-float(np.mean(z * z)) / 2.0))


class SpeakerIdentifier:
    """Enrolled voiceprints, and which one an utterance is closest to."""

    def __init__(self, config: SpeechConfig, embedder: SpeakerEmbedder | None = None) -> None:
        self._config = config
        self._embedder = embedder or SpectralEmbedder()
        self._path = Path(config.voiceprints_path)
        self._voiceprints: dict[str, Voiceprint] = {}
        self._loaded_mtime: float | None = None

    @property
    def enrolled(self) -> dict[str, int]:
        """Every voiceprint, and how many utterances it was enrolled from."""
        self._reload_if_changed()
        return {profile_id: print_.utterances for profile_id, print_ in sorted(self._voiceprints.items())}

    def matchable(self, profile_id: str) -> bool:
        """Whether a voice has enough enrolment to be recognised."""
        self._reload_if_changed()
        print_ = self._voiceprints.get(profile_id)
        return print_ is not None and print_.utterances >= self._config.speaker_min_enrolment_utterances

    def identify(self, audio: np.ndarray) -> Identification:
        """The enrolled voice this utterance is closest to, or a guest.

        A guest whenever there is any doubt: too short, too little voice,
        nobody matchable, or nobody close enough. Being a guest costs
        personalisation and nothing else.
        """
        if len(audio) < self._config.speaker_min_audio_s * self._config.sample_rate_hz:
            return GUEST
        self._reload_if_changed()
        candidates = [profile_id for profile_id in self._voiceprints if self.matchable(profile_id)]
        if not candidates:
            return GUEST
        vector = self._embedder.embed(audio, self._config.sample_rate_hz)
        if vector is None:
            return GUEST
        scores = {profile_id: self._voiceprints[profile_id].score(vector) for profile_id in candidates}
        best_id = max(scores, key=scores.get)
        best = scores[best_id]
        if best < self._config.speaker_match_threshold:
            LOGGER.info("voice not recognised (closest %s at %.3f)", best_id, best)
            return Identification(profile_id="", score=best, judged=True)
        LOGGER.info("voice recognised as %s (%.3f)", best_id, best)
        return Identification(profile_id=best_id, score=best, judged=True)

    def scores(self, audio: np.ndarray) -> dict[str, float]:
        """How close an utterance is to every voice with any enrolment."""
        self._reload_if_changed()
        vector = self._embedder.embed(audio, self._config.sample_rate_hz)
        if vector is None:
            return {}
        return {
            profile_id: print_.score(vector)
            for profile_id, print_ in self._voiceprints.items()
            if print_.utterances >= 2
        }

    def enroll(self, profile_id: str, audio: np.ndarray) -> int:
        """Add one utterance to a voiceprint. Returns the utterances it now holds.

        :raises VoiceprintError: for an unsafe id or an utterance with too
            little voice in it.
        """
        if not re.match(PROFILE_ID_PATTERN, profile_id):
            raise VoiceprintError(f"{profile_id!r} is not a valid profile id")
        if len(audio) < self._config.speaker_min_audio_s * self._config.sample_rate_hz:
            raise VoiceprintError(
                f"at least {self._config.speaker_min_audio_s:g} s of speech is needed"
            )
        vector = self._embedder.embed(audio, self._config.sample_rate_hz)
        if vector is None:
            raise VoiceprintError("there was too little voice in that recording")
        self._reload_if_changed()
        empty = Voiceprint(0, np.zeros_like(vector), np.zeros_like(vector))
        updated = self._voiceprints.get(profile_id, empty).added(vector)
        self._voiceprints[profile_id] = updated
        self._save()
        return updated.utterances

    def remove(self, profile_id: str) -> bool:
        """Delete a voiceprint. Returns whether there was one."""
        self._reload_if_changed()
        if self._voiceprints.pop(profile_id, None) is None:
            return False
        self._save()
        return True

    def _reload_if_changed(self) -> None:
        """Pick up an enrolment made by another process, on the next utterance."""
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            self._voiceprints, self._loaded_mtime = {}, None
            return
        if mtime == self._loaded_mtime:
            return
        try:
            raw = json.loads(self._path.read_text(encoding=_ENCODING))
            self._voiceprints = {
                profile_id: Voiceprint(
                    int(entry["utterances"]),
                    np.asarray(entry["mean"], dtype=np.float64),
                    np.asarray(entry["squares"], dtype=np.float64),
                )
                for profile_id, entry in raw["voiceprints"].items()
            }
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Recognition is a nicety; an unreadable file makes everyone a
            # guest rather than taking the speech pipeline down.
            LOGGER.warning("voiceprints at %s unreadable; everyone is a guest: %s", self._path, exc)
            self._voiceprints = {}
        self._loaded_mtime = mtime

    def _save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "note": "Feature statistics only, never audio (FR-51). Recognition grants nothing (FR-52).",
            "voiceprints": {
                profile_id: {
                    "utterances": print_.utterances,
                    "mean": [round(float(value), 6) for value in print_.mean],
                    "squares": [round(float(value), 6) for value in print_.squares],
                }
                for profile_id, print_ in sorted(self._voiceprints.items())
            },
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + _TEMPORARY_SUFFIX)
        temporary.write_text(json.dumps(payload, indent=2), encoding=_ENCODING)
        os.replace(temporary, self._path)
        self._loaded_mtime = self._path.stat().st_mtime
