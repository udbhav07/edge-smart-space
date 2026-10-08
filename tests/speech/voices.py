"""Synthetic voices for testing speaker recognition without a microphone.

A voice is glottal pulses at the speaker's pitch, shaped by the resonances
of a vocal tract whose length scales every formant, moving through vowels.
Two voices differ the way two people do -- pitch and tract length -- and two
utterances of one voice differ in which vowels, how long, and in noise. It
is a test of the mechanism, not a claim about accuracy on real people.
"""

import numpy as np

SAMPLE_RATE_HZ = 16000

#: Formant frequencies of a few vowels for an average adult tract, in Hz.
_VOWELS = {
    "a": (730.0, 1090.0, 2440.0),
    "i": (270.0, 2290.0, 3010.0),
    "u": (300.0, 870.0, 2240.0),
    "e": (530.0, 1840.0, 2480.0),
    "o": (570.0, 840.0, 2410.0),
}
_BANDWIDTHS_HZ = (90.0, 110.0, 170.0)


def _resonate(signal: np.ndarray, centre_hz: float, bandwidth_hz: float) -> np.ndarray:
    """A two-pole resonator: one formant."""
    radius = np.exp(-np.pi * bandwidth_hz / SAMPLE_RATE_HZ)
    theta = 2.0 * np.pi * centre_hz / SAMPLE_RATE_HZ
    a1, a2 = -2.0 * radius * np.cos(theta), radius * radius
    out = np.zeros_like(signal)
    y1 = y2 = 0.0
    for index, x in enumerate(signal):
        y = x - a1 * y1 - a2 * y2
        out[index] = y
        y2, y1 = y1, y
    return out


def utterance(pitch_hz: float, tract_scale: float, seed: int, seconds: float = 2.0) -> np.ndarray:
    """One utterance of the voice (pitch, tract scale), as float32 in [-1, 1]."""
    rng = np.random.default_rng(seed)
    vowels = list(_VOWELS)
    pieces = []
    remaining = int(seconds * SAMPLE_RATE_HZ)
    while remaining > 0:
        length = min(remaining, int(rng.uniform(0.15, 0.35) * SAMPLE_RATE_HZ))
        f0 = pitch_hz * rng.uniform(0.92, 1.08)
        period = SAMPLE_RATE_HZ / f0
        pulses = np.zeros(length)
        position = rng.uniform(0, period)
        while position < length:
            pulses[int(position)] = 1.0
            position += period * rng.uniform(0.98, 1.02)
        sound = pulses
        for (formant, bandwidth) in zip(_VOWELS[vowels[rng.integers(len(vowels))]], _BANDWIDTHS_HZ):
            sound = _resonate(sound, formant / tract_scale, bandwidth)
        pieces.append(sound * np.hanning(length))
        remaining -= length
    audio = np.concatenate(pieces)
    audio += rng.normal(0.0, 0.002 * np.abs(audio).max(), len(audio))
    return (0.5 * audio / np.abs(audio).max()).astype(np.float32)


#: Three household voices and a stranger.
RAVI = (115.0, 1.10)
ANU = (215.0, 0.88)
DEV = (150.0, 1.00)
STRANGER = (180.0, 0.95)
