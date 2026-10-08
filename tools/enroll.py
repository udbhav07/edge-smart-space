"""Enrol a person's voice, so the room knows who is speaking (FR-52).

    python -m tools.enroll ravi --name Ravi one.wav two.wav three.wav
    python -m tools.enroll ravi --name Ravi --record 3      # three from the microphone

Enrolment adds utterances to the person's voiceprint and, the first time,
starts their long-term memory (``state/profiles/ravi.md``). Recordings are
16 kHz mono 16-bit WAV, a few seconds of ordinary speech each; at least
``speech.speaker_min_enrolment_utterances`` are needed before the voice can
be recognised. Only feature statistics are kept, never the audio (FR-51).

It reports how close the new voice is to everyone already enrolled. Two
people who sound alike to this recogniser show up here, at enrolment, rather
than as a reply addressed to the wrong name later. Recognition personalises
replies and grants nothing (FR-52), so a confusable household is a nuisance,
not a hazard -- but it is better known than discovered.
"""

from __future__ import annotations

import argparse
import logging
import wave
from pathlib import Path

import numpy as np

from src.assistance.__main__ import build_store
from src.assistance.memory import ProfileError
from src.common.clock import RealClock
from src.common.config import Config, ConfigError, load_config
from src.speech.speaker_profile import SpeakerIdentifier, VoiceprintError

LOGGER = logging.getLogger("tools.enroll")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

_SAMPLE_WIDTH_BYTES = 2
_INT16_FULL_SCALE = 32768.0


class RecordingError(ValueError):
    """A recording that cannot be enrolled from, and why."""


def read_wav(path: Path, sample_rate_hz: int) -> np.ndarray:
    """A 16-bit mono WAV at the pipeline's rate, as float32 in [-1, 1].

    :raises RecordingError: for any other format. Resampling is not done
        here: a voiceprint enrolled at one rate and matched at another would
        compare different things.
    """
    try:
        with wave.open(str(path), "rb") as recording:
            if recording.getnchannels() != 1:
                raise RecordingError(f"{path.name}: mono only, got {recording.getnchannels()} channels")
            if recording.getsampwidth() != _SAMPLE_WIDTH_BYTES:
                raise RecordingError(f"{path.name}: 16-bit samples only")
            if recording.getframerate() != sample_rate_hz:
                raise RecordingError(
                    f"{path.name}: recorded at {recording.getframerate()} Hz; "
                    f"{sample_rate_hz} Hz is needed"
                )
            raw = recording.readframes(recording.getnframes())
    except (OSError, wave.Error, EOFError) as exc:
        raise RecordingError(f"{path.name}: not a readable WAV file ({exc})") from exc
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE


def record(config: Config, seconds: float) -> np.ndarray:
    """One utterance from the microphone, using the speech pipeline's capture.

    :raises RecordingError: when the speech extra is not installed.
    """
    try:
        from src.speech.audio_capture import AudioCapture
        from src.speech.asr import to_float_audio
    except ImportError as exc:
        raise RecordingError(f"recording needs the speech extra: {exc}") from exc
    clock = RealClock()
    capture = AudioCapture(config.speech, clock)
    frames = []
    needed = int(seconds * config.speech.sample_rate_hz / config.speech.chunk_samples)
    capture.start()
    try:
        while len(frames) < needed:
            frames.append(capture.frames.get(timeout=2.0))
    finally:
        capture.stop()
    return to_float_audio(frames)


def enroll(
    config: Config, profile_id: str, display_name: str, utterances: list[np.ndarray]
) -> list[str]:
    """Add utterances to a voiceprint, starting the profile if it is new.

    :returns: lines to report.
    :raises ProfileError, VoiceprintError: when the enrolment cannot be made.
    """
    identifier = SpeakerIdentifier(config.speech)
    store = build_store(config, RealClock())
    lines = []
    if not store.exists(profile_id):
        store.create(profile_id, display_name)
        lines.append(f"started the profile {profile_id} ({display_name})")

    others = {name for name in identifier.enrolled if name != profile_id}
    closest: dict[str, float] = {}
    for audio in utterances:
        for name, score in identifier.scores(audio).items():
            if name in others:
                closest[name] = max(closest.get(name, 0.0), score)
        count = identifier.enroll(profile_id, audio)
    lines.append(f"voiceprint for {profile_id} now holds {count} utterance(s)")

    needed = config.speech.speaker_min_enrolment_utterances
    if count < needed:
        lines.append(f"{needed - count} more before the voice can be recognised")
    else:
        lines.append("the voice can be recognised")

    threshold = config.speech.speaker_match_threshold
    for name, score in sorted(closest.items(), key=lambda item: -item[1]):
        warning = "  <- would be taken for them" if score >= threshold else ""
        lines.append(f"  closeness to {name}: {score:.3f} (threshold {threshold:g}){warning}")
    lines.append("recognition personalises replies only; it unlocks nothing (FR-52)")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.enroll", description=__doc__.splitlines()[0])
    parser.add_argument("profile_id", help="Short lower-case id, such as ravi")
    parser.add_argument("recordings", nargs="*", type=Path, help="16 kHz mono 16-bit WAV files")
    parser.add_argument("--name", default="", help="How the assistant addresses them")
    parser.add_argument("--record", type=int, default=0, metavar="N", help="Record N utterances")
    parser.add_argument("--seconds", type=float, default=4.0, help="Length of each recording")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    # Intermixed: "ravi --name Ravi a.wav b.wav" must work as written.
    arguments = parser.parse_intermixed_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    try:
        config = load_config(arguments.config)
        utterances = [read_wav(path, config.speech.sample_rate_hz) for path in arguments.recordings]
        for index in range(arguments.record):
            input(f"press Enter, then speak for {arguments.seconds:g} s ({index + 1}/{arguments.record})")
            utterances.append(record(config, arguments.seconds))
        if not utterances:
            print("nothing to enrol from: give WAV files, or --record N")
            return 2
        for line in enroll(config, arguments.profile_id, arguments.name or arguments.profile_id, utterances):
            print(line)
    except (ConfigError, RecordingError, ProfileError, VoiceprintError) as exc:
        print(f"cannot enrol: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
