"""How well each wake word detector hears "Hey Jarvis" over music (FR-98).

No published benchmark covers wake words said over music, or said in an
Indian accent, so the choice of detector is measured here, in this room's
terms:

- **heard**: clips of the wake phrase -- synthesised, or your own recordings
  -- mixed into real songs at several loudnesses (SNR, in dB: speech over
  music; 0 dB is as loud as the music), and how many each detector catches;
- **false wakes**: the same songs alone, and how often each wakes on them,
  per hour.

Compare detectors by what they catch at the false-wake rate you will put up
with, not at equal thresholds: their scores mean different things.

    python -m tools.wake_eval --youtube "hotel california eagles" --youtube "tum hi ho arijit singh"
    python -m tools.wake_eval --music song.mp3 --voices recordings/   # your own "Hey Jarvis" WAVs

Record your own with ``python -m tools.record`` (16 kHz mono WAV), a few
dozen takes per person, at a normal distance from the microphone.
"""

from __future__ import annotations

import argparse
import io
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.config import SpeechConfig, load_config

LOGGER = logging.getLogger("wake_eval")

RATE = 16000
#: Where the phrase starts in each mixed clip, and how long a clip is, in s.
_LEAD_S, _CLIP_S = 1.5, 5.0
DEFAULT_PHRASES = (
    "Hey Jarvis.",
    "Hey Jarvis, play some music.",
    "Hey Jarvis, stop.",
    "Hey Jarvis, what's on today?",
    "Hey Jarvis, turn on the fan.",
)
#: Each phrase is spoken at these speeds, so the set is not one rhythm.
_LENGTH_SCALES = (0.85, 1.0, 1.2)


@dataclass(frozen=True)
class Engine:
    label: str
    config: SpeechConfig


def engines(config: SpeechConfig) -> list[Engine]:
    """Each detector at the settings worth comparing."""
    with_ = lambda **changes: config.model_copy(update=changes)  # noqa: E731
    return [
        *(Engine(f"openwakeword @{t}", with_(wake_engine="openwakeword", wake_word_threshold=t)) for t in (0.5, 0.3, 0.1)),
        *(Engine(f"sherpa @{t}", with_(wake_engine="sherpa", kws_threshold=t)) for t in (0.25, 0.1, 0.05)),
        Engine("both (oww 0.3, sherpa 0.1)", with_(wake_engine="both", wake_word_threshold=0.3, kws_threshold=0.1)),
        Engine("either (oww 0.5, sherpa 0.1)", with_(wake_engine="either", wake_word_threshold=0.5, kws_threshold=0.1)),
        Engine("either (oww 0.3, sherpa 0.1)", with_(wake_engine="either", wake_word_threshold=0.3, kws_threshold=0.1)),
    ]


def synthesised(voices: list[Path], phrases: tuple[str, ...] = DEFAULT_PHRASES) -> list[np.ndarray]:
    """The phrases in each Piper voice at each speed, as 16 kHz float audio."""
    from piper import PiperVoice
    from piper.config import SynthesisConfig

    clips = []
    for path in voices:
        voice = PiperVoice.load(str(path))
        for phrase in phrases:
            for scale in _LENGTH_SCALES:
                chunks = list(voice.synthesize(phrase, syn_config=SynthesisConfig(length_scale=scale)))
                clips.append(_resampled(np.concatenate([c.audio_float_array for c in chunks]), chunks[0].sample_rate))
    return clips


def recorded(folder: Path) -> list[np.ndarray]:
    """Your own takes: every WAV in a folder."""
    return [decoded(path.read_bytes()) for path in sorted(folder.glob("*.wav"))]


def decoded(data: bytes, limit_s: float | None = None) -> np.ndarray:
    """Any audio file as 16 kHz mono float, via PyAV."""
    import av

    container = av.open(io.BytesIO(data)) if isinstance(data, bytes) else av.open(data)
    return _decode(container, limit_s)


def _decode(container, limit_s: float | None) -> np.ndarray:
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=RATE)
    parts, total = [], 0
    for frame in container.decode(container.streams.audio[0]):
        for out in resampler.resample(frame):
            samples = out.to_ndarray().reshape(-1)
            parts.append(samples)
            total += samples.size
        if limit_s is not None and total >= limit_s * RATE:
            break
    container.close()
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return audio[: int(limit_s * RATE)] if limit_s else audio


def from_youtube(query: str, limit_s: float) -> np.ndarray:
    """A song's first minutes, streamed and decoded; nothing is kept."""
    import av
    import yt_dlp

    options = {"quiet": True, "no_warnings": True, "format": "bestaudio/best", "noplaylist": True}
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(f"ytsearch1:{query}", download=False)["entries"][0]
    headers = "".join(f"{k}: {v}\r\n" for k, v in (info.get("http_headers") or {}).items())
    LOGGER.info("music: %s", info.get("title"))
    return _decode(av.open(info["url"], options={"headers": headers}, timeout=15), limit_s)


def mixed(speech: np.ndarray, music: np.ndarray, snr_db: float, offset: int) -> np.ndarray:
    """The phrase over a stretch of music, at a given speech-to-music ratio."""
    length = int(_CLIP_S * RATE)
    bed = np.resize(music[offset: offset + length], length).astype(np.float32)
    start = int(_LEAD_S * RATE)
    speech = speech[: length - start]
    speech_rms = float(np.sqrt(np.mean(speech**2))) or 1e-6
    bed_rms = float(np.sqrt(np.mean(bed[start: start + speech.size] ** 2))) or 1e-6
    bed = bed * (speech_rms / bed_rms) / (10 ** (snr_db / 20))
    out = bed.copy()
    out[start: start + speech.size] += speech
    peak = float(np.max(np.abs(out))) or 1.0
    return out / max(peak, 1.0) * 0.9 if peak > 1.0 else out


def frames_of(audio: np.ndarray, chunk: int) -> list[np.ndarray]:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    return [pcm[i: i + chunk] for i in range(0, pcm.size - chunk + 1, chunk)]


def heard_in(detector, audio: np.ndarray, chunk: int) -> bool:
    detector.reset()
    return any(detector.heard(frame) for frame in frames_of(audio, chunk))


def false_wakes(detector, audio: np.ndarray, chunk: int) -> int:
    detector.reset()
    count = 0
    for frame in frames_of(audio, chunk):
        if detector.heard(frame):
            count += 1
            detector.reset()
    return count


def _resampled(audio: np.ndarray, rate: int) -> np.ndarray:
    if rate == RATE:
        return audio.astype(np.float32)
    n = int(audio.size * RATE / rate)
    return np.interp(np.linspace(0, audio.size, n, endpoint=False), np.arange(audio.size), audio).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.wake_eval", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--voices", type=Path, help="A folder of your own 'Hey Jarvis' WAVs; synthesised if not given")
    parser.add_argument("--piper", type=Path, action="append", default=[], help="Piper voices to synthesise with")
    parser.add_argument("--music", type=Path, action="append", default=[], help="Audio files to mix the phrase into")
    parser.add_argument("--youtube", action="append", default=[], help="Songs to stream from YouTube instead")
    parser.add_argument("--music-s", type=float, default=180.0, help="How much of each song to use")
    parser.add_argument("--snr", type=float, action="append", default=[], help="Speech-to-music ratios in dB")
    parser.add_argument("--only", action="append", default=[], help="Run only engines whose label starts with this")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config(arguments.config).speech
    voices = arguments.piper or [Path(config.voice_model_path)]
    positives = recorded(arguments.voices) if arguments.voices else synthesised(voices)
    songs = [decoded(path.read_bytes(), arguments.music_s) for path in arguments.music]
    songs += [from_youtube(query, arguments.music_s) for query in arguments.youtube]
    if not songs:
        parser.error("give some music: --music FILE or --youtube QUERY")
    snrs = arguments.snr or [20.0, 10.0, 5.0, 0.0, -5.0]
    music_s = sum(song.size for song in songs) / RATE
    LOGGER.info("%d clips of the phrase, %d songs (%.0f s of music)", len(positives), len(songs), music_s)

    from src.speech.wakeword import build_wake_detector

    rng = np.random.default_rng(7)
    print(f"\n{'engine':32s} " + " ".join(f"{snr:>+6.0f}dB" for snr in snrs) + "   false/h")
    for engine in engines(config):
        if arguments.only and not any(engine.label.startswith(prefix) for prefix in arguments.only):
            continue
        detector = build_wake_detector(engine.config)
        rates = []
        for snr in snrs:
            hits = 0
            for index, clip in enumerate(positives):
                song = songs[index % len(songs)]
                offset = int(rng.integers(0, max(1, song.size - int(_CLIP_S * RATE))))
                hits += heard_in(detector, mixed(clip, song, snr, offset), config.chunk_samples)
            rates.append(hits / len(positives))
        wrong = sum(false_wakes(detector, song, config.chunk_samples) for song in songs)
        print(f"{engine.label:32s} " + " ".join(f"{rate:>7.0%} " for rate in rates) + f"   {wrong / (music_s / 3600):6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
