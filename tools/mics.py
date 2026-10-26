"""List the microphones, and which one speech will use.

    python -m tools.mics

For each input device: whether it opens at the pipeline's rate, and how loud
two seconds of it are, so the one you speak into is easy to pick out. Put
part of its name in ``speech.device_name``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.common.config import ConfigError, load_config
from src.speech.audio_capture import MicrophoneError, choose_device, input_devices

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
_LISTEN_S = 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.mics", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config).speech
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    try:
        import pyaudio
    except ImportError as exc:
        print(f"listing microphones needs the speech extra: {exc}")
        return 2

    audio = pyaudio.PyAudio()
    try:
        devices = input_devices(audio)
        try:
            chosen = choose_device(devices, config)
        except MicrophoneError as exc:
            chosen = None
            print(f"warning: {exc}\n")
        reads = int(_LISTEN_S * config.sample_rate_hz / config.chunk_samples)
        seen = set()
        print(f"{'index':>5}  {'level':>6}  name")
        for index, host, name in devices:
            if name in seen:
                continue  # the same device under another host API
            try:
                stream = audio.open(
                    format=pyaudio.paInt16, channels=config.channels, rate=config.sample_rate_hz,
                    input=True, input_device_index=index, frames_per_buffer=config.chunk_samples,
                )
            except OSError:
                continue
            samples = np.frombuffer(
                b"".join(stream.read(config.chunk_samples, exception_on_overflow=False) for _ in range(reads)),
                dtype=np.int16,
            ).astype(np.float64)
            stream.close()
            seen.add(name)
            marker = "  <- speech uses this" if index == chosen else ""
            print(f"{index:>5}  {np.sqrt(np.mean(samples ** 2)):6.0f}  {name} ({host}){marker}")
        if chosen is None and config.device_index is None and not config.device_name:
            print("\nspeech uses the system default; set speech.device_name to pin one")
    finally:
        audio.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
