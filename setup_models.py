"""Fetch every model the node needs, once, while it still has internet.

    python setup_models.py
    python setup_models.py --config path/to/config.yaml

The deployed node has no outbound connection (NFR-06), so nothing may be
downloaded at startup. The wake-word models, the keyword spotter, the Whisper
model and the Piper voice named in configuration are fetched here, into the local caches the services then
load from with network access refused.
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.request
from pathlib import Path

from src.common.config import load_config

DEFAULT_CONFIG_PATH = Path("config/default.yaml")


def fetch_wake_word_models() -> None:
    import openwakeword

    print("Downloading OpenWakeWord models...")
    openwakeword.utils.download_models()
    print("OpenWakeWord models are ready.")


def fetch_whisper_model(model: str) -> None:
    from faster_whisper import download_model

    print(f"Downloading Whisper {model}...")
    download_model(model)
    print(f"Whisper {model} is ready.")


def fetch_piper_voice(model_path: str) -> None:
    """The voice replies are spoken in (FR-79), into the configured folder."""
    from piper.download_voices import download_voice

    path = Path(model_path)
    print(f"Downloading the Piper voice {path.stem}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    download_voice(path.stem, path.parent)
    print(f"Piper voice {path.stem} is ready.")


#: Where sherpa-onnx publishes its keyword spotting models (FR-98).
_KWS_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"


def fetch_keyword_spotter(model_dir: str) -> None:
    """sherpa-onnx's keyword spotter, which listens for "hey jarvis" as words (FR-98)."""
    folder = Path(model_dir)
    if (folder / "tokens.txt").is_file():
        print(f"Keyword spotter {folder.name} is ready.")
        return
    print(f"Downloading the keyword spotter {folder.name}...")
    with urllib.request.urlopen(f"{_KWS_RELEASES}{folder.name}.tar.bz2", timeout=120) as response:
        archive = tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:bz2")
    folder.parent.mkdir(parents=True, exist_ok=True)
    archive.extractall(folder.parent, filter="data")
    print(f"Keyword spotter {folder.name} is ready.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the speech models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    fetch_wake_word_models()
    if config.speech.wake_engine != "openwakeword":
        fetch_keyword_spotter(config.speech.kws_model_dir)
    fetch_whisper_model(config.speech.asr_model)
    if config.speech.voice_engine == "piper":
        fetch_piper_voice(config.speech.voice_model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
