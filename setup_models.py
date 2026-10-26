"""Fetch every model the node needs, once, while it still has internet.

    python setup_models.py
    python setup_models.py --config path/to/config.yaml

The deployed node has no outbound connection (NFR-06), so nothing may be
downloaded at startup. The wake-word models and the Whisper model named in
configuration are fetched here, into the local caches the services then
load from with network access refused.
"""

from __future__ import annotations

import argparse
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
    """The voice replies are spoken in (FR-79), into the configured folder.

    Without the voice extra this says so and moves on: the room then speaks
    with the system's voices, or answers on screen only, and the speech
    models fetched above are still what the node needs.
    """
    try:
        from piper.download_voices import download_voice
    except ImportError:
        print('Skipping the Piper voice: install it with pip install -e ".[voice]", then run this again.')
        return

    path = Path(model_path)
    print(f"Downloading the Piper voice {path.stem}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    download_voice(path.stem, path.parent)
    print(f"Piper voice {path.stem} is ready.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the speech models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    fetch_wake_word_models()
    fetch_whisper_model(config.speech.asr_model)
    if config.speech.voice_engine == "piper":
        fetch_piper_voice(config.speech.voice_model_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
