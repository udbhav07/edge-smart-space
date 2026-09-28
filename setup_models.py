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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the speech models.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args(argv)
    config = load_config(arguments.config)
    fetch_wake_word_models()
    fetch_whisper_model(config.speech.asr_model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
