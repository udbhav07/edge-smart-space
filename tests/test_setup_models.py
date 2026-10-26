"""Unit tests for the one-off model fetch.

Nothing is downloaded: every fetcher is replaced. What is checked is that
the Whisper model fetched is the one the transcriber will later load
offline, since a mismatch would only surface on the node, without internet.
"""

from pathlib import Path

import pytest

import setup_models


@pytest.fixture(name="fetched")
def _fetched(monkeypatch) -> list[str]:
    fetched: list[str] = []
    monkeypatch.setattr(
        setup_models, "fetch_wake_word_models", lambda: fetched.append("wake")
    )
    monkeypatch.setattr(
        setup_models, "fetch_whisper_model", lambda model: fetched.append(model)
    )
    monkeypatch.setattr(
        setup_models, "fetch_piper_voice", lambda path: fetched.append(path)
    )
    return fetched


def test_the_wake_word_models_are_fetched(fetched):
    setup_models.main([])
    assert "wake" in fetched


def test_the_configured_whisper_model_is_fetched(fetched):
    from pathlib import Path

    from src.common.config import load_config

    setup_models.main([])
    assert load_config(Path("config/default.yaml")).speech.asr_model in fetched


def test_a_successful_fetch_exits_zero(fetched):
    assert setup_models.main([]) == 0


def test_the_configured_voice_is_fetched(fetched):
    from pathlib import Path

    from src.common.config import load_config

    setup_models.main([])
    assert load_config(Path("config/default.yaml")).speech.voice_model_path in fetched


def test_no_voice_is_fetched_when_speaking_is_off(fetched, tmp_path):
    from src.common.config import load_config

    config = Path("config/default.yaml").read_text(encoding="utf-8")
    overlay = tmp_path / "quiet.yaml"
    overlay.write_text(config.replace("voice_engine: piper", "voice_engine: none"), encoding="utf-8")
    setup_models.main(["--config", str(overlay)])
    assert load_config(Path("config/default.yaml")).speech.voice_model_path not in fetched
