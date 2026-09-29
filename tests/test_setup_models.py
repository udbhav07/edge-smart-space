"""Unit tests for the one-off model fetch.

Nothing is downloaded: both fetchers are replaced. What is checked is that
the Whisper model fetched is the one the transcriber will later load
offline, since a mismatch would only surface on the node, without internet.
"""

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
