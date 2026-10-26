"""Unit tests for the one-off model fetch.

Nothing is downloaded: every fetcher is replaced. What is checked is that
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


def test_without_the_voice_extra_the_voice_is_skipped(monkeypatch, tmp_path, capsys):
    """A machine without piper still finishes; it says how to add the voice."""
    import sys

    monkeypatch.setitem(sys.modules, "piper", None)
    setup_models.fetch_piper_voice(str(tmp_path / "voice.onnx"))
    assert ".[voice]" in capsys.readouterr().out
    assert not (tmp_path / "voice.onnx").exists()
