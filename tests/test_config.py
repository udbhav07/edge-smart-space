import importlib

from voiceAssistant import config


def reload_config(monkeypatch, **environment):
    for name in ("SMART_SPACE_STT_DEVICE", "SMART_SPACE_STT_COMPUTE_TYPE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    return importlib.reload(config)


def test_stt_defaults_to_cuda_float16(monkeypatch):
    loaded_config = reload_config(monkeypatch)

    assert loaded_config.STT_DEVICE == "cuda"
    assert loaded_config.STT_COMPUTE_TYPE == "float16"


def test_stt_settings_can_be_overridden(monkeypatch):
    loaded_config = reload_config(
        monkeypatch,
        SMART_SPACE_STT_DEVICE="cpu",
        SMART_SPACE_STT_COMPUTE_TYPE="int8",
    )

    assert loaded_config.STT_DEVICE == "cpu"
    assert loaded_config.STT_COMPUTE_TYPE == "int8"