import sys
import types


openai_stub = types.ModuleType("openai")
setattr(openai_stub, "OpenAI", object)
sys.modules.setdefault("openai", openai_stub)

silero_vad_stub = types.ModuleType("silero_vad")
setattr(silero_vad_stub, "load_silero_vad", object)
setattr(silero_vad_stub, "VADIterator", object)
sys.modules.setdefault("silero_vad", silero_vad_stub)

from voiceAssistant import agent_core


class FakeWakeWordModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self):
        pass


class FakeVadIterator:
    def __init__(self, model, sampling_rate, min_silence_duration_ms):
        self.args = (model, sampling_rate, min_silence_duration_ms)

    def reset_states(self):
        pass


class FakeWhisperModel:
    calls = []

    def __init__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class FakeLlm:
    pass


def test_agent_loads_whisper_once_with_configured_settings(monkeypatch):
    FakeWhisperModel.calls = []
    monkeypatch.setattr(agent_core, "Model", FakeWakeWordModel)
    monkeypatch.setattr(agent_core, "load_silero_vad", lambda: object())
    monkeypatch.setattr(agent_core, "VADIterator", FakeVadIterator)
    monkeypatch.setattr(agent_core, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(agent_core, "SmartAgentLLM", FakeLlm)
    monkeypatch.setattr(
        "openwakeword.utils.download_models",
        lambda: (_ for _ in ()).throw(AssertionError("startup download")),
    )

    agent_core.SmartAgent(object())

    assert FakeWhisperModel.calls == [
        (("base.en",), {"device": "cuda", "compute_type": "float16"})
    ]