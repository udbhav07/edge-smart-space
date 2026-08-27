"""Shared pytest configuration.

The voice pipeline depends on torch, onnxruntime, PortAudio and an OpenAI
client. Those are an optional install (the ``speech`` extra), and torch has
no wheel for Python 3.14 at all, so on a core-only machine those modules
cannot even be imported.

Without this file an unimportable test module is a *collection error*, which
aborts the entire run — the core tests never execute, and the failure looks
identical whether a dependency is merely absent or the code is genuinely
broken. Ignoring the modules whose dependencies are missing keeps the two
cases distinguishable: a missing extra is silence, a real failure is a
failure.

Note what is *not* listed here. The pipeline's state machine and timeout
arithmetic, and the validator's post-decode checks, are pure logic and are
tested on every machine. Only the modules that load a model or open a device
are optional.

CI installs the speech extra, so nothing is skipped there.
"""

from __future__ import annotations

import importlib.util

#: Test module -> the import that module cannot survive without.
_OPTIONAL_DEPENDENCIES = {
    "speech/test_asr.py": "torch",
    "speech/test_wakeword.py": "openwakeword",
    "speech/test_audio_capture.py": "pyaudio",
    "reasoning/test_single_shot.py": "openai",
}


def _is_missing(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is None
    except (ImportError, ValueError):
        return True


collect_ignore = [
    test_module
    for test_module, dependency in _OPTIONAL_DEPENDENCIES.items()
    if _is_missing(dependency)
]
