"""Shared pytest configuration.

The voice pipeline depends on torch, onnxruntime and PortAudio. Those are an
optional install (the ``speech`` extra), and torch has no wheel for Python
3.14 at all, so on a core-only machine those modules cannot even be imported.

Without this file an unimportable test module is a *collection error*, which
aborts the entire run — the core tests never execute, and the failure looks
identical whether a dependency is merely absent or the code is genuinely
broken. Ignoring the modules whose dependencies are missing keeps the two
cases distinguishable: a missing extra is silence, a real failure is a
failure.

CI installs the speech extra, so nothing is skipped there.
"""

from __future__ import annotations

import importlib.util

#: Test module -> the import that module cannot survive without.
_OPTIONAL_DEPENDENCIES = {
    "test_agent_core.py": "torch",
    "test_config.py": "pyaudio",
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
