"""Unit tests for a terminal that never crashes on what was typed (U-10)."""

import io
import sys

from tools.console import safe_console


def test_text_the_console_cannot_encode_is_replaced_not_fatal(monkeypatch):
    narrow = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", narrow)
    safe_console()
    print("\u0939\u0935\u093e \u0920\u0902\u0921\u0940 \u0915\u0930\u094b")
    narrow.flush()
    assert narrow.buffer.getvalue().startswith(b"?")
