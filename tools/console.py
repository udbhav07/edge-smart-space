"""Terminal output that never crashes on what a person typed (U-10).

A Windows console or a pipe may use an encoding that cannot represent
Hindi, emoji or other text an occupant can perfectly well say; printing it
raised UnicodeEncodeError and took the tool down before the system saw a
word. Unencodable characters are replaced instead.
"""

from __future__ import annotations

import sys


def safe_console() -> None:
    """Make stdout and stderr replace what they cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
