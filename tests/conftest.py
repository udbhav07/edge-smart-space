"""Shared pytest configuration.

Every module in the tree imports without torch, PortAudio, openWakeWord or
an OpenAI client. Those packages are needed only to *build the defaults* --
to load a model or open a device -- so each is imported inside the function
that does that, and every collaborator is injectable.

The effect is that the parts most likely to break are tested on every
machine: the endpointing state machine, the FR-50 timeout arithmetic, the
wake-word threshold, the queue's drop-oldest policy, and the post-decode
validation in FR-44. None of them need a GPU, a microphone or a server.

Nothing is skipped, so this file deliberately declares no ignores. If a
module ever has to import a heavy dependency at module scope again, add it
here with the reason -- but prefer making it injectable instead.
"""

from __future__ import annotations

collect_ignore: list[str] = []
