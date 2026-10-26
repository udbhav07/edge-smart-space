"""``python -m src.devices.music`` -- the music player, as a service (FR-87)."""

from __future__ import annotations

from src.devices.music import build
from src.devices.runner import run


def main(argv: list[str] | None = None) -> int:
    return run(argv, "Play music from the occupant's library.", "device-music", build)


if __name__ == "__main__":
    raise SystemExit(main())
