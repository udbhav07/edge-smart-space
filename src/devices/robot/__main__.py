"""``python -m src.devices.robot`` -- the home robot, as a service (FR-100)."""

from __future__ import annotations

from src.devices.robot import build
from src.devices.runner import run


def main(argv: list[str] | None = None) -> int:
    return run(argv, "Run the home robot.", "device-robot", build)


if __name__ == "__main__":
    raise SystemExit(main())
