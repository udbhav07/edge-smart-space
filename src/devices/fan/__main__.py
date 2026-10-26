"""``python -m src.devices.fan`` -- the simulated ceiling fan, as a service (FR-87)."""

from __future__ import annotations

from src.devices.fan import build
from src.devices.runner import run


def main(argv: list[str] | None = None) -> int:
    return run(argv, "Run the simulated fan.", "device-fan", build)


if __name__ == "__main__":
    raise SystemExit(main())
