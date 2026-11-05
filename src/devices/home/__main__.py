"""``python -m src.devices.home`` -- the simulated house's fans, lights and TV, as a service (FR-99)."""

from __future__ import annotations

from src.devices.home import build
from src.devices.runner import run


def main(argv: list[str] | None = None) -> int:
    return run(argv, "Run the simulated house's devices.", "device-home", build)


if __name__ == "__main__":
    raise SystemExit(main())
