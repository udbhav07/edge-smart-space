"""``python -m src.devices.mcp_bridge`` -- every configured MCP server, as a device (FR-90)."""

from __future__ import annotations

from src.devices.mcp_bridge import build
from src.devices.runner import run


def main(argv: list[str] | None = None) -> int:
    return run(argv, "Mount the configured MCP servers as devices.", "device-mcp", build)


if __name__ == "__main__":
    raise SystemExit(main())
