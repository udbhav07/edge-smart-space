"""A simulated desk lamp as an MCP server: what any third-party server looks like to the bridge.

    python examples/mcp_lamp/server.py        # speaks MCP over stdio

It is an ordinary MCP server built with the official SDK and knows nothing
about the room; the bridge (``src/devices/mcp_bridge``) mounts it as the
device ``lamp`` because ``config/default.yaml`` says so. ``factory_reset`` is
deliberately left out of that configuration, to show that a tool a server
offers is not a tool the room exposes.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from mcp.server.mcpserver import MCPServer

server = MCPServer("Desk lamp", instructions="A simulated desk lamp. It lights nothing.")

_lamp = {"on": False, "brightness": 100}


def _report(message: str) -> dict:
    return {"message": f"{message} (simulated)", "on": _lamp["on"], "brightness": _lamp["brightness"]}


@server.tool()
def turn_on() -> dict:
    """Turn the lamp on."""
    _lamp["on"] = True
    return _report(f"The lamp is on at {_lamp['brightness']}%.")


@server.tool()
def turn_off() -> dict:
    """Turn the lamp off."""
    _lamp["on"] = False
    return _report("The lamp is off.")


@server.tool()
def set_brightness(level: Annotated[int, Field(ge=0, le=100, description="Brightness in percent")]) -> dict:
    """Set how bright the lamp is."""
    _lamp["brightness"] = level
    _lamp["on"] = level > 0
    return _report(f"The lamp is at {level}%.")


@server.tool()
def get_status() -> dict:
    """Say whether the lamp is on and how bright."""
    return _report(f"The lamp is {'on at ' + str(_lamp['brightness']) + '%' if _lamp['on'] else 'off'}.")


@server.tool()
def factory_reset() -> dict:
    """Erase the lamp's settings. Not exposed by the room's configuration."""
    _lamp.update(on=False, brightness=100)
    return _report("The lamp was reset.")


if __name__ == "__main__":
    server.run()
