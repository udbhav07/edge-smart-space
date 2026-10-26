"""Run the occupant console: ``python -m src.console`` (FR-56).

Serves the page and a push stream on ``console.host:console.port`` -- this
node, or its address on the room's network, never anything beyond -- and
makes no outbound connection: the page's scripts and fonts are served from
here, built ahead of time.

Stopping it, or never starting it, changes nothing else: it is a blackboard
client like the recorder, with no authority of its own.
"""

from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

from src.common.clock import RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.console.actions import ConsoleActions
from src.console.google_signin import GoogleSignIn
from src.console.server import STATIC_DIRECTORY, Hub, Link, create_app
from src.console.state import ConsoleState

LOGGER = logging.getLogger("console")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")
CLIENT_ID = "console"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the occupant console (FR-56).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--host", default=None, help="Overrides console.host")
    parser.add_argument("--port", type=int, default=None, help="Overrides console.port")
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    import uvicorn

    clock = RealClock()
    state = ConsoleState(
        config.console, clock, inbox_path=Path(config.console.inbox_path), utc_offset_h=config.site.utc_offset_h
    )
    lock = threading.Lock()
    hub = Hub()

    holder: list = []
    transport = build_transport(config.mqtt, CLIENT_ID, holder)
    blackboard = Blackboard(config.mqtt, transport)
    link = Link(state, hub, lock, blackboard)
    link.attach(transport)
    holder.append(link)

    actions = ConsoleActions(config, clock, blackboard, state)
    app = create_app(state, actions, hub, link, lock, signin=GoogleSignIn(config.google))

    host = arguments.host or config.console.host
    port = arguments.port or config.console.port
    if not (STATIC_DIRECTORY / "index.html").is_file():
        LOGGER.warning("the frontend is not built; serving a placeholder page (see console/web)")
    blackboard.start()
    LOGGER.info("console on http://%s:%d -- it has no authority; the room runs without it", host, port)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    finally:
        blackboard.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
