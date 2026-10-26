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
import asyncio
import datetime as dt
import logging
import threading
from pathlib import Path

from src.common.clock import RealClock
from src.common.config import ConfigError, load_config
from src.common.mqtt_client import Blackboard, build_transport
from src.console.actions import ConsoleActions
from src.console.google_signin import GoogleSignIn
from src.console.remote_voice import RemoteVoice
from src.console.server import STATIC_DIRECTORY, Hub, Link, create_app
from src.console.state import ConsoleState
from src.console.tls import ensure_certificate, local_names

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
        config.console, clock, inbox_path=Path(config.console.inbox_path), utc_offset_h=config.site.utc_offset_h,
        bounds_c=(config.validator.setpoint_bounds_c.low, config.validator.setpoint_bounds_c.high),
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
    remote_voice = RemoteVoice(config.speech, config.console.talk_max_seconds)
    app = create_app(
        state, actions, hub, link, lock,
        signin=GoogleSignIn(config.google), remote_voice=remote_voice, talk_max_bytes=config.console.talk_max_bytes,
    )

    host = arguments.host or config.console.host
    port = arguments.port or config.console.port
    if not (STATIC_DIRECTORY / "index.html").is_file():
        LOGGER.warning("the frontend is not built; serving a placeholder page (see console/web)")
    servers = [uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))]
    if config.console.tls_port:
        # The same console over HTTPS, so a phone may use its microphone (FR-94).
        names, addresses = local_names()
        now = dt.datetime.fromtimestamp(clock.now(), tz=dt.timezone.utc)
        cert, key = ensure_certificate(Path(config.console.tls_directory), names, addresses, now)
        servers.append(
            uvicorn.Server(
                uvicorn.Config(app, host=host, port=config.console.tls_port, log_level="warning", ssl_certfile=str(cert), ssl_keyfile=str(key))
            )
        )
    blackboard.start()
    LOGGER.info("console on http://%s:%d -- it has no authority; the room runs without it", host, port)
    if config.console.tls_port:
        LOGGER.info("and on https://%s:%d, for talking from a phone (accept its certificate once)", host, config.console.tls_port)
    try:
        asyncio.run(_serve(servers))
    except KeyboardInterrupt:
        LOGGER.info("stopping")
    finally:
        blackboard.stop()
    return 0


async def _serve(servers) -> None:
    """Every server on one event loop, so they share the push stream."""
    await asyncio.gather(*(server.serve() for server in servers))


if __name__ == "__main__":
    raise SystemExit(main())
