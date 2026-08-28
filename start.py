"""Start every service the system needs, in one command.

    python start.py                 # everything
    python start.py --only speech   # one service
    python start.py --skip speech   # everything else
    python start.py --check         # preflight only, start nothing

This is the development and demonstration launcher. On the Jetson the
deployment path is process supervision, one systemd unit per component with
``Restart=always`` (DESIGN.md section 9.3), because restart independence is a
requirement rather than a convenience: killing the reasoning process and
watching regulatory control continue is part of the demonstration. This
script deliberately mirrors that shape — separate processes, no shared
memory, each one killable on its own.

Two dependencies are checked but never started, because they are not ours to
own: the MQTT broker and the inference server. The broker is required, since
the blackboard is the only coupling between components (section 4.4). The
inference server is *not*: total reasoning unavailability must not stop
regulatory control (FR-47), so a missing one is a warning and the run
continues.

Written to be OS-agnostic. Processes are launched through ``sys.executable``
and stopped with terminate-then-kill, both of which behave the same on Linux,
macOS and Windows. Only the printed remediation hints differ by platform.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import platform
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.common.config import Config, ConfigError, load_config

LOGGER = logging.getLogger("start")

DEFAULT_CONFIG_PATH = Path("config/default.yaml")

#: How long a dependency has to accept a TCP connection before it counts as
#: absent. Local services answer in milliseconds; this is generous.
PROBE_TIMEOUT_S = 2.0

#: Grace period between asking a child to stop and killing it.
SHUTDOWN_GRACE_S = 5.0

#: Polling interval while supervising children.
_SUPERVISE_INTERVAL_S = 0.5

LINUX = "Linux"
WINDOWS = "Windows"
MACOS = "Darwin"


@dataclass(frozen=True)
class Service:
    """One process this script owns."""

    name: str
    module: str
    description: str

    def command(self, config_path: Path) -> list[str]:
        return [sys.executable, "-m", self.module, "--config", str(config_path)]


@dataclass(frozen=True)
class Dependency:
    """An external service this script checks but does not start."""

    name: str
    host: str
    port: int
    required: bool
    hints: dict[str, str] = field(default_factory=dict)

    def hint(self, system: str) -> str:
        return self.hints.get(system, self.hints.get(LINUX, ""))


SERVICES = (
    Service(
        name="simulator",
        module="sim.run_sim",
        description="Room plant, sensors and actuator (simulation mode)",
    ),
    Service(
        name="speech",
        module="src.speech",
        description="Wake word, transcription, and the Personal Context call",
    ),
)

SERVICE_NAMES = tuple(service.name for service in SERVICES)


def _broker_hints() -> dict[str, str]:
    return {
        LINUX: "sudo systemctl start mosquitto   (Ubuntu 22.04: sudo apt install mosquitto)",
        MACOS: "brew services start mosquitto",
        WINDOWS: "net start mosquitto   (or run mosquitto.exe -v)",
    }


def _inference_hints() -> dict[str, str]:
    return {
        LINUX: "ollama serve   (or llama-server --jinja -m <model.gguf> --port 11434)",
        MACOS: "ollama serve",
        WINDOWS: "ollama serve",
    }


def dependencies(config: Config) -> tuple[Dependency, ...]:
    """External services, resolved from configuration."""
    endpoint = urlparse(config.reasoning.base_url)
    return (
        Dependency(
            name="mqtt broker",
            host=config.mqtt.host,
            port=config.mqtt.port,
            required=True,
            hints=_broker_hints(),
        ),
        Dependency(
            name="inference server",
            host=endpoint.hostname or "localhost",
            port=endpoint.port or 80,
            required=False,
            hints=_inference_hints(),
        ),
    )


def is_reachable(host: str, port: int, timeout_s: float = PROBE_TIMEOUT_S) -> bool:
    """Whether something is accepting connections there."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except (OSError, ValueError):
        return False


def preflight(config: Config, system: str) -> bool:
    """Report on every dependency.

    :returns: True when everything *required* is present. An absent optional
        dependency is reported and tolerated: FR-47 requires the control loop
        to survive total reasoning unavailability, so refusing to start
        without it would contradict the design.
    """
    satisfied = True
    for dependency in dependencies(config):
        if is_reachable(dependency.host, dependency.port):
            LOGGER.info(
                "ok       %s at %s:%s", dependency.name, dependency.host, dependency.port
            )
            continue

        level = LOGGER.error if dependency.required else LOGGER.warning
        label = "MISSING " if dependency.required else "absent  "
        # ASCII only: a Windows console using cp1252 renders an em dash as a
        # replacement character, and a launcher's first job is legible output.
        level(
            "%s %s at %s:%s -> %s",
            label,
            dependency.name,
            dependency.host,
            dependency.port,
            dependency.hint(system),
        )
        if dependency.required:
            satisfied = False
        else:
            LOGGER.warning(
                "         continuing without it; regulatory control does not "
                "depend on reasoning (FR-47)"
            )
    return satisfied


def select(only: list[str] | None, skip: list[str] | None) -> tuple[Service, ...]:
    """Choose which services to run.

    :raises ValueError: if a name does not match a known service, rather than
        silently starting nothing.
    """
    for name in (only or []) + (skip or []):
        if name not in SERVICE_NAMES:
            raise ValueError(
                f"unknown service {name!r}; choose from {list(SERVICE_NAMES)}"
            )

    chosen = SERVICES if not only else tuple(s for s in SERVICES if s.name in only)
    if skip:
        chosen = tuple(service for service in chosen if service.name not in skip)
    return chosen


def _pump(name: str, stream) -> None:
    """Prefix a child's output so one terminal stays readable."""
    for line in iter(stream.readline, ""):
        print(f"[{name}] {line.rstrip()}", flush=True)
    with contextlib.suppress(Exception):
        stream.close()


def launch(service: Service, config_path: Path) -> subprocess.Popen:
    """Start one service with its output piped back here."""
    process = subprocess.Popen(
        service.command(config_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    threading.Thread(
        target=_pump, args=(service.name, process.stdout), daemon=True
    ).start()
    LOGGER.info("started  %s (pid %s)", service.name, process.pid)
    return process


def shutdown(running: dict[str, subprocess.Popen]) -> None:
    """Ask every child to stop, then insist.

    terminate() then kill() is used rather than POSIX signals so the same
    code path works on Windows.
    """
    for name, process in running.items():
        if process.poll() is None:
            LOGGER.info("stopping %s", name)
            process.terminate()

    deadline = time.monotonic() + SHUTDOWN_GRACE_S
    for name, process in running.items():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            LOGGER.warning("killing  %s, it did not stop in time", name)
            process.kill()


def supervise(running: dict[str, subprocess.Popen]) -> int:
    """Watch children until one exits or the operator interrupts.

    A child exiting is reported but does not tear the others down. That is
    the point of separate processes: the demonstration includes killing the
    reasoning process and showing regulatory control continue (section 9.3).
    """
    try:
        while running:
            for name, process in list(running.items()):
                code = process.poll()
                if code is not None:
                    LOGGER.warning("%s exited with code %s", name, code)
                    del running[name]
            time.sleep(_SUPERVISE_INTERVAL_S)
        LOGGER.info("all services have exited")
        return 0
    except KeyboardInterrupt:
        LOGGER.info("interrupted")
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the edge smart space services.",
        epilog="Services: " + ", ".join(f"{s.name} ({s.description})" for s in SERVICES),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--only", nargs="+", metavar="NAME", help="Run only these.")
    parser.add_argument("--skip", nargs="+", metavar="NAME", help="Run all but these.")
    parser.add_argument(
        "--check", action="store_true", help="Run the preflight checks and stop."
    )
    parser.add_argument(
        "--no-preflight", action="store_true", help="Start without checking anything."
    )
    arguments = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    system = platform.system()

    try:
        config = load_config(arguments.config)
    except ConfigError as exc:
        LOGGER.error("%s", exc)
        return 2

    try:
        services = select(arguments.only, arguments.skip)
    except ValueError as exc:
        LOGGER.error("%s", exc)
        return 2

    if not arguments.no_preflight:
        if not preflight(config, system) and not arguments.check:
            LOGGER.error("required dependencies are missing; not starting")
            return 1
    if arguments.check:
        return 0

    if not services:
        LOGGER.error("no services selected")
        return 2

    running: dict[str, subprocess.Popen] = {}
    try:
        for service in services:
            running[service.name] = launch(service, arguments.config)
        LOGGER.info("watch the blackboard with: mosquitto_sub -t 'space/#' -v")
        return supervise(running)
    finally:
        shutdown(running)


if __name__ == "__main__":
    raise SystemExit(main())
