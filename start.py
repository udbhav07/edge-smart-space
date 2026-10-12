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
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.common import topics
from src.common.config import Config, ConfigError, DeviceSource, load_config
from src.common.mqtt_client import topic_matches

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
    #: Run only when Layer 1 is this source; None runs whatever the source.
    source: DeviceSource | None = None

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
        source=DeviceSource.SIMULATED,
    ),
    Service(
        name="devices",
        module="src.io",
        description="ESPHome nodes and the air conditioner (hardware mode)",
        source=DeviceSource.ESPHOME,
    ),
    Service(
        name="estimator",
        module="src.estimation",
        description="Online RC identification: coefficients, prediction, residual",
    ),
    Service(
        name="control",
        module="src.control",
        description="Regulatory loop: deadband tracking behind the safety gate",
    ),
    Service(
        name="detectors",
        module="src.faults",
        description="Fault detector bank D1-D5 and the degradation mode",
    ),
    Service(
        name="assistance",
        module="src.assistance",
        description="Tool executor: local calendar and mock travel bookings",
    ),
    Service(
        name="reasoning",
        module="src.reasoning",
        description="Supervisor agent and assistant (needs the inference server)",
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
            if dependency.name == _INFERENCE:
                LOGGER.info("         %s", describe_model_state(config))
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


_INFERENCE = "inference server"

#: Ollama's list of models held in memory. Other servers lack it, and then
#: nothing can be said about loading, which is reported as such.
_LOADED_MODELS_PATH = "/api/ps"


def loaded_models(config: Config) -> tuple[str, ...] | None:
    """Models the inference server holds in memory, or None if it cannot say."""
    endpoint = urlparse(config.reasoning.base_url)
    url = f"{endpoint.scheme}://{endpoint.netloc}{_LOADED_MODELS_PATH}"
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT_S) as response:
            listing = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None
    return tuple(str(item.get("name", "")) for item in listing.get("models", ()))


def describe_model_state(config: Config, loaded: tuple[str, ...] | None = ...) -> str:
    """Whether the first reply will be quick (U-09)."""
    if loaded is ...:
        loaded = loaded_models(config)
    model = config.reasoning.model
    if loaded is None:
        return f"cannot tell whether {model} is loaded"
    if model in loaded:
        return f"{model} is loaded"
    return (
        f"{model} is not loaded yet; reasoning loads it at startup, and a reply "
        f"before that finishes takes about a minute"
    )


#: A topic each periodic service publishes on every tick, live rather than
#: retained, so traffic on it means a copy is running now (U-12).
LIVE_SIGNATURES = {
    "simulator": topics.SENSOR_STATE,
    "devices": topics.SENSOR_STATE,
    "estimator": topics.ESTIMATE_THERMAL,
    "control": topics.ACTUATOR_COMMAND,
}


def already_running(
    services: tuple[Service, ...], live_topics: set[str]
) -> dict[str, str]:
    """Services about to start whose output is already on the broker.

    :param live_topics: topics heard as live traffic, not retained replay.
    :returns: service name -> the topic that gave it away.
    """
    found = {}
    for service in services:
        signature = LIVE_SIGNATURES.get(service.name)
        if signature is None:
            continue
        heard = sorted(topic for topic in live_topics if topic_matches(signature.wildcard(), topic))
        if heard:
            found[service.name] = heard[0]
    return found


def listen_live(config: Config, seconds: float) -> set[str]:
    """Topics with live (not retained) traffic during a short listen."""
    import paho.mqtt.client as mqtt

    heard: set[str] = set()
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"start-probe-{id(heard)}")

    def on_connect(_client, _userdata, _flags, _reason_code, _properties=None) -> None:
        client.subscribe(topics.ALL_TOPICS, 0)

    def on_message(_client, _userdata, message) -> None:
        if not message.retain:
            heard.add(message.topic)

    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(config.mqtt.host, config.mqtt.port, int(config.mqtt.keepalive_s))
    except OSError:
        return heard
    client.loop_start()
    try:
        time.sleep(seconds)
    finally:
        client.loop_stop()
        client.disconnect()
    return heard


def select(
    only: list[str] | None,
    skip: list[str] | None,
    source: DeviceSource | None = None,
) -> tuple[Service, ...]:
    """Choose which services to run.

    Layer 1 follows ``devices.source`` -- the simulator or the real devices,
    never both -- unless a service is named explicitly with ``--only``.

    :raises ValueError: if a name does not match a known service, rather than
        silently starting nothing.
    """
    for name in (only or []) + (skip or []):
        if name not in SERVICE_NAMES:
            raise ValueError(
                f"unknown service {name!r}; choose from {list(SERVICE_NAMES)}"
            )

    if only:
        chosen = tuple(s for s in SERVICES if s.name in only)
    else:
        chosen = tuple(
            s for s in SERVICES if s.source is None or source is None or s.source is source
        )
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


#: Seconds to wait before relaunching a service that exited (NFR-08).
RESTART_DELAY_S = 2.0

#: Restarts allowed within RESTART_WINDOW_S before a crash-looping service is
#: left down. A service that dies on start would otherwise spin forever, and
#: hide the one line of the log that says why.
MAX_RESTARTS = 5
RESTART_WINDOW_S = 60.0


def supervise(
    running: dict[str, subprocess.Popen],
    relaunch=None,
    restart: bool = True,
) -> int:
    """Watch children, restarting any that exit, until interrupted (NFR-08).

    Separate processes are the point: killing the reasoning process during
    the demonstration shows regulatory control carry on (section 9.3). On the
    Jetson systemd restarts it (``Restart=always``); this does the same on a
    development machine, so the demonstration behaves the same on both.

    :param relaunch: starts a service again by name; None disables restarts.
    """
    history: dict[str, list[float]] = {}
    try:
        while running:
            for name, process in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                LOGGER.warning("%s exited with code %s", name, code)
                del running[name]
                if not restart or relaunch is None:
                    continue
                now = time.monotonic()
                recent = [t for t in history.get(name, []) if now - t < RESTART_WINDOW_S]
                if len(recent) >= MAX_RESTARTS:
                    LOGGER.error(
                        "%s exited %d times in %.0f s; leaving it down", name,
                        len(recent), RESTART_WINDOW_S,
                    )
                    continue
                time.sleep(RESTART_DELAY_S)
                history[name] = recent + [now]
                running[name] = relaunch(name)
                LOGGER.warning("restarted %s", name)
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
    parser.add_argument(
        "--allow-running",
        action="store_true",
        help="Start even if a copy of a service is already publishing.",
    )
    parser.add_argument(
        "--no-restart",
        action="store_true",
        help="Leave a service down if it exits (the default restarts it, NFR-08).",
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
        services = select(arguments.only, arguments.skip, config.devices.source)
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

    if not arguments.allow_running:
        # Two simulators publish two rooms onto one set of topics, and the
        # estimator then fits a model to both (U-12).
        busy = already_running(services, listen_live(config, config.loop.sensor_period_s + 1.0))
        if busy:
            for name, topic in busy.items():
                LOGGER.error("%s is already running: live traffic on %s", name, topic)
            LOGGER.error("stop the running copy first, or pass --allow-running")
            return 1

    running: dict[str, subprocess.Popen] = {}
    try:
        for service in services:
            running[service.name] = launch(service, arguments.config)
        LOGGER.info("watch the blackboard with: python -m tools.blackboard_view")
        by_name = {service.name: service for service in services}
        return supervise(
            running,
            relaunch=lambda name: launch(by_name[name], arguments.config),
            restart=not arguments.no_restart,
        )
    finally:
        shutdown(running)


if __name__ == "__main__":
    raise SystemExit(main())
