"""Any MCP server, mounted as a device on the blackboard (FR-90).

The room does not speak MCP to a model. A 7B model shown every tool of every
server would have no context left (FR-89), and a call made point to point
would never reach the blackboard (FR-60). Instead each server named in
``actuators.mcp_servers`` is started over stdio, its tools are listed, and
the ones the configuration allows -- with the effect and safety class the
configuration gives them, never the server's own say-so -- become the
capabilities of an ordinary device. Commands then arrive on the board like
any device's, and are carried to the server as ``call_tool``.

MCP's client is asynchronous; the device service is not. Each server gets
one event loop on a thread of its own, holding the session open for the life
of the process, and calls cross to it with ``run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
from collections.abc import Mapping
from typing import Any

from src.common.clock import Clock
from src.common.config import Config, McpServerConfig
from src.common.devices import DeviceCapability, DeviceManifest
from src.common.tools import ArgumentValue, ParameterType, ProviderOutcome, ToolParameter
from src.devices.runner import DeviceParts

LOGGER = logging.getLogger(__name__)

#: How long a server has to start and list its tools, in s.
STARTUP_TIMEOUT_S = 30.0

_JSON_TYPES = {
    "string": ParameterType.STRING,
    "number": ParameterType.NUMBER,
    "integer": ParameterType.INTEGER,
    "boolean": ParameterType.BOOLEAN,
}


class BridgeError(RuntimeError):
    """A configured server cannot be mounted; said at startup, not to an occupant."""


def parameters_from_schema(tool: str, schema: Mapping[str, Any]) -> tuple[ToolParameter, ...]:
    """A tool's JSON-schema input as the room's flat parameters.

    :raises BridgeError: for anything that is not a flat object of strings,
        numbers, integers and booleans. A tool taking a nested object cannot
        be checked by the room's gate, so it is refused rather than passed
        through unchecked.
    """
    required = set(schema.get("required", ()))
    parameters = []
    for name, spec in schema.get("properties", {}).items():
        optional_by_type = False
        if "anyOf" in spec:
            # Optional[int] in pydantic: anyOf [{"type": "integer"}, {"type": "null"}].
            options = [option for option in spec["anyOf"] if option.get("type") != "null"]
            if len(options) != 1:
                raise BridgeError(f"{tool}: parameter {name!r} accepts several types; only one is supported")
            optional_by_type = len(options) < len(spec["anyOf"])
            spec = {**spec, **options[0]}
        kind = _JSON_TYPES.get(spec.get("type", ""))
        if kind is None:
            raise BridgeError(f"{tool}: parameter {name!r} is {spec.get('type', 'untyped')}; only flat values are supported")
        choices = tuple(str(choice) for choice in spec.get("enum", ())) if kind is ParameterType.STRING else ()
        parameters.append(
            ToolParameter(
                name=name,
                type=kind,
                description=spec.get("description") or spec.get("title") or name.replace("_", " "),
                required=name in required and not optional_by_type,
                choices=choices,
                minimum=spec.get("minimum") if kind in (ParameterType.NUMBER, ParameterType.INTEGER) else None,
                maximum=spec.get("maximum") if kind in (ParameterType.NUMBER, ParameterType.INTEGER) else None,
            )
        )
    return tuple(parameters)


def outcome_from_result(result: Any) -> ProviderOutcome:
    """A ``CallToolResult`` as a sentence and flat detail.

    A server that returns an object -- as structured content, or as JSON text,
    which is how many send it -- may say a ``message``; its other flat values
    become the detail and the device's state.

    :raises ValueError: if the server reports an error; the device service
        passes the sentence on.
    """
    text = " ".join(getattr(block, "text", "") for block in result.content if getattr(block, "type", "") == "text").strip()
    if result.is_error:
        raise ValueError(f"The device reported a problem: {text.splitlines()[0] if text else 'no reason given'}")
    data = result.structured_content if isinstance(result.structured_content, dict) else _json_object(text)
    detail = {key: value for key, value in (data or {}).items() if key != "message" and _flat(value)}
    message = str((data or {}).get("message") or text or "Done.")
    return ProviderOutcome(message=message, detail=detail)


def _json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _innermost(exc: BaseException) -> str:
    """The first concrete reason inside an exception group, in words."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return str(exc) or type(exc).__name__


def _flat(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


class McpConnection:
    """One MCP server over stdio, held open on a thread with its own event loop."""

    def __init__(self, server: McpServerConfig, timeout_s: float) -> None:
        self._server = server
        self._timeout_s = timeout_s
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=f"mcp-{server.device_id}", daemon=True)
        self._client: Any = None
        self._opened = asyncio.Event()
        self._closing = asyncio.Event()
        self._failure: BaseException | None = None
        self._session: Any = None

    def open(self) -> list[Any]:
        """Start the server and list its tools.

        :raises BridgeError: if it does not start or answer in time.
        """
        self._thread.start()
        ready: asyncio.Future = asyncio.run_coroutine_threadsafe(self._ready(), self._loop)
        self._session = asyncio.run_coroutine_threadsafe(self._hold(), self._loop)
        try:
            return ready.result(STARTUP_TIMEOUT_S)
        except (BridgeError, TimeoutError, OSError, RuntimeError) as exc:
            self.close()
            raise BridgeError(f"{self._server.device_id}: the MCP server did not start: {exc}") from exc

    def call(self, name: str, arguments: Mapping[str, ArgumentValue]) -> Any:
        if self._client is None:
            raise ValueError("The device is not connected.")
        future = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, dict(arguments)), self._loop)
        try:
            return future.result(self._timeout_s)
        except TimeoutError as exc:
            future.cancel()
            raise ValueError("The device didn't answer in time.") from exc

    def close(self) -> None:
        if not self._loop.is_running():
            return
        self._loop.call_soon_threadsafe(self._closing.set)
        if self._session is not None:
            try:
                self._session.result(5.0)
            except (TimeoutError, OSError, RuntimeError, BridgeError, asyncio.CancelledError) as exc:
                # The server is going away either way; a slow or broken exit is logged, not raised.
                LOGGER.debug("closing %s: %s", self._server.device_id, exc)
        self._loop.call_soon_threadsafe(self._loop.stop)

    async def _ready(self) -> list[Any]:
        await self._opened.wait()
        if self._client is None:
            raise BridgeError(str(self._failure) or "it exited")
        tools, cursor = [], None
        while True:
            page = await self._client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = getattr(page, "next_cursor", None)
            if not cursor:
                return tools

    async def _hold(self) -> None:
        """Keep the session open in one task, as the SDK's context managers require."""
        from mcp import Client, MCPError, StdioServerParameters

        command = sys.executable if self._server.command in ("python", "python3") else self._server.command
        parameters = StdioServerParameters(command=command, args=list(self._server.args), env=self._environment())
        try:
            async with Client(parameters, read_timeout_seconds=self._timeout_s) as client:
                self._client = client
                self._opened.set()
                await self._closing.wait()
        except (OSError, RuntimeError, MCPError, ExceptionGroup) as exc:
            # The SDK runs on anyio task groups, so a server that exits at
            # startup arrives as a group around "Connection closed".
            self._failure = BridgeError(_innermost(exc))
            raise self._failure from exc
        finally:
            self._client = None
            self._opened.set()

    def _environment(self) -> dict[str, str] | None:
        if not self._server.env:
            return None
        from mcp.client.stdio import get_default_environment

        return {**get_default_environment(), **self._server.env}


def mount(server: McpServerConfig, tools: list[Any], now: float) -> DeviceManifest:
    """The device a server becomes: only its configured tools, as configured.

    :raises BridgeError: if a configured tool is not offered by the server,
        or its name cannot be a capability. Either way the configuration and
        the server disagree, and it is better found now than by an occupant.
    """
    offered = {tool.name: tool for tool in tools}
    missing = sorted(set(server.tools) - set(offered))
    if missing:
        raise BridgeError(f"{server.device_id}: the server offers no {missing}; it offers {sorted(offered)}")
    capabilities = []
    for name, declared in server.tools.items():
        tool = offered[name]
        try:
            capabilities.append(
                DeviceCapability(
                    name=name,
                    purpose=declared.purpose or tool.description or name.replace("_", " "),
                    effect=declared.effect,
                    device_class=declared.device_class,
                    parameters=parameters_from_schema(name, tool.input_schema or {}),
                    min_interval_s=declared.min_interval_s,
                )
            )
        except ValueError as exc:
            raise BridgeError(f"{server.device_id}: tool {name!r} cannot be a capability: {exc}") from exc
    return DeviceManifest(
        ts=now,
        device_id=server.device_id,
        name=server.name,
        kind=server.kind,
        room=server.room,
        aliases=tuple(server.aliases),
        simulated=server.simulated,
        provider=f"mcp:{server.device_id}",
        capabilities=tuple(capabilities),
    )


def device(server: McpServerConfig, connection: McpConnection, now: float) -> DeviceParts:
    """One mounted server as device parts: its handlers call the server, its state is the last answer."""
    manifest = mount(server, connection.open(), now)
    state: dict[str, ArgumentValue] = {}

    def handler(name: str):
        def run(arguments: Mapping[str, ArgumentValue]) -> ProviderOutcome:
            outcome = outcome_from_result(connection.call(name, arguments))
            state.update(outcome.detail)
            return outcome

        return run

    handlers = {capability.name: handler(capability.name) for capability in manifest.capabilities}
    return DeviceParts(manifest=manifest, handlers=handlers, state=lambda: dict(state), close=connection.close)


def build(config: Config, clock: Clock) -> list[DeviceParts]:
    """Every configured server, mounted. One that fails to mount is logged and left out."""
    mounted = []
    for server in config.actuators.mcp_servers:
        try:
            mounted.append(device(server, McpConnection(server, config.actuators.command_timeout_s), clock.now()))
        except BridgeError as exc:
            LOGGER.error("%s", exc)
    return mounted
