"""The console's web server: a page, a push stream, and a few actions (FR-56).

Two threads meet here. The MQTT client delivers messages on its own thread;
the web server runs an asyncio loop. :class:`Link` folds each message into the
shared :class:`ConsoleState` under one lock and hands the resulting events to
the :class:`Hub`, which carries them onto the loop and out to every open
browser. A browser that has just connected is given the whole snapshot, then
the stream; one that cannot keep a socket open can poll ``/api/state``.

Nothing the rest of the system does waits on this: if no browser is open, or
the console process is not running at all, every other component behaves
exactly as before.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from src.common import topics
from src.common.mqtt_client import Blackboard
from src.console.actions import ActionRefused, ConsoleActions
from src.console.google_signin import GoogleSignIn, SignInError
from src.console.state import ConsoleState, Event, encode

LOGGER = logging.getLogger(__name__)

STATIC_DIRECTORY = Path(__file__).with_name("static")

#: Events buffered per browser. A tab left in the background must not grow
#: without limit; the oldest events go first, and the next snapshot heals it.
_CLIENT_QUEUE = 500

#: What is subscribed to: the whole tree, as the recorder does.
_EVERYTHING_QOS = 1

_NOT_BUILT = """<!doctype html><meta charset="utf-8"><title>Smart Space</title>
<body style="font:15px system-ui;background:#0b0e12;color:#d8dee6;padding:40px">
<h1 style="font-weight:600">The console is not built yet</h1>
<p>Build the frontend once: <code>cd console/web &amp;&amp; npm install &amp;&amp; npm run build</code>.
The backend is running regardless; <a style="color:#7fe0d0" href="/api/state">/api/state</a> shows it.</p>"""


class Hub:
    """Carries events from the MQTT thread to every connected browser."""

    def __init__(self) -> None:
        self._clients: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def clients(self) -> int:
        return len(self._clients)

    def push(self, events: list[Event]) -> None:
        """Thread-safe: may be called from the MQTT thread."""
        if not events or self._loop is None or self._loop.is_closed():
            return
        encoded = [encode(event) for event in events]
        self._loop.call_soon_threadsafe(self._fan_out, encoded)

    def _fan_out(self, encoded: list[str]) -> None:
        for queue in tuple(self._clients):
            for text in encoded:
                if queue.full():
                    queue.get_nowait()
                queue.put_nowait(text)

    def open(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_CLIENT_QUEUE)
        self._clients.add(queue)
        return queue

    def close(self, queue: asyncio.Queue) -> None:
        self._clients.discard(queue)


class Link:
    """The MQTT transport's target: subscribes to everything, folds it in."""

    def __init__(self, state: ConsoleState, hub: Hub, lock: threading.Lock, blackboard: Blackboard) -> None:
        self._state = state
        self._hub = hub
        self._lock = lock
        self._blackboard = blackboard
        self._transport = None
        self.connected = False

    def attach(self, transport) -> None:
        self._transport = transport

    # Called by the transport, exactly as it calls a Blackboard.
    def on_connected(self) -> None:
        self.connected = True
        self._blackboard.on_connected()
        if self._transport is not None:
            self._transport.subscribe(topics.ALL_TOPICS, _EVERYTHING_QOS)
        self._hub.push([{"type": "link", "data": {"broker": True}}])

    def on_disconnected(self) -> None:
        self.connected = False
        self._blackboard.on_disconnected()
        self._hub.push([{"type": "link", "data": {"broker": False}}])

    def dispatch(self, topic: str, payload: bytes) -> None:
        try:
            with self._lock:
                events = self._state.accept(topic, payload)
        except Exception:
            # One malformed or unexpected message must not stop the console
            # listening; it is logged with the topic that carried it.
            LOGGER.exception("console could not fold a message on %s", topic)
            return
        self._hub.push(events)


class SayRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    profile_id: str = ""


class TargetRequest(BaseModel):
    target_c: float
    profile_id: str = ""


class AnswerRequest(BaseModel):
    yes: bool


class RemoveRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    starts_at: str = ""


class KeepRequest(BaseModel):
    event_id: str = Field(min_length=1, max_length=40)


class SettleRequest(BaseModel):
    profile_id: str = Field(min_length=1, max_length=64)
    fact: str = Field(min_length=1, max_length=400)
    keep: bool


class EntryRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    starts_at: str = Field(min_length=1, max_length=40)
    duration_min: int | None = Field(default=None, gt=0, le=1440)


class BookingRequest(BaseModel):
    kind: str
    destination: str = Field(min_length=1, max_length=120)
    depart_on: str = Field(min_length=1, max_length=40)
    origin: str = ""
    nights: int | None = Field(default=None, gt=0, le=60)


class MailRequest(BaseModel):
    to: str = Field(min_length=1, max_length=200)
    subject: str = Field(default="", max_length=200)
    body: str = Field(default="", max_length=4000)


class ComfortRequest(BaseModel):
    direction: str = Field(min_length=1, max_length=16)
    profile_id: str = ""


class AutomaticRequest(BaseModel):
    profile_id: str = ""


class CoolingOffRequest(BaseModel):
    hours: float = Field(gt=0.0, le=24.0)
    profile_id: str = ""


class ResetRequest(BaseModel):
    reason: str = ""


class InjectRequest(BaseModel):
    subject: str = Field(min_length=1)
    fault: str = Field(min_length=1)
    magnitude: float | None = None


def create_app(
    state: ConsoleState,
    actions: ConsoleActions,
    hub: Hub,
    link: Link,
    lock: threading.Lock,
    signin: GoogleSignIn | None = None,
) -> FastAPI:
    """The web application; every dependency injected so it can be tested."""
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        hub.bind(asyncio.get_running_loop())
        yield

    app = FastAPI(
        title="Smart Space console", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )

    def snapshot() -> dict[str, Any]:
        with lock:
            data = json.loads(encode(state.snapshot()))
        data["broker"] = link.connected
        return data

    def act(function, *args) -> JSONResponse:
        try:
            with lock:
                result = function(*args)
        except (ActionRefused, ValidationError) as exc:
            return JSONResponse({"ok": False, "error": _plain(exc)}, status_code=400)
        return JSONResponse({"ok": True, "result": _jsonable(result)})

    @app.get("/api/state")
    async def get_state() -> JSONResponse:
        return JSONResponse(snapshot())

    @app.get("/api/health")
    async def get_health() -> JSONResponse:
        return JSONResponse({"broker": link.connected, "browsers": hub.clients})

    @app.post("/api/say")
    async def post_say(request: SayRequest) -> JSONResponse:
        return act(actions.say, request.text, request.profile_id)

    @app.post("/api/target")
    async def post_target(request: TargetRequest) -> JSONResponse:
        return act(actions.set_target, request.target_c, request.profile_id)

    @app.post("/api/comfort")
    async def post_comfort(request: ComfortRequest) -> JSONResponse:
        return act(actions.comfort, request.direction, request.profile_id)

    @app.post("/api/automatic")
    async def post_automatic(request: AutomaticRequest) -> JSONResponse:
        return act(actions.automatic, request.profile_id)

    @app.post("/api/cooling-off")
    async def post_cooling_off(request: CoolingOffRequest) -> JSONResponse:
        return act(actions.cooling_off, request.hours, request.profile_id)

    @app.post("/api/speech/settings")
    async def post_speech_settings(changes: dict[str, Any]) -> JSONResponse:
        return act(actions.speech_settings, changes)

    # --- Google sign-in (FR-84): the token goes to a file, never the board ---

    @app.get("/api/google")
    async def get_google() -> JSONResponse:
        with lock:
            status = state.snapshot()["google"]
        return JSONResponse({"configured": bool(signin and signin.configured), "status": status})

    @app.get("/api/google/connect", response_model=None)
    async def get_google_connect() -> RedirectResponse | HTMLResponse:
        if signin is None:
            return _signin_page("Google is not available in this console.", ok=False)
        try:
            return RedirectResponse(signin.start(), status_code=302)
        except SignInError as exc:
            return _signin_page(str(exc), ok=False)

    @app.get("/api/google/callback", response_model=None)
    async def get_google_callback(
        state_value: str = Query("", alias="state"), code: str = "", error: str = ""
    ) -> HTMLResponse:
        if signin is None:
            return _signin_page("Google is not available in this console.", ok=False)
        if error:
            return _signin_page("Google sign-in was cancelled; nothing was connected.", ok=False)
        try:
            email = await asyncio.to_thread(signin.finish, state_value, code)
        except SignInError as exc:
            return _signin_page(str(exc), ok=False)
        return _signin_page(f"Connected {email or 'your Google account'}.", ok=True)

    @app.post("/api/google/disconnect")
    async def post_google_disconnect() -> JSONResponse:
        if signin is None:
            return JSONResponse({"ok": False, "error": "Google is not available in this console."}, status_code=400)
        return JSONResponse({"ok": True, "result": signin.disconnect()})

    @app.post("/api/bookings/{invocation_id}")
    async def post_answer(invocation_id: str, request: AnswerRequest) -> JSONResponse:
        response = act(actions.answer, invocation_id, request.yes)
        with lock:
            hub.push([{"type": "waiting", "data": state.snapshot()["waiting"]}])
        return response

    @app.post("/api/calendar/refresh")
    async def post_calendar_refresh() -> JSONResponse:
        return act(actions.refresh_calendar)

    @app.post("/api/calendar/remove")
    async def post_calendar_remove(request: RemoveRequest) -> JSONResponse:
        return act(actions.remove_entry, request.subject, request.starts_at)

    @app.post("/api/calendar/keep")
    async def post_calendar_keep(request: KeepRequest) -> JSONResponse:
        return act(actions.keep_entry, request.event_id)

    @app.post("/api/memory/settle")
    async def post_memory_settle(request: SettleRequest) -> JSONResponse:
        return act(actions.settle_fact, request.profile_id, request.fact, request.keep)

    @app.post("/api/calendar/add")
    async def post_calendar_add(request: EntryRequest) -> JSONResponse:
        return act(actions.add_entry, request.subject, request.starts_at, request.duration_min)

    @app.post("/api/bookings")
    async def post_booking(request: BookingRequest) -> JSONResponse:
        return act(actions.book, request.kind, request.destination, request.depart_on, request.origin, request.nights)

    @app.post("/api/mail")
    async def post_mail(request: MailRequest) -> JSONResponse:
        return act(actions.send_mail, request.to, request.subject, request.body)

    @app.post("/api/operator/reset")
    async def post_reset(request: ResetRequest) -> JSONResponse:
        return act(actions.reset, request.reason)

    @app.post("/api/operator/inject")
    async def post_inject(request: InjectRequest) -> JSONResponse:
        return act(actions.inject, request.subject, request.fault, request.magnitude)

    @app.websocket("/api/events")
    async def events(socket: WebSocket) -> None:
        await socket.accept()
        queue = hub.open()
        try:
            await socket.send_text(encode({"type": "snapshot", "data": snapshot()}))
            while True:
                await socket.send_text(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.close(queue)

    assets = STATIC_DIRECTORY / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def page(path: str):
        candidate = (STATIC_DIRECTORY / path).resolve()
        if path and candidate.is_file() and STATIC_DIRECTORY.resolve() in candidate.parents:
            return FileResponse(candidate)
        index = STATIC_DIRECTORY / "index.html"
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse(_NOT_BUILT)

    return app


def _plain(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        return f"{'.'.join(str(part) for part in first.get('loc', ()))}: {first.get('msg', 'invalid')}"
    return str(exc)


def _jsonable(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump(mode="json", by_alias=True)
    return result


def _signin_page(message: str, ok: bool) -> HTMLResponse:
    """Where the browser lands after Google: what happened, then back to Settings."""
    title = "Connected" if ok else "Not connected"
    body = (
        "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>Google: {title}</title>"
        "<meta http-equiv=refresh content='3;url=/settings'>"
        "<body style='font-family:system-ui,sans-serif;background:#f5f5f7;color:#1d1d1f;"
        "display:grid;place-items:center;min-height:100vh;margin:0'>"
        "<div style='background:#fff;border-radius:18px;padding:32px 40px;"
        "box-shadow:0 4px 24px rgba(0,0,0,.08);max-width:28rem'>"
        f"<h1 style='font-size:24px;margin:0 0 8px'>{title}</h1>"
        f"<p style='margin:0 0 16px'>{html.escape(message)}</p>"
        "<a href='/settings' style='color:#0071e3'>Back to Settings</a></div></body>"
    )
    return HTMLResponse(body, status_code=200 if ok else 400)
