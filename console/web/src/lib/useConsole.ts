import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { ServerEvent, Snapshot } from "./types";

// Push first, pull as the fallback. The page opens a WebSocket and is given a
// whole snapshot, then every change. While the socket is down it polls
// /api/state so the screen never freezes on stale numbers, and it keeps
// trying to reconnect with a growing delay.

const POLL_MS = 3000;
const MAX_BACKOFF_MS = 15000;
const KEEP_TURNS = 120;
const KEEP_STEPS = 200;
const KEEP_POINTS = 240;

export type LinkState = "connecting" | "live" | "polling" | "offline";

type Action = { kind: "snapshot"; data: Snapshot } | { kind: "event"; event: ServerEvent };

function reduce(state: Snapshot | null, action: Action): Snapshot | null {
  if (action.kind === "snapshot") return action.data;
  if (!state) return state;
  const { type, data } = action.event as { type: string; data: any };
  const room = state.room;
  switch (type) {
    case "snapshot":
      return data as Snapshot;
    case "turn":
      if (state.conversation.some((turn) => turn.id === data.id)) return state;
      return { ...state, conversation: [...state.conversation, data].slice(-KEEP_TURNS) };
    case "step":
      if (state.activity.some((step) => step.id === data.id)) return state;
      return { ...state, activity: [...state.activity, data].slice(-KEEP_STEPS) };
    case "sensor":
      return { ...state, room: { ...room, sensors: { ...room.sensors, [data.sensor_id]: data } } };
    case "health":
      return { ...state, room: { ...room, health: { ...room.health, [data.sensor_id]: data } } };
    case "thermal":
    case "coefficients":
    case "actuator":
    case "mode":
    case "goal":
    case "tracking":
    case "tariff":
      return { ...state, room: { ...room, [type]: data } };
    case "speech":
      return { ...state, speech: data };
    case "point":
      return { ...state, history: [...state.history, data].slice(-KEEP_POINTS) };
    case "faults":
    case "diagnoses":
    case "profiles":
    case "injections":
    case "waiting":
    case "calendar":
    case "tools":
      return { ...state, [type]: data };
    case "link":
      return { ...state, broker: data.broker };
    default:
      return state;
  }
}

export function useConsole() {
  const [state, dispatch] = useReducer(reduce, null);
  const [link, setLink] = useState<LinkState>("connecting");
  const [calendarStale, setCalendarStale] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const backoff = useRef(1000);
  const pollTimer = useRef<number | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const alive = useRef(true);

  const poll = useCallback(async () => {
    try {
      const response = await fetch("/api/state", { cache: "no-store" });
      if (!response.ok) throw new Error(String(response.status));
      dispatch({ kind: "snapshot", data: await response.json() });
      setLink((current) => (current === "live" ? current : "polling"));
    } catch {
      setLink((current) => (current === "live" ? current : "offline"));
    }
  }, []);

  const stopPolling = () => {
    if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
    pollTimer.current = null;
  };
  const startPolling = useCallback(() => {
    if (pollTimer.current !== null) return;
    void poll();
    pollTimer.current = window.setInterval(() => void poll(), POLL_MS);
  }, [poll]);

  const connect = useCallback(() => {
    if (!alive.current) return;
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${window.location.host}/api/events`);
    socketRef.current = socket;
    socket.onopen = () => {
      backoff.current = 1000;
      stopPolling();
      setLink("live");
    };
    socket.onmessage = (message) => {
      const event = JSON.parse(message.data) as ServerEvent;
      if (event.type === "snapshot") dispatch({ kind: "snapshot", data: event.data as Snapshot });
      else if (event.type === "calendar_stale") setCalendarStale((n) => n + 1);
      else dispatch({ kind: "event", event });
    };
    socket.onclose = () => {
      if (!alive.current) return;
      setLink((current) => (current === "live" ? "polling" : current));
      startPolling();
      reconnectTimer.current = window.setTimeout(connect, backoff.current);
      backoff.current = Math.min(backoff.current * 2, MAX_BACKOFF_MS);
    };
    socket.onerror = () => socket.close();
  }, [startPolling]);

  useEffect(() => {
    alive.current = true;
    connect();
    return () => {
      alive.current = false;
      stopPolling();
      if (reconnectTimer.current !== null) window.clearTimeout(reconnectTimer.current);
      socketRef.current?.close();
    };
  }, [connect]);

  return { state, link, calendarStale };
}
