import { createContext, useCallback, useContext, useEffect, useReducer, useRef, useState } from "react";
import type { BoardMessage, ServerEvent, Snapshot, TopicRow } from "./types";

// One connection for every page. Push first, pull as the fallback: the page
// opens a WebSocket and is given a whole snapshot, then every change. While
// the socket is down it polls /api/state so no page freezes on stale numbers,
// and it keeps trying to reconnect with a growing delay.
//
// Events arrive several a second (every MQTT message is one), so they are
// applied in batches every BATCH_MS rather than one render each.

const POLL_MS = 3000;
const MAX_BACKOFF_MS = 15000;
const BATCH_MS = 200;
const KEEP_TURNS = 120;
const KEEP_STEPS = 300;
const KEEP_POINTS = 360;
const KEEP_TAIL = 400;
const KEEP_INBOX = 100;

export type LinkState = "connecting" | "live" | "polling" | "offline";

export interface ConsoleData {
  state: Snapshot | null;
  tail: BoardMessage[];
  link: LinkState;
  calendarStale: number;
}

type Action = { kind: "snapshot"; data: Snapshot } | { kind: "batch"; events: ServerEvent[] };
interface Store { state: Snapshot | null; tail: BoardMessage[] }

function applyOne(state: Snapshot, tail: BoardMessage[], event: ServerEvent): [Snapshot, BoardMessage[]] {
  const { type, data } = event as { type: string; data: any };
  const room = state.room;
  switch (type) {
    case "turn":
      if (state.conversation.some((turn) => turn.id === data.id)) return [state, tail];
      return [{ ...state, conversation: [...state.conversation, data].slice(-KEEP_TURNS) }, tail];
    case "step":
      if (state.activity.some((step) => step.id === data.id)) return [state, tail];
      return [{ ...state, activity: [...state.activity, data].slice(-KEEP_STEPS) }, tail];
    case "inbox":
      if (state.inbox.some((item) => item.id === data.id)) return [state, tail];
      return [{ ...state, inbox: [...state.inbox, data].slice(-KEEP_INBOX) }, tail];
    case "sensor":
      return [{ ...state, room: { ...room, sensors: { ...room.sensors, [data.sensor_id]: data } } }, tail];
    case "health":
      return [{ ...state, room: { ...room, health: { ...room.health, [data.sensor_id]: data } } }, tail];
    case "thermal":
    case "coefficients":
    case "actuator":
    case "mode":
    case "goal":
    case "tracking":
    case "tariff":
      return [{ ...state, room: { ...room, [type]: data } }, tail];
    case "speech":
      return [{ ...state, speech: data }, tail];
    case "speech_settings":
    case "google":
    case "energy":
    case "devices":
      return [{ ...state, [type]: data }, tail];
    case "series": {
      const points = state.series[data.key] ?? [];
      return [{ ...state, series: { ...state.series, [data.key]: [...points, data.point].slice(-KEEP_POINTS) } }, tail];
    }
    case "mqtt": {
      const message = data as BoardMessage;
      const rows = state.topics.filter((row) => row.topic !== message.topic);
      if (!message.withdrawn) {
        const previous = state.topics.find((row) => row.topic === message.topic);
        const row: TopicRow = {
          topic: message.topic,
          count: message.count ?? (previous?.count ?? 0) + 1,
          first_ts: previous?.first_ts ?? message.ts,
          last_ts: message.ts,
          retained: message.retained ?? previous?.retained ?? false,
          payload: message.payload,
          per_minute: previous?.per_minute ?? 0,
          arrivals: [...(previous?.arrivals ?? []), message.ts].slice(-30),
        };
        rows.push(row);
      }
      return [{ ...state, topics: rows }, [...tail, message].slice(-KEEP_TAIL)];
    }
    case "faults":
    case "diagnoses":
    case "profiles":
    case "injections":
    case "waiting":
    case "calendar":
    case "tools":
    case "bookings":
    case "sent":
      return [{ ...state, [type]: data }, tail];
    case "link":
      return [{ ...state, broker: data.broker }, tail];
    default:
      return [state, tail];
  }
}

/** Each topic's last message, oldest first: a stream that is not blank on arrival. */
function seedTail(snapshot: Snapshot): BoardMessage[] {
  return [...snapshot.topics]
    .sort((a, b) => a.last_ts - b.last_ts)
    .map((row) => ({ topic: row.topic, ts: row.last_ts, payload: row.payload, count: row.count, retained: row.retained }));
}

function reduce(store: Store, action: Action): Store {
  if (action.kind === "snapshot") return { state: action.data, tail: store.tail.length ? store.tail : seedTail(action.data) };
  if (!store.state) return store;
  let state = store.state;
  let tail = store.tail;
  for (const event of action.events) {
    if (event.type === "snapshot") {
      state = event.data as Snapshot;
      if (!tail.length) tail = seedTail(state);
    }
    else [state, tail] = applyOne(state, tail, event);
  }
  return { state, tail };
}

function useConnection(): ConsoleData {
  const [store, dispatch] = useReducer(reduce, { state: null, tail: [] });
  const [link, setLink] = useState<LinkState>("connecting");
  const [calendarStale, setCalendarStale] = useState(0);
  const socketRef = useRef<WebSocket | null>(null);
  const backoff = useRef(1000);
  const pollTimer = useRef<number | null>(null);
  const reconnectTimer = useRef<number | null>(null);
  const pending = useRef<ServerEvent[]>([]);
  const flushTimer = useRef<number | null>(null);
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

  const flush = () => {
    flushTimer.current = null;
    const events = pending.current;
    pending.current = [];
    if (events.length) dispatch({ kind: "batch", events });
  };

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
      if (event.type === "snapshot") {
        pending.current = [];
        dispatch({ kind: "snapshot", data: event.data as Snapshot });
        return;
      }
      if (event.type === "calendar_stale") setCalendarStale((n) => n + 1);
      if (event.type === "device_result") window.dispatchEvent(new CustomEvent(DEVICE_RESULT_EVENT, { detail: event.data }));
      pending.current.push(event);
      if (flushTimer.current === null) flushTimer.current = window.setTimeout(flush, BATCH_MS);
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
      if (flushTimer.current !== null) window.clearTimeout(flushTimer.current);
      socketRef.current?.close();
    };
  }, [connect]);

  return { state: store.state, tail: store.tail, link, calendarStale };
}

const ConsoleContext = createContext<ConsoleData>({ state: null, tail: [], link: "connecting", calendarStale: 0 });

export function ConsoleProvider({ children }: { children: React.ReactNode }) {
  return <ConsoleContext.Provider value={useConnection()}>{children}</ConsoleContext.Provider>;
}

export const useConsole = () => useContext(ConsoleContext);

/** A device's answer to a call made from this page, for whoever made it. */
export const DEVICE_RESULT_EVENT = "smart-space:device-result";

const MAIL_SEEN_KEY = "mail-seen";
const MAIL_SEEN_EVENT = "smart-space:mail-seen";

function readMailSeen(): number {
  try {
    return Number(localStorage.getItem(MAIL_SEEN_KEY) ?? 0);
  } catch {
    return 0;
  }
}

/** When this browser last opened the inbox; newer messages count as unread. */
export function useMailSeen(): number {
  const [seen, setSeen] = useState(readMailSeen);
  useEffect(() => {
    const update = () => setSeen(readMailSeen());
    window.addEventListener(MAIL_SEEN_EVENT, update);
    return () => window.removeEventListener(MAIL_SEEN_EVENT, update);
  }, []);
  return seen;
}

export function markMailSeen(): void {
  try {
    localStorage.setItem(MAIL_SEEN_KEY, String(Date.now() / 1000));
  } catch {
    /* private window: unread counts reset each visit */
  }
  window.dispatchEvent(new Event(MAIL_SEEN_EVENT));
}

/** Wall-clock seconds, refreshed every second, for ages and staleness. */
export function useNow(): number {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, []);
  return now;
}
