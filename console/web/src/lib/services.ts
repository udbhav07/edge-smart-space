import type { TopicRow } from "./types";
import type { Health } from "@/components/ui";

// Which process publishes what, and how often it should. A service is judged
// by its own output: one that publishes every tick and has gone quiet is
// silent; one that only publishes on change is quiet, not dead, when it has
// had nothing to say. Topic patterns follow src/common/topics.py.

export interface ServiceSpec {
  id: string;
  name: string;
  role: string;
  layer: string;
  /** Topics it publishes. */
  publishes: string[];
  /** Seconds between messages when it is working; null when it speaks only on change. */
  cadence_s: number | null;
}

export const SERVICES: ServiceSpec[] = [
  {
    id: "simulator",
    name: "Room simulator",
    role: "The plant and its sensors: temperature, humidity, outdoor, presence, power, and the air conditioner's state",
    layer: "Layer 1",
    publishes: ["space/sensor/+/state", "space/actuator/+/state"],
    cadence_s: 5,
  },
  {
    id: "estimator",
    name: "Thermal estimator",
    role: "Recursive least squares on the room's four-coefficient model: prediction, residual, coefficients",
    layer: "Layer 2",
    publishes: ["space/estimate/thermal", "space/estimate/coefficients"],
    cadence_s: 5,
  },
  {
    id: "control",
    name: "Regulatory control",
    role: "Deadband control behind the safety validator; every command and every verdict",
    layer: "Layer 2–3",
    publishes: ["space/actuator/+/command", "space/audit/validation", "space/goal/active", "space/control/tracking", "space/tariff/state"],
    cadence_s: 5,
  },
  {
    id: "detectors",
    name: "Fault detectors",
    role: "D1–D5 and the mode manager: sensor health, faults, degradation mode",
    layer: "Layer 3",
    publishes: ["space/sensor/+/health", "space/fault/+", "space/system/mode"],
    cadence_s: null,
  },
  {
    id: "reasoning",
    name: "Reasoning",
    role: "The supervisor (every 300 s and on events), the assistant, and fault diagnosis",
    layer: "Layer 4",
    publishes: ["space/goal/proposed", "space/audit/reasoning", "space/context/reply", "space/diagnosis/+", "space/assist/proposed"],
    cadence_s: 300,
  },
  {
    id: "assistance",
    name: "Tool executor",
    role: "Runs tool calls behind their gates: calendar, mock bookings, mock mail, memory",
    layer: "Layer 1",
    publishes: ["space/assist/result", "space/assist/catalogue", "space/context/profile/+"],
    cadence_s: 30,
  },
  {
    id: "speech",
    name: "Speech",
    role: "Wake word, Whisper and extraction; publishes what it heard and its own status",
    layer: "Layer 1",
    publishes: ["space/speech/status", "space/context/preference"],
    cadence_s: 15,
  },
  {
    id: "console",
    name: "Console",
    role: "This dashboard's gateway: what you type, confirm and inject",
    layer: "Layer 1",
    publishes: ["space/context/utterance", "space/assist/confirmed", "space/inject/+", "space/system/reset"],
    cadence_s: null,
  },
];

export function matches(pattern: string, topic: string): boolean {
  const want = pattern.split("/");
  const have = topic.split("/");
  for (let index = 0; index < want.length; index++) {
    if (want[index] === "#") return true;
    if (index >= have.length) return false;
    if (want[index] !== "+" && want[index] !== have[index]) return false;
  }
  return want.length === have.length;
}

export function topicsOf(service: ServiceSpec, rows: TopicRow[]): TopicRow[] {
  return rows.filter((row) => service.publishes.some((pattern) => matches(pattern, row.topic)));
}

export function producerOf(topic: string): ServiceSpec | undefined {
  return SERVICES.find((service) => service.publishes.some((pattern) => matches(pattern, topic)));
}

export interface ServiceHealth {
  health: Health;
  word: string;
  last_ts: number | null;
  per_minute: number;
  messages: number;
}

/** Live, quiet or silent, against the service's own cadence. */
export function judge(service: ServiceSpec, rows: TopicRow[], now: number): ServiceHealth {
  const mine = topicsOf(service, rows);
  const last = mine.reduce<number | null>((latest, row) => (latest === null || row.last_ts > latest ? row.last_ts : latest), null);
  const per_minute = mine.reduce((sum, row) => sum + rateOf(row, now), 0);
  const messages = mine.reduce((sum, row) => sum + row.count, 0);
  if (last === null) return { health: "idle", word: service.cadence_s === null ? "nothing yet" : "not heard", last_ts: null, per_minute, messages };
  const age = now - last;
  if (service.cadence_s === null) return { health: "good", word: "on change", last_ts: last, per_minute, messages };
  if (age <= service.cadence_s * 3) return { health: "good", word: "live", last_ts: last, per_minute, messages };
  if (age <= service.cadence_s * 12) return { health: "warn", word: "late", last_ts: last, per_minute, messages };
  return { health: "alert", word: "silent", last_ts: last, per_minute, messages };
}

/** Messages a minute, preferring what this browser has seen arrive. */
export function rateOf(row: TopicRow, now: number): number {
  const arrivals = (row.arrivals ?? []).filter((ts) => now - ts < 120);
  if (arrivals.length > 2) {
    const span = arrivals[arrivals.length - 1] - arrivals[0];
    if (span > 0) return ((arrivals.length - 1) * 60) / span;
  }
  return now - row.last_ts < 120 ? row.per_minute : 0;
}
