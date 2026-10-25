// What the gateway sends: src/console/state.py is the source of truth.

export type Tone = "neutral" | "info" | "good" | "warn" | "alert";

export interface Turn {
  id: string;
  ts: number;
  role: "occupant" | "assistant" | "system";
  text: string;
  source?: "voice" | "console";
  profile_id?: string;
  intent?: string;
  waiting?: string;
  tone?: Tone;
  fault_id?: string;
}

export interface Step {
  id: string;
  ts: number;
  kind: string;
  text: string;
  detail: string;
  tone: Tone;
  ref: string;
}

export interface Reading {
  ts: number;
  sensor_id: string;
  value: number;
  unit: string;
  quality: "ok" | "suspect" | "faulted";
}

export interface Health {
  sensor_id: string;
  quality: "ok" | "suspect" | "faulted";
  active_fault_id: string | null;
}

export interface Thermal {
  t_in: number;
  t_pred: number;
  residual: number;
  residual_sigma: number;
  model_confidence: number;
  adaptation: "active" | "frozen";
}

export interface ModeState {
  mode: "INIT" | "NORMAL" | "DEGRADED_SENSOR" | "DEGRADED_ACTUATOR" | "SAFE_HOLD";
  since_ts: number;
  active_fault_ids: string[];
  reason: string;
}

export interface Tracking {
  setpoint_c: number;
  tracked_c: number;
  band: "normal" | "peak";
  reason: string;
}

export interface Actuator {
  actuator_id: string;
  simulated: boolean;
  kind: string;
  ack: string;
}

export interface Goal {
  source: string;
  setpoint_c: number;
  rationale: string;
}

export interface Fault {
  fault_id: string;
  detector: string;
  subject: string;
  class: string;
  detected_ts: number;
  mode_impact: string;
}

export interface Diagnosis {
  fault_id: string;
  user_message: string;
  primary_hypothesis: string;
  confidence: string;
}

export interface SpeechStatus {
  ts: number;
  activity: "waiting" | "listening" | "transcribing" | "understanding";
  heard: string;
  microphone: string;
  wake_word: string;
  heartbeat_s: number;
}

export interface Waiting {
  invocation_id: string;
  tool: string;
  arguments: Record<string, string | number | boolean | null>;
  expires_ts: number;
  summary: string;
}

export interface CalendarEntry {
  event_id: string;
  starts_at: string;
  subject: string;
}

export interface Profile {
  profile_id: string;
  display_name: string;
  facts: string[];
  preferences: Record<string, string>;
}

export interface Room {
  sensors: Record<string, Reading>;
  health: Record<string, Health>;
  thermal: Thermal | null;
  coefficients: Record<string, number> | null;
  actuator: Actuator | null;
  mode: ModeState | null;
  goal: Goal | null;
  tracking: Tracking | null;
  tariff: { band: "normal" | "peak"; next_transition_ts: number } | null;
}

export interface Snapshot {
  server_ts: number;
  broker: boolean;
  room: Room;
  speech: SpeechStatus | null;
  series: Record<string, [number, number][]>;
  topics: TopicRow[];
  bookings: Booking[];
  sent: SentMail[];
  inbox: InboxItem[];
  faults: Fault[];
  diagnoses: Record<string, Diagnosis>;
  profiles: Profile[];
  injections: Record<string, { subject: string; kind: string; magnitude: number }>;
  conversation: Turn[];
  activity: Step[];
  waiting: Waiting[];
  calendar: { entries: CalendarEntry[]; as_of: number | null; note: string };
  tools: { name: string; purpose: string; effect: string }[];
  operator_controls: boolean;
}

export interface TopicRow {
  topic: string;
  count: number;
  first_ts: number;
  last_ts: number;
  retained: boolean;
  payload: unknown;
  per_minute: number;
  /** Arrival times seen by this browser, to keep the rate live between snapshots. */
  arrivals?: number[];
}

export interface BoardMessage {
  topic: string;
  ts: number;
  payload?: unknown;
  count?: number;
  retained?: boolean;
  withdrawn?: boolean;
}

export interface Booking {
  reference: string;
  kind: string;
  destination: string;
  origin: string;
  depart_on: string;
  nights: number | null;
  ts: number;
  message: string;
}

export interface SentMail {
  reference: string;
  to: string;
  subject: string;
  body: string;
  ts: number;
}

export interface InboxItem {
  id: string;
  ts: number;
  sender: string;
  subject: string;
  body: string;
  kind: string;
  tone: Tone;
}

export type ServerEvent =
  | { type: "snapshot"; data: Snapshot }
  | { type: "turn"; data: Turn }
  | { type: "step"; data: Step }
  | { type: "link"; data: { broker: boolean } }
  | { type: string; data: unknown };
