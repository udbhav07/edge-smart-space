import {
  Activity,
  BadgeCheck,
  Bot,
  BrainCircuit,
  CalendarClock,
  CircleAlert,
  Cpu,
  Droplets,
  FlaskConical,
  Gauge,
  Mic,
  MessageSquare,
  Power,
  RotateCcw,
  ShieldCheck,
  Sigma,
  Sparkles,
  Sun,
  Target,
  Thermometer,
  UserRound,
  Wrench,
  Zap,
  type LucideIcon,
} from "lucide-react";
import { ago, clockSeconds, cn, modeLabel, sensorLabel } from "./format";
import { matches } from "./services";

// How each topic reads to a person. Every message on the blackboard has a
// schema (src/common/schemas.py), so each gets a one-line summary for lists
// and a detail card for the board, written in the room's terms rather than
// as the JSON it travels as. The raw message stays one click away.

/* eslint-disable @typescript-eslint/no-explicit-any */
type Payload = Record<string, any>;
type Tone = "good" | "warn" | "alert" | "info" | "muted";

export interface Renderer {
  pattern: string;
  icon: LucideIcon | ((topic: string) => LucideIcon);
  title: (topic: string) => string;
  summary: (payload: Payload, topic: string) => React.ReactNode;
  detail: (payload: Payload, topic: string, now: number) => React.ReactNode;
}

// --- shared pieces -----------------------------------------------------------

const toneClass: Record<Tone, string> = {
  good: "bg-good/10 text-good",
  warn: "bg-ember/10 text-ember",
  alert: "bg-alert/10 text-alert",
  info: "bg-glacier/10 text-glacier",
  muted: "bg-raised text-muted",
};

export function Chip({ tone = "muted", children }: { tone?: Tone; children: React.ReactNode }) {
  return <span className={cn("inline-flex items-center whitespace-nowrap rounded-full px-2 py-0.5 text-[11.5px] font-medium", toneClass[tone])}>{children}</span>;
}

function Big({ value, unit, tone }: { value: React.ReactNode; unit?: string; tone?: Tone }) {
  return (
    <div className="flex items-baseline gap-1.5">
      <span className={cn("font-mono text-[34px] font-light leading-none tabular", tone === "alert" ? "text-alert" : tone === "warn" ? "text-ember" : "text-fg")}>{value}</span>
      {unit && <span className="text-[14px] text-muted">{unit}</span>}
    </div>
  );
}

function Fields({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(110px,auto)_minmax(0,1fr)] gap-x-4 gap-y-1.5 text-[13px]">
      {rows.filter(([, value]) => value !== undefined && value !== null && value !== "").map(([label, value]) => (
        <div key={label} className="contents">
          <dt className="text-muted">{label}</dt>
          <dd className="min-w-0 break-words text-fg">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Meter({ value, tone = "info" }: { value: number; tone?: Tone }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-raised">
        <div className={cn("h-full rounded-full", tone === "good" ? "bg-good" : tone === "warn" ? "bg-ember" : tone === "alert" ? "bg-alert" : "bg-glacier")} style={{ width: `${pct}%` }} />
      </div>
      <span className="w-10 text-right font-mono text-[12px] text-muted tabular">{pct.toFixed(0)}%</span>
    </div>
  );
}

function Quote({ children }: { children: React.ReactNode }) {
  return <blockquote className="border-l-2 border-glacier/50 pl-3 text-[14.5px] leading-relaxed text-fg">{children}</blockquote>;
}

function Stack({ children }: { children: React.ReactNode }) {
  return <div className="grid gap-4">{children}</div>;
}

const when = (ts: number | null | undefined, now: number) => (ts ? `${clockSeconds(ts)} · ${ago(ts, now)}` : "—");
const num = (value: unknown, digits = 2) => (typeof value === "number" ? value.toFixed(digits) : "—");
const signed = (value: number, digits = 3) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const words = (value: unknown) => String(value ?? "").replace(/_/g, " ").toLowerCase();
const sentence = (value: unknown) => { const text = words(value); return text.charAt(0).toUpperCase() + text.slice(1); };
const segment = (topic: string, index: number) => topic.split("/")[index] ?? "";
const nameOf = (id: string) => sensorLabel[id] ?? (id === "ac" ? "Air conditioner" : id);
const qualityTone = (quality: string): Tone => (quality === "ok" ? "good" : quality === "suspect" ? "warn" : "alert");
const statusTone = (status: string): Tone => (status === "OK" ? "good" : status === "CONFIRMATION_REQUIRED" ? "info" : "alert");
const verdictTone = (verdict: string): Tone => (verdict === "ACCEPTED" ? "good" : verdict === "CLAMPED" ? "warn" : "alert");
const modeTone = (mode: string): Tone => (mode === "NORMAL" ? "good" : mode === "SAFE_HOLD" ? "alert" : "warn");

function reading(value: number, unit: string): string {
  if (unit === "bool") return value >= 0.5 ? "Occupied" : "Empty";
  if (unit === "W") return `${Math.round(value)} W`;
  if (unit === "%RH") return `${value.toFixed(1)} %RH`;
  return `${value.toFixed(1)} °C`;
}

const sensorIcon = (topic: string): LucideIcon => {
  const id = segment(topic, 2);
  return id.startsWith("hum") ? Droplets : id.startsWith("pir") ? UserRound : id.startsWith("pwr") ? Zap : id.startsWith("outdoor") ? Sun : Thermometer;
};

/** An argument as a person would write it: ISO local times become "Sun 25 Oct, 15:00". */
function argument(value: unknown): string {
  if (typeof value === "string" && /^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2})?)?$/.test(value)) {
    const date = new Date(value.length === 10 ? `${value}T00:00:00` : value);
    if (!Number.isNaN(date.getTime())) {
      const day = date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
      return value.length === 10 || value.endsWith("T00:00:00") ? day : `${day}, ${date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })}`;
    }
  }
  return String(value);
}

function argumentsList(args: Payload | undefined): string {
  return Object.entries(args ?? {}).filter(([key, value]) => key !== "profile_id" && value !== null && value !== "").map(([key, value]) => `${key.replace(/_/g, " ")} ${argument(value)}`).join(" · ");
}

/** A command or goal put to the safety gate, in one phrase. */
function ruling(value: Payload | undefined): string {
  if (!value) return "—";
  const setpoint = typeof value.setpoint_c === "number" ? `${value.setpoint_c.toFixed(1)}°` : "";
  if (value.kind) return setpoint ? `${value.kind} to ${setpoint}` : value.kind;
  return setpoint ? `goal ${setpoint}` : argumentsList(value);
}

// --- one renderer per topic ---------------------------------------------------

export const RENDERERS: Renderer[] = [
  {
    pattern: "space/sensor/+/state",
    icon: sensorIcon,
    title: (topic) => `${nameOf(segment(topic, 2))} · reading`,
    summary: (p) => <><span className="font-mono text-fg tabular">{reading(p.value, p.unit)}</span> <Chip tone={qualityTone(p.quality)}>{p.quality}</Chip></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-3"><Big value={reading(p.value, p.unit).replace(/ .*$/, "")} unit={p.unit === "bool" ? undefined : p.unit === "C" ? "°C" : p.unit} /><Chip tone={qualityTone(p.quality)}>{p.quality}</Chip></div>
        <Fields rows={[["Sensor", `${nameOf(p.sensor_id)} (${p.sensor_id})`], ["Measured", when(p.ts, now)], ["Quality", p.quality === "ok" ? "Trusted" : p.quality === "suspect" ? "Outside what the instrument can report; held back" : "Faulted"]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/sensor/+/health",
    icon: ShieldCheck,
    title: (topic) => `${nameOf(segment(topic, 2))} · health`,
    summary: (p) => <><Chip tone={qualityTone(p.quality)}>{p.quality}</Chip>{p.active_fault_id && <span className="text-alert"> {p.active_fault_id}</span>}</>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-2"><Chip tone={qualityTone(p.quality)}>{p.quality === "ok" ? "Trusted" : p.quality}</Chip><span className="text-[14px] text-fg">{nameOf(p.sensor_id)}</span></div>
        <Fields rows={[["Judged", when(p.ts, now)], ["Last reading", when(p.last_reading_ts, now)], ["Fault", p.active_fault_id ?? "none"]]} />
        <p className="text-[12.5px] text-muted">Published by the detectors whenever their judgement of this sensor changes. The estimator stops learning from a sensor marked faulted (FR-29).</p>
      </Stack>
    ),
  },
  {
    pattern: "space/estimate/thermal",
    icon: Gauge,
    title: () => "Thermal estimate",
    summary: (p) => <><span className="font-mono text-fg tabular">{num(p.t_in, 1)}°</span> <span className="text-muted">predicted</span> <span className="font-mono tabular">{num(p.t_pred, 2)}°</span> <span className="text-muted">residual</span> <span className="font-mono tabular">{signed(p.residual, 2)}</span> <Chip tone={p.adaptation === "active" ? "good" : "warn"}>{p.adaptation === "active" ? "learning" : "frozen"}</Chip></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="grid grid-cols-2 gap-4">
          <div><div className="text-[11px] uppercase tracking-[0.08em] text-muted">Measured</div><Big value={num(p.t_in, 2)} unit="°C" /></div>
          <div><div className="text-[11px] uppercase tracking-[0.08em] text-muted">Predicted</div><Big value={num(p.t_pred, 2)} unit="°C" /></div>
        </div>
        <Fields rows={[
          ["Residual", <span className="font-mono">{signed(p.residual)} °C (σ {num(p.residual_sigma, 3)})</span>],
          ["Over the horizon", p.horizon_residual !== null && p.horizon_residual !== undefined ? <span className="font-mono">{signed(p.horizon_residual)} °C (σ {num(p.horizon_residual_sigma, 3)})</span> : "—"],
          ["Learning", p.adaptation === "active" ? "Active" : "Frozen: a sensor it learns from is faulted, or the mode forbids it"],
          ["At", when(p.ts, now)],
        ]} />
        <div><div className="mb-1 text-[11px] uppercase tracking-[0.08em] text-muted">Model confidence</div><Meter value={p.model_confidence} tone={p.model_confidence > 0.8 ? "good" : "warn"} /></div>
      </Stack>
    ),
  },
  {
    pattern: "space/estimate/coefficients",
    icon: Sigma,
    title: () => "Model coefficients",
    summary: (p) => <span className="font-mono tabular">a2 {num(p.a2, 5)} · a3 {num(p.a3, 4)} · a4 {num(p.a4, 4)}</span>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[["a1", p.a1, 5, "kept per step"], ["a2", p.a2, 5, p.a2 > 0 ? `τ ≈ ${(5 / p.a2 / 60).toFixed(0)} min` : ""], ["a3", p.a3, 4, `${(p.a3 * 12).toFixed(2)} °C/min cooling`], ["a4", p.a4, 4, "occupant gain"]].map(([name, value, digits, note]) => (
            <div key={name as string} className="rounded-xl bg-raised/50 p-3">
              <div className="font-mono text-[12px] text-muted">{name}</div>
              <div className="font-mono text-[18px] text-fg tabular">{(value as number).toFixed(digits as number)}</div>
              <div className="text-[11.5px] text-muted">{note}</div>
            </div>
          ))}
        </div>
        <Fields rows={[["Uncertainty", <span className="font-mono">trace(P) {num(p.trace_p, 4)}</span>], ["Samples", (p.samples_since_reset as number).toLocaleString()], ["At", when(p.ts, now)]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/actuator/+/command",
    icon: Power,
    title: (topic) => `${nameOf(segment(topic, 2))} · command`,
    summary: (p) => <><Chip tone={p.kind === "COOL" ? "info" : p.kind === "OFF" ? "muted" : "muted"}>{p.kind}</Chip>{p.setpoint_c !== null && <span className="font-mono"> to {num(p.setpoint_c, 1)}°</span>}</>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-3"><Big value={p.kind} />{p.setpoint_c !== null && <span className="font-mono text-[15px] text-muted">→ {num(p.setpoint_c, 1)} °C</span>}</div>
        <p className="text-[13px] text-muted">
          {p.kind === "MAINTAIN" ? "Keep doing what you are doing: sent every tick while nothing needs to change." : p.kind === "HOLD" ? "Hold: actuation is suspended by the current mode." : p.kind === "COOL" ? "Run the compressor. Re-asserted every 60 s, because an infrared command can be lost." : "Switch the compressor off."}
        </p>
        <Fields rows={[["Actuator", nameOf(p.actuator_id)], ["Sent", when(p.ts, now)]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/actuator/+/state",
    icon: Zap,
    title: (topic) => `${nameOf(segment(topic, 2))} · state`,
    summary: (p) => <><Chip tone={p.kind === "COOL" ? "info" : "muted"}>{p.kind}</Chip> <span className="text-muted">ack</span> <Chip tone={p.ack === "ACKNOWLEDGED" ? "good" : p.ack === "FAILED" ? "alert" : "muted"}>{words(p.ack)}</Chip>{p.simulated && <span className="text-muted"> · simulated</span>}</>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex flex-wrap items-center gap-2"><Big value={p.kind} /><Chip tone={p.ack === "ACKNOWLEDGED" ? "good" : p.ack === "FAILED" ? "alert" : "muted"}>{words(p.ack)}</Chip>{p.simulated && <Chip>simulated</Chip>}</div>
        <Fields rows={[
          ["Acknowledgement", p.ack === "UNKNOWN" ? "Not known: an infrared remote never answers, and the meter is still in its grace period" : p.ack === "ACKNOWLEDGED" ? "Confirmed by what the unit draws from the mains" : "The unit's draw disagrees with the command"],
          ["Last command", when(p.last_command_ts, now)],
          ["Reported", when(p.ts, now)],
        ]} />
      </Stack>
    ),
  },
  {
    pattern: "space/fault/+",
    icon: CircleAlert,
    title: (topic) => `Fault ${segment(topic, 2)}`,
    summary: (p) => <><Chip tone="alert">{words(p.detector)}</Chip> <span className="text-fg">{nameOf(p.subject)}</span> <span className="text-muted">→ {words(p.mode_impact)}</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex flex-wrap items-center gap-2"><Chip tone="alert">{sentence(p.detector)}</Chip><span className="text-[15px] text-fg">{nameOf(p.subject)}</span></div>
        <Fields rows={[["Class", p.class], ["Raised", when(p.detected_ts, now)], ["Mode it implies", modeLabel[p.mode_impact] ?? words(p.mode_impact)]]} />
        <div><div className="mb-1 text-[11px] uppercase tracking-[0.08em] text-muted">Confidence</div><Meter value={p.confidence} tone="alert" /></div>
        <div>
          <div className="mb-1.5 text-[11px] uppercase tracking-[0.08em] text-muted">Evidence</div>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {Object.entries(p.evidence ?? {}).map(([key, value]) => (
              <div key={key} className="rounded-lg bg-raised/50 px-2.5 py-1.5">
                <div className="text-[11px] text-muted">{key.replace(/_/g, " ")}</div>
                <div className="font-mono text-[13px] text-fg tabular">{key.endsWith("_ts") ? clockSeconds(value as number) : Number(value).toPrecision(4)}</div>
              </div>
            ))}
          </div>
        </div>
      </Stack>
    ),
  },
  {
    pattern: "space/diagnosis/+",
    icon: BrainCircuit,
    title: (topic) => `Diagnosis of ${segment(topic, 2)}`,
    summary: (p) => <span className="text-fg">{p.user_message}</span>,
    detail: (p, _t, now) => (
      <Stack>
        <Quote>{p.user_message}</Quote>
        <Fields rows={[
          ["Hypothesis", sentence(p.primary_hypothesis)],
          ["Confidence", <Chip tone={p.confidence === "high" ? "good" : p.confidence === "medium" ? "warn" : "muted"}>{p.confidence}</Chip>],
          ["Recommends", `${modeLabel[p.recommended_mode] ?? p.recommended_mode}${p.recommendation_legal ? "" : " (not a legal transition; the detectors' mode stands)"}`],
          ["Evidence used", (p.supporting_evidence ?? []).map(words).join(", ")],
          ["Written", `${when(p.ts, now)}${p.generic ? " · fixed wording, the model gave nothing usable" : ""}`],
        ]} />
      </Stack>
    ),
  },
  {
    pattern: "space/system/mode",
    icon: Activity,
    title: () => "System mode",
    summary: (p) => <><Chip tone={modeTone(p.mode)}>{modeLabel[p.mode] ?? p.mode}</Chip> <span className="text-muted">{p.reason}</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-2"><Big value={modeLabel[p.mode] ?? p.mode} tone={modeTone(p.mode) === "good" ? undefined : modeTone(p.mode)} /></div>
        <Fields rows={[["Why", p.reason || "—"], ["Since", when(p.since_ts, now)], ["Active faults", (p.active_fault_ids ?? []).length ? p.active_fault_ids.join(", ") : "none"]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/system/reset",
    icon: RotateCcw,
    title: () => "Operator reset",
    summary: (p) => <><span className="text-fg">{p.requester}</span> <span className="text-muted">{p.reason ? `: ${p.reason}` : ""}</span></>,
    detail: (p, _t, now) => <Fields rows={[["By", p.requester], ["Reason", p.reason || "none given"], ["At", when(p.ts, now)]]} />,
  },
  {
    pattern: "space/inject/+",
    icon: FlaskConical,
    title: (topic) => `Injection · ${nameOf(segment(topic, 2))}`,
    summary: (p) => (p.kind === "NONE" ? <Chip tone="good">cleared</Chip> : <><Chip tone="warn">{words(p.kind)}</Chip>{p.magnitude ? <span className="font-mono"> {p.magnitude}</span> : null} <span className="text-muted">by {p.requester}</span></>),
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-2">{p.kind === "NONE" ? <Chip tone="good">Nothing injected</Chip> : <Chip tone="warn">{sentence(p.kind)}</Chip>}<span className="text-[15px] text-fg">{nameOf(p.subject)}</span></div>
        <Fields rows={[["Magnitude", p.kind === "NONE" ? "—" : String(p.magnitude)], ["Requested by", p.requester], ["At", when(p.ts, now)]]} />
        <p className="text-[12.5px] text-muted">Applied at the sensor exactly as a real fault would appear. Retained, so it answers what is being injected right now.</p>
      </Stack>
    ),
  },
  {
    pattern: "space/goal/+",
    icon: Target,
    title: (topic) => (segment(topic, 2) === "active" ? "Goal in force" : "Goal proposed"),
    summary: (p) => <><span className="font-mono text-fg tabular">{num(p.setpoint_c, 1)}°</span> <span className="text-muted">from {p.source}</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-3"><Big value={num(p.setpoint_c, 1)} unit="°C" /><Chip tone="info">{p.source}</Chip></div>
        {p.rationale && <Quote>{p.rationale}</Quote>}
        <Fields rows={[["Mode", modeLabel[p.mode] ?? p.mode], ["Set", when(p.ts, now)], ["Lapses", when(p.expires_ts, now).replace(" ago", " ago (expired)")]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/control/tracking",
    icon: Target,
    title: () => "Setpoint held",
    summary: (p) => <><span className="text-muted">holding</span> <span className="font-mono text-glacier tabular">{num(p.tracked_c, 1)}°</span>{p.tracked_c !== p.setpoint_c && <span className="text-muted"> (asked {num(p.setpoint_c, 1)}°)</span>} <Chip tone={p.band === "peak" ? "warn" : "muted"}>{p.band}</Chip></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="grid grid-cols-2 gap-4">
          <div><div className="text-[11px] uppercase tracking-[0.08em] text-muted">Asked</div><Big value={num(p.setpoint_c, 1)} unit="°C" /></div>
          <div><div className="text-[11px] uppercase tracking-[0.08em] text-muted">Held</div><Big value={num(p.tracked_c, 1)} unit="°C" /></div>
        </div>
        <Fields rows={[["Tariff", p.band], ["Why they differ", p.reason || "They do not"], ["Since", when(p.ts, now)]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/tariff/state",
    icon: CalendarClock,
    title: () => "Tariff",
    summary: (p) => <><Chip tone={p.band === "peak" ? "warn" : "good"}>{p.band}</Chip> <span className="text-muted">until {clockSeconds(p.next_transition_ts).slice(0, 5)}</span></>,
    detail: (p, _t, now) => <Stack><Big value={p.band === "peak" ? "Peak" : "Normal"} tone={p.band === "peak" ? "warn" : undefined} /><Fields rows={[["Changes at", when(p.next_transition_ts, now).replace(/ · .*/, "")], ["Published", when(p.ts, now)]]} /><p className="text-[12.5px] text-muted">At peak the comfort band shifts up by a configured offset, and the validator still bounds it (FR-16).</p></Stack>,
  },
  {
    pattern: "space/speech/status",
    icon: Mic,
    title: () => "Speech",
    summary: (p) => <><Chip tone={p.activity === "waiting" ? "muted" : "info"}>{p.activity}</Chip>{p.heard && <span className="text-fg"> “{p.heard}”</span>}</>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-2"><Chip tone={p.activity === "waiting" ? "muted" : "info"}>{p.activity === "waiting" ? `Waiting for “${(p.wake_word || "").replace(/_/g, " ")}”` : sentence(p.activity)}</Chip></div>
        {p.heard && <Quote>{p.heard}</Quote>}
        <Fields rows={[["Microphone", p.microphone || "—"], ["Heartbeat", `every ${p.heartbeat_s} s`], ["Last said", when(p.ts, now)]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/context/preference",
    icon: Sparkles,
    title: () => "Understood",
    summary: (p) => <><Chip tone="info">{p.intent}</Chip>{p.target_c !== null && p.target_c !== undefined && <span className="font-mono"> {num(p.target_c, 1)}°</span>}{p.comfort !== "unchanged" && <span className="text-muted"> {p.comfort}</span>} <span className="text-fg">“{p.transcript || p.rationale}”</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <Quote>{p.transcript || p.rationale}</Quote>
        <Fields rows={[
          ["Understood as", <span className="flex flex-wrap gap-1.5"><Chip tone="info">{p.intent}</Chip>{p.subject && <Chip>{p.subject}</Chip>}{p.comfort !== "unchanged" && <Chip tone="info">{p.comfort}</Chip>}</span>],
          ["Temperature named", p.target_c !== null && p.target_c !== undefined ? `${num(p.target_c, 1)} °C` : "none"],
          ["Speaker", p.profile_id || "guest"],
          ["At", when(p.ts, now)],
        ]} />
        <p className="text-[12.5px] text-muted">A preference, not a command: the goal manager and the safety validator decide what is applied (FR-53).</p>
      </Stack>
    ),
  },
  {
    pattern: "space/context/reply",
    icon: Bot,
    title: () => "Jarvis replied",
    summary: (p) => <span className="text-fg">{p.reply}</span>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="rounded-xl bg-raised/50 p-3 text-[13px] text-muted">“{p.transcript}”</div>
        <Quote>{p.reply}</Quote>
        <Fields rows={[["Intent", p.intent], ["Tools called", (p.invocation_ids ?? []).length ? p.invocation_ids.join(", ") : "none"], ["Waiting for a yes", p.awaiting_confirmation || "no"], ["At", when(p.ts, now)]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/context/utterance",
    icon: MessageSquare,
    title: () => "Typed in the console",
    summary: (p) => <span className="text-fg">“{p.text}”</span>,
    detail: (p, _t, now) => <Stack><Quote>{p.text}</Quote><Fields rows={[["Speaking as", p.profile_id || "guest"], ["At", when(p.ts, now)]]} /></Stack>,
  },
  {
    pattern: "space/context/profile/+",
    icon: UserRound,
    title: (topic) => `Profile · ${segment(topic, 3)}`,
    summary: (p) => <><span className="text-fg">{p.display_name || "Guest"}</span> <span className="text-muted">· {(p.facts ?? []).length} facts · {(p.recent ?? []).length} recent</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="text-[18px] font-semibold text-fg">{p.display_name || "Guest (unrecognised voices)"}</div>
        {Object.keys(p.preferences ?? {}).length > 0 && <div className="flex flex-wrap gap-1.5">{Object.entries(p.preferences).map(([key, value]) => <Chip key={key} tone="info">{key.replace(/_/g, " ")}: {String(value)}</Chip>)}</div>}
        {(p.facts ?? []).length > 0 && <ul className="list-disc space-y-0.5 pl-5 text-[13.5px] text-fg">{p.facts.map((fact: string) => <li key={fact}>{fact}</li>)}</ul>}
        <Fields rows={[["Recent exchanges", String((p.recent ?? []).length)], ["Published", when(p.ts, now)]]} />
        <p className="text-[12.5px] text-muted">Personalises replies only; recognising a voice unlocks nothing (FR-52).</p>
      </Stack>
    ),
  },
  {
    pattern: "space/assist/proposed",
    icon: Wrench,
    title: () => "Tool requested",
    summary: (p) => <><Chip tone="info">{p.tool}</Chip> <span className="text-muted">{argumentsList(p.arguments)}</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-2"><Chip tone="info">{p.tool}</Chip><span className="text-[13px] text-muted">by {words(p.requester)}</span></div>
        <Fields rows={Object.entries(p.arguments ?? {}).filter(([key]) => key !== "profile_id").map(([key, value]) => [sentence(key), argument(value)] as [string, React.ReactNode])} />
        {p.rationale && <div className="rounded-xl bg-raised/50 p-3 text-[13px] text-muted">“{p.rationale}”</div>}
        <Fields rows={[["Invocation", <span className="font-mono">{p.invocation_id}</span>], ["Requested", when(p.ts, now)], ["Expires", when(p.expires_ts, now)]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/assist/confirmed",
    icon: BadgeCheck,
    title: () => "Confirmed by the occupant",
    summary: (p) => <><Chip tone="good">{p.tool}</Chip> <span className="text-muted">{argumentsList(p.arguments)}</span></>,
    detail: (p, _t, now) => <Stack><div className="flex items-center gap-2"><Chip tone="good">Yes, go ahead</Chip><Chip tone="info">{p.tool}</Chip></div><Fields rows={[["What", argumentsList(p.arguments)], ["Invocation", <span className="font-mono">{p.invocation_id}</span>], ["At", when(p.ts, now)]]} /><p className="text-[12.5px] text-muted">The occupant's yes is the topic itself: only a person confirming republishes the invocation here (FR-54).</p></Stack>,
  },
  {
    pattern: "space/assist/result",
    icon: Wrench,
    title: () => "Tool result",
    summary: (p) => <><Chip tone={statusTone(p.status)}>{words(p.status)}</Chip> <span className="text-muted">{p.tool}</span> <span className="text-fg">{p.message}</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex flex-wrap items-center gap-2"><Chip tone={statusTone(p.status)}>{sentence(p.status)}</Chip><Chip>{p.tool}</Chip>{p.simulated && <Chip tone="warn">mock</Chip>}</div>
        {p.message && <Quote>{p.message}</Quote>}
        {Object.keys(p.detail ?? {}).length > 0 && <Fields rows={Object.entries(p.detail).map(([key, value]) => [key.replace(/_/g, " "), String(value)] as [string, React.ReactNode])} />}
        <Fields rows={[["Provider", p.provider || "—"], ["Invocation", <span className="font-mono">{p.invocation_id}</span>], ["At", when(p.ts, now)]]} />
      </Stack>
    ),
  },
  {
    pattern: "space/assist/catalogue",
    icon: Cpu,
    title: () => "Tool catalogue",
    summary: (p) => <span className="text-muted">{(p.tools ?? []).map((tool: Payload) => tool.name).join(" · ")}</span>,
    detail: (p) => (
      <ul className="grid gap-2">
        {(p.tools ?? []).map((tool: Payload) => (
          <li key={tool.name} className="rounded-xl bg-raised/50 p-3">
            <div className="flex items-center gap-2"><span className="font-mono text-[13px] text-fg">{tool.name}</span><Chip tone={tool.effect === "commit" ? "warn" : tool.effect === "write" ? "info" : "muted"}>{tool.effect}</Chip></div>
            <p className="mt-1 text-[12.5px] text-muted">{tool.purpose}</p>
          </li>
        ))}
      </ul>
    ),
  },
  {
    pattern: "space/audit/validation",
    icon: ShieldCheck,
    title: () => "Safety gate",
    summary: (p) => {
      const what = ruling(p.proposed);
      const applied = ruling(p.applied);
      return <><Chip tone={verdictTone(p.verdict)}>{words(p.verdict)}</Chip> <span className="font-mono text-fg">{what}</span>{what !== applied && <span className="font-mono text-ember"> → {applied}</span>}{p.reason !== "NONE" && <span className="text-muted"> {words(p.reason)}</span>}</>;
    },
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex items-center gap-2"><Chip tone={verdictTone(p.verdict)}>{sentence(p.verdict)}</Chip>{p.reason !== "NONE" && <Chip tone="warn">{words(p.reason)}</Chip>}</div>
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl bg-raised/50 p-3"><div className="text-[11px] uppercase tracking-[0.08em] text-muted">Proposed</div><div className="mt-1 font-mono text-[15px] text-fg">{ruling(p.proposed)}</div></div>
          <div className="rounded-xl bg-raised/50 p-3"><div className="text-[11px] uppercase tracking-[0.08em] text-muted">Applied</div><div className={cn("mt-1 font-mono text-[15px]", ruling(p.proposed) === ruling(p.applied) ? "text-fg" : "text-ember")}>{ruling(p.applied)}</div></div>
        </div>
        <Fields rows={[["Ruled", when(p.ts, now)]]} />
        <p className="text-[12.5px] text-muted">Every command and every goal passes here: bounds, rate of change, compressor dwell, command frequency (FR-13, FR-14).</p>
      </Stack>
    ),
  },
  {
    pattern: "space/audit/reasoning",
    icon: BrainCircuit,
    title: () => "Model call",
    summary: (p) => <><Chip tone="info">{p.caller}</Chip> <span className="font-mono tabular">{num(p.latency_s, 1)} s</span> <span className="text-muted">· {p.prompt_tokens + p.completion_tokens} tokens · {p.trigger}</span></>,
    detail: (p, _t, now) => (
      <Stack>
        <div className="flex flex-wrap items-center gap-2"><Chip tone="info">{sentence(p.caller)}</Chip><span className="font-mono text-[13px] text-muted">{p.model}</span></div>
        <div className="grid grid-cols-3 gap-3">
          <div className="rounded-xl bg-raised/50 p-3"><div className="text-[11px] text-muted">Latency</div><div className="font-mono text-[17px] text-fg tabular">{num(p.latency_s, 1)} s</div></div>
          <div className="rounded-xl bg-raised/50 p-3"><div className="text-[11px] text-muted">Prompt</div><div className="font-mono text-[17px] text-fg tabular">{p.prompt_tokens}</div></div>
          <div className="rounded-xl bg-raised/50 p-3"><div className="text-[11px] text-muted">Answer</div><div className="font-mono text-[17px] text-fg tabular">{p.completion_tokens}</div></div>
        </div>
        <Fields rows={[["Trigger", p.trigger], ["Tools called", (p.tool_calls ?? []).join(", ") || "none"], ["Verdict", p.verdict], ["Applied", p.applied || "—"], ["At", when(p.ts, now)]]} />
        {p.raw_output && <details className="rounded-xl bg-raised/40 p-3"><summary className="cursor-pointer text-[12.5px] text-muted">What the model said</summary><pre className="mt-2 whitespace-pre-wrap font-mono text-[12px] text-fg/85">{p.raw_output}</pre></details>}
      </Stack>
    ),
  },
];

const FALLBACK: Renderer = {
  pattern: "#",
  icon: Activity,
  title: (topic) => topic.split("/").slice(1).join(" · "),
  summary: (p) => <span className="text-muted">{typeof p === "object" && p ? Object.keys(p).filter((key) => key !== "ts").slice(0, 5).join(", ") : String(p)}</span>,
  detail: (p, _t, now) => (
    typeof p === "object" && p
      ? <Fields rows={Object.entries(p).map(([key, value]) => [key.replace(/_/g, " "), key.endsWith("ts") && typeof value === "number" ? when(value, now) : typeof value === "object" ? JSON.stringify(value) : String(value)] as [string, React.ReactNode])} />
      : <span className="text-muted">{String(p)}</span>
  ),
};

export function rendererFor(topic: string): Renderer {
  return RENDERERS.find((renderer) => matches(renderer.pattern, topic)) ?? FALLBACK;
}

export function iconFor(topic: string): LucideIcon {
  // A lucide icon is a forwardRef object; a plain function picks one per topic.
  const icon = rendererFor(topic).icon;
  return typeof icon === "function" ? (icon as (t: string) => LucideIcon)(topic) : icon;
}

/** A renderer never takes the page down: a payload it does not expect falls back. */
export function safely(render: () => React.ReactNode, fallback: React.ReactNode): React.ReactNode {
  try {
    return render();
  } catch {
    return fallback;
  }
}
