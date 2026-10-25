import * as Tabs from "@radix-ui/react-tabs";
import {
  Activity,
  BrainCircuit,
  CalendarDays,
  CircleAlert,
  Cpu,
  Gauge,
  Mic,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Trash2,
  Wrench,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { Snapshot, Step } from "@/lib/types";
import { ago, celsius, clock, cn, entryWhen, formatReading, modeLabel, sensorLabel, speechAlive } from "@/lib/format";

export type Point = { ts: number; t_in: number; t_pred: number };

const qualityDot: Record<string, string> = {
  ok: "bg-good",
  suspect: "bg-ember",
  faulted: "bg-alert",
};

function Sparkline({ points }: { points: Point[] }) {
  if (points.length < 2) return <div className="h-16 rounded-lg bg-raised/40" />;
  const width = 300;
  const height = 64;
  const values = points.flatMap((point) => [point.t_in, point.t_pred]);
  const low = Math.min(...values) - 0.2;
  const high = Math.max(...values) + 0.2;
  const x = (index: number) => (index / (points.length - 1)) * width;
  const y = (value: number) => height - ((value - low) / (high - low)) * height;
  const line = (key: "t_in" | "t_pred") => points.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point[key]).toFixed(1)}`).join(" ");
  const area = `${line("t_in")} L${width},${height} L0,${height} Z`;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-16 w-full" preserveAspectRatio="none" role="img" aria-label="Indoor temperature and the model's prediction">
      <defs>
        <linearGradient id="fill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="rgb(var(--glacier))" stopOpacity="0.28" />
          <stop offset="1" stopColor="rgb(var(--glacier))" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path d={area} fill="url(#fill)" />
      <path d={line("t_pred")} fill="none" stroke="rgb(var(--muted))" strokeWidth="1.25" strokeDasharray="3 3" vectorEffect="non-scaling-stroke" />
      <path d={line("t_in")} fill="none" stroke="rgb(var(--glacier))" strokeWidth="1.75" vectorEffect="non-scaling-stroke" />
      <circle cx={x(points.length - 1)} cy={y(points[points.length - 1].t_in)} r="3" fill="rgb(var(--glacier))" />
    </svg>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] uppercase tracking-[0.08em] text-muted">{children}</div>;
}

function RoomTab({ state, history, now }: { state: Snapshot; history: Point[]; now: number }) {
  const room = state.room;
  const indoor = room.sensors.temp_01;
  const indoorHealth = room.health.temp_01;
  const onModel = indoorHealth?.quality === "faulted";
  return (
    <div className="space-y-5">
      <section>
        <Label>Indoor</Label>
        <div className="mt-1 flex items-end justify-between">
          <div className="font-mono text-[44px] font-light leading-none text-fg tabular">
            {celsius(onModel ? room.thermal?.t_pred : indoor?.value ?? room.thermal?.t_in)}
          </div>
          <div className="pb-1 text-right text-[12.5px] text-muted">
            {onModel ? <span className="text-ember">from the model — sensor faulted</span> : indoor ? ago(indoor.ts, now) : "waiting"}
          </div>
        </div>
        <div className="mt-3">
          <Sparkline points={history} />
          <div className="mt-1 flex gap-4 text-[11px] text-muted">
            <span className="flex items-center gap-1.5"><span className="h-0.5 w-3 bg-glacier" /> measured</span>
            <span className="flex items-center gap-1.5"><span className="h-0.5 w-3 border-t border-dashed border-muted" /> model's prediction</span>
          </div>
        </div>
      </section>

      <section className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border hairline bg-raised/40 p-3">
          <Label>Holding</Label>
          <div className="mt-1 font-mono text-xl text-fg tabular">{celsius(room.tracking?.tracked_c)}</div>
          <div className="text-[12px] text-muted">
            {room.tracking?.reason || (room.goal ? `asked ${celsius(room.goal.setpoint_c)} by ${room.goal.source}` : "no goal yet")}
          </div>
        </div>
        <div className="rounded-xl border hairline bg-raised/40 p-3">
          <Label>Tariff</Label>
          <div className={cn("mt-1 text-xl font-medium capitalize", room.tariff?.band === "peak" ? "text-ember" : "text-fg")}>
            {room.tariff?.band ?? "--"}
          </div>
          <div className="text-[12px] text-muted">{room.tariff ? `changes at ${clock(room.tariff.next_transition_ts)}` : "unknown"}</div>
        </div>
      </section>

      <section>
        <Label>Sensors</Label>
        <ul className="mt-2 divide-y divide-[rgb(var(--line)/var(--line-alpha))]">
          {Object.keys({ temp_01: 1, hum_01: 1, outdoor_01: 1, pir_01: 1, pwr_01: 1, ...room.sensors }).map((id) => {
            const reading = room.sensors[id];
            const health = room.health[id];
            const quality = health?.quality ?? reading?.quality ?? "ok";
            return (
              <li key={id} className="flex items-center gap-3 py-2 text-[13.5px]">
                <span className={cn("h-2 w-2 rounded-full", reading ? qualityDot[quality] : "bg-muted/40")} />
                <span className="text-fg/90">{sensorLabel[id] ?? id}</span>
                <span className="ml-auto font-mono text-fg tabular">{reading ? formatReading(reading.value, reading.unit) : "--"}</span>
              </li>
            );
          })}
        </ul>
      </section>

      <section className="rounded-xl border hairline bg-raised/40 p-3">
        <div className="flex items-center gap-2"><Zap className="h-4 w-4 text-glacier" /><Label>Air conditioner</Label></div>
        <div className="mt-1.5 flex items-baseline justify-between">
          <span className="text-[15px] font-medium text-fg">{room.actuator?.kind ?? "--"}</span>
          <span className="text-[12px] text-muted">
            {room.actuator ? `ack ${room.actuator.ack.toLowerCase()}${room.actuator.simulated ? " · simulated" : ""}` : "no state yet"}
          </span>
        </div>
      </section>

      <section className="rounded-xl border hairline bg-raised/40 p-3">
        <div className="flex items-center gap-2"><Gauge className="h-4 w-4 text-glacier" /><Label>Thermal model</Label></div>
        <div className="mt-2 grid grid-cols-3 gap-2 text-center">
          <div><div className="font-mono text-[15px] text-fg tabular">{room.thermal ? `${(room.thermal.model_confidence * 100).toFixed(0)}%` : "--"}</div><div className="text-[11px] text-muted">confidence</div></div>
          <div><div className="font-mono text-[15px] text-fg tabular">{room.thermal ? `${room.thermal.residual >= 0 ? "+" : ""}${room.thermal.residual.toFixed(2)}` : "--"}</div><div className="text-[11px] text-muted">residual °C</div></div>
          <div><div className={cn("text-[15px] capitalize", room.thermal?.adaptation === "frozen" ? "text-ember" : "text-fg")}>{room.thermal?.adaptation ?? "--"}</div><div className="text-[11px] text-muted">learning</div></div>
        </div>
      </section>
    </div>
  );
}

const stepIcon: Record<string, typeof Sparkles> = {
  understood: Sparkles, tool: Wrench, result: CalendarDays, safety: ShieldCheck, model: BrainCircuit,
  fault: CircleAlert, mode: Activity, goal: Gauge, command: Zap, operator: RotateCcw, booking: CalendarDays,
};
const toneText: Record<string, string> = { neutral: "text-muted", info: "text-muted", good: "text-good", warn: "text-ember", alert: "text-alert" };

function AgentTab({ steps }: { steps: Step[] }) {
  const recent = useMemo(() => [...steps].reverse().slice(0, 120), [steps]);
  if (!recent.length) return <p className="text-[13px] text-muted">Nothing yet. Every decision any part of the system makes appears here as it happens.</p>;
  return (
    <ol className="relative space-y-0.5 border-l hairline pl-4">
      {recent.map((step) => {
        const Icon = stepIcon[step.kind] ?? Sparkles;
        return (
          <li key={step.id} className="relative py-1.5">
            <span className={cn("absolute -left-[21px] top-3 h-1.5 w-1.5 rounded-full", step.tone === "alert" ? "bg-alert" : step.tone === "warn" ? "bg-ember" : step.tone === "good" ? "bg-good" : "bg-muted/60")} />
            <div className="flex items-start gap-2 text-[13px]">
              <Icon className={cn("mt-0.5 h-3.5 w-3.5 shrink-0", toneText[step.tone])} />
              <div className="min-w-0 flex-1">
                <div className="text-fg/90">{step.text}</div>
                {step.detail && <div className="break-words font-mono text-[11px] leading-snug text-muted">{step.detail}</div>}
              </div>
              <span className="shrink-0 font-mono text-[11px] text-muted tabular">{clock(step.ts)}</span>
            </div>
          </li>
        );
      })}
    </ol>
  );
}

function CalendarTab({ state, onRefresh, onRemove }: { state: Snapshot; onRefresh: () => void; onRemove: (subject: string, starts: string) => void }) {
  const { entries, as_of, note } = state.calendar;
  const days = useMemo(() => {
    const grouped = new Map<string, typeof entries>();
    [...entries].sort((a, b) => a.starts_at.localeCompare(b.starts_at)).forEach((entry) => {
      const { day } = entryWhen(entry.starts_at);
      grouped.set(day, [...(grouped.get(day) ?? []), entry]);
    });
    return [...grouped.entries()];
  }, [entries]);
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <Label>Next 7 days</Label>
        <button onClick={onRefresh} className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] text-muted hover:bg-raised hover:text-fg">
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </button>
      </div>
      {!days.length && <p className="text-[13px] text-muted">{as_of ? note || "Nothing in the calendar." : "Loading the calendar…"}</p>}
      {days.map(([day, list]) => (
        <section key={day}>
          <div className="mb-1.5 text-[12px] font-medium text-fg/80">{day}</div>
          <ul className="space-y-1.5">
            {list.map((entry) => (
              <li key={entry.event_id} className="group flex items-center gap-3 rounded-xl border hairline bg-raised/40 px-3 py-2">
                <span className="font-mono text-[13px] text-glacier tabular">{entryWhen(entry.starts_at).time}</span>
                <span className="min-w-0 flex-1 truncate text-[13.5px] text-fg">{entry.subject}</span>
                <button
                  onClick={() => onRemove(entry.subject, entry.starts_at)}
                  aria-label={`Remove ${entry.subject}`}
                  className="rounded-md p-1 text-muted opacity-0 transition hover:bg-alert/10 hover:text-alert group-hover:opacity-100 focus:opacity-100"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </li>
            ))}
          </ul>
        </section>
      ))}
      <p className="text-[11.5px] text-muted">Your own calendar, kept on this node. Removing an entry needs no confirmation (FR-58).</p>
    </div>
  );
}

function SystemTab({ state, now, onReset, onInject }: { state: Snapshot; now: number; onReset: () => void; onInject: (subject: string, fault: string, magnitude?: number) => void }) {
  const mode = state.room.mode;
  const alive = speechAlive(state.speech, now);
  const injections = Object.values(state.injections);
  return (
    <div className="space-y-5">
      <section className="rounded-xl border hairline bg-raised/40 p-3">
        <Label>Mode</Label>
        <div className={cn("mt-1 text-[16px] font-medium", mode?.mode === "NORMAL" ? "text-good" : "text-ember")}>{mode ? modeLabel[mode.mode] : "--"}</div>
        <div className="text-[12.5px] text-muted">{mode?.reason}</div>
      </section>

      <section>
        <Label>Faults</Label>
        {!state.faults.length && <p className="mt-1.5 text-[13px] text-muted">None active.</p>}
        <ul className="mt-2 space-y-2">
          {state.faults.map((fault) => (
            <li key={fault.fault_id} className="rounded-xl border border-alert/25 bg-alert/5 p-3">
              <div className="flex items-center gap-2 text-[13px] font-medium text-alert">
                <CircleAlert className="h-4 w-4" />
                {fault.detector.replace(/_/g, " ")} on {sensorLabel[fault.subject] ?? fault.subject}
              </div>
              <p className="mt-1 text-[13px] text-fg/90">{state.diagnoses[fault.fault_id]?.user_message ?? "Diagnosing…"}</p>
              <div className="mt-1 text-[11.5px] text-muted">raised {ago(fault.detected_ts, now)} · {fault.mode_impact.replace(/_/g, " ").toLowerCase()}</div>
            </li>
          ))}
        </ul>
      </section>

      <section className="space-y-2">
        <Label>Services</Label>
        <div className="flex items-center gap-3 text-[13.5px]">
          <span className={cn("h-2 w-2 rounded-full", state.broker ? "bg-good" : "bg-alert")} />
          Blackboard
          <span className="ml-auto text-[12px] text-muted">{state.broker ? "connected" : "unreachable"}</span>
        </div>
        <div className="flex items-center gap-3 text-[13.5px]">
          <span className={cn("h-2 w-2 rounded-full", alive ? "bg-good" : "bg-muted/50")} />
          <Mic className="h-3.5 w-3.5 text-muted" /> Voice
          <span className="ml-auto max-w-[60%] truncate text-right text-[12px] text-muted">
            {alive ? `${state.speech?.activity} · ${state.speech?.microphone || "microphone"}` : "not running — typing still works"}
          </span>
        </div>
        <div className="flex items-center gap-3 text-[13.5px]">
          <span className={cn("h-2 w-2 rounded-full", state.tools.length ? "bg-good" : "bg-muted/50")} />
          <Cpu className="h-3.5 w-3.5 text-muted" /> Tools
          <span className="ml-auto text-[12px] text-muted">{state.tools.length ? `${state.tools.length} declared` : "executor not seen"}</span>
        </div>
      </section>

      {state.operator_controls && (
        <section className="rounded-xl border border-dashed hairline p-3">
          <Label>Operator · demonstration</Label>
          <p className="mt-1 text-[12px] text-muted">Faults are applied at the sensor; the detectors still have to find them.</p>
          <div className="mt-2.5 flex flex-wrap gap-1.5">
            {[
              ["Drop the thermometer", "temp_01", "DROPOUT"],
              ["Freeze it at 27°", "temp_01", "STUCK_AT", 27],
              ["Impossible reading", "temp_01", "OUT_OF_RANGE", 999],
              ["Break the AC", "ac", "NO_RESPONSE"],
            ].map(([label, subject, fault, magnitude]) => (
              <button
                key={String(label)}
                onClick={() => onInject(String(subject), String(fault), magnitude as number | undefined)}
                className="rounded-lg border hairline px-2.5 py-1 text-[12px] text-fg hover:border-ember/50 hover:text-ember"
              >
                {label}
              </button>
            ))}
          </div>
          {injections.length > 0 && (
            <div className="mt-2.5 space-y-1">
              {injections.map((injection) => (
                <div key={injection.subject} className="flex items-center justify-between text-[12.5px]">
                  <span className="text-ember">{injection.subject}: {injection.kind.replace(/_/g, " ").toLowerCase()}</span>
                  <button onClick={() => onInject(injection.subject, "NONE")} className="rounded-md px-2 py-0.5 text-muted hover:bg-raised hover:text-fg">Clear</button>
                </div>
              ))}
            </div>
          )}
          <button onClick={onReset} className="mt-3 flex items-center gap-1.5 rounded-lg bg-raised px-3 py-1.5 text-[12.5px] text-fg hover:brightness-125">
            <RotateCcw className="h-3.5 w-3.5" /> Reset after checking
          </button>
        </section>
      )}
    </div>
  );
}

export type PanelTab = "room" | "agent" | "calendar" | "system";

export function ContextPanel(props: {
  state: Snapshot;
  history: Point[];
  tab: PanelTab;
  onTab: (tab: PanelTab) => void;
  onRefreshCalendar: () => void;
  onRemove: (subject: string, starts: string) => void;
  onReset: () => void;
  onInject: (subject: string, fault: string, magnitude?: number) => void;
}) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const tabs: [PanelTab, string][] = [["room", "Room"], ["agent", "Agent"], ["calendar", "Calendar"], ["system", "System"]];
  const alerts = props.state.faults.length;
  return (
    <Tabs.Root value={props.tab} onValueChange={(value) => props.onTab(value as PanelTab)} className="flex h-full flex-col">
      <Tabs.List className="flex gap-1 px-4 pt-4" aria-label="Context">
        {tabs.map(([value, label]) => (
          <Tabs.Trigger
            key={value}
            value={value}
            className="relative rounded-lg px-2.5 py-1.5 text-[13px] text-muted transition hover:text-fg data-[state=active]:bg-raised data-[state=active]:text-fg"
          >
            {label}
            {value === "system" && alerts > 0 && <span className="absolute right-1 top-1 h-1.5 w-1.5 rounded-full bg-alert" />}
          </Tabs.Trigger>
        ))}
      </Tabs.List>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-4 pb-6 pt-4">
        <Tabs.Content value="room"><RoomTab state={props.state} history={props.history} now={now} /></Tabs.Content>
        <Tabs.Content value="agent"><AgentTab steps={props.state.activity} /></Tabs.Content>
        <Tabs.Content value="calendar"><CalendarTab state={props.state} onRefresh={props.onRefreshCalendar} onRemove={props.onRemove} /></Tabs.Content>
        <Tabs.Content value="system"><SystemTab state={props.state} now={now} onReset={props.onReset} onInject={props.onInject} /></Tabs.Content>
      </div>
    </Tabs.Root>
  );
}
