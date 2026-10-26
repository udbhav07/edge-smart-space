import { CircleAlert, ListFilter, RotateCcw, ShieldAlert, ShieldCheck, Thermometer, Unplug, Zap } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { ago, celsius, clock, cn, modeLabel, sensorLabel } from "@/lib/format";
import { useConsole, useNow } from "@/lib/useConsole";
import { Button, Card, Dot, Empty, Label, PageHeader, Pill, inputClass, type Health } from "@/components/ui";

const KINDS: [string, string][] = [
  ["all", "Everything"],
  ["fault", "Faults"],
  ["mode", "Mode"],
  ["safety", "Safety gate"],
  ["goal", "Goals"],
  ["model", "Model calls"],
  ["tool", "Tools"],
  ["result", "Tool results"],
  ["command", "Commands"],
  ["operator", "Operator"],
];
const toneHealth: Record<string, Health> = { good: "good", warn: "warn", alert: "alert", info: "idle", neutral: "idle" };

const INJECTIONS: { label: string; subject: string; fault: string; magnitude?: number; note: string }[] = [
  { label: "Thermometer drops out", subject: "temp_01", fault: "DROPOUT", note: "found in about 15 s" },
  { label: "Impossible reading (999°)", subject: "temp_01", fault: "OUT_OF_RANGE", magnitude: 999, note: "about 10 s" },
  { label: "Thermometer stuck at 27°", subject: "temp_01", fault: "STUCK_AT", magnitude: 27, note: "about 5 min" },
  { label: "Air conditioner dead", subject: "ac", fault: "NO_RESPONSE", note: "about 10 min" },
  { label: "Humidity sensor drops out", subject: "hum_01", fault: "DROPOUT", note: "advisory only" },
];

export function Faults() {
  const { state } = useConsole();
  const now = useNow();
  const [kind, setKind] = useState("all");
  const [reason, setReason] = useState("");
  if (!state) return null;
  const room = state.room;
  const mode = room.mode;
  const guard = (promise: Promise<unknown>, done: string) => promise.then(() => toast.success(done)).catch((error: Refused) => toast.error(error.message));
  const steps = [...state.activity].reverse().filter((step) => kind === "all" || step.kind === kind);
  const injections = Object.values(state.injections);

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Faults & control"
        subtitle="What the detectors have found, what the system is allowed to do because of it, and every decision on the way. Injected faults are applied at the sensor; the detectors still have to find them."
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <Card title="Mode" icon={ShieldCheck}>
          <div className="flex items-center gap-3">
            {mode && <Pill health={mode.mode === "NORMAL" ? "good" : "warn"} pulse={mode.mode !== "NORMAL"}>{modeLabel[mode.mode]}</Pill>}
            <span className="font-mono text-[12px] text-muted">{mode?.mode}</span>
          </div>
          <p className="mt-2 text-[13.5px] text-fg/90">{mode?.reason || "—"}</p>
          {mode && <div className="mt-1 text-[12px] text-muted">since {clock(mode.since_ts)} · {ago(mode.since_ts, now)}</div>}
          <div className="mt-4 grid grid-cols-2 gap-3">
            <div className="rounded-xl bg-raised/50 p-3"><Label>Goal</Label><div className="mt-0.5 font-mono text-[18px] text-fg tabular">{celsius(room.goal?.setpoint_c)}</div><div className="text-[11.5px] text-muted">from {room.goal?.source ?? "—"}</div></div>
            <div className="rounded-xl bg-raised/50 p-3"><Label>Held</Label><div className="mt-0.5 font-mono text-[18px] text-glacier tabular">{celsius(room.tracking?.tracked_c)}</div><div className="truncate text-[11.5px] text-muted">{room.tracking?.reason || `${room.tracking?.band ?? "—"} tariff`}</div></div>
          </div>
        </Card>

        <Card title="Active faults" icon={CircleAlert} className="xl:col-span-2">
          {state.faults.length === 0 ? (
            <Empty icon={ShieldCheck} title="No active faults">Every detector is satisfied with what it is seeing.</Empty>
          ) : (
            <ul className="grid gap-3 md:grid-cols-2">
              {state.faults.map((fault) => {
                const diagnosis = state.diagnoses[fault.fault_id];
                return (
                  <li key={fault.fault_id} className="rounded-xl border border-alert/25 bg-alert/5 p-4">
                    <div className="flex items-center gap-2 text-[13px] font-medium text-alert"><CircleAlert className="h-4 w-4" />{fault.detector.replace(/_/g, " ")}</div>
                    <div className="mt-0.5 text-[12.5px] text-muted">{sensorLabel[fault.subject] ?? fault.subject} · raised {ago(fault.detected_ts, now)} · {fault.mode_impact.replace(/_/g, " ").toLowerCase()}</div>
                    <p className="mt-2 text-[13.5px] text-fg/90">{diagnosis?.user_message ?? "Diagnosing…"}</p>
                    {diagnosis && <div className="mt-1 text-[11.5px] text-muted">hypothesis {diagnosis.primary_hypothesis.replace(/_/g, " ")} · confidence {diagnosis.confidence}</div>}
                  </li>
                );
              })}
            </ul>
          )}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="Sensor health" icon={Thermometer}>
          <ul className="divide-y divide-[rgb(var(--line)/var(--line-alpha))]">
            {["temp_01", "hum_01", "outdoor_01", "pir_01", "pwr_01"].map((id) => {
              const health = room.health[id];
              const quality = health?.quality ?? "unknown";
              return (
                <li key={id} className="flex items-center gap-3 py-2 text-[13.5px]">
                  <Dot health={quality === "ok" ? "good" : quality === "suspect" ? "warn" : quality === "faulted" ? "alert" : "idle"} />
                  <span className="text-fg">{sensorLabel[id]}</span>
                  <span className="ml-auto text-[12px] text-muted">{quality}{health?.active_fault_id ? ` · ${health.active_fault_id}` : ""}</span>
                </li>
              );
            })}
          </ul>
          <div className="mt-3 rounded-xl bg-raised/50 p-3">
            <div className="flex items-center gap-2"><Zap className="h-4 w-4 text-muted" /><Label>Air conditioner</Label></div>
            <div className="mt-1 text-[13.5px] text-fg">{room.actuator?.kind ?? "—"} <span className="text-muted">· ack {room.actuator?.ack.toLowerCase() ?? "—"}{room.actuator?.simulated ? " · simulated" : ""}</span></div>
          </div>
        </Card>

        {state.operator_controls && (
          <Card title="Operator" icon={Unplug} className="xl:col-span-2">
            <p className="text-[12.5px] text-muted">For demonstration. A fault is applied at the sensor exactly as a real one would appear; nothing above the sensor layer is told.</p>
            <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {INJECTIONS.map((injection) => (
                <button
                  key={injection.label}
                  onClick={() => void guard(api.inject(injection.subject, injection.fault, injection.magnitude), `Injected: ${injection.label.toLowerCase()}`)}
                  className="rounded-xl border hairline bg-raised/40 p-3 text-left transition hover:border-ember/40"
                >
                  <div className="text-[13px] font-medium text-fg">{injection.label}</div>
                  <div className="mt-0.5 text-[11.5px] text-muted">{injection.note}</div>
                </button>
              ))}
            </div>
            {injections.length > 0 && (
              <div className="mt-4">
                <Label>Being injected now</Label>
                <ul className="mt-1.5 grid gap-1.5">
                  {injections.map((injection) => (
                    <li key={injection.subject} className="flex items-center gap-3 rounded-lg bg-ember/5 px-3 py-2 text-[13px]">
                      <span className="text-ember">{sensorLabel[injection.subject] ?? injection.subject}: {injection.kind.replace(/_/g, " ").toLowerCase()}</span>
                      <Button variant="ghost" className="ml-auto" onClick={() => void guard(api.inject(injection.subject, "NONE"), `Cleared ${injection.subject}`)}>Clear</Button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            <div className="mt-4 flex flex-wrap items-end gap-2">
              <input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="What did you check? (recorded with the reset)" className={cn(inputClass, "max-w-[420px]")} id="reset-reason" />
              <Button icon={RotateCcw} onClick={() => void guard(api.reset(reason || "checked from the console"), "Reset sent; faults are re-tested")}>Reset after checking</Button>
            </div>
          </Card>
        )}
      </div>

      <Card className="mt-4" title="Decision log" icon={ShieldAlert} action={
        <div className="flex items-center gap-2">
          <ListFilter className="h-3.5 w-3.5 text-muted" />
          <select value={kind} onChange={(event) => setKind(event.target.value)} className="rounded-lg border hairline bg-ink/60 px-2 py-1 text-[12.5px] text-fg" aria-label="Filter decisions">
            {KINDS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
      }>
        <ul className="grid gap-0.5">
          {steps.slice(0, 150).map((step) => (
            <li key={step.id} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-3 rounded-lg px-2 py-1.5 hover:bg-raised/40">
              <span className="mt-1.5"><Dot health={toneHealth[step.tone] ?? "idle"} /></span>
              <div className="min-w-0">
                <div className="text-[13px] text-fg/90">{step.text}</div>
                {step.detail && <div className="truncate font-mono text-[11.5px] text-muted">{step.detail}</div>}
              </div>
              <span className="font-mono text-[11.5px] text-muted tabular">{clock(step.ts)}</span>
            </li>
          ))}
          {steps.length === 0 && <li className="px-2 py-4 text-[13px] text-muted">Nothing of this kind yet.</li>}
        </ul>
      </Card>
    </div>
  );
}
