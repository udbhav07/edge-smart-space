import { Droplets, Minus, Plus, Power, RotateCcw, Snowflake, Sun, Thermometer } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { ago, celsius, cn } from "@/lib/format";
import type { Goal, Snapshot } from "@/lib/types";
import { useNow } from "@/lib/useConsole";
import { Button, Card } from "@/components/ui";

// The room's temperature, as the occupant controls it (FR-80, FR-81). Every
// control is a request through the safety gate: a preference becomes a goal,
// and the validator decides what reaches the air conditioner (FR-45). Shown
// on Home, and as the air conditioner's remote.

const STEP_C = 0.5;
const SEND_AFTER_MS = 700;
const COOLING_OFF_HOURS = [1, 3, 8];

const guard = (promise: Promise<unknown>, done?: string) =>
  promise.then(() => { if (done) toast.success(done); return true; }).catch((error: Refused) => { toast.error(error.message); return false; });

/** Who set the target, in a person's words. */
function setBy(goal: Goal | null, now: number): string {
  if (!goal) return "Waiting for the room to choose a target.";
  const when = goal.ts ? ` · ${ago(goal.ts, now)}` : "";
  if (goal.source === "preference") return `You${when}`;
  if (goal.source === "supervisor") return `AI plan: ${goal.rationale || "planning ahead"}${when}`;
  if (goal.source === "operator") return `Operator${when}`;
  return `Automatic${when}`;
}

function sentence(indoor: number | undefined, target: number | undefined, coolingOff: boolean): string {
  if (coolingOff) return "Cooling is off unless the room gets too warm.";
  if (indoor === undefined || target === undefined) return "Reading the room…";
  const gap = indoor - target;
  if (gap > 1) return `Cooling toward ${celsius(target, 1)}`;
  if (gap < -1) return `Warming toward ${celsius(target, 1)}`;
  return "Comfortable";
}

export function Climate({ state, className = "lg:col-span-2" }: { state: Snapshot; className?: string }) {
  const now = useNow();
  const room = state.room;
  const [low, high] = state.bounds_c ?? [18, 30];
  const indoorFaulted = room.health.temp_01?.quality === "faulted";
  const indoor = indoorFaulted ? room.thermal?.t_pred : room.sensors.temp_01?.value;
  const held = room.goal?.setpoint_c ?? room.tracking?.tracked_c;
  const [draft, setDraft] = useState<number | null>(null);
  const timer = useRef<number | null>(null);
  const yours = room.goal?.source === "preference";
  // Recognised by the request, not the value: the gate climbs to it 2° at a time.
  const coolingOff = yours && (room.goal?.rationale ?? "").startsWith("Cooling off");
  const shown = draft ?? held;

  // A new goal on the board is the answer to the last request: show it, not the draft.
  useEffect(() => setDraft(null), [room.goal?.ts]);
  useEffect(() => () => { if (timer.current !== null) window.clearTimeout(timer.current); }, []);

  const ask = (value: number) => {
    const target = Math.min(high, Math.max(low, Math.round(value / STEP_C) * STEP_C));
    setDraft(target);
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      timer.current = null;
      void guard(api.setTarget(target, ""), `Asked for ${target}°`);
    }, SEND_AFTER_MS);
  };

  return (
    <Card className={className}>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-6 md:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <div className="grid content-start gap-4">
          <div className="flex items-center gap-2 text-[13px] text-muted"><Thermometer className="h-4 w-4" /> Room 1</div>
          <div>
            <div className="display text-[72px] font-semibold leading-none text-fg tabular sm:text-[84px]">{celsius(indoor)}</div>
            <div className="mt-2 text-[17px] font-medium text-fg">{sentence(indoor, held, coolingOff)}</div>
            {indoorFaulted && <div className="mt-1 text-[13px] text-ember">The sensor is faulted; this is the model's estimate.</div>}
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-1 text-[13.5px] text-muted">
            <span className="flex items-center gap-1.5"><Droplets className="h-4 w-4" />{room.sensors.hum_01 ? `${Math.round(room.sensors.hum_01.value)}% humidity` : "--"}</span>
            <span className="flex items-center gap-1.5"><Sun className="h-4 w-4" />{celsius(room.sensors.outdoor_01?.value)} outside</span>
          </div>
        </div>

        <div className="grid content-start gap-4">
          <div>
            <div className="text-[12px] font-medium text-muted">Target</div>
            <div className="mt-1 flex items-center gap-3">
              <button aria-label="Lower the target" onClick={() => shown !== undefined && ask(shown - STEP_C)} disabled={shown === undefined || shown <= low} className="flex h-11 w-11 items-center justify-center rounded-full bg-raised text-fg transition hover:bg-raised/70 active:scale-95 disabled:opacity-40"><Minus className="h-5 w-5" /></button>
              <span className={cn("display w-[110px] text-center text-[44px] font-semibold leading-none tabular", draft !== null ? "text-glacier" : "text-fg")}>{celsius(shown)}</span>
              <button aria-label="Raise the target" onClick={() => shown !== undefined && ask(shown + STEP_C)} disabled={shown === undefined || shown >= high} className="flex h-11 w-11 items-center justify-center rounded-full bg-raised text-fg transition hover:bg-raised/70 active:scale-95 disabled:opacity-40"><Plus className="h-5 w-5" /></button>
            </div>
            <input
              type="range"
              aria-label="Target temperature"
              min={low}
              max={high}
              step={STEP_C}
              value={shown ?? (low + high) / 2}
              onChange={(event) => ask(Number(event.target.value))}
              className="mt-3 w-full accent-[rgb(var(--glacier))]"
            />
            <div className="flex justify-between text-[11.5px] text-muted tabular"><span>{low}°</span><span>{high}°</span></div>
            <div className="mt-1 text-[12.5px] text-muted">{draft !== null ? "Sending…" : `Set by ${setBy(room.goal, now)}`}</div>
          </div>

          <div className="flex flex-wrap gap-2">
            {Object.entries(state.presets ?? {}).map(([name, value]) => (
              <button key={name} onClick={() => void guard(api.setTarget(value, ""), `${name}: asked for ${value}°`)} className={cn("rounded-full px-3.5 py-1.5 text-[13px] font-medium transition", held === value && yours ? "bg-glacier text-white" : "bg-raised text-fg hover:bg-raised/70")}>
                {name} <span className="tabular opacity-70">{value}°</span>
              </button>
            ))}
          </div>

          <div className="flex flex-wrap gap-2">
            <Button icon={Snowflake} onClick={() => void guard(api.comfort("cooler"), "A little cooler")}>Cooler</Button>
            <Button icon={Sun} onClick={() => void guard(api.comfort("warmer"), "A little warmer")}>Warmer</Button>
            {yours && <Button variant="ghost" icon={RotateCcw} onClick={() => void guard(api.automatic(), "Back to automatic")}>Back to automatic</Button>}
          </div>

          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            <Power className="h-4 w-4 text-muted" />
            <span className="text-muted">Cooling off for</span>
            {COOLING_OFF_HOURS.map((hours) => (
              <button key={hours} onClick={() => void guard(api.coolingOff(hours), `Cooling off for ${hours} h`)} className="rounded-full bg-raised px-3 py-1 text-[12.5px] font-medium text-fg hover:bg-raised/70">{hours} h</button>
            ))}
          </div>
          {coolingOff && <p className="text-[12px] text-muted">The target climbs to {state.cooling_off_c}° in safe steps; the air conditioner stays off unless the room passes it. Press Back to automatic to end it early.</p>}
        </div>
      </div>
    </Card>
  );
}
