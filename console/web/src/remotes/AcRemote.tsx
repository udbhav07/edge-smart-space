import { Minus, Plus } from "lucide-react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { celsius } from "@/lib/format";
import { useConsole } from "@/lib/useConsole";
import { Climate } from "@/components/Climate";
import type { RemoteProps, RemoteSpec } from "./types";

// The air conditioner. Its capabilities are plant, so this remote never calls
// the device: every control is a preference the safety gate decides on, the
// same as Home's climate controls (FR-45, FR-88).

const STEP_C = 0.5;

function AcRemote({ device, size }: RemoteProps) {
  const { state } = useConsole();
  if (!state) return null;
  if (size === "full") return <Climate state={state} className="" />;
  const room = state.room;
  const indoor = room.health.temp_01?.quality === "faulted" ? room.thermal?.t_pred : room.sensors.temp_01?.value;
  const target = room.goal?.setpoint_c ?? room.tracking?.tracked_c;
  const [low, high] = state.bounds_c ?? [18, 30];
  const ask = (value: number) =>
    api
      .setTarget(Math.min(high, Math.max(low, value)), "")
      .then(() => toast.success(`Asked for ${Math.min(high, Math.max(low, value))}°; the safety gate decides`))
      .catch((error: Refused) => toast.error(error.message));
  const command = String(device.state.command ?? "").toLowerCase();
  return (
    <div className="flex items-center gap-4">
      <div>
        <div className="display text-[32px] font-semibold leading-none text-fg tabular">{celsius(indoor)}</div>
        <div className="mt-1 text-[12px] text-muted">{command ? `unit: ${command}` : "room"}</div>
      </div>
      <div className="ml-auto flex items-center gap-2">
        <button aria-label="Lower the target" disabled={target === undefined || target <= low} onClick={() => target !== undefined && void ask(target - STEP_C)} className="flex h-9 w-9 items-center justify-center rounded-full bg-raised text-fg hover:bg-raised/70 disabled:opacity-40"><Minus className="h-4 w-4" /></button>
        <span className="w-16 text-center text-[20px] font-semibold text-glacier tabular">{celsius(target)}</span>
        <button aria-label="Raise the target" disabled={target === undefined || target >= high} onClick={() => target !== undefined && void ask(target + STEP_C)} className="flex h-9 w-9 items-center justify-center rounded-full bg-raised text-fg hover:bg-raised/70 disabled:opacity-40"><Plus className="h-4 w-4" /></button>
      </div>
    </div>
  );
}

export const acRemote: RemoteSpec = { component: AcRemote };
