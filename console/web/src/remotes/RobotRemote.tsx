import { Bath, Battery, BatteryLow, Dog, Home as Dock, Navigation, Square, Tv } from "lucide-react";
import { Button } from "@/components/ui";
import { cn } from "@/lib/format";
import type { Device } from "@/lib/types";
import { capability, has, label } from "./parts";
import type { RemoteProps, RemoteSpec } from "./types";

// The home robot (FR-100): a button per errand, where to send it, and what
// it is doing now. An errand answers at once and runs on, so the robot's
// progress comes from its state, not from the call's answer.

const ERRANDS = [
  { name: "feed_the_dog", title: "Feed the dog", icon: Dog },
  { name: "clean_the_bathroom", title: "Clean the bathroom", icon: Bath },
  { name: "switch_on_tv", title: "Switch on the TV", icon: Tv },
] as const;

function RobotRemote({ device, size, call }: RemoteProps) {
  const off = !device.online;
  const doing = String(device.state.doing ?? "");
  const busy = doing !== "";
  const battery = Number(device.state.battery ?? 0);
  const progress = Number(device.state.progress ?? 0);
  const said = String(device.state.last_said ?? "");
  const rooms = capability(device, "go_to")?.parameters.find((parameter) => parameter.name === "room")?.choices ?? [];
  const BatteryIcon = battery < 25 ? BatteryLow : Battery;
  return (
    <div className={cn("grid gap-3", size === "full" && "gap-5 py-2")}>
      <div className="grid gap-1.5">
        <div className="flex items-center gap-2 text-[13px]">
          <span className={cn("font-medium", busy ? "text-glacier" : "text-muted")}>
            {busy ? doing[0].toUpperCase() + doing.slice(1) : device.state.docked ? "On its dock" : `Waiting in the ${label(String(device.state.room ?? "house")).toLowerCase()}`}
          </span>
          <span className={cn("ml-auto flex items-center gap-1 tabular text-[12px]", battery < 25 ? "text-ember" : "text-muted")} title="Battery">
            <BatteryIcon className="h-3.5 w-3.5" /> {battery}%
          </span>
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-raised" aria-label="Errand progress" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className="h-full rounded-full bg-glacier transition-[width] duration-700" style={{ width: `${busy ? progress : 0}%` }} />
        </div>
        {said && <p className="truncate text-[12px] text-muted" title={said}>{said}</p>}
      </div>
      <div className={cn("grid gap-2", size === "full" ? "grid-cols-3" : "grid-cols-1 sm:grid-cols-3")}>
        {ERRANDS.filter((errand) => has(device, errand.name)).map((errand) => (
          <Button key={errand.name} icon={errand.icon} disabled={off || busy} onClick={() => void call(errand.name)} className="px-3 text-[13px]">
            {errand.title}
          </Button>
        ))}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {rooms.length > 0 && (
          <label className="flex items-center gap-1.5 text-[13px] text-muted">
            <Navigation className="h-3.5 w-3.5" />
            <select
              className="rounded-full bg-raised px-3 py-[6px] text-[13px] text-fg disabled:opacity-40"
              disabled={off || busy}
              value=""
              onChange={(event) => event.target.value && void call("go_to", { room: event.target.value })}
              aria-label="Send the robot to a room"
            >
              <option value="">Send to…</option>
              {rooms.map((room) => <option key={room} value={room}>{label(room)}</option>)}
            </select>
          </label>
        )}
        {has(device, "return_to_dock") && (
          <Button icon={Dock} variant="ghost" disabled={off || Boolean(device.state.docked && !busy)} onClick={() => void call("return_to_dock")} className="px-3 text-[13px]">
            Dock
          </Button>
        )}
        {has(device, "stop") && (
          <Button icon={Square} variant="danger" disabled={off || !busy} onClick={() => void call("stop")} className="ml-auto px-3 text-[13px]">
            Stop
          </Button>
        )}
      </div>
    </div>
  );
}

export const robotRemote: RemoteSpec = {
  component: RobotRemote,
  fits: (device: Device) => has(device, "stop", "status"),
};
