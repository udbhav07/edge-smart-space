import { cn } from "@/lib/format";
import type { Device } from "@/lib/types";
import { PowerButton, boundedNumber, capability, has } from "./parts";
import type { RemoteProps, RemoteSpec } from "./types";

// A fan: one big switch and its speeds as steps, from 1 to the top its
// manifest declares. Speed 0 is the switch's job.

function FanRemote({ device, size, call }: RemoteProps) {
  const speed = boundedNumber(capability(device, "set_speed"))!;
  const on = Boolean(device.state.on);
  const current = on ? Number(device.state.speed ?? 0) : 0;
  const off = !device.online;
  const steps = Array.from({ length: Number(speed.maximum) - Math.max(1, Number(speed.minimum)) + 1 }, (_, n) => Math.max(1, Number(speed.minimum)) + n);
  return (
    <div className={cn("flex items-center gap-4", size === "full" && "flex-col gap-6 py-4")}>
      <PowerButton on={on} disabled={off} size={size} onToggle={() => void call(on ? "turn_off" : "turn_on")} />
      <div className={cn("grid gap-1.5", size === "full" && "justify-items-center")}>
        <span className="text-[12px] text-muted">{on ? `Speed ${current} of ${speed.maximum}` : "Off"}</span>
        <div className="flex gap-1.5" role="group" aria-label="Fan speed">
          {steps.map((step) => (
            <button
              key={step}
              disabled={off}
              aria-pressed={current === step}
              onClick={() => void call("set_speed", { [speed.name]: step })}
              className={cn(
                "rounded-full font-semibold tabular transition active:scale-95 disabled:opacity-40",
                size === "full" ? "h-14 w-14 text-[18px]" : "h-9 w-9 text-[13px]",
                current === step ? "bg-glacier text-white" : current > step ? "bg-glacier/20 text-glacier" : "bg-raised text-fg hover:bg-raised/70",
              )}
            >
              {step}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export const fanRemote: RemoteSpec = {
  component: FanRemote,
  fits: (device: Device) => has(device, "turn_on", "turn_off") && boundedNumber(capability(device, "set_speed")) !== undefined,
};
