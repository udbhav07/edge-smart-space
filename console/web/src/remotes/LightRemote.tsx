import { Sun } from "lucide-react";
import { cn } from "@/lib/format";
import type { Device, DeviceCapability } from "@/lib/types";
import { PowerButton, Slider, boundedNumber, has, valueOf } from "./parts";
import type { RemoteProps, RemoteSpec } from "./types";

// A light: one big switch and, if it has one, the level it sets -- whatever
// the capability is called (set_brightness, set_level, set_dim).

function dimmer(device: Device): DeviceCapability | undefined {
  return device.manifest.capabilities.find((c) => c.device_class !== "plant" && c.name.startsWith("set_") && boundedNumber(c) !== undefined);
}

function LightRemote({ device, size, call }: RemoteProps) {
  const on = Boolean(device.state.on);
  const off = !device.online;
  const level = dimmer(device);
  const parameter = level ? boundedNumber(level) : undefined;
  const current = level && parameter ? valueOf(device, level, parameter) : undefined;
  return (
    <div className={cn("flex items-center gap-4", size === "full" && "flex-col gap-6 py-4")}>
      <PowerButton on={on} disabled={off} size={size} onToggle={() => void call(on ? "turn_off" : "turn_on")} />
      {level && parameter && (
        <div className={cn("min-w-0 flex-1", size === "full" && "w-full max-w-md")}>
          <Slider
            title={level.name.replace(/^set_/, "").replace(/_/g, " ")}
            parameter={parameter}
            value={typeof current === "number" ? current : Number(parameter.maximum)}
            disabled={off}
            onSet={(value) => void call(level.name, { [parameter.name]: value })}
          />
          {size === "full" && <div className="mt-1 flex justify-between text-muted"><Sun className="h-3.5 w-3.5 opacity-50" /><Sun className="h-4 w-4" /></div>}
        </div>
      )}
    </div>
  );
}

export const lightRemote: RemoteSpec = {
  component: LightRemote,
  fits: (device: Device) => has(device, "turn_on", "turn_off"),
};
