import { Bot, CloudOff, Fan, Lightbulb, Music, Plug, Power, Thermometer, Tv, type LucideIcon } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/format";
import type { Device, DeviceCapability, DeviceParameter } from "@/lib/types";
import { Pill } from "@/components/ui";

// The pieces remotes share. A remote written for a new kind of device should
// be mostly these, arranged for that device.

export const KIND_ICON: Record<string, LucideIcon> = { ac: Thermometer, fan: Fan, light: Lightbulb, media: Music, tv: Tv, robot: Bot, other: Plug };

export const iconOf = (device: Device): LucideIcon => KIND_ICON[device.manifest.kind] ?? Plug;

/** A capability the device declares, or undefined. Plant capabilities are never offered. */
export function capability(device: Device, name: string): DeviceCapability | undefined {
  return device.manifest.capabilities.find((c) => c.name === name && c.device_class !== "plant");
}

export const has = (device: Device, ...names: string[]) => names.every((name) => capability(device, name) !== undefined);

/** The one parameter a capability takes, if it takes exactly one. */
export function onlyParameter(found: DeviceCapability | undefined): DeviceParameter | undefined {
  const shown = (found?.parameters ?? []).filter((p) => !p.supplied_by_system);
  return shown.length === 1 ? shown[0] : undefined;
}

/** A bounded number the capability sets: the slider or the steps it is drawn as. */
export function boundedNumber(found: DeviceCapability | undefined): DeviceParameter | undefined {
  const parameter = onlyParameter(found);
  return parameter && (parameter.type === "integer" || parameter.type === "number") && parameter.minimum != null && parameter.maximum != null ? parameter : undefined;
}

/** What a parameter is now: under its own name, or -- "set_brightness(level)" -- under what the capability sets. */
export function valueOf(device: Device, found: DeviceCapability, parameter: DeviceParameter): unknown {
  return device.state[parameter.name] ?? device.state[found.name.replace(/^set_/, "")];
}

export const label = (name: string) => {
  const text = name.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
};

/** The device's state in a line of words. */
export function stateLine(device: Device): string {
  const said = (key: string, value: unknown) => {
    const name = key.replace(/_/g, " ");
    if (typeof value !== "boolean") return `${name} ${value}`;
    if (key === "on") return value ? "on" : "off";
    return value ? name : `not ${name}`;
  };
  return Object.entries(device.state)
    .filter(([key, value]) => !key.startsWith("max_") && value !== "" && value !== null)
    .map(([key, value]) => said(key, value))
    .join(" · ");
}

/** Online or offline, simulated, behind MCP: never colour alone. */
export function Badges({ device }: { device: Device }) {
  return (
    <span className="flex flex-wrap items-center gap-1.5">
      {device.online ? <Pill health="good">Online</Pill> : <Pill health="idle"><CloudOff className="h-3 w-3" /> Offline</Pill>}
      {device.manifest.simulated && <span className="rounded-full border border-dashed border-line/30 px-2 py-px text-[11px] text-muted">simulated</span>}
      {device.manifest.provider.startsWith("mcp:") && <span className="rounded-full bg-raised px-2 py-px text-[11px] text-muted" title="Mounted through the MCP bridge">MCP</span>}
    </span>
  );
}

/** A number with both bounds, sent when let go so a drag is one request, not fifty. */
export function Slider({ parameter, value, onSet, disabled, title }: { parameter: DeviceParameter; value: number; onSet: (value: number) => void; disabled: boolean; title?: string }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <label className="grid gap-1">
      <span className="flex justify-between text-[12px] text-muted"><span>{title ?? parameter.description}</span><span className="tabular text-fg">{draft}</span></span>
      <input
        type="range"
        aria-label={title ?? parameter.description}
        min={parameter.minimum ?? 0}
        max={parameter.maximum ?? 100}
        step={parameter.type === "integer" ? 1 : 0.1}
        value={draft}
        disabled={disabled}
        onChange={(event) => setDraft(Number(event.target.value))}
        onPointerUp={() => onSet(draft)}
        onKeyUp={() => onSet(draft)}
        className="w-full accent-[rgb(var(--glacier))]"
      />
    </label>
  );
}

/** The big round power button of a remote. */
export function PowerButton({ on, disabled, onToggle, size }: { on: boolean; disabled: boolean; onToggle: () => void; size: "compact" | "full" }) {
  return (
    <button
      type="button"
      aria-pressed={on}
      aria-label={on ? "Turn off" : "Turn on"}
      disabled={disabled}
      onClick={onToggle}
      className={cn(
        "flex shrink-0 items-center justify-center rounded-full shadow-sm transition active:scale-95 disabled:opacity-40",
        size === "full" ? "h-24 w-24" : "h-12 w-12",
        on ? "bg-glacier text-white" : "bg-raised text-muted hover:text-fg",
      )}
    >
      <Power className={size === "full" ? "h-10 w-10" : "h-5 w-5"} />
    </button>
  );
}
