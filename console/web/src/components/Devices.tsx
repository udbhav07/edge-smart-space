import { CloudOff, Fan, Lightbulb, Music, Pause, Play, Plug, Search, Square, Upload, type LucideIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { cn } from "@/lib/format";
import type { Device, DeviceCapability, DeviceParameter } from "@/lib/types";
import { DEVICE_RESULT_EVENT } from "@/lib/useConsole";
import { Button, Card, Pill, Switch, inputClass } from "@/components/ui";

// Devices beyond the thermal loop, drawn from what each one says it can do
// (FR-91). Nothing here is written for a particular fan or lamp: a device
// plugged in tomorrow -- natively, on an ESP32, or behind the MCP bridge --
// gets controls from its manifest. The air conditioner is plant, and is set
// through the climate controls above.

const KIND_ICON: Record<string, LucideIcon> = { fan: Fan, light: Lightbulb, media: Music, other: Plug };

/** Calls a capability, and says the device's own answer when it arrives. */
function useDeviceCall() {
  const pending = useRef(new Set<string>());
  useEffect(() => {
    const onResult = (event: Event) => {
      const answer = (event as CustomEvent<{ invocation_id: string; status: string; message: string }>).detail;
      if (!pending.current.delete(answer.invocation_id)) return;
      if (answer.status === "OK") toast.success(answer.message || "Done");
      else toast.error(answer.message || "The device couldn't do that.");
    };
    window.addEventListener(DEVICE_RESULT_EVENT, onResult);
    return () => window.removeEventListener(DEVICE_RESULT_EVENT, onResult);
  }, []);
  return (deviceId: string, capability: string, args: Record<string, unknown> = {}) =>
    api
      .device(deviceId, capability, args)
      .then((id) => pending.current.add(String(id)))
      .catch((error: Refused) => toast.error(error.message));
}

const label = (name: string) => {
  const text = name.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
};

/** A number with both bounds: a slider, sent when let go. */
function Slider({ parameter, value, onSet, disabled }: { parameter: DeviceParameter; value: number; onSet: (value: number) => void; disabled: boolean }) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <label className="grid gap-1">
      <span className="flex justify-between text-[12px] text-muted"><span>{parameter.description}</span><span className="tabular text-fg">{draft}</span></span>
      <input
        type="range"
        aria-label={parameter.description}
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

/** One capability, as a control its parameters suggest. */
function Control({ device, capability, call }: { device: Device; capability: DeviceCapability; call: ReturnType<typeof useDeviceCall> }) {
  const [text, setText] = useState("");
  const shown = capability.parameters.filter((p) => !p.supplied_by_system);
  const off = !device.online;
  const id = device.device_id;
  if (shown.length === 0) {
    return <Button variant={capability.effect === "read" ? "ghost" : "secondary"} disabled={off} onClick={() => void call(id, capability.name)}>{label(capability.name)}</Button>;
  }
  if (shown.length === 1) {
    const parameter = shown[0];
    // A parameter's value is in the state under its own name, or -- for
    // "set_brightness(level)" -- under what the capability sets.
    const current = device.state[parameter.name] ?? device.state[capability.name.replace(/^set_/, "")];
    if ((parameter.type === "integer" || parameter.type === "number") && parameter.minimum != null && parameter.maximum != null) {
      return <div className="min-w-[220px] flex-1"><Slider parameter={parameter} value={typeof current === "number" ? current : parameter.minimum} disabled={off} onSet={(value) => void call(id, capability.name, { [parameter.name]: value })} /></div>;
    }
    if (parameter.type === "boolean") {
      return <div className="min-w-[220px]"><Switch checked={Boolean(current)} disabled={off} label={label(capability.name)} onChange={(next) => void call(id, capability.name, { [parameter.name]: next })} /></div>;
    }
    if (parameter.choices.length) {
      return (
        <div className="flex flex-wrap gap-1.5">
          {parameter.choices.map((choice) => (
            <button key={choice} disabled={off} onClick={() => void call(id, capability.name, { [parameter.name]: choice })} className={cn("rounded-full px-3 py-1 text-[12.5px] font-medium", current === choice ? "bg-glacier text-white" : "bg-raised text-fg hover:bg-raised/70")}>{choice}</button>
          ))}
        </div>
      );
    }
    return (
      <form onSubmit={(event) => { event.preventDefault(); if (text.trim()) void call(id, capability.name, { [parameter.name]: text.trim() }).then(() => setText("")); }} className="flex min-w-[240px] flex-1 gap-2">
        <input value={text} disabled={off} onChange={(event) => setText(event.target.value)} placeholder={parameter.description} aria-label={parameter.description} className={cn(inputClass, "py-1.5")} />
        <Button type="submit" disabled={off || !text.trim()}>{label(capability.name)}</Button>
      </form>
    );
  }
  return null; // several parameters: asked for in words, through Jarvis
}

function stateLine(device: Device): string {
  const values = Object.entries(device.state).filter(([key, value]) => key !== "max_speed" && value !== "" && value !== null);
  const said = (key: string, value: unknown) => {
    const name = key.replace(/_/g, " ");
    if (typeof value !== "boolean") return `${name} ${value}`;
    if (key === "on") return value ? "on" : "off";
    return value ? name : `not ${name}`;
  };
  return values.map(([key, value]) => said(key, value)).join(" · ");
}

function Music_({ device, call }: { device: Device; call: ReturnType<typeof useDeviceCall> }) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const state = device.state as { playing?: boolean; paused?: boolean; track?: string; volume?: number; tracks?: number };
  const off = !device.online;
  const volume = device.manifest.capabilities.find((c) => c.name === "set_volume")?.parameters[0];
  const upload = async (chosen: File | undefined) => {
    if (!chosen) return;
    setBusy(true);
    try {
      await api.uploadMusic(chosen);
      toast.success(`Uploaded ${chosen.name}; playing it`);
    } catch (error) {
      toast.error((error as Refused).message);
    } finally {
      setBusy(false);
      if (file.current) file.current.value = "";
    }
  };
  return (
    <div className="grid gap-3">
      <div className="flex items-center gap-3">
        <span className={cn("flex h-11 w-11 items-center justify-center rounded-2xl", state.playing ? "bg-glacier text-white" : "bg-raised text-muted")}><Music className="h-5 w-5" /></span>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[15px] font-medium text-fg">{state.track || "Nothing playing"}</div>
          <div className="text-[12px] text-muted">{state.playing ? "Playing" : state.paused ? "Paused" : "Stopped"} · {state.tracks ?? 0} tracks in your library</div>
        </div>
        <div className="flex gap-1.5">
          {state.playing
            ? <Button icon={Pause} disabled={off} onClick={() => void call(device.device_id, "pause")}>Pause</Button>
            : <Button icon={Play} variant="primary" disabled={off} onClick={() => void call(device.device_id, state.paused ? "resume" : "play")}>Play</Button>}
          <Button icon={Square} variant="ghost" disabled={off || (!state.playing && !state.paused)} onClick={() => void call(device.device_id, "stop")}>Stop</Button>
        </div>
      </div>
      <form onSubmit={(event) => { event.preventDefault(); if (query.trim()) void call(device.device_id, "play", { query: query.trim() }).then(() => setQuery("")); }} className="flex gap-2">
        <input value={query} disabled={off} onChange={(event) => setQuery(event.target.value)} placeholder="Play something… an artist, a song, a mood" aria-label="What to play" className={cn(inputClass, "py-1.5")} />
        <Button type="submit" icon={Search} disabled={off || !query.trim()}>Play</Button>
        <input ref={file} type="file" accept=".mp3,.flac,.wav,.ogg,audio/*" className="hidden" onChange={(event) => void upload(event.target.files?.[0])} />
        <Button icon={Upload} disabled={off || busy} onClick={() => file.current?.click()}>{busy ? "Uploading…" : "Upload"}</Button>
      </form>
      {volume && <Slider parameter={{ ...volume, description: "Volume" }} value={state.volume ?? 60} disabled={off} onSet={(value) => void call(device.device_id, "set_volume", { volume: value })} />}
    </div>
  );
}

export function Devices({ devices }: { devices: Device[] }) {
  const call = useDeviceCall();
  const shown = devices.filter((device) => device.manifest.kind !== "ac");
  if (!shown.length) return null;
  return (
    <Card title="Devices" icon={Plug} action={<span className="text-[12px] text-muted">{shown.filter((d) => d.online).length} of {shown.length} online</span>}>
      <ul className="grid gap-4 md:grid-cols-2">
        {shown.map((device) => {
          const Icon = KIND_ICON[device.manifest.kind] ?? Plug;
          const isMusic = device.manifest.kind === "media" && device.manifest.capabilities.some((c) => c.name === "play");
          return (
            <li key={device.device_id} className={cn("grid content-start gap-3 rounded-2xl border hairline p-4", !device.online && "opacity-60", isMusic && "md:col-span-2")}>
              <div className="flex flex-wrap items-center gap-2">
                <Icon className="h-4 w-4 text-muted" />
                <span className="text-[15px] font-semibold text-fg">{device.manifest.name}</span>
                {device.online ? <Pill health="good">Online</Pill> : <Pill health="idle"><CloudOff className="h-3 w-3" /> Offline</Pill>}
                {device.manifest.simulated && <span className="rounded-full border border-dashed border-line/30 px-2 py-px text-[11px] text-muted">simulated</span>}
                {device.manifest.provider.startsWith("mcp:") && <span className="rounded-full bg-raised px-2 py-px text-[11px] text-muted" title="Mounted through the MCP bridge">MCP</span>}
              </div>
              {isMusic ? (
                <Music_ device={device} call={call} />
              ) : (
                <>
                  {stateLine(device) && <div className="text-[12.5px] text-muted">{stateLine(device)}</div>}
                  <div className="flex flex-wrap items-center gap-2">
                    {device.manifest.capabilities.filter((c) => c.device_class !== "plant").map((capability) => (
                      <Control key={capability.name} device={device} capability={capability} call={call} />
                    ))}
                  </div>
                </>
              )}
            </li>
          );
        })}
      </ul>
      <p className="mt-3 text-[11.5px] text-muted">Say it too: “Hey Jarvis, turn on the fan”. Each device here announced itself on the board; plug in another and it appears.</p>
    </Card>
  );
}
