import { useState } from "react";
import { cn } from "@/lib/format";
import type { DeviceCapability } from "@/lib/types";
import { Button, Switch, inputClass } from "@/components/ui";
import { Slider, label, onlyParameter, stateLine, valueOf } from "./parts";
import type { RemoteProps } from "./types";

// The remote every device gets, drawn from its manifest alone (FR-92). A
// device plugged in with no remote written for it is still controllable.

function Control({ capability, ...props }: RemoteProps & { capability: DeviceCapability }) {
  const { device, call } = props;
  const [text, setText] = useState("");
  const off = !device.online;
  const shown = capability.parameters.filter((p) => !p.supplied_by_system);
  if (shown.length === 0) {
    return <Button variant={capability.effect === "read" ? "ghost" : "secondary"} disabled={off} onClick={() => void call(capability.name)}>{label(capability.name)}</Button>;
  }
  const parameter = onlyParameter(capability);
  if (!parameter) return null; // several parameters: asked for in words, through Jarvis
  const current = valueOf(device, capability, parameter);
  if ((parameter.type === "integer" || parameter.type === "number") && parameter.minimum != null && parameter.maximum != null) {
    return (
      <div className="min-w-[220px] flex-1">
        <Slider parameter={parameter} value={typeof current === "number" ? current : parameter.minimum} disabled={off} onSet={(value) => void call(capability.name, { [parameter.name]: value })} />
      </div>
    );
  }
  if (parameter.type === "boolean") {
    return <div className="min-w-[220px]"><Switch checked={Boolean(current)} disabled={off} label={label(capability.name)} onChange={(next) => void call(capability.name, { [parameter.name]: next })} /></div>;
  }
  if (parameter.choices.length) {
    return (
      <div className="flex flex-wrap gap-1.5">
        {parameter.choices.map((choice) => (
          <button key={choice} disabled={off} onClick={() => void call(capability.name, { [parameter.name]: choice })} className={cn("rounded-full px-3 py-1 text-[12.5px] font-medium", current === choice ? "bg-glacier text-white" : "bg-raised text-fg hover:bg-raised/70")}>{choice}</button>
        ))}
      </div>
    );
  }
  return (
    <form onSubmit={(event) => { event.preventDefault(); if (text.trim()) void call(capability.name, { [parameter.name]: text.trim() }).then(() => setText("")); }} className="flex min-w-[240px] flex-1 gap-2">
      <input value={text} disabled={off} onChange={(event) => setText(event.target.value)} placeholder={parameter.description} aria-label={parameter.description} className={cn(inputClass, "py-1.5")} />
      <Button type="submit" disabled={off || !text.trim()}>{label(capability.name)}</Button>
    </form>
  );
}

export function GenericRemote(props: RemoteProps) {
  const { device, size } = props;
  const capabilities = device.manifest.capabilities.filter((c) => c.device_class !== "plant");
  const line = stateLine(device);
  return (
    <div className="grid gap-3">
      {line && <div className={cn("text-muted", size === "full" ? "text-[15px]" : "text-[12.5px]")}>{line}</div>}
      <div className="flex flex-wrap items-center gap-2">
        {capabilities.map((capability) => <Control key={capability.name} capability={capability} {...props} />)}
      </div>
    </div>
  );
}
