import { ArrowLeft, History, ListChecks, Plug, Radio } from "lucide-react";
import { useEffect, useRef } from "react";
import { Link, useParams } from "react-router-dom";
import { ago, clock, cn } from "@/lib/format";
import type { Device } from "@/lib/types";
import { useConsole, useNow } from "@/lib/useConsole";
import { Card, Dot, Empty, PageHeader, type Health } from "@/components/ui";
import { Badges, iconOf, label } from "@/remotes/parts";
import { remoteFor } from "@/remotes/registry";
import { useDeviceCall } from "@/remotes/useDeviceCall";

// One device's full remote, and what there is to know about it (FR-92).

const toneHealth: Record<string, Health> = { good: "good", warn: "warn", alert: "alert", info: "idle", neutral: "idle" };

/** The capability that reads the device without changing it, if it has one that takes nothing. */
function reader(device: Device): string | undefined {
  return device.manifest.capabilities.find((c) => c.effect === "read" && c.device_class !== "plant" && c.parameters.every((p) => !p.required))?.name;
}

export function Remote() {
  const { deviceId = "" } = useParams();
  const { state } = useConsole();
  const now = useNow();
  const call = useDeviceCall(deviceId);
  const device = state?.devices?.find((d) => d.device_id === deviceId);
  const refreshed = useRef(false);

  // A device behind MCP knows its state only once asked: read it quietly on
  // arrival, so the remote shows what is true rather than nothing.
  useEffect(() => {
    if (!device || refreshed.current || !device.online || Object.keys(device.state).length) return;
    const read = reader(device);
    if (read) {
      refreshed.current = true;
      void call(read, {}, { quiet: true });
    }
  }, [device, call]);

  if (!state) return null;
  if (!device) {
    return (
      <div className="mx-auto max-w-[900px] px-4 py-6 sm:px-8">
        <Card>
          <Empty icon={Plug} title={`No device called “${deviceId}”`}>
            It may never have announced itself, or its service may not be running. <Link to="/remotes" className="text-glacier hover:underline">All remotes</Link>
          </Empty>
        </Card>
      </div>
    );
  }

  const Remote = remoteFor(device);
  const Icon = iconOf(device);
  const activity = state.activity.filter((step) => step.text.startsWith(`${device.device_id}_`)).slice(-12).reverse();
  return (
    <div className="mx-auto max-w-[1000px] px-4 py-6 sm:px-8">
      <Link to="/remotes" className="mb-2 inline-flex items-center gap-1 text-[13px] text-muted hover:text-fg"><ArrowLeft className="h-3.5 w-3.5" /> All remotes</Link>
      <PageHeader
        title={device.manifest.name}
        subtitle={<span className="flex flex-wrap items-center gap-2"><Icon className="h-4 w-4" />{label(device.manifest.kind)}{device.manifest.aliases.length ? ` · also “${device.manifest.aliases.join("”, “")}”` : ""}</span>}
        actions={<Badges device={device} />}
      />
      <div className="grid gap-4">
        <Card className={cn(!device.online && "opacity-60")}>
          {!device.online && <p className="mb-3 text-[13px] text-ember">Offline: its service is not running or lost its connection. The controls come back when it does.</p>}
          <Remote device={device} size="full" call={call} />
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          <Card title="Now" icon={Radio}>
            {Object.keys(device.state).length ? (
              <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-[13.5px]">
                {Object.entries(device.state).map(([key, value]) => (
                  <div key={key} className="contents"><dt className="text-muted">{key.replace(/_/g, " ")}</dt><dd className="text-fg tabular">{String(value)}</dd></div>
                ))}
              </dl>
            ) : (
              <p className="text-[13px] text-muted">Nothing reported yet.</p>
            )}
            <p className="mt-3 text-[12px] text-muted">
              {device.state_ts ? `Reported ${ago(device.state_ts, now)}` : ""}
              {device.last_result ? ` · last answer: “${device.last_result.message || device.last_result.status}”` : ""}
            </p>
          </Card>
          <Card title="What it can do" icon={ListChecks}>
            <ul className="grid gap-2">
              {device.manifest.capabilities.map((capability) => (
                <li key={capability.name} className="text-[13px]">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-fg">{capability.name}</span>
                    <span className={cn("rounded-full px-1.5 text-[11px]", capability.device_class === "plant" ? "bg-alert/10 text-alert" : "bg-raised text-muted")}>{capability.device_class}</span>
                    <span className="rounded-full bg-raised px-1.5 text-[11px] text-muted">{capability.effect}</span>
                  </div>
                  <p className="text-[12.5px] text-muted">
                    {capability.purpose}
                    {capability.parameters.filter((p) => !p.supplied_by_system).map((p) => ` · ${p.name}${p.minimum != null ? ` ${p.minimum}–${p.maximum}` : ""}`).join("")}
                  </p>
                </li>
              ))}
            </ul>
            {device.manifest.capabilities.some((c) => c.device_class === "plant") && (
              <p className="mt-3 text-[12px] text-muted">Plant capabilities are reached only through the temperature controls and the safety gate.</p>
            )}
          </Card>
        </div>

        <Card title="Recent activity" icon={History}>
          {activity.length ? (
            <ul className="grid gap-1.5">
              {activity.map((step) => (
                <li key={step.id} className="flex items-start gap-2.5 text-[13px]">
                  <span className="mt-1.5"><Dot health={toneHealth[step.tone] ?? "idle"} /></span>
                  <span className="min-w-0 flex-1"><span className="text-fg/90">{step.text}</span>{step.detail && <span className="text-muted"> — {step.detail}</span>}</span>
                  <span className="shrink-0 font-mono text-[11.5px] text-muted tabular">{clock(step.ts)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-[13px] text-muted">Nothing asked of it since this console started.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
