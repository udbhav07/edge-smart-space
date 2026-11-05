import { ArrowUpRight, TriangleAlert } from "lucide-react";
import { Link } from "react-router-dom";
import { cn } from "@/lib/format";
import type { Device } from "@/lib/types";
import { Badges, iconOf } from "./parts";
import { remoteFor } from "./registry";
import { useDeviceCall } from "./useDeviceCall";

/** One device's compact remote, as the grid and Home show it. */
export function RemoteTile({ device, className }: { device: Device; className?: string }) {
  const Remote = remoteFor(device);
  const call = useDeviceCall(device.device_id);
  const Icon = iconOf(device);
  const failed = device.last_result && device.last_result.status !== "OK" ? device.last_result : null;
  return (
    <section className={cn("card grid min-w-0 content-start gap-3 rounded-[22px] p-5", !device.online && "opacity-60", className)} aria-label={`${device.manifest.name} remote`}>
      <header className="flex flex-wrap items-center gap-2">
        <Icon className="h-4 w-4 text-muted" />
        <Link to={`/remotes/${encodeURIComponent(device.device_id)}`} className="group flex items-center gap-1 text-[16px] font-semibold text-fg hover:text-glacier">
          {device.manifest.name}
          <ArrowUpRight className="h-3.5 w-3.5 text-muted group-hover:text-glacier" />
        </Link>
        <span className="ml-auto"><Badges device={device} /></span>
      </header>
      <Remote device={device} size="compact" call={call} />
      {failed && (
        <p className="flex items-center gap-1.5 text-[12px] text-ember" title={failed.message}>
          <TriangleAlert className="h-3.5 w-3.5 shrink-0" /> <span className="truncate">Last: {inWords(failed)}</span>
        </p>
      )}
    </section>
  );
}

/**
 * A failure as a person would put it. A call refused for how it was written --
 * "'robot_go_to' does not take ['speed']" -- was Jarvis asking the wrong way,
 * and the exact refusal stays in the tooltip for whoever wants it.
 */
function inWords(failed: { tool: string; status: string; message: string }): string {
  if (failed.status === "BAD_ARGUMENTS" && failed.message.startsWith(`'${failed.tool}'`)) return "Jarvis asked it the wrong way; nothing changed.";
  if (failed.status === "UNKNOWN_TOOL") return "It was asked for something it can't do.";
  return failed.message || failed.status.toLowerCase();
}
