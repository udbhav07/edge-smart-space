import { ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import type { Device } from "@/lib/types";
import { RemoteTile } from "@/remotes/RemoteTile";

// Home's strip of remotes: the same remotes as the Remotes page, from the
// same registry, so the two cannot drift (FR-91, FR-92). The air conditioner
// is left to the climate controls above it.

export function Devices({ devices }: { devices: Device[] }) {
  const shown = devices.filter((device) => device.manifest.kind !== "ac");
  if (!shown.length) return null;
  return (
    <section aria-label="Devices" className="grid gap-3">
      <div className="flex items-baseline justify-between px-1">
        <h2 className="text-[17px] font-semibold tracking-[-0.02em] text-fg">Devices</h2>
        <Link to="/remotes" className="flex items-center gap-1 text-[13px] text-muted hover:text-fg">All remotes <ArrowRight className="h-3 w-3" /></Link>
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 md:grid-cols-2 xl:grid-cols-3">
        {shown.map((device) => <RemoteTile key={device.device_id} device={device} />)}
      </div>
    </section>
  );
}
