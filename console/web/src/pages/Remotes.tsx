import { Plug } from "lucide-react";
import { useConsole } from "@/lib/useConsole";
import { Card, Empty, PageHeader } from "@/components/ui";
import { RemoteTile } from "@/remotes/RemoteTile";

// Every device's remote, usable from here (FR-92). Each is the same remote as
// the device's own page, compact; the name opens the full one.

const ORDER: Record<string, number> = { ac: 0, fan: 1, light: 2, tv: 3, media: 4, robot: 5, other: 6 };

export function Remotes() {
  const { state } = useConsole();
  if (!state) return null;
  const devices = [...(state.devices ?? [])].sort(
    (a, b) => (ORDER[a.manifest.kind] ?? 9) - (ORDER[b.manifest.kind] ?? 9) || a.manifest.name.localeCompare(b.manifest.name),
  );
  const online = devices.filter((device) => device.online).length;
  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Remotes"
        subtitle={`Control every device by hand. ${devices.length ? `${online} of ${devices.length} online.` : ""} Each request goes through the same checks as one Jarvis makes, and the air conditioner always through the safety gate.`}
      />
      {devices.length === 0 ? (
        <Card>
          <Empty icon={Plug} title="No devices yet">
            A device appears here as soon as its service announces itself on the board: the fan, the music player, or any MCP server in the configuration. Start them with python start.py.
          </Empty>
        </Card>
      ) : (
        <div className="grid grid-cols-[minmax(0,1fr)] gap-4 md:grid-cols-2 xl:grid-cols-3">
          {devices.map((device) => <RemoteTile key={device.device_id} device={device} />)}
        </div>
      )}
    </div>
  );
}
