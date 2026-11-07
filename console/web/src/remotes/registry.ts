import type { Device } from "@/lib/types";
import { acRemote } from "./AcRemote";
import { fanRemote } from "./FanRemote";
import { GenericRemote } from "./GenericRemote";
import { lightRemote } from "./LightRemote";
import { musicRemote } from "./MusicRemote";
import { robotRemote } from "./RobotRemote";
import type { RemoteProps, RemoteSpec } from "./types";

// Which remote a device gets (FR-92). The first that fits wins:
//
//   1. one written for this device, by id;
//   2. one written for its kind;
//   3. the generic remote, drawn from its manifest -- so a device plugged in
//      tomorrow, with nothing written for it, can still be controlled.
//
// To give a new device a remote of its own: write remotes/XRemote.tsx
// exporting a RemoteSpec, and add one line below.

const BY_ID: Record<string, RemoteSpec> = {};

const BY_KIND: Record<string, RemoteSpec> = {
  ac: acRemote,
  fan: fanRemote,
  light: lightRemote,
  media: musicRemote,
  robot: robotRemote,
};

function specFor(device: Device): RemoteSpec | undefined {
  for (const spec of [BY_ID[device.device_id], BY_KIND[device.manifest.kind]]) {
    if (spec && (!spec.fits || spec.fits(device))) return spec;
  }
  return undefined;
}

export function remoteFor(device: Device): React.ComponentType<RemoteProps> {
  return specFor(device)?.component ?? GenericRemote;
}

/** Grid classes for a device's tile: two columns for a wide remote, once the grid has them. */
export function tileSpan(device: Device): string | undefined {
  return specFor(device)?.wide ? "md:col-span-2" : undefined;
}
