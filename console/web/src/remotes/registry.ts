import type { Device } from "@/lib/types";
import { acRemote } from "./AcRemote";
import { fanRemote } from "./FanRemote";
import { GenericRemote } from "./GenericRemote";
import { lightRemote } from "./LightRemote";
import { musicRemote } from "./MusicRemote";
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
};

export function remoteFor(device: Device): React.ComponentType<RemoteProps> {
  for (const spec of [BY_ID[device.device_id], BY_KIND[device.manifest.kind]]) {
    if (spec && (!spec.fits || spec.fits(device))) return spec.component;
  }
  return GenericRemote;
}
