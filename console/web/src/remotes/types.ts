import type { Device } from "@/lib/types";

// What every remote is given (FR-92). A remote draws controls for one device
// and asks it to do things through `call`; it never talks to the device, the
// broker or the API itself, so every remote goes through the executor's gates
// the same way the assistant does.

export interface CallOptions {
  /** Read the device without saying anything: a refresh, not a request. */
  quiet?: boolean;
}

/** Ask the device to run one of its capabilities. */
export type DeviceCall = (capability: string, args?: Record<string, unknown>, options?: CallOptions) => Promise<void>;

export interface RemoteProps {
  device: Device;
  /** "compact" in the grid and on Home; "full" on the device's own page. */
  size: "compact" | "full";
  call: DeviceCall;
}

export interface RemoteSpec {
  component: React.ComponentType<RemoteProps>;
  /** Whether this remote can drive the device; if not, the generic remote does. */
  fits?: (device: Device) => boolean;
  /** Whether its compact remote needs two columns of a multi-column grid. */
  wide?: boolean;
}
