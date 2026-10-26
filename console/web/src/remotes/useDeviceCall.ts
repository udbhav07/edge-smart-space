import { useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { DEVICE_RESULT_EVENT } from "@/lib/useConsole";
import type { DeviceCall } from "./types";

/** Calls one device's capabilities, and says the device's own answer when it arrives. */
export function useDeviceCall(deviceId: string): DeviceCall {
  const said = useRef(new Set<string>());
  useEffect(() => {
    const onResult = (event: Event) => {
      const answer = (event as CustomEvent<{ invocation_id: string; status: string; message: string }>).detail;
      if (!said.current.delete(answer.invocation_id)) return;
      if (answer.status === "OK") toast.success(answer.message || "Done");
      else toast.error(answer.message || "The device couldn't do that.");
    };
    window.addEventListener(DEVICE_RESULT_EVENT, onResult);
    return () => window.removeEventListener(DEVICE_RESULT_EVENT, onResult);
  }, []);
  return useCallback(
    (capability, args = {}, options = {}) =>
      api
        .device(deviceId, capability, args)
        .then((id) => {
          if (!options.quiet) said.current.add(String(id));
        })
        .catch((error: Refused) => {
          if (!options.quiet) toast.error(error.message);
        }),
    [deviceId],
  );
}
