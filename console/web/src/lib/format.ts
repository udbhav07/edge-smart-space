import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { SpeechStatus } from "./types";

export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));

export const clock = (epochS: number) =>
  new Date(epochS * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

export const clockSeconds = (epochS: number) =>
  new Date(epochS * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });

export function ago(epochS: number, nowS: number): string {
  const seconds = Math.max(0, Math.round(nowS - epochS));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  return `${Math.round(seconds / 3600)} h ago`;
}

export const celsius = (value: number | undefined | null, digits = 1) =>
  value === undefined || value === null || Number.isNaN(value) ? "--" : `${value.toFixed(digits)}°`;

export const modeLabel: Record<string, string> = {
  INIT: "Starting up",
  NORMAL: "All good",
  DEGRADED_SENSOR: "Running on the model",
  DEGRADED_ACTUATOR: "Air conditioner on hold",
  SAFE_HOLD: "Safe hold",
};

export const sensorLabel: Record<string, string> = {
  temp_01: "Indoor temperature",
  hum_01: "Humidity",
  outdoor_01: "Outside",
  pir_01: "Presence",
  pwr_01: "AC power",
};

/** Whether the speech pipeline has spoken within two heartbeats. */
export function speechAlive(speech: SpeechStatus | null, nowS: number): boolean {
  if (!speech) return false;
  return nowS - speech.ts < speech.heartbeat_s * 2.5;
}

export function formatReading(value: number, unit: string): string {
  if (unit === "bool") return value >= 0.5 ? "Occupied" : "Empty";
  if (unit === "W") return `${Math.round(value)} W`;
  if (unit === "%RH") return `${Math.round(value)}%`;
  return `${value.toFixed(1)}°`;
}

export function entryWhen(iso: string): { day: string; time: string } {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return { day: iso.slice(0, 10), time: iso.slice(11, 16) };
  return {
    day: at.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" }),
    time: at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
  };
}
