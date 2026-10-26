// Every action is a request to publish something the rest of the system
// already understands. The gateway refuses what cannot be done and says why.

export class Refused extends Error {}

async function post<T = unknown>(path: string, body: unknown = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Refused("The console server is not reachable. The room keeps running on its own.");
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Refused(payload.error ?? payload.detail?.[0]?.msg ?? `Request failed (${response.status})`);
  }
  return payload.result as T;
}

export const api = {
  say: (text: string, profile_id: string) => post("/api/say", { text, profile_id }),
  setTarget: (target_c: number, profile_id: string) => post("/api/target", { target_c, profile_id }),
  comfort: (direction: "warmer" | "cooler") => post("/api/comfort", { direction, profile_id: "" }),
  automatic: () => post("/api/automatic", { profile_id: "" }),
  coolingOff: (hours: number) => post("/api/cooling-off", { hours, profile_id: "" }),
  speechSettings: (changes: Partial<import("./types").SpeechSettings>) => post("/api/speech/settings", changes),
  disconnectGoogle: () => post("/api/google/disconnect"),
  device: (deviceId: string, capability: string, args: Record<string, unknown> = {}) =>
    post<string>(`/api/devices/${encodeURIComponent(deviceId)}/${encodeURIComponent(capability)}`, { arguments: args }),
  uploadMusic: async (file: File) => {
    let response: Response;
    try {
      response = await fetch(`/api/music/upload?name=${encodeURIComponent(file.name)}`, {
        method: "POST",
        headers: { "content-type": "application/octet-stream" },
        body: file,
      });
    } catch {
      throw new Refused("The console server is not reachable.");
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) throw new Refused(payload.error ?? `Upload failed (${response.status})`);
    return payload.result as string;
  },
  talk: async (recording: Blob) => {
    let response: Response;
    try {
      response = await fetch("/api/talk", { method: "POST", headers: { "content-type": recording.type || "application/octet-stream" }, body: recording });
    } catch {
      throw new Refused("The console server is not reachable.");
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) throw new Refused(payload.error ?? `Request failed (${response.status})`);
    return payload.result as { utterance_id: string; transcript: string };
  },
  sendMessage: async (sender: string, recording: Blob) => {
    let response: Response;
    try {
      response = await fetch(`/api/messages?sender=${encodeURIComponent(sender)}`, { method: "POST", headers: { "content-type": recording.type || "application/octet-stream" }, body: recording });
    } catch {
      throw new Refused("The console server is not reachable.");
    }
    const payload = await response.json().catch(() => ({}));
    if (!response.ok || payload.ok === false) throw new Refused(payload.error ?? `Sending failed (${response.status})`);
    return payload.result as import("./types").HomeMessage;
  },
  replayMessage: (messageId: string) => post(`/api/messages/${encodeURIComponent(messageId)}/replay`),
  messageAudio: (messageId: string) => `/api/messages/${encodeURIComponent(messageId)}/audio`,
  replyAudio: (messageId: string) => `/api/messages/${encodeURIComponent(messageId)}/reply/audio`,
  talkText: (text: string) => post<{ utterance_id: string; text: string }>("/api/talk/text", { text, profile_id: "" }),
  speak: async (text: string) => {
    const response = await fetch("/api/voice/speak", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text }) });
    if (!response.ok) throw new Refused("The room's voice is not available; the reply is on screen.");
    return response.blob();
  },
  answer: (invocationId: string, yes: boolean) => post(`/api/bookings/${encodeURIComponent(invocationId)}`, { yes }),
  refreshCalendar: () => post("/api/calendar/refresh"),
  removeEntry: (subject: string, starts_at: string) => post("/api/calendar/remove", { subject, starts_at }),
  keepEntry: (event_id: string) => post("/api/calendar/keep", { event_id }),
  settleFact: (profile_id: string, fact: string, keep: boolean) => post("/api/memory/settle", { profile_id, fact, keep }),
  addEntry: (subject: string, starts_at: string, duration_min?: number) =>
    post("/api/calendar/add", { subject, starts_at, duration_min: duration_min ?? null }),
  book: (booking: { kind: string; destination: string; depart_on: string; origin?: string; nights?: number | null }) =>
    post("/api/bookings", { origin: "", nights: null, ...booking }),
  sendMail: (to: string, subject: string, body: string) => post("/api/mail", { to, subject, body }),
  reset: (reason: string) => post("/api/operator/reset", { reason }),
  inject: (subject: string, fault: string, magnitude?: number) =>
    post("/api/operator/inject", { subject, fault, magnitude: magnitude ?? null }),
};
