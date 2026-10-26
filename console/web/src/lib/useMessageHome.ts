import { useState } from "react";
import { api, Refused } from "@/lib/api";
import { useRecorder } from "@/lib/useRecorder";

// A message home from this device (FR-95): recorded here, played in the room
// in this person's own voice, answered by voice. Who it is from, and which
// messages this device sent, are remembered on this device only.

const SENDER_KEY = "smart-space.message-sender";
const SENT_KEY = "smart-space.messages-sent";
const SENT_KEPT = 20;

function remembered<T>(key: string, fallback: T): T {
  try {
    const value = window.localStorage.getItem(key);
    return value === null ? fallback : (JSON.parse(value) as T);
  } catch {
    return fallback;
  }
}

function remember(key: string, value: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // A private window keeps nothing; the page works regardless.
  }
}

export function useMessageHome() {
  const [sender, setSenderState] = useState<string>(() => remembered(SENDER_KEY, ""));
  const [sent, setSent] = useState<string[]>(() => remembered(SENT_KEY, []));
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");

  const setSender = (name: string) => {
    setSenderState(name);
    remember(SENDER_KEY, name);
  };

  const send = async (recording: Blob) => {
    setSending(true);
    try {
      const { message_id } = await api.sendMessage(sender.trim(), recording);
      const next = [...sent.filter((id) => id !== message_id), message_id].slice(-SENT_KEPT);
      setSent(next);
      remember(SENT_KEY, next);
    } catch (refused) {
      setError((refused as Refused).message);
    } finally {
      setSending(false);
    }
  };

  const microphone = useRecorder((recording) => void send(recording));
  const ready = sender.trim().length > 0;

  const press = () => {
    if (!microphone.recording) setError(ready ? "" : "Write your name first, so home knows who it is from.");
    microphone.press(ready && !sending);
  };

  return {
    sender,
    setSender,
    sent,
    ready,
    recording: microphone.recording,
    tapped: microphone.tapped,
    sending,
    error: error || microphone.error,
    press,
    release: microphone.release,
  };
}
