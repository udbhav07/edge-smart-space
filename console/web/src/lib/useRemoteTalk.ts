import { useCallback, useEffect, useRef, useState } from "react";
import { api, Refused } from "@/lib/api";
import { REMOTE_REPLY_EVENT } from "@/lib/useConsole";
import { useRecorder } from "@/lib/useRecorder";

export { microphoneAllowed } from "@/lib/useRecorder";

// Talking to the room from this device, away from it (FR-94): record on this
// device's microphone, send it, wait for Jarvis's answer to *this* request,
// and say it here -- the room stays silent.

export type TalkPhase = "idle" | "recording" | "sending" | "thinking" | "speaking";

export interface Exchange {
  asked: string;
  answer?: string;
  spoken: boolean;
}

/** How long to wait for Jarvis before saying it did not answer, in ms. */
const REPLY_WAIT_MS = 90_000;
/** A short silent WAV, played on the press itself so the device lets this page speak later. */
const SILENCE =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=";

export function useRemoteTalk() {
  const [phase, setPhase] = useState<Exclude<TalkPhase, "recording">>("idle");
  const [error, setError] = useState("");
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const player = useRef<HTMLAudioElement | null>(null);
  const waiting = useRef<{ id: string; spoken: boolean; timer: number } | null>(null);

  const record = (patch: Partial<Exchange>) =>
    setExchanges((list) => (list.length ? [...list.slice(0, -1), { ...list[list.length - 1], ...patch }] : list));

  const say = useCallback(async (text: string) => {
    try {
      const audio = await api.speak(text);
      const element = player.current ?? new Audio();
      player.current = element;
      element.src = URL.createObjectURL(audio);
      setPhase("speaking");
      element.onended = () => setPhase("idle");
      await element.play();
    } catch {
      setPhase("idle"); // the answer is on screen either way
    }
  }, []);

  // Jarvis's answer, matched to the request this page made.
  useEffect(() => {
    const onReply = (event: Event) => {
      const answer = (event as CustomEvent<{ utterance_id: string; reply: string }>).detail;
      const pending = waiting.current;
      if (!pending || pending.id !== answer.utterance_id) return;
      window.clearTimeout(pending.timer);
      waiting.current = null;
      record({ answer: answer.reply });
      if (pending.spoken) void say(answer.reply);
      else setPhase("idle");
    };
    window.addEventListener(REMOTE_REPLY_EVENT, onReply);
    return () => window.removeEventListener(REMOTE_REPLY_EVENT, onReply);
  }, [say]);

  const await_ = (id: string, spoken: boolean) => {
    setPhase("thinking");
    const timer = window.setTimeout(() => {
      waiting.current = null;
      record({ answer: "No answer came; the reasoning service may be busy. Try again." });
      setPhase("idle");
    }, REPLY_WAIT_MS);
    waiting.current = { id, spoken, timer };
  };

  const send = async (recording: Blob) => {
    setPhase("sending");
    try {
      const { utterance_id, transcript } = await api.talk(recording);
      setExchanges((list) => [...list.slice(-19), { asked: transcript, spoken: true }]);
      await_(utterance_id, true);
    } catch (refused) {
      setError((refused as Refused).message);
      setPhase("idle");
    }
  };

  const microphone = useRecorder((recording) => void send(recording));

  /** Press: start. Release after holding: send. A quick tap: keep going until the next tap. */
  const press = () => {
    const mayStart = phase === "idle" || phase === "speaking";
    if (mayStart && !microphone.recording) {
      setError("");
      // Unlock playback on the press itself: a phone lets a page speak only after a touch.
      const element = player.current ?? new Audio();
      player.current = element;
      element.pause();
      element.src = SILENCE;
      void element.play().catch(() => undefined);
    }
    microphone.press(mayStart);
  };

  const type = async (text: string) => {
    setError("");
    try {
      const { utterance_id } = await api.talkText(text);
      setExchanges((list) => [...list.slice(-19), { asked: text, spoken: false }]);
      await_(utterance_id, false);
    } catch (refused) {
      setError((refused as Refused).message);
    }
  };

  return {
    phase: (microphone.recording ? "recording" : phase) as TalkPhase,
    error: error || microphone.error,
    exchanges,
    press,
    release: microphone.release,
    type,
    replay: say,
    tapped: microphone.tapped,
  };
}
