import { useCallback, useEffect, useRef, useState } from "react";
import { api, Refused } from "@/lib/api";
import { REMOTE_REPLY_EVENT } from "@/lib/useConsole";

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
/** A press shorter than this is a tap: recording continues until the next tap. */
const TAP_MS = 350;
/** A short silent WAV, played on the press itself so the device lets this page speak later. */
const SILENCE =
  "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=";

/** Whether this page may use the microphone here: HTTPS, or this machine itself. */
export const microphoneAllowed = () => typeof window !== "undefined" && window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;

export function useRemoteTalk() {
  const [phase, setPhase] = useState<TalkPhase>("idle");
  const [error, setError] = useState("");
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [held, setHeld] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const pressedAt = useRef(0);
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

  const start = async () => {
    setError("");
    if (!microphoneAllowed()) {
      setError("This page can't use the microphone over plain http.");
      return;
    }
    // Unlock playback on the press itself: a phone lets a page speak only after a touch.
    const element = player.current ?? new Audio();
    player.current = element;
    element.src = SILENCE;
    void element.play().catch(() => undefined);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const media = new MediaRecorder(stream);
      chunks.current = [];
      media.ondataavailable = (event) => event.data.size && chunks.current.push(event.data);
      media.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        const recording = new Blob(chunks.current, { type: media.mimeType || "audio/webm" });
        void send(recording);
      };
      media.start();
      recorder.current = media;
      pressedAt.current = Date.now();
      setPhase("recording");
    } catch {
      setError("The microphone was not allowed. Allow it for this page in the browser's settings.");
      setPhase("idle");
    }
  };

  const stop = () => {
    recorder.current?.stop();
    recorder.current = null;
    setHeld(false);
  };

  /** Press: start. Release after holding: send. A quick tap: keep going until the next tap. */
  const press = () => {
    if (phase === "recording") stop();
    else if (phase === "idle" || phase === "speaking") {
      player.current?.pause();
      void start();
      setHeld(true);
    }
  };
  const release = () => {
    if (phase === "recording" && held && Date.now() - pressedAt.current >= TAP_MS) stop();
    else setHeld(false);
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

  return { phase, error, exchanges, press, release, type, replay: say, tapped: phase === "recording" && !held };
}
