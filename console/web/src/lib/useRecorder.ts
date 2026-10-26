import { useRef, useState } from "react";

// This device's microphone, held to record (FR-94, FR-95): press and hold,
// release to finish; or tap once to start and again to finish. What was
// recorded is handed on whole and kept nowhere here.

/** A press shorter than this is a tap: recording continues until the next tap. */
const TAP_MS = 350;

/** Whether this page may use the microphone here: HTTPS, or this machine itself. */
export const microphoneAllowed = () => typeof window !== "undefined" && window.isSecureContext && !!navigator.mediaDevices?.getUserMedia;

export function useRecorder(onRecorded: (recording: Blob) => void) {
  const [recording, setRecording] = useState(false);
  const [held, setHeld] = useState(false);
  const [error, setError] = useState("");
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const pressedAt = useRef(0);
  const handOn = useRef(onRecorded);
  handOn.current = onRecorded;

  const start = async () => {
    setError("");
    if (!microphoneAllowed()) {
      setError("This page can't use the microphone over plain http.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const media = new MediaRecorder(stream);
      chunks.current = [];
      media.ondataavailable = (event) => event.data.size && chunks.current.push(event.data);
      media.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        setRecording(false);
        handOn.current(new Blob(chunks.current, { type: media.mimeType || "audio/webm" }));
      };
      media.start();
      recorder.current = media;
      pressedAt.current = Date.now();
      setRecording(true);
    } catch {
      setError("The microphone was not allowed. Allow it for this page in the browser's settings.");
    }
  };

  const stop = () => {
    recorder.current?.stop();
    recorder.current = null;
    setHeld(false);
  };

  /** Press: start, if it may. Pressed while recording: finish. */
  const press = (mayStart: boolean) => {
    if (recording) stop();
    else if (mayStart) {
      void start();
      setHeld(true);
    }
  };
  /** Release after holding: finish. Release after a quick tap: keep going until the next tap. */
  const release = () => {
    if (recording && held && Date.now() - pressedAt.current >= TAP_MS) stop();
    else setHeld(false);
  };

  return { recording, tapped: recording && !held, error, press, release };
}
