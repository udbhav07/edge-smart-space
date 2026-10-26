import { AnimatePresence, motion } from "framer-motion";
import { speechAlive } from "@/lib/format";
import type { Snapshot } from "@/lib/types";
import { useConsole, useNow } from "@/lib/useConsole";
import { Orb, type Presence } from "@/components/Orb";

// What the room is doing with your voice, on every page (FR-93). "Hey
// Jarvis" lights it at once; it stays through thinking and the spoken reply,
// and lights again when the room listens for your answer. Overheard speech --
// nobody talking to the assistant -- never lights it.

/** How long after a spoken request without a reply it still reads "thinking". */
const THINKING_S = 60;

interface Shown {
  presence: Presence;
  label: string;
  detail?: string;
}

function shown(state: Snapshot | null, now: number): Shown | null {
  const speech = state?.speech;
  if (!state || !speech || !speechAlive(speech, now)) return null;
  switch (speech.activity) {
    case "listening":
      return { presence: "listening", label: "Listening…", detail: "Go ahead" };
    case "transcribing":
      return { presence: "transcribing", label: "Got it…" };
    case "understanding":
      return { presence: "thinking", label: "Thinking…", detail: speech.heard ? `“${speech.heard}”` : undefined };
    case "speaking":
      return { presence: "speaking", label: "Speaking" };
  }
  // Between the request and the spoken reply, the assistant is working.
  const last = state.conversation[state.conversation.length - 1];
  if (last?.role === "occupant" && last.source === "voice" && now - last.ts < THINKING_S) {
    return { presence: "thinking", label: "Thinking…", detail: `“${last.text}”` };
  }
  return null;
}

export function VoiceIndicator() {
  const { state } = useConsole();
  const now = useNow();
  const current = shown(state, now);
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-6 z-40 flex justify-center px-4" aria-live="polite">
      <AnimatePresence>
        {current && (
          <motion.div
            key="voice"
            initial={{ opacity: 0, y: 16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.97 }}
            transition={{ type: "spring", stiffness: 420, damping: 32 }}
            className="glass flex max-w-[min(560px,100%)] items-center gap-3 rounded-full border hairline py-2 pl-2 pr-5 shadow-[0_18px_50px_-18px_rgb(0_0_0/0.45)]"
            role="status"
          >
            <Orb presence={current.presence} size={36} />
            <div className="min-w-0 leading-tight">
              <div className="text-[14px] font-semibold text-fg">{current.label}</div>
              {current.detail && <div className="truncate text-[12px] text-muted">{current.detail}</div>}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/** The sidebar orb follows the same: alive while the room is listening to you. */
export function voicePresence(state: Snapshot | null, now: number): Presence | null {
  return shown(state, now)?.presence ?? null;
}
