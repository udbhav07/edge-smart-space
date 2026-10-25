import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { speechAlive } from "@/lib/format";
import type { Snapshot } from "@/lib/types";
import { useConsole, useNow, type LinkState } from "@/lib/useConsole";
import { Composer } from "@/components/Composer";
import { Conversation } from "@/components/Conversation";
import { Orb, type Presence } from "@/components/Orb";

// Talking to the room without speaking aloud. Typed words take exactly the
// path speech does: the same extraction, the same gates (FR-57).

function presenceOf(state: Snapshot | null, link: LinkState, now: number): Presence {
  if (!state || link === "offline") return "offline";
  const speech = state.speech;
  if (speech && speechAlive(speech, now)) {
    if (speech.activity === "listening") return "listening";
    if (speech.activity === "transcribing") return "transcribing";
    if (speech.activity === "understanding") return "thinking";
  }
  const last = state.conversation[state.conversation.length - 1];
  if (last?.role === "occupant" && now - last.ts < 90) return "thinking";
  return "idle";
}

export function Chat() {
  const { state, link } = useConsole();
  const now = useNow();
  const [speaker, setSpeaker] = useState(() => {
    try { return localStorage.getItem("speaker") ?? ""; } catch { return ""; }
  });
  const composer = useRef<HTMLTextAreaElement>(null);
  const names = useMemo(
    () => Object.fromEntries((state?.profiles ?? []).map((profile) => [profile.profile_id, profile.display_name || profile.profile_id])),
    [state?.profiles],
  );
  if (!state) return null;
  const presence = presenceOf(state, link, now);
  const voice = speechAlive(state.speech, now);
  const heard = voice && state.speech?.activity === "understanding" ? state.speech.heard : "";
  const chooseSpeaker = (id: string) => {
    setSpeaker(id);
    try { localStorage.setItem("speaker", id); } catch { /* private window */ }
  };
  const guard = (promise: Promise<unknown>) => promise.then(() => true).catch((error: Refused) => { toast.error(error.message); return false; });

  const empty = (
    <div className="mx-auto flex h-full max-w-[680px] flex-col items-center justify-center px-6 pb-8 text-center">
      <Orb presence={presence} size={96} />
      <h1 className="mt-7 text-balance text-[24px] font-semibold tracking-tight text-fg">Ask Jarvis anything about the room.</h1>
      <p className="mt-2 max-w-[52ch] text-balance text-[14px] text-muted">
        Type here when you'd rather not speak aloud. A temperature, your calendar, a trip or an email: anything that commits you to someone else waits for your yes.
      </p>
    </div>
  );

  return (
    <div className="flex h-full flex-col">
      <div className="hairline-b flex items-center gap-3 px-4 py-3 sm:px-8">
        <Orb presence={presence} size={26} />
        <div className="min-w-0">
          <div className="text-[14px] font-medium text-fg">Jarvis</div>
          <div className="truncate text-[12px] text-muted">
            {presence === "listening" ? "Listening…" : presence === "thinking" ? "Thinking…" : voice ? "Voice is on; say “Hey Jarvis”, or type below" : "Voice is off; typing works the same"}
          </div>
        </div>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
        <Conversation
          turns={state.conversation}
          steps={state.activity}
          waiting={state.waiting}
          names={names}
          presence={presence}
          heard={heard}
          onAnswer={(id, yes) => void guard(api.answer(id, yes)).then((ok) => ok && toast.success(yes ? "Confirmed" : "Declined; nothing was done"))}
          empty={empty}
        />
      </div>
      <Composer
        ref={composer}
        disabled={link === "offline" || !state.broker}
        disabledReason={link === "offline" ? "The console server is not reachable" : "Waiting for the blackboard…"}
        profiles={state.profiles}
        speaker={speaker}
        onSpeaker={chooseSpeaker}
        onSend={(text) => guard(api.say(text, speaker))}
        onTarget={(value) => void guard(api.setTarget(value, speaker))}
        goal={state.room.goal?.setpoint_c ?? null}
        held={state.room.tracking?.tracked_c ?? null}
        voiceHint={voice ? "Voice on — say “Hey Jarvis”" : "Voice off"}
        showSuggestions={state.conversation.length === 0}
      />
    </div>
  );
}
