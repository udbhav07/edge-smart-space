import { Command as CommandIcon, PanelRight, Wifi, WifiOff } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Toaster, toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { celsius, cn, modeLabel, speechAlive } from "@/lib/format";
import type { Snapshot } from "@/lib/types";
import { useConsole, type LinkState } from "@/lib/useConsole";
import { CommandPalette } from "@/components/CommandPalette";
import { Composer } from "@/components/Composer";
import { ContextPanel, type PanelTab, type Point } from "@/components/ContextPanel";
import { Conversation } from "@/components/Conversation";
import { Orb, type Presence } from "@/components/Orb";

function greeting(): string {
  const hour = new Date().getHours();
  if (hour >= 5 && hour < 12) return "Good morning.";
  if (hour >= 12 && hour < 18) return "Good afternoon.";
  return "Good evening.";
}

function presenceOf(state: Snapshot | null, link: LinkState, now: number, replying: boolean): Presence {
  if (!state || link === "offline") return "offline";
  const speech = state.speech;
  if (speech && speechAlive(speech, now)) {
    if (speech.activity === "listening") return "listening";
    if (speech.activity === "transcribing") return "transcribing";
    if (speech.activity === "understanding") return "thinking";
  }
  const last = state.conversation[state.conversation.length - 1];
  if (last?.role === "occupant" && now - last.ts < 90) return "thinking";
  return replying ? "speaking" : "idle";
}

function statusLine(state: Snapshot | null, link: LinkState, presence: Presence, now: number): string {
  if (link === "offline") return "Can't reach the console server. The room keeps running on its own.";
  if (!state) return "Connecting to the room…";
  if (!state.broker) return "The console can't reach the blackboard; showing the last known state.";
  switch (presence) {
    case "listening": return "Listening…";
    case "transcribing": return "Catching what you said…";
    case "thinking": return "Thinking…";
    default:
      return speechAlive(state.speech, now)
        ? `Say “Hey Jarvis”, or type below`
        : "Type below. Voice is not running right now.";
  }
}

export default function App() {
  const { state, link, calendarStale } = useConsole();
  const [now, setNow] = useState(() => Date.now() / 1000);
  const [panelOpen, setPanelOpen] = useState(() => window.innerWidth >= 1100);
  const [tab, setTab] = useState<PanelTab>("room");
  const [palette, setPalette] = useState(false);
  const [speaker, setSpeaker] = useState(() => localStorage.getItem("speaker") ?? "");
  const [replying, setReplying] = useState(false);
  const composer = useRef<HTMLTextAreaElement>(null);
  const seenFaults = useRef<Set<string> | null>(null);
  const lastReply = useRef<string | null>(null);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    try { localStorage.setItem("speaker", speaker); } catch { /* private window */ }
  }, [speaker]);

  // A new fault is worth interrupting for; the first snapshot is not.
  useEffect(() => {
    if (!state) return;
    const ids = new Set(state.faults.map((fault) => fault.fault_id));
    if (seenFaults.current) {
      state.faults
        .filter((fault) => !seenFaults.current!.has(fault.fault_id))
        .forEach((fault) => toast.warning(`${fault.detector.replace(/_/g, " ")} on ${fault.subject}`, { description: "Mode changed at once; the explanation follows." }));
    }
    seenFaults.current = ids;
  }, [state?.faults]);

  // A brief glow on the orb when Jarvis answers.
  useEffect(() => {
    const last = state?.conversation.filter((turn) => turn.role === "assistant").pop();
    if (!last || last.id === lastReply.current) return;
    const first = lastReply.current === null;
    lastReply.current = last.id;
    if (first) return;
    setReplying(true);
    const timer = window.setTimeout(() => setReplying(false), 2400);
    return () => window.clearTimeout(timer);
  }, [state?.conversation]);

  const refreshCalendar = useCallback(() => {
    api.refreshCalendar().catch((error: Refused) => toast.error(error.message));
  }, []);
  useEffect(() => {
    if (state?.broker && tab === "calendar") refreshCalendar();
  }, [tab, state?.broker, calendarStale, refreshCalendar]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPalette((open) => !open);
      } else if (event.key === "/" && document.activeElement?.tagName !== "TEXTAREA" && document.activeElement?.tagName !== "INPUT") {
        event.preventDefault();
        composer.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const guard = (promise: Promise<unknown>, done?: string) =>
    promise.then(() => { if (done) toast.success(done); return true; }).catch((error: Refused) => { toast.error(error.message); return false; });

  const say = (text: string) => guard(api.say(text, speaker));
  const setTarget = (value: number) => void guard(api.setTarget(value, speaker));
  const inject = (subject: string, fault: string, magnitude?: number) =>
    void guard(api.inject(subject, fault, magnitude), fault === "NONE" ? `Cleared ${subject}` : `Injected at ${subject}; the detectors have to find it`);
  const reset = () => void guard(api.reset("checked from the console"), "Reset sent; faults are re-tested");
  const removeEntry = (subject: string, starts: string) => void guard(api.removeEntry(subject, starts), `Removing “${subject}”`);
  const answer = (id: string, yes: boolean) => void guard(api.answer(id, yes), yes ? "Confirmed" : "Declined; nothing was booked");

  const presence = presenceOf(state, link, now, replying);
  const names = useMemo(
    () => Object.fromEntries((state?.profiles ?? []).map((profile) => [profile.profile_id, profile.display_name || profile.profile_id])),
    [state?.profiles],
  );
  const heard =
    state?.speech && speechAlive(state.speech, now) && state.speech.activity === "understanding" ? state.speech.heard : "";
  const room = state?.room;
  const indoor = room?.health.temp_01?.quality === "faulted" ? room?.thermal?.t_pred : room?.sensors.temp_01?.value ?? room?.thermal?.t_in;
  const mode = room?.mode?.mode;
  const voiceUp = speechAlive(state?.speech ?? null, now);

  const empty = (
    <div className="mx-auto flex h-full max-w-[760px] flex-col items-center justify-center px-6 pb-10 text-center">
      <Orb presence={presence} size={112} />
      <h1 className="mt-8 text-balance text-[30px] font-semibold tracking-tight text-fg">
        {greeting()} The room is {room?.mode?.mode === "NORMAL" ? "comfortable" : "being looked after"}.
      </h1>
      <p className="mt-3 max-w-[52ch] text-balance text-[15px] text-muted">
        {voiceUp ? "Say “Hey Jarvis” and speak normally, or type below." : "Type below to talk to the room."} Ask for a temperature, check your calendar, or plan a trip; anything that commits you to someone else waits for your yes.
      </p>
      <div className="mt-6 flex items-center gap-6 font-mono text-[13px] text-muted tabular">
        <span><span className="text-[22px] text-fg">{celsius(indoor)}</span> indoors</span>
        <span><span className="text-[22px] text-fg">{celsius(room?.sensors.outdoor_01?.value)}</span> outside</span>
        <span><span className="text-[22px] text-fg">{celsius(room?.tracking?.tracked_c)}</span> held</span>
      </div>
    </div>
  );

  return (
    <div className="atmosphere flex h-full flex-col">
      <header className="hairline-b glass sticky top-0 z-30 flex items-center gap-3 px-4 py-2.5 sm:px-5">
        <Orb presence={presence} size={30} />
        <div className="min-w-0">
          <div className="flex items-baseline gap-2">
            <span className="text-[15px] font-semibold tracking-tight text-fg">Jarvis</span>
            <span className="hidden text-[12px] text-muted sm:inline">Smart Space</span>
          </div>
          <div className={cn("truncate text-[12.5px]", presence === "listening" || presence === "thinking" ? "text-glacier" : "text-muted")}>
            {statusLine(state, link, presence, now)}
          </div>
        </div>
        <div className="ml-auto flex items-center gap-2">
          {room && (
            <button onClick={() => { setTab("room"); setPanelOpen(true); }} className="hidden items-center gap-2 rounded-full border hairline px-3 py-1 text-[12.5px] text-fg hover:bg-raised md:flex">
              <span className="font-mono tabular">{celsius(indoor)}</span>
              <span className="text-muted">→</span>
              <span className="font-mono text-glacier tabular">{celsius(room.tracking?.tracked_c)}</span>
            </button>
          )}
          {mode && (
            <button onClick={() => { setTab("system"); setPanelOpen(true); }} className={cn("hidden items-center gap-1.5 rounded-full px-3 py-1 text-[12.5px] sm:flex", mode === "NORMAL" ? "bg-good/10 text-good" : "bg-ember/10 text-ember")}>
              <span className={cn("h-1.5 w-1.5 rounded-full", mode === "NORMAL" ? "bg-good" : "bg-ember animate-pulse")} />
              {modeLabel[mode]}
            </button>
          )}
          <span title={link === "live" ? "Live" : link === "polling" ? "Reconnecting; polling meanwhile" : "Offline"} className={cn("flex h-8 w-8 items-center justify-center rounded-lg", link === "live" ? "text-good" : link === "offline" ? "text-alert" : "text-ember")}>
            {link === "offline" ? <WifiOff className="h-4 w-4" /> : <Wifi className="h-4 w-4" />}
          </span>
          <button onClick={() => setPalette(true)} className="hidden items-center gap-1.5 rounded-lg border hairline px-2 py-1 text-[12px] text-muted hover:text-fg sm:flex" aria-label="Commands">
            <CommandIcon className="h-3.5 w-3.5" />K
          </button>
          <button onClick={() => setPanelOpen((open) => !open)} className={cn("flex h-8 w-8 items-center justify-center rounded-lg text-muted hover:bg-raised hover:text-fg", panelOpen && "text-fg")} aria-label="Toggle the context panel">
            <PanelRight className="h-4 w-4" />
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <main className="flex min-w-0 flex-1 flex-col">
          <div className="scroll-thin min-h-0 flex-1 overflow-y-auto">
            {state ? (
              <Conversation
                turns={state.conversation}
                steps={state.activity}
                waiting={state.waiting}
                names={names}
                presence={presence}
                heard={heard}
                onAnswer={answer}
                empty={empty}
              />
            ) : (
              empty
            )}
          </div>
          <Composer
            ref={composer}
            disabled={link === "offline" || !state || !state.broker}
            disabledReason={link === "offline" ? "The console server is not reachable" : "Waiting for the blackboard…"}
            profiles={state?.profiles ?? []}
            speaker={speaker}
            onSpeaker={setSpeaker}
            onSend={say}
            onTarget={setTarget}
            goal={room?.goal?.setpoint_c ?? null}
            held={room?.tracking?.tracked_c ?? null}
            voiceHint={voiceUp ? "Voice on — say “Hey Jarvis”" : "Voice off"}
            showSuggestions={!!state && state.conversation.length === 0}
          />
        </main>

        <AnimatePresence initial={false}>
          {panelOpen && state && (
            <motion.aside
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 380, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ type: "spring", stiffness: 380, damping: 38 }}
              className="hairline-l glass hidden shrink-0 overflow-hidden lg:block"
            >
              <div className="h-full w-[380px]">
                <ContextPanel
                  state={state}
                  history={state.history as Point[]}
                  tab={tab}
                  onTab={setTab}
                  onRefreshCalendar={refreshCalendar}
                  onRemove={removeEntry}
                  onReset={reset}
                  onInject={inject}
                />
              </div>
            </motion.aside>
          )}
        </AnimatePresence>
      </div>

      <CommandPalette
        open={palette}
        onOpenChange={setPalette}
        operator={!!state?.operator_controls}
        actions={{
          ask: (text) => void say(text),
          focusComposer: () => composer.current?.focus(),
          setTarget,
          openTab: (next) => { setTab(next); setPanelOpen(true); },
          refreshCalendar,
          inject,
          reset,
        }}
      />
      <Toaster theme="system" position="top-center" toastOptions={{ className: "!bg-surface !text-fg !border-[rgb(var(--line)/0.08)]" }} />
    </div>
  );
}
