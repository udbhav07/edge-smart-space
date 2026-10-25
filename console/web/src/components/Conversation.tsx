import { AnimatePresence, motion } from "framer-motion";
import {
  AlertTriangle,
  BrainCircuit,
  CalendarCheck,
  ChevronRight,
  Keyboard,
  Mic,
  Plane,
  ShieldCheck,
  Sparkles,
  Wrench,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Step, Turn, Waiting } from "@/lib/types";
import { clock, cn } from "@/lib/format";
import { Orb, type Presence } from "./Orb";

/** Steps worth showing beside a conversation: what the agent did to answer. */
const TRACE_KINDS = new Set(["understood", "tool", "result", "safety", "model"]);
const TRACE_WINDOW_S = 90;

type Item = { kind: "turn"; turn: Turn } | { kind: "trace"; id: string; steps: Step[] };

function interleave(turns: Turn[], steps: Step[]): Item[] {
  const items: Item[] = [];
  const pool = steps.filter((step) => TRACE_KINDS.has(step.kind) && !(step.kind === "model" && !/assistant/i.test(step.text)));
  let cursor = 0;
  turns.forEach((turn, index) => {
    items.push({ kind: "turn", turn });
    if (turn.role !== "occupant") return;
    const next = turns.slice(index + 1).find((t) => t.role === "assistant");
    const until = next ? next.ts + 0.5 : turn.ts + TRACE_WINDOW_S;
    const mine: Step[] = [];
    while (cursor < pool.length && pool[cursor].ts < turn.ts - 0.5) cursor++;
    let scan = cursor;
    while (scan < pool.length && pool[scan].ts <= until) mine.push(pool[scan++]);
    cursor = scan;
    if (mine.length) {
      // The trace sits after the occupant's turn, before the reply.
      items.push({ kind: "trace", id: `trace-${turn.id}`, steps: mine });
    }
  });
  return items;
}

const stepIcon: Record<string, typeof Sparkles> = {
  understood: Sparkles,
  tool: Wrench,
  result: CalendarCheck,
  safety: ShieldCheck,
  model: BrainCircuit,
};

const toneText: Record<string, string> = {
  neutral: "text-muted",
  info: "text-muted",
  good: "text-good",
  warn: "text-ember",
  alert: "text-alert",
};

function Trace({ steps, live }: { steps: Step[]; live: boolean }) {
  const [open, setOpen] = useState(false);
  const headline = steps[steps.length - 1];
  return (
    <div className="ml-11 animate-rise">
      <button
        onClick={() => setOpen((value) => !value)}
        className="group flex items-center gap-2 rounded-md px-2 py-1 text-[13px] text-muted hover:text-fg"
        aria-expanded={open}
      >
        <ChevronRight className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-90")} />
        <span className={cn(live && "shimmer-text")}>
          {live ? "Working" : "Worked"} · {steps.length} step{steps.length === 1 ? "" : "s"}
        </span>
        {!open && <span className="truncate opacity-70">— {headline.text}</span>}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.ol
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="ml-3 overflow-hidden border-l hairline pl-4"
          >
            {steps.map((step) => {
              const Icon = stepIcon[step.kind] ?? Sparkles;
              return (
                <li key={step.id} className="relative py-1.5">
                  <span className="absolute -left-[22px] top-2.5 h-2 w-2 rounded-full bg-raised ring-2 ring-[rgb(var(--ink))]" />
                  <div className={cn("flex items-start gap-2 text-[13px]", toneText[step.tone])}>
                    <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                    <div className="min-w-0">
                      <div className="text-fg/90">{step.text}</div>
                      {step.detail && <div className="truncate font-mono text-[11.5px] text-muted">{step.detail}</div>}
                    </div>
                    <span className="ml-auto shrink-0 font-mono text-[11px] text-muted tabular">{clock(step.ts)}</span>
                  </div>
                </li>
              );
            })}
          </motion.ol>
        )}
      </AnimatePresence>
    </div>
  );
}

function Bubble({ turn, names, presence }: { turn: Turn; names: Record<string, string>; presence: Presence }) {
  if (turn.role === "system") {
    return (
      <div className="flex animate-rise justify-center">
        <div
          className={cn(
            "flex max-w-[88%] items-start gap-2 rounded-xl border px-3.5 py-2 text-[13.5px]",
            turn.tone === "good" ? "border-good/25 bg-good/5 text-good" : "border-ember/25 bg-ember/5 text-ember",
          )}
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          <span className="text-fg/90">{turn.text}</span>
        </div>
      </div>
    );
  }
  if (turn.role === "occupant") {
    const Source = turn.source === "voice" ? Mic : Keyboard;
    const who = turn.profile_id ? names[turn.profile_id] ?? turn.profile_id : "Guest";
    return (
      <div className="flex animate-rise flex-col items-end gap-1">
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-raised px-4 py-2.5 text-[15px] text-fg">{turn.text}</div>
        <div className="flex items-center gap-1.5 pr-1 text-[11.5px] text-muted">
          <Source className="h-3 w-3" />
          {who} · {turn.source === "voice" ? "spoken" : "typed"} · {clock(turn.ts)}
        </div>
      </div>
    );
  }
  return (
    <div className="flex animate-rise items-start gap-3">
      <Orb presence={presence === "speaking" ? "speaking" : "idle"} size={32} className="mt-0.5" />
      <div className="min-w-0 max-w-[85%]">
        <div className="text-[15px] leading-relaxed text-fg">{turn.text}</div>
        <div className="mt-1 text-[11.5px] text-muted">Jarvis · {clock(turn.ts)}</div>
      </div>
    </div>
  );
}

function ConfirmCard({ waiting, onAnswer }: { waiting: Waiting; onAnswer: (id: string, yes: boolean) => void }) {
  const [busy, setBusy] = useState<"yes" | "no" | null>(null);
  const answer = (yes: boolean) => {
    setBusy(yes ? "yes" : "no");
    onAnswer(waiting.invocation_id, yes);
  };
  const details = Object.entries(waiting.arguments).filter(([, value]) => value !== null && value !== "");
  return (
    <div className="ml-11 animate-rise rounded-2xl border border-glacier/25 bg-glacier/[0.04] p-4">
      <div className="flex items-center gap-2 text-[13px] font-medium text-glacier">
        <Plane className="h-4 w-4" /> Needs your confirmation
      </div>
      <div className="mt-1.5 text-[15px] text-fg">{waiting.summary}</div>
      <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-[12.5px]">
        {details.map(([key, value]) => (
          <div key={key} className="contents">
            <dt className="text-muted">{key.replace(/_/g, " ")}</dt>
            <dd className="font-mono text-fg/85">{String(value)}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-[12px] text-muted">A mock booking: no real reservation is made, and nothing happens until you say yes.</p>
      <div className="mt-3 flex gap-2">
        <button
          disabled={busy !== null}
          onClick={() => answer(true)}
          className="rounded-lg bg-glacier px-3.5 py-1.5 text-[13px] font-medium text-white hover:brightness-110 disabled:opacity-50"
        >
          {busy === "yes" ? "Confirming…" : "Confirm"}
        </button>
        <button
          disabled={busy !== null}
          onClick={() => answer(false)}
          className="rounded-lg border hairline px-3.5 py-1.5 text-[13px] text-fg hover:bg-raised disabled:opacity-50"
        >
          Decline
        </button>
      </div>
    </div>
  );
}

export function Conversation({
  turns,
  steps,
  waiting,
  names,
  presence,
  heard,
  onAnswer,
  empty,
}: {
  turns: Turn[];
  steps: Step[];
  waiting: Waiting[];
  names: Record<string, string>;
  presence: Presence;
  heard: string;
  onAnswer: (id: string, yes: boolean) => void;
  empty: React.ReactNode;
}) {
  const items = useMemo(() => interleave(turns, steps), [turns, steps]);
  const end = useRef<HTMLDivElement>(null);
  const lastTurn = turns[turns.length - 1];
  const awaitingReply = lastTurn?.role === "occupant";

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items.length, waiting.length, heard, awaitingReply]);

  if (!turns.length && !waiting.length && !heard) return <>{empty}</>;

  return (
    <div className="mx-auto flex w-full max-w-[760px] flex-col gap-5 px-4 pb-6 pt-8 sm:px-6">
      {items.map((item, index) =>
        item.kind === "turn" ? (
          <Bubble key={item.turn.id} turn={item.turn} names={names} presence={index === items.length - 1 ? presence : "idle"} />
        ) : (
          <Trace key={item.id} steps={item.steps} live={awaitingReply && index === items.length - 1} />
        ),
      )}
      {heard && (
        <div className="flex animate-rise flex-col items-end gap-1">
          <div className="max-w-[80%] rounded-2xl rounded-br-md border border-dashed border-glacier/30 px-4 py-2.5 text-[15px] text-fg/80">
            {heard}
          </div>
          <div className="flex items-center gap-1.5 pr-1 text-[11.5px] text-glacier">
            <Mic className="h-3 w-3" /> heard just now
          </div>
        </div>
      )}
      {(awaitingReply || presence === "thinking") && (
        <div className="flex items-center gap-3">
          <Orb presence="thinking" size={32} />
          <span className="shimmer-text text-[14px]">Thinking…</span>
        </div>
      )}
      {waiting.map((entry) => (
        <ConfirmCard key={entry.invocation_id} waiting={entry} onAnswer={onAnswer} />
      ))}
      <div ref={end} />
    </div>
  );
}
