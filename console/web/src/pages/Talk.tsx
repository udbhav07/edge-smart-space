import { ArrowRight, CalendarDays, Lock, Mic, Send, Volume2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "@/lib/api";
import { cn, entryWhen } from "@/lib/format";
import type { CalendarEntry } from "@/lib/types";
import { useConsole } from "@/lib/useConsole";
import { microphoneAllowed, useRemoteTalk, type TalkPhase } from "@/lib/useRemoteTalk";
import { Button, Card, inputClass } from "@/components/ui";
import { Orb, type Presence } from "@/components/Orb";
import { RemoteTile } from "@/remotes/RemoteTile";

// The room, from a phone or a laptop anywhere on the network (FR-94): talk to
// Jarvis on this device's microphone and hear the answer here, see what is
// coming up, and reach every remote. Nothing said here is spoken in the room.

const PHASE: Record<TalkPhase, { label: string; presence: Presence }> = {
  idle: { label: "Hold to talk", presence: "idle" },
  recording: { label: "Listening…", presence: "listening" },
  sending: { label: "Got it…", presence: "transcribing" },
  thinking: { label: "Thinking…", presence: "thinking" },
  speaking: { label: "Speaking", presence: "speaking" },
};

function Schedule({ entries, asOf }: { entries: CalendarEntry[]; asOf: number | null }) {
  const now = Date.now();
  const coming = [...entries]
    .filter((entry) => new Date(entry.starts_at).getTime() >= now - 3600_000)
    .sort((a, b) => a.starts_at.localeCompare(b.starts_at))
    .slice(0, 8);
  const days = new Map<string, CalendarEntry[]>();
  for (const entry of coming) {
    const day = entryWhen(entry.starts_at).day;
    days.set(day, [...(days.get(day) ?? []), entry]);
  }
  if (!coming.length) return <p className="text-[13.5px] text-muted">{asOf ? "Nothing coming up this week." : "Loading…"}</p>;
  return (
    <div className="grid gap-3">
      {[...days.entries()].map(([day, list]) => (
        <div key={day}>
          <div className="mb-1 text-[12px] font-semibold text-muted">{day}</div>
          <ul className="grid gap-1">
            {list.map((entry) => (
              <li key={entry.event_id} className={cn("flex items-baseline gap-3 rounded-lg px-2 py-1.5 text-[14px]", entry.tentative && "border border-dashed border-ember/40")}>
                <span className={cn("w-[64px] shrink-0 font-mono text-[12.5px] tabular", entry.tentative ? "text-ember" : "text-glacier")}>{entryWhen(entry.starts_at).time}</span>
                <span className={cn("min-w-0 flex-1", entry.tentative ? "italic text-fg/80" : "text-fg")}>{entry.subject}</span>
                {entry.tentative && <span className="shrink-0 text-[11px] text-ember">pencilled in</span>}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

export function Talk() {
  const { state, calendarStale } = useConsole();
  const talk = useRemoteTalk();
  const [text, setText] = useState("");
  useEffect(() => { void api.refreshCalendar().catch(() => undefined); }, [calendarStale]);
  if (!state) return null;
  const allowed = microphoneAllowed();
  const secure = state.tls_port ? `https://${window.location.hostname}:${state.tls_port}/talk` : "";
  const shown = PHASE[talk.phase];
  const devices = state.devices ?? [];
  return (
    <div className="mx-auto grid max-w-[720px] gap-4 px-4 py-6 sm:px-8">
      <div className="pt-2">
        <h1 className="display text-[32px] font-bold leading-tight text-fg sm:text-[38px]">Talk to the room</h1>
        <p className="mt-1 text-[14.5px] text-muted">From wherever you are. Jarvis answers on this device; nobody in the room hears it.</p>
      </div>

      <Card>
        <div className="grid justify-items-center gap-3 py-2">
          <button
            type="button"
            aria-label={talk.phase === "recording" ? "Stop and send" : "Hold to talk"}
            disabled={!allowed || talk.phase === "sending" || talk.phase === "thinking"}
            onPointerDown={(event) => { event.preventDefault(); talk.press(); }}
            onPointerUp={talk.release}
            onPointerLeave={() => talk.phase === "recording" && !talk.tapped && talk.release()}
            onContextMenu={(event) => event.preventDefault()}
            className={cn(
              "relative flex h-36 w-36 select-none items-center justify-center rounded-full transition active:scale-95 disabled:opacity-40",
              talk.phase === "recording" ? "bg-glacier text-white shadow-[0_0_0_10px_rgb(var(--glacier)/0.18)]" : "bg-raised text-fg",
            )}
            style={{ touchAction: "none" }}
          >
            {talk.phase === "idle" ? <Mic className="h-14 w-14" /> : <Orb presence={shown.presence} size={84} />}
          </button>
          <div className="text-center">
            <div className="text-[17px] font-semibold text-fg">{talk.tapped ? "Listening… tap to send" : shown.label}</div>
            {talk.phase === "idle" && allowed && <div className="text-[12.5px] text-muted">Hold while you speak, or tap once to start and again to send</div>}
          </div>
          {!allowed && (
            <div className="flex max-w-[460px] items-start gap-2 rounded-xl bg-ember/10 p-3 text-[13px] text-fg">
              <Lock className="mt-0.5 h-4 w-4 shrink-0 text-ember" />
              <span>
                A browser lets a page use the microphone only over a secure connection.
                {secure ? <> Open <a href={secure} className="font-medium text-glacier underline">{secure}</a> — the first time, your browser warns that the room's certificate is its own; continue to the site.</> : " Ask whoever runs the room to turn on HTTPS (console.tls_port)."}
                {" "}You can type below meanwhile.
              </span>
            </div>
          )}
          {talk.error && <p className="text-center text-[13px] text-alert">{talk.error}</p>}
        </div>

        {talk.exchanges.length > 0 && (
          <ul className="mt-3 grid gap-2 border-t hairline pt-4">
            {talk.exchanges.slice(-4).map((exchange, index) => (
              <li key={index} className="grid gap-1.5 text-[14px]">
                <div className="max-w-[85%] justify-self-end rounded-2xl bg-glacier px-3 py-1.5 text-white">{exchange.asked}</div>
                {exchange.answer ? (
                  <div className="flex max-w-[85%] items-start gap-2 justify-self-start">
                    <div className="rounded-2xl bg-raised px-3 py-1.5 text-fg">{exchange.answer}</div>
                    <button aria-label="Say it again" onClick={() => void talk.replay(exchange.answer!)} className="mt-1 rounded-full p-1.5 text-muted hover:bg-raised hover:text-fg"><Volume2 className="h-4 w-4" /></button>
                  </div>
                ) : (
                  <div className="justify-self-start text-[12.5px] text-muted">Thinking…</div>
                )}
              </li>
            ))}
          </ul>
        )}

        <form onSubmit={(event) => { event.preventDefault(); if (text.trim()) void talk.type(text.trim()).then(() => setText("")); }} className="mt-4 flex gap-2">
          <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Or type… “make it 23”, “what's on tomorrow?”" aria-label="Type to Jarvis" className={cn(inputClass, "py-2")} />
          <Button type="submit" variant="primary" icon={Send} disabled={!text.trim()}>Ask</Button>
        </form>
      </Card>

      <Card title="Coming up" icon={CalendarDays} action={<Link to="/calendar" className="flex items-center gap-1 text-[12px] text-muted hover:text-fg">Calendar <ArrowRight className="h-3 w-3" /></Link>}>
        <Schedule entries={state.calendar.entries} asOf={state.calendar.as_of} />
      </Card>

      {devices.length > 0 && (
        <section className="grid gap-3">
          <div className="flex items-baseline justify-between px-1">
            <h2 className="text-[17px] font-semibold tracking-[-0.02em] text-fg">The room</h2>
            <Link to="/remotes" className="flex items-center gap-1 text-[13px] text-muted hover:text-fg">All remotes <ArrowRight className="h-3 w-3" /></Link>
          </div>
          <div className="grid grid-cols-[minmax(0,1fr)] gap-3">
            {devices.map((device) => <RemoteTile key={device.device_id} device={device} />)}
          </div>
        </section>
      )}
    </div>
  );
}
