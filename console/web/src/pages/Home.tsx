import { ArrowRight, CalendarDays, Check, CheckCircle2, Ear, Mic, MicOff, RotateCcw, Send, ShieldAlert, Trash2, Volume2, VolumeX, X, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { cn, entryWhen, modeLabel } from "@/lib/format";
import { DEFAULT_SPEECH_SETTINGS, type Snapshot } from "@/lib/types";
import { useConsole } from "@/lib/useConsole";
import { Button, Card, Switch } from "@/components/ui";
import { Devices } from "@/components/Devices";
import { Climate } from "@/components/Climate";

// Home: what someone living in the room reaches for, most often first (FR-80).
// Every control here is a request: it goes through the same safety gate as
// a spoken one, and the room may answer with less than was asked.


const guard = (promise: Promise<unknown>, done?: string) =>
  promise.then(() => { if (done) toast.success(done); return true; }).catch((error: Refused) => { toast.error(error.message); return false; });

function greeting(): string {
  const hour = new Date().getHours();
  return hour < 5 ? "Good night" : hour < 12 ? "Good morning" : hour < 18 ? "Good afternoon" : "Good evening";
}

function Status({ state }: { state: Snapshot }) {
  const mode = state.room.mode?.mode;
  const waitsOnAPerson = mode === "DEGRADED_ACTUATOR" || mode === "SAFE_HOLD";
  const good = mode === "NORMAL";
  // The inbox carries each mode change in the shared plain wording (src/common/plain.py).
  const said = [...state.inbox].reverse().find((item) => item.kind === "mode")?.body;
  return (
    <Card>
      <div className="flex items-center gap-2">
        {good ? <CheckCircle2 className="h-5 w-5 text-good" /> : <ShieldAlert className={cn("h-5 w-5", waitsOnAPerson ? "text-alert" : "text-ember")} />}
        <span className="text-[17px] font-semibold text-fg">{mode ? modeLabel[mode] : "Starting up"}</span>
      </div>
      <p className="mt-2 text-[14px] leading-relaxed text-fg/85">
        {good ? "Everything is working. The room looks after itself." : said ?? state.room.mode?.reason ?? ""}
      </p>
      {waitsOnAPerson && (
        <div className="mt-3 grid gap-2">
          <p className="text-[12.5px] text-muted">Check the air conditioner first, then press Reset; the faults are tested again.</p>
          <Button variant="primary" icon={RotateCcw} onClick={() => void guard(api.reset("checked from Home"), "Reset sent")}>Reset</Button>
        </div>
      )}
      {state.faults.length > 0 && (
        <Link to="/faults" className="mt-3 flex items-center gap-1 text-[13px] text-glacier hover:underline">{state.faults.length} active {state.faults.length === 1 ? "fault" : "faults"} <ArrowRight className="h-3 w-3" /></Link>
      )}
    </Card>
  );
}

function Talk({ state }: { state: Snapshot }) {
  const [text, setText] = useState("");
  const settings = state.speech_settings ?? DEFAULT_SPEECH_SETTINGS;
  const turns = state.conversation.filter((turn) => turn.role !== "system");
  const asked = [...turns].reverse().find((turn) => turn.role === "occupant");
  const answered = [...turns].reverse().find((turn) => turn.role === "assistant");
  const activity = state.speech?.activity;
  const voiceWord = !state.speech ? "Voice is off" : activity === "muted" ? "Microphone muted" : activity === "speaking" ? "Speaking…" : activity === "waiting" ? (state.speech.listen_mode === "continuous" ? "Listening" : `Say “${state.speech.wake_word || "Hey Jarvis"}”`) : `${activity?.[0].toUpperCase()}${activity?.slice(1)}…`;
  const send = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!text.trim()) return;
    if (await guard(api.say(text, ""))) setText("");
  };
  const change = (changes: Partial<typeof settings>) => void guard(api.speechSettings(changes));
  return (
    <Card title="Talk to Jarvis" icon={Mic} action={<Link to="/chat" className="flex items-center gap-1 text-[12px] text-muted hover:text-fg">Chat <ArrowRight className="h-3 w-3" /></Link>}>
      <form onSubmit={send} className="flex gap-2">
        <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Ask Jarvis… “make it 23”, “what's on tomorrow?”" aria-label="Ask Jarvis" className="min-w-0 flex-1 rounded-full border hairline bg-raised/60 px-4 py-2 text-[14px] text-fg placeholder:text-muted/70 focus:border-glacier focus:outline-none" />
        <Button type="submit" variant="primary" icon={Send} disabled={!text.trim()}>Ask</Button>
      </form>
      {(asked || answered) && (
        <div className="mt-3 grid gap-1.5 text-[13.5px]">
          {asked && <div className="max-w-[85%] justify-self-end rounded-2xl bg-glacier px-3 py-1.5 text-white">{asked.text}</div>}
          {answered && <div className="max-w-[85%] justify-self-start rounded-2xl bg-raised px-3 py-1.5 text-fg">{answered.text}</div>}
        </div>
      )}
      <div className="mt-3 text-[12.5px] text-muted">{voiceWord}{state.speech?.voice ? ` · voice: ${state.speech.voice}` : ""}</div>
      <div className="mt-1 grid divide-y divide-[rgb(var(--line)/var(--line-alpha))]">
        <Switch checked={settings.listening} onChange={(listening) => change({ listening })} label={settings.listening ? "Microphone on" : "Microphone muted"} hint={settings.listening ? "Mute and nothing is captured at all." : "Nothing is being captured."} />
        <Switch checked={settings.speak_replies} onChange={(speak_replies) => change({ speak_replies })} label="Speak replies aloud" hint="Answers to what you say out loud." />
      </div>
      <div className="mt-1 flex items-center gap-1.5 text-[12px] text-muted">
        {settings.listening ? <Mic className="h-3.5 w-3.5" /> : <MicOff className="h-3.5 w-3.5" />}
        {settings.speak_replies ? <Volume2 className="h-3.5 w-3.5" /> : <VolumeX className="h-3.5 w-3.5" />}
        <Link to="/settings" className="ml-auto hover:text-fg">More voice settings</Link>
      </div>
    </Card>
  );
}

function Today({ state }: { state: Snapshot }) {
  const { calendarStale } = useConsole();
  useEffect(() => { void api.refreshCalendar().catch(() => undefined); }, [calendarStale]);
  const now = new Date();
  const upcoming = [...state.calendar.entries]
    .filter((entry) => new Date(entry.starts_at).getTime() >= now.getTime() - 3600_000)
    .sort((a, b) => a.starts_at.localeCompare(b.starts_at))
    .slice(0, 4);
  const overheard = state.profiles.reduce((count, profile) => count + (profile.suggestions?.length ?? 0), 0);
  return (
    <Card title="Today and next" icon={CalendarDays} action={<Link to="/calendar" className="flex items-center gap-1 text-[12px] text-muted hover:text-fg">Calendar <ArrowRight className="h-3 w-3" /></Link>}>
      {state.waiting.length > 0 && (
        <ul className="mb-3 grid gap-2">
          {state.waiting.map((item) => (
            <li key={item.invocation_id} className="rounded-xl border border-glacier/25 bg-glacier/[0.04] p-3">
              <div className="text-[13.5px] font-medium text-fg">{item.summary}</div>
              <div className="mt-2 flex gap-2">
                <Button variant="primary" icon={Check} className="px-3 py-1 text-[12.5px]" onClick={() => void guard(api.answer(item.invocation_id, true), "Confirmed")}>Yes</Button>
                <Button icon={X} className="px-3 py-1 text-[12.5px]" onClick={() => void guard(api.answer(item.invocation_id, false), "Declined; nothing was done")}>No</Button>
              </div>
            </li>
          ))}
        </ul>
      )}
      <ul className="grid gap-1">
        {upcoming.map((entry) => {
          const when = entryWhen(entry.starts_at);
          return (
            <li key={entry.event_id} className={cn("flex min-w-0 items-center gap-3 rounded-lg px-2 py-1.5 text-[13.5px]", entry.tentative && "border border-dashed border-ember/40")}>
              <span className={cn("w-[92px] shrink-0 font-mono text-[12px] tabular", entry.tentative ? "text-ember" : "text-glacier")}>{when.day.split(",")[0]} {when.time}</span>
              <span className={cn("min-w-0 flex-1 truncate", entry.tentative ? "italic text-fg/80" : "text-fg")}>{entry.subject}</span>
              {entry.google && <span title="In your Google Calendar" className="shrink-0 rounded bg-raised px-1 text-[10.5px] font-semibold text-muted">G</span>}
              {entry.tentative && (
                <span className="flex shrink-0 gap-1">
                  <button aria-label={`Keep ${entry.subject}`} title="Keep" onClick={() => void guard(api.keepEntry(entry.event_id), `Keeping “${entry.subject}”`)} className="rounded-md p-1 text-glacier hover:bg-glacier/10"><Check className="h-3.5 w-3.5" /></button>
                  <button aria-label={`Discard ${entry.subject}`} title="Discard" onClick={() => void guard(api.removeEntry(entry.subject, entry.starts_at), `Discarding “${entry.subject}”`)} className="rounded-md p-1 text-muted hover:bg-raised"><Trash2 className="h-3.5 w-3.5" /></button>
                </span>
              )}
            </li>
          );
        })}
        {upcoming.length === 0 && <li className="px-2 text-[13px] text-muted">{state.calendar.as_of ? "Nothing coming up this week." : "Loading…"}</li>}
      </ul>
      {overheard > 0 && (
        <Link to="/chat" className="mt-3 flex items-center gap-1.5 rounded-xl border border-dashed border-ember/40 px-3 py-2 text-[13px] text-fg hover:bg-ember/[0.04]">
          <Ear className="h-3.5 w-3.5 text-ember" /> Overheard · {overheard} to keep or dismiss <ArrowRight className="ml-auto h-3 w-3 text-muted" />
        </Link>
      )}
    </Card>
  );
}

function Energy({ state }: { state: Snapshot }) {
  const power = state.room.sensors.pwr_01?.value;
  const peak = state.room.tariff?.band === "peak";
  const energy = state.energy;
  return (
    <Card title="Energy today" icon={Zap}>
      <div className="flex items-baseline gap-1.5">
        <span className="display text-[36px] font-semibold leading-none text-fg tabular">{energy ? energy.kwh.toFixed(2) : "--"}</span>
        <span className="text-[13px] text-muted">kWh</span>
        <span className="ml-auto text-[17px] font-medium text-fg tabular">{energy ? `${energy.currency} ${energy.cost.toFixed(2)}` : ""}</span>
      </div>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-muted">
        <span>Now {power !== undefined ? `${Math.round(power)} W` : "--"}</span>
        <span className={cn(peak && "text-ember")}>{state.room.tariff ? (peak ? "Peak price" : "Normal price") : "Price unknown"}</span>
      </div>
      <p className="mt-2 text-[11.5px] text-muted">An estimate from the power meter since midnight, counted while this console is running.</p>
    </Card>
  );
}

function GoogleChip({ state }: { state: Snapshot }) {
  const google = state.google;
  if (google?.connected) {
    return (
      <Link to="/settings" className="flex items-center gap-2 rounded-full bg-surface px-3.5 py-1.5 text-[13px] shadow-sm hover:text-fg">
        <span className={cn("h-1.5 w-1.5 rounded-full", google.online ? "bg-good" : "bg-ember")} />
        <span className="text-fg">{google.account || "Google connected"}</span>
        {google.outbox > 0 && <span className="text-muted">· {google.outbox} waiting to send</span>}
      </Link>
    );
  }
  if (google?.needs_reconnect) return <a href="/api/google/connect" className="rounded-full bg-ember/10 px-3.5 py-1.5 text-[13px] font-medium text-ember">Reconnect Google</a>;
  if (google?.configured) return <a href="/api/google/connect" className="rounded-full bg-glacier px-3.5 py-1.5 text-[13px] font-medium text-white hover:brightness-110">Connect Google</a>;
  return <Link to="/settings" className="rounded-full bg-surface px-3.5 py-1.5 text-[13px] text-muted shadow-sm hover:text-fg">Google: not set up</Link>;
}

export function Home() {
  const { state } = useConsole();
  if (!state) return null;
  return (
    <div className="mx-auto max-w-[1200px] px-4 py-6 sm:px-8">
      <div className="flex flex-wrap items-end justify-between gap-4 pb-6 pt-2">
        <div>
          <h1 className="display text-[34px] font-bold leading-[1.1] text-fg sm:text-[40px]">{greeting()}</h1>
          <p className="mt-2 text-[15px] text-muted">Your room, and what's next.</p>
        </div>
        <GoogleChip state={state} />
      </div>
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-3">
        <Climate state={state} />
        <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] content-start gap-4">
          <Status state={state} />
          <Energy state={state} />
        </div>
      </div>
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)] gap-4 lg:grid-cols-2">
        <Talk state={state} />
        <Today state={state} />
      </div>
      <div className="mt-4 grid grid-cols-[minmax(0,1fr)]">
        <Devices devices={state.devices ?? []} />
      </div>
    </div>
  );
}
