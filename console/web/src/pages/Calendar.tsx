import { CalendarDays, Check, Ear, Plus, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { cn } from "@/lib/format";
import type { CalendarEntry } from "@/lib/types";
import { useConsole } from "@/lib/useConsole";
import { Button, Card, Field, PageHeader, inputClass } from "@/components/ui";

// The occupant's own calendar, kept on this node (FR-58). Writes need no
// confirmation: an unwanted entry is deleted, not undone with a third party.
// An entry pencilled in from overheard conversation (FR-77) is drawn apart --
// dashed and hatched, labelled model tentative -- until it is kept.

/** Hatching for a tentative entry: a texture, so it never relies on colour alone. */
const HATCH = { backgroundImage: "repeating-linear-gradient(135deg, rgb(var(--ember) / 0.16) 0 6px, transparent 6px 12px)" };

export function TentativeChip() {
  return <span className="inline-flex items-center gap-1 whitespace-nowrap rounded-full border border-dashed border-ember/60 px-1.5 py-px text-[10.5px] font-medium text-ember"><Ear className="h-3 w-3" /> Model tentative</span>;
}

const DAY_START = 7;
const DAY_END = 23;
const HOUR_PX = 44;
const DEFAULT_MINUTES = 60;

function localDate(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function AddEntry({ onDone }: { onDone: () => void }) {
  const [subject, setSubject] = useState("");
  const [day, setDay] = useState(() => localDate(new Date()));
  const [time, setTime] = useState("15:00");
  const [minutes, setMinutes] = useState(60);
  const [busy, setBusy] = useState(false);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      await api.addEntry(subject, `${day}T${time}:00`, minutes);
      toast.success(`Adding “${subject}”`);
      setSubject("");
      onDone();
    } catch (error) {
      toast.error((error as Refused).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} className="grid gap-3">
      <Field label="What"><input id="entry-subject" required value={subject} onChange={(event) => setSubject(event.target.value)} placeholder="Meeting with Anu" className={inputClass} /></Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Day"><input id="entry-day" type="date" required value={day} onChange={(event) => setDay(event.target.value)} className={inputClass} /></Field>
        <Field label="Time"><input id="entry-time" type="time" required value={time} onChange={(event) => setTime(event.target.value)} className={inputClass} /></Field>
      </div>
      <Field label="Length" hint="Minutes">
        <select id="entry-length" value={minutes} onChange={(event) => setMinutes(Number(event.target.value))} className={inputClass}>
          {[15, 30, 45, 60, 90, 120].map((value) => <option key={value} value={value}>{value} min</option>)}
        </select>
      </Field>
      <Button type="submit" variant="primary" icon={Plus} disabled={busy || !subject.trim()}>Add to calendar</Button>
      <p className="text-[11.5px] text-muted">Goes through the tool executor like every action; it appears here once the calendar has written it.</p>
    </form>
  );
}

export function Calendar() {
  const { state, calendarStale } = useConsole();
  const [selected, setSelected] = useState<CalendarEntry | null>(null);
  const refresh = () => api.refreshCalendar().catch((error: Refused) => toast.error(error.message));
  useEffect(() => {
    void refresh();
  }, [calendarStale]);

  const days = useMemo(() => {
    const start = new Date();
    start.setHours(0, 0, 0, 0);
    return Array.from({ length: 7 }, (_, index) => {
      const day = new Date(start);
      day.setDate(start.getDate() + index);
      return day;
    });
  }, []);
  if (!state) return null;
  const entries = state.calendar.entries;
  const byDay = (day: Date) => entries.filter((entry) => entry.starts_at.slice(0, 10) === localDate(day));
  const hours = Array.from({ length: DAY_END - DAY_START }, (_, index) => DAY_START + index);
  const remove = (entry: CalendarEntry) =>
    api.removeEntry(entry.subject, entry.starts_at).then(() => { toast.success(`Removing “${entry.subject}”`); setSelected(null); }).catch((error: Refused) => toast.error(error.message));
  const keep = (entry: CalendarEntry) =>
    api.keepEntry(entry.event_id).then(() => { toast.success(`Keeping “${entry.subject}”`); setSelected(null); }).catch((error: Refused) => toast.error(error.message));
  const tentative = entries.filter((entry) => entry.tentative);

  return (
    <div className="mx-auto max-w-[1500px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Calendar"
        subtitle={state.google?.connected ? "Your calendar, kept on this node and mirrored with Google Calendar both ways. Jarvis adds to it when you ask, and pencils in plans it overhears as model tentative for you to keep or discard." : "Your own calendar, kept on this node. Jarvis adds to it when you ask, and pencils in plans it overhears as model tentative for you to keep or discard. Connect Google in Settings to mirror it to Google Calendar."}
        actions={<Button icon={RefreshCw} onClick={() => void refresh()}>Refresh</Button>}
      />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Card padded={false} className="overflow-hidden">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-b hairline px-4 py-2 text-[12px] text-muted">
            <span className="flex items-center gap-1.5"><span className="h-3 w-4 rounded-sm border-l-2 border-glacier bg-glacier/15" /> Firm: added on request</span>
            <span className="flex items-center gap-1.5"><span className="h-3 w-4 rounded-sm border border-dashed border-ember/70" style={HATCH} /> Model tentative: overheard, waiting for you</span>
          </div>
          <div className="scroll-thin overflow-x-auto">
            <div className="min-w-[760px]">
              <div className="grid grid-cols-[56px_repeat(7,minmax(0,1fr))] border-b hairline">
                <div />
                {days.map((day, index) => (
                  <div key={index} className={cn("px-2 py-2.5 text-center", index === 0 && "text-glacier")}>
                    <div className="text-[11px] uppercase tracking-[0.06em] text-muted">{day.toLocaleDateString([], { weekday: "short" })}</div>
                    <div className={cn("text-[18px] font-medium tabular", index === 0 ? "text-glacier" : "text-fg")}>{day.getDate()}</div>
                  </div>
                ))}
              </div>
              <div className="grid grid-cols-[56px_repeat(7,minmax(0,1fr))]">
                <div>
                  {hours.map((hour) => (
                    <div key={hour} className="pr-2 text-right font-mono text-[10.5px] text-muted tabular" style={{ height: HOUR_PX }}>{String(hour).padStart(2, "0")}:00</div>
                  ))}
                </div>
                {days.map((day, index) => (
                  <div key={index} className="relative border-l hairline" style={{ height: HOUR_PX * hours.length }}>
                    {hours.map((hour) => <div key={hour} className="border-t hairline" style={{ height: HOUR_PX }} />)}
                    {byDay(day).map((entry) => {
                      const at = new Date(entry.starts_at);
                      const offset = (at.getHours() + at.getMinutes() / 60 - DAY_START) * HOUR_PX;
                      const top = Math.max(0, Math.min(offset, HOUR_PX * hours.length - 24));
                      return (
                        <button
                          key={entry.event_id}
                          onClick={() => setSelected(entry)}
                          title={entry.tentative ? `Model tentative: heard “${entry.heard ?? ""}”` : undefined}
                          className={cn(
                            "absolute inset-x-1 overflow-hidden rounded-md px-1.5 py-1 text-left text-[11.5px] leading-tight text-fg",
                            entry.tentative
                              ? "border border-dashed border-ember/70 bg-surface text-fg/85 hover:border-ember"
                              : "border-l-2 border-glacier bg-glacier/15 hover:bg-glacier/25",
                            selected?.event_id === entry.event_id && (entry.tentative ? "ring-1 ring-ember" : "ring-1 ring-glacier"),
                          )}
                          style={{ top, height: Math.max(24, (DEFAULT_MINUTES / 60) * HOUR_PX - 2), ...(entry.tentative ? HATCH : {}) }}
                        >
                          <div className={cn("flex items-center gap-1 font-mono text-[10.5px]", entry.tentative ? "text-ember" : "text-glacier")}>
                            {entry.tentative && <Ear className="h-2.5 w-2.5 shrink-0" />}
                            {at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                            {entry.google && <span title="From Google Calendar" className="ml-auto rounded bg-raised px-0.5 font-sans text-[9.5px] font-semibold text-muted">G</span>}
                          </div>
                          <div className={cn("truncate", entry.tentative && "italic")}>{entry.subject}</div>
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
        <div className="grid min-w-0 content-start gap-4">
          {selected && (
            <Card title={selected.tentative ? "Pencilled in" : "Entry"} icon={selected.tentative ? Ear : CalendarDays} action={selected.tentative ? <TentativeChip /> : undefined}>
              <div className="text-[15px] font-medium text-fg">{selected.subject}</div>
              <div className="mt-1 text-[13px] text-muted">{new Date(selected.starts_at).toLocaleString([], { weekday: "long", day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" })}</div>
              {selected.tentative && (
                <div className="mt-3 rounded-xl border border-dashed border-ember/40 p-3" style={HATCH}>
                  <div className="text-[11px] font-medium uppercase tracking-[0.08em] text-ember">Heard</div>
                  <p className="mt-0.5 text-[13.5px] italic text-fg/90">“{selected.heard || "…"}”</p>
                  <p className="mt-2 text-[12px] text-muted">Nobody asked for this: Jarvis overheard it and pencilled it in. Keep it to make it a firm entry.</p>
                </div>
              )}
              <div className="mt-3 flex flex-wrap gap-2">
                {selected.tentative && <Button variant="primary" icon={Check} onClick={() => void keep(selected)}>Keep</Button>}
                <Button variant="danger" icon={Trash2} onClick={() => void remove(selected)}>{selected.tentative ? "Discard" : "Remove"}</Button>
                <Button variant="ghost" onClick={() => setSelected(null)}>Close</Button>
              </div>
            </Card>
          )}
          {tentative.length > 0 && (
            <Card title={`Pencilled in · ${tentative.length}`} icon={Ear}>
              <ul className="grid gap-2">
                {tentative.map((entry) => (
                  <li key={entry.event_id} className="min-w-0 overflow-hidden rounded-xl border border-dashed border-ember/40 p-2.5" style={HATCH}>
                    <button onClick={() => setSelected(entry)} className="block w-full min-w-0 text-left">
                      <div className="flex items-center gap-2 text-[13px]"><span className="min-w-0 truncate font-medium text-fg">{entry.subject}</span><span className="ml-auto shrink-0 whitespace-nowrap font-mono text-[11.5px] text-ember tabular">{new Date(entry.starts_at).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })}</span></div>
                      {entry.heard && <div className="mt-0.5 truncate text-[12px] italic text-muted">“{entry.heard}”</div>}
                    </button>
                    <div className="mt-2 flex gap-1.5">
                      <Button variant="primary" icon={Check} className="px-2 py-1 text-[12px]" onClick={() => void keep(entry)}>Keep</Button>
                      <Button variant="ghost" icon={Trash2} className="px-2 py-1 text-[12px]" onClick={() => void remove(entry)}>Discard</Button>
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          )}
          <Card title="Add an entry" icon={Plus}><AddEntry onDone={() => undefined} /></Card>
          <Card title="Coming up" icon={CalendarDays}>
            <ul className="grid gap-1.5">
              {[...entries].sort((a, b) => a.starts_at.localeCompare(b.starts_at)).slice(0, 8).map((entry) => (
                <li key={entry.event_id}>
                  <button onClick={() => setSelected(entry)} className="flex w-full min-w-0 items-center gap-3 rounded-lg px-2 py-1.5 text-left text-[13px] hover:bg-raised/50">
                    <span className={cn("shrink-0 whitespace-nowrap font-mono text-[12px] tabular", entry.tentative ? "text-ember" : "text-glacier")}>{new Date(entry.starts_at).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })}</span>
                    <span className={cn("min-w-0 flex-1 truncate", entry.tentative ? "italic text-fg/80" : "text-fg")}>{entry.subject}</span>
                    {entry.tentative && <span className="shrink-0" title="Model tentative: overheard, not yet kept"><Ear className="h-3.5 w-3.5 text-ember" /></span>}
                    {entry.google && <span title="From Google Calendar" className="shrink-0 rounded bg-raised px-1 text-[10.5px] font-semibold text-muted">G</span>}
                  </button>
                </li>
              ))}
              {entries.length === 0 && <li className="text-[13px] text-muted">{state.calendar.as_of ? "Nothing in the next seven days." : "Loading…"}</li>}
            </ul>
          </Card>
        </div>
      </div>
    </div>
  );
}
