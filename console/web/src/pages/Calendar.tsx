import { CalendarDays, Plus, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { cn } from "@/lib/format";
import type { CalendarEntry } from "@/lib/types";
import { useConsole } from "@/lib/useConsole";
import { Button, Card, Field, PageHeader, inputClass } from "@/components/ui";

// The occupant's own calendar, kept on this node (FR-58). Writes need no
// confirmation: an unwanted entry is deleted, not undone with a third party.

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

  return (
    <div className="mx-auto max-w-[1500px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Calendar"
        subtitle="Your own calendar, kept on this node. Jarvis adds to it when you ask; you can add and remove here. Nothing is shared with any hosted service."
        actions={<Button icon={RefreshCw} onClick={() => void refresh()}>Refresh</Button>}
      />
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Card padded={false} className="overflow-hidden">
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
                          className={cn(
                            "absolute inset-x-1 overflow-hidden rounded-md border-l-2 border-glacier bg-glacier/15 px-1.5 py-1 text-left text-[11.5px] leading-tight text-fg hover:bg-glacier/25",
                            selected?.event_id === entry.event_id && "ring-1 ring-glacier",
                          )}
                          style={{ top, height: Math.max(24, (DEFAULT_MINUTES / 60) * HOUR_PX - 2) }}
                        >
                          <div className="font-mono text-[10.5px] text-glacier">{at.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</div>
                          <div className="truncate">{entry.subject}</div>
                        </button>
                      );
                    })}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Card>
        <div className="grid content-start gap-4">
          {selected && (
            <Card title="Entry" icon={CalendarDays}>
              <div className="text-[15px] font-medium text-fg">{selected.subject}</div>
              <div className="mt-1 text-[13px] text-muted">{new Date(selected.starts_at).toLocaleString([], { weekday: "long", day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" })}</div>
              <div className="mt-3 flex gap-2">
                <Button variant="danger" icon={Trash2} onClick={() => void remove(selected)}>Remove</Button>
                <Button variant="ghost" onClick={() => setSelected(null)}>Close</Button>
              </div>
            </Card>
          )}
          <Card title="Add an entry" icon={Plus}><AddEntry onDone={() => undefined} /></Card>
          <Card title="Coming up" icon={CalendarDays}>
            <ul className="grid gap-1.5">
              {[...entries].sort((a, b) => a.starts_at.localeCompare(b.starts_at)).slice(0, 8).map((entry) => (
                <li key={entry.event_id}>
                  <button onClick={() => setSelected(entry)} className="flex w-full items-center gap-3 rounded-lg px-2 py-1.5 text-left text-[13px] hover:bg-raised/50">
                    <span className="font-mono text-[12px] text-glacier tabular">{new Date(entry.starts_at).toLocaleString([], { weekday: "short", hour: "2-digit", minute: "2-digit" })}</span>
                    <span className="truncate text-fg">{entry.subject}</span>
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
