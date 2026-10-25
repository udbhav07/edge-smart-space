import { Braces, Network, Pause, Play, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ago, clockSeconds, cn } from "@/lib/format";
import { iconFor, rendererFor, safely } from "@/lib/renderers";
import { matches, producerOf, rateOf } from "@/lib/services";
import type { BoardMessage, TopicRow } from "@/lib/types";
import { useConsole, useNow } from "@/lib/useConsole";
import { Button, Card, Empty, Json, Label, PageHeader, inputClass } from "@/components/ui";

// The blackboard as it is: every topic anything has published since the
// console started, how often, whether the topic contract retains it, and what
// the last message said, read through its topic's renderer. Below it, the
// messages as they arrive (FR-60). The JSON as published is one click away.

function filterBy(query: string) {
  const needle = query.trim().toLowerCase();
  if (!needle) return () => true;
  if (needle.includes("+") || needle.includes("#")) return (topic: string) => matches(needle, topic.toLowerCase());
  return (topic: string) => topic.toLowerCase().includes(needle);
}

function Summary({ topic, payload }: { topic: string; payload: unknown }) {
  if (payload === undefined || payload === null) return <span className="text-muted">—</span>;
  const fallback = <span className="text-muted">unreadable message</span>;
  return <>{safely(() => rendererFor(topic).summary(payload as Record<string, unknown>, topic), fallback)}</>;
}

function TopicName({ topic }: { topic: string }) {
  const Icon = iconFor(topic);
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <span className="grid h-7 w-7 shrink-0 place-items-center rounded-lg bg-raised text-glacier"><Icon className="h-3.5 w-3.5" /></span>
      <div className="min-w-0">
        <div className="truncate text-[13px] font-medium text-fg">{rendererFor(topic).title(topic)}</div>
        <div className="truncate font-mono text-[11px] text-muted">{topic}</div>
      </div>
    </div>
  );
}

export function Mqtt() {
  const { state, tail } = useConsole();
  const now = useNow();
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const [paused, setPaused] = useState<BoardMessage[] | null>(null);
  const [raw, setRaw] = useState(false);
  const selected = params.get("topic");
  const match = useMemo(() => filterBy(query), [query]);

  const rows = useMemo(
    () => (state?.topics ?? []).filter((row) => match(row.topic)).sort((a, b) => a.topic.localeCompare(b.topic)),
    [state?.topics, match],
  );
  const shownTail = (paused ?? tail).filter((message) => match(message.topic)).slice(-120).reverse();
  if (!state) return null;
  const selectedRow: TopicRow | undefined = state.topics.find((row) => row.topic === selected);
  const total = state.topics.reduce((sum, row) => sum + row.count, 0);
  const SelectedIcon = selectedRow ? iconFor(selectedRow.topic) : Network;

  return (
    <div className="mx-auto max-w-[1500px] px-4 py-6 sm:px-8">
      <PageHeader
        title="MQTT board"
        subtitle="Every component talks only through these topics, so this is the whole system's conversation. Filter by text or by an MQTT pattern such as space/sensor/+/state."
        actions={
          <>
            <span className="text-[12.5px] text-muted tabular">{state.topics.length} topics · {total.toLocaleString()} messages</span>
            <Button icon={paused ? Play : Pause} onClick={() => setPaused(paused ? null : tail)}>{paused ? "Resume" : "Pause"}</Button>
          </>
        }
      />

      <div className="relative mb-4 max-w-[560px]">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
        <input id="topic-filter" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Filter topics, e.g. sensor or space/fault/#" className={cn(inputClass, "pl-9")} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <Card padded={false} className="overflow-hidden">
          <div className="scroll-thin max-h-[640px] overflow-auto">
            <table className="w-full min-w-[600px] table-fixed text-[12.5px]">
              <thead className="sticky top-0 z-10 bg-surface">
                <tr className="text-left text-[11px] uppercase tracking-[0.06em] text-muted">
                  <th className="w-[38%] px-4 py-2.5 font-medium">Topic</th>
                  <th className="px-3 py-2.5 font-medium">Latest</th>
                  <th className="w-[60px] px-3 py-2.5 text-right font-medium">/min</th>
                  <th className="w-[84px] px-3 py-2.5 font-medium">Last</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.topic}
                    onClick={() => setParams({ topic: row.topic })}
                    className={cn("cursor-pointer border-t hairline hover:bg-raised/50", row.topic === selected && "bg-raised")}
                  >
                    <td className="px-4 py-2"><TopicName topic={row.topic} /></td>
                    <td className="truncate px-3 py-2 text-[12.5px]"><Summary topic={row.topic} payload={row.payload} /></td>
                    <td className="px-3 py-2 text-right font-mono text-fg tabular">{rateOf(row, now).toFixed(1)}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-muted">{ago(row.last_ts, now)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {rows.length === 0 && <Empty icon={Network} title="No topics match">Nothing published matches that filter since the console started.</Empty>}
          </div>
        </Card>

        <Card
          title={selectedRow ? rendererFor(selectedRow.topic).title(selectedRow.topic) : "Topic detail"}
          icon={SelectedIcon}
          action={selectedRow && (
            <button onClick={() => setRaw(!raw)} className={cn("flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12px] transition-colors", raw ? "bg-glacier/10 text-glacier" : "text-muted hover:bg-raised hover:text-fg")}>
              <Braces className="h-3.5 w-3.5" /> {raw ? "Readable" : "Raw JSON"}
            </button>
          )}
        >
          {selectedRow ? (
            <div className="grid gap-4">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px] text-muted">
                <span className="font-mono text-fg/80">{selectedRow.topic}</span>
                <span>from {producerOf(selectedRow.topic)?.name ?? "unknown"}</span>
                {selectedRow.retained && <span className="rounded bg-glacier/10 px-1.5 text-[10.5px] font-medium text-glacier">retained</span>}
              </div>
              {raw || selectedRow.payload === undefined || selectedRow.payload === null
                ? <Json value={selectedRow.payload} />
                : safely(() => rendererFor(selectedRow.topic).detail(selectedRow.payload as Record<string, unknown>, selectedRow.topic, now), <Json value={selectedRow.payload} />)}
              <div className="grid grid-cols-3 gap-3 border-t hairline pt-3">
                <div><Label>Messages</Label><div className="mt-0.5 font-mono text-[13px] text-fg tabular">{selectedRow.count.toLocaleString()}</div></div>
                <div><Label>Rate</Label><div className="mt-0.5 font-mono text-[13px] text-fg tabular">{rateOf(selectedRow, now).toFixed(1)}/min</div></div>
                <div><Label>Received</Label><div className="mt-0.5 text-[13px] text-fg">{ago(selectedRow.last_ts, now)}</div></div>
              </div>
            </div>
          ) : (
            <Empty icon={Network} title="Pick a topic">What it last said appears here, written out; the JSON as published is a click away.</Empty>
          )}
        </Card>
      </div>

      <Card className="mt-4" title={paused ? "Message stream · paused" : "Message stream"} icon={Network} padded={false}>
        <div className="scroll-thin max-h-[480px] overflow-auto">
          <ul>
            {shownTail.map((message, index) => {
              const Icon = iconFor(message.topic);
              return (
                <li
                  key={`${message.ts}-${message.topic}-${index}`}
                  onClick={() => setParams({ topic: message.topic })}
                  className={cn("flex cursor-pointer items-center gap-3 border-t hairline px-4 py-2 text-[12.5px] first:border-t-0 hover:bg-raised/40", message.topic === selected && "bg-raised/60")}
                >
                  <span className="w-[92px] shrink-0 whitespace-nowrap font-mono text-[11.5px] text-muted tabular">{clockSeconds(message.ts)}</span>
                  <Icon className="h-3.5 w-3.5 shrink-0 text-glacier" />
                  <span className="w-[200px] shrink-0 truncate font-medium text-fg/90" title={message.topic}>{rendererFor(message.topic).title(message.topic)}</span>
                  <span className="min-w-0 flex-1 truncate">{message.withdrawn ? <span className="text-ember">withdrawn: the retained message was cleared</span> : <Summary topic={message.topic} payload={message.payload} />}</span>
                </li>
              );
            })}
          </ul>
          {shownTail.length === 0 && <Empty icon={Network} title="Waiting for messages">Messages appear here as they are published.</Empty>}
        </div>
      </Card>
    </div>
  );
}
