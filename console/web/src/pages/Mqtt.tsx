import { Network, Pause, Play, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ago, clockSeconds, cn } from "@/lib/format";
import { matches, producerOf, rateOf } from "@/lib/services";
import type { BoardMessage, TopicRow } from "@/lib/types";
import { useConsole, useNow } from "@/lib/useConsole";
import { Button, Card, Empty, Json, Label, PageHeader, inputClass } from "@/components/ui";

// The blackboard as it is: every topic anything has published since the
// console started, how often, whether the topic contract retains it, and the
// last payload verbatim. Below it, the messages as they arrive (FR-60).

function short(payload: unknown): string {
  if (payload === undefined) return "";
  const text = typeof payload === "string" ? payload : JSON.stringify(payload);
  return text.length > 160 ? `${text.slice(0, 160)}…` : text;
}

function filterBy(query: string) {
  const needle = query.trim().toLowerCase();
  if (!needle) return () => true;
  if (needle.includes("+") || needle.includes("#")) return (topic: string) => matches(needle, topic.toLowerCase());
  return (topic: string) => topic.toLowerCase().includes(needle);
}

export function Mqtt() {
  const { state, tail } = useConsole();
  const now = useNow();
  const [params, setParams] = useSearchParams();
  const [query, setQuery] = useState("");
  const [paused, setPaused] = useState<BoardMessage[] | null>(null);
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
          <div className="scroll-thin max-h-[560px] overflow-auto">
            <table className="w-full min-w-[640px] text-[12.5px]">
              <thead className="sticky top-0 bg-surface">
                <tr className="text-left text-[11px] uppercase tracking-[0.06em] text-muted">
                  <th className="px-4 py-2.5 font-medium">Topic</th>
                  <th className="px-3 py-2.5 font-medium">From</th>
                  <th className="px-3 py-2.5 text-right font-medium">/min</th>
                  <th className="px-3 py-2.5 text-right font-medium">Count</th>
                  <th className="px-3 py-2.5 font-medium">Last</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.topic}
                    onClick={() => setParams({ topic: row.topic })}
                    className={cn("cursor-pointer border-t hairline hover:bg-raised/50", row.topic === selected && "bg-raised")}
                  >
                    <td className="max-w-[340px] px-4 py-2">
                      <div className="flex items-center gap-2">
                        <span className="truncate font-mono text-fg/90">{row.topic}</span>
                        {row.retained && <span className="shrink-0 rounded bg-glacier/10 px-1.5 text-[10.5px] font-medium text-glacier">retained</span>}
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-3 py-2 text-muted">{producerOf(row.topic)?.name ?? "—"}</td>
                    <td className="px-3 py-2 text-right font-mono text-fg tabular">{rateOf(row, now).toFixed(1)}</td>
                    <td className="px-3 py-2 text-right font-mono text-muted tabular">{row.count.toLocaleString()}</td>
                    <td className="whitespace-nowrap px-3 py-2 text-muted">{ago(row.last_ts, now)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {rows.length === 0 && <Empty icon={Network} title="No topics match">Nothing published matches that filter since the console started.</Empty>}
          </div>
        </Card>

        <Card title={selectedRow ? "Last message" : "Topic detail"} icon={Network}>
          {selectedRow ? (
            <div className="grid gap-3">
              <div className="break-all font-mono text-[13px] text-fg">{selectedRow.topic}</div>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <div><Label>From</Label><div className="mt-0.5 text-[13px] text-fg">{producerOf(selectedRow.topic)?.name ?? "—"}</div></div>
                <div><Label>Retained</Label><div className="mt-0.5 text-[13px] text-fg">{selectedRow.retained ? "Yes" : "No"}</div></div>
                <div><Label>Messages</Label><div className="mt-0.5 font-mono text-[13px] text-fg tabular">{selectedRow.count.toLocaleString()}</div></div>
                <div><Label>Received</Label><div className="mt-0.5 font-mono text-[13px] text-fg">{clockSeconds(selectedRow.last_ts)}</div></div>
              </div>
              <Json value={selectedRow.payload} />
            </div>
          ) : (
            <Empty icon={Network} title="Pick a topic">Its last payload appears here exactly as it was published.</Empty>
          )}
        </Card>
      </div>

      <Card className="mt-4" title={paused ? "Message stream · paused" : "Message stream"} icon={Network} padded={false}>
        <div className="scroll-thin max-h-[420px] overflow-auto">
          <table className="w-full min-w-[720px] text-[12px]">
            <tbody>
              {shownTail.map((message, index) => (
                <tr key={`${message.ts}-${message.topic}-${index}`} className="border-t hairline align-top hover:bg-raised/40" onClick={() => setParams({ topic: message.topic })}>
                  <td className="whitespace-nowrap px-4 py-1.5 font-mono text-muted tabular">{clockSeconds(message.ts)}</td>
                  <td className="whitespace-nowrap px-3 py-1.5 font-mono text-glacier">{message.topic}</td>
                  <td className="px-3 py-1.5 font-mono text-fg/80">{message.withdrawn ? <span className="text-ember">withdrawn</span> : short(message.payload)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {shownTail.length === 0 && <Empty icon={Network} title="Waiting for messages">Messages appear here as they are published.</Empty>}
        </div>
      </Card>
    </div>
  );
}
