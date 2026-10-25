import { ArrowUpRight, Cpu } from "lucide-react";
import { Link } from "react-router-dom";
import { ago, cn } from "@/lib/format";
import { SERVICES, judge, rateOf, topicsOf } from "@/lib/services";
import type { TopicRow } from "@/lib/types";
import { useConsole, useNow } from "@/lib/useConsole";
import { Card, Label, PageHeader, Pill } from "@/components/ui";

/** The most telling field of a payload, in a few words. */
function summarise(row: TopicRow): string {
  const payload = row.payload as Record<string, unknown> | null;
  if (!payload || typeof payload !== "object") return String(row.payload ?? "").slice(0, 80);
  const pick = (...keys: string[]) => keys.filter((key) => key in payload).map((key) => `${key} ${format(payload[key])}`).join(" · ");
  if ("sensor_id" in payload && "value" in payload) return pick("value", "unit", "quality");
  if ("t_in" in payload) return pick("t_in", "t_pred", "residual", "adaptation");
  if ("a1" in payload) return pick("a1", "a2", "a3", "a4");
  if ("verdict" in payload) return pick("verdict", "reason");
  if ("mode" in payload && "reason" in payload) return pick("mode", "reason");
  if ("kind" in payload && "actuator_id" in payload) return pick("kind", "ack");
  if ("activity" in payload) return pick("activity", "microphone");
  if ("caller" in payload) return pick("caller", "trigger", "latency_s");
  if ("tool" in payload && "status" in payload) return pick("tool", "status");
  if ("reply" in payload) return pick("reply");
  return Object.keys(payload).slice(0, 4).join(", ");
}

function format(value: unknown): string {
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(Math.abs(value) < 1 ? 4 : 2);
  if (typeof value === "string") return value.length > 40 ? `${value.slice(0, 40)}…` : value;
  return JSON.stringify(value)?.slice(0, 40) ?? "";
}

export function Services() {
  const { state } = useConsole();
  const now = useNow();
  if (!state) return null;
  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Services"
        subtitle="Each process and what it is producing right now. A service is judged by its own output against its own cadence: a tick-rate service that has gone quiet is silent, one that speaks only on change is not."
      />
      <div className="grid gap-4 lg:grid-cols-2">
        {SERVICES.map((service) => {
          const judged = judge(service, state.topics, now);
          const rows = topicsOf(service, state.topics).sort((a, b) => b.last_ts - a.last_ts);
          return (
            <Card key={service.id} padded={false}>
              <div className="flex items-start gap-3 px-5 pb-3 pt-4">
                <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-raised"><Cpu className="h-4 w-4 text-muted" /></span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h2 className="text-[15px] font-semibold text-fg">{service.name}</h2>
                    <span className="text-[11.5px] text-muted">{service.layer}</span>
                    <span className="ml-auto"><Pill health={judged.health}>{judged.word}</Pill></span>
                  </div>
                  <p className="mt-0.5 text-[12.5px] text-muted">{service.role}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-px border-y hairline bg-[rgb(var(--line)/var(--line-alpha))]">
                {[
                  ["Messages/min", judged.per_minute.toFixed(judged.per_minute < 10 ? 1 : 0)],
                  ["Seen", judged.messages.toLocaleString()],
                  ["Last output", judged.last_ts ? ago(judged.last_ts, now) : "never"],
                ].map(([label, value]) => (
                  <div key={label} className="bg-surface px-5 py-2.5">
                    <Label>{label}</Label>
                    <div className="mt-0.5 font-mono text-[15px] text-fg tabular">{value}</div>
                  </div>
                ))}
              </div>
              <ul className="divide-y divide-[rgb(var(--line)/var(--line-alpha))]">
                {rows.slice(0, 7).map((row) => (
                  <li key={row.topic}>
                    <Link to={`/mqtt?topic=${encodeURIComponent(row.topic)}`} className="group grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 px-5 py-2 hover:bg-raised/40">
                      <span className="truncate font-mono text-[12px] text-fg/90">{row.topic}</span>
                      <span className="flex items-center gap-1 font-mono text-[11.5px] text-muted tabular">
                        {rateOf(row, now).toFixed(1)}/min <ArrowUpRight className="h-3 w-3 opacity-0 group-hover:opacity-100" />
                      </span>
                      <span className="col-span-2 truncate text-[12px] text-muted">{summarise(row)}</span>
                    </Link>
                  </li>
                ))}
                {rows.length === 0 && <li className="px-5 py-4 text-[13px] text-muted">Nothing published by this service since the console started{service.cadence_s ? "; if it is running it should speak within a few seconds" : ""}.</li>}
                {rows.length > 7 && <li className={cn("px-5 py-2 text-[12px] text-muted")}>and {rows.length - 7} more topics on the MQTT board</li>}
              </ul>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
