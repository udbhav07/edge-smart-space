import { ArrowLeft, ArrowRight, BookOpen, Network, Radio } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { ACTOR_NAMES, TOPIC_DOCS, docBySlug, type TopicDoc } from "@/lib/docs";
import { ago } from "@/lib/format";
import { iconFor, rendererFor, safely } from "@/lib/renderers";
import { matches } from "@/lib/services";
import { useConsole, useNow } from "@/lib/useConsole";
import { Card, Empty, Json, Label, PageHeader } from "@/components/ui";
import { ActorChip, Badge, ContractBadges } from "@/components/TopicAbout";

// Every topic explained: an index grouped by what the topics are about, and a
// page per topic with who writes it, who reads it and why, its fields, and
// what it says right now.

const GROUPS: { group: TopicDoc["group"]; intro: string }[] = [
  { group: "Sensing", intro: "What the room's instruments measure, and whether they can be trusted." },
  { group: "Model", intro: "What the system has learned about the room, and how well it predicts it." },
  { group: "Control", intro: "The setpoint, the safety gate's input and output, and what the air conditioner is told." },
  { group: "Faults", intro: "Faults found, the mode they put the system in, and the operator's hand on it." },
  { group: "Conversation", intro: "Speech and chat: what was heard, understood and answered, and who said it." },
  { group: "Tools", intro: "Calendar, bookings, mail and memory, each behind its own gate." },
  { group: "Audit", intro: "The record of every decision the safety gate and the language model made." },
];

export function Topics() {
  const { state } = useConsole();
  const now = useNow();
  const live = (doc: TopicDoc) => (state?.topics ?? []).filter((row) => matches(doc.pattern, row.topic));

  return (
    <div className="mx-auto max-w-[1200px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Topic guide"
        subtitle={<>The services never call one another. Each one writes what it knows to a topic and reads the topics it needs: that shared board is the whole system's wiring. Here is every topic, what it means, who writes it and who reads it. See <Link to="/architecture" className="text-glacier hover:underline">How it works</Link> for the bigger picture.</>}
        actions={<Link to="/mqtt" className="inline-flex items-center gap-1.5 rounded-lg border hairline bg-raised/60 px-3 py-1.5 text-[13px] font-medium text-fg hover:bg-raised"><Network className="h-3.5 w-3.5" /> Live board</Link>}
      />

      <div className="grid gap-6">
        {GROUPS.map(({ group, intro }) => (
          <section key={group}>
            <div className="mb-2.5 flex flex-wrap items-baseline gap-x-3">
              <h2 className="text-[15px] font-semibold text-fg">{group}</h2>
              <p className="text-[13px] text-muted">{intro}</p>
            </div>
            <div className="grid gap-2.5 md:grid-cols-2">
              {TOPIC_DOCS.filter((doc) => doc.group === group).map((doc) => {
                const rows = live(doc);
                const last = rows.reduce<number | null>((latest, row) => (latest === null || row.last_ts > latest ? row.last_ts : latest), null);
                const Icon = iconFor(doc.pattern.replace(/\+/g, "temp_01"));
                return (
                  <Link key={doc.slug} to={`/topics/${doc.slug}`} className="group card rounded-[22px] p-4 transition-colors hover:border-glacier/40">
                    <div className="flex items-start gap-3">
                      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-raised text-glacier"><Icon className="h-4 w-4" /></span>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-[14px] font-medium text-fg">{doc.name}</span>
                          <ArrowRight className="ml-auto h-3.5 w-3.5 text-muted opacity-0 transition-opacity group-hover:opacity-100" />
                        </div>
                        <div className="font-mono text-[11.5px] text-muted">{doc.pattern}</div>
                        <p className="mt-1.5 text-[13px] leading-relaxed text-fg/80">{doc.short}</p>
                        <div className="mt-2.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px] text-muted">
                          <span>{[...new Set(doc.producers.map((p) => ACTOR_NAMES[p.who]))].join(", ")}</span>
                          <ArrowRight className="h-3 w-3" />
                          <span>{[...new Set(doc.consumers.map((c) => ACTOR_NAMES[c.who]))].join(", ")}</span>
                        </div>
                        <div className="mt-2 flex flex-wrap items-center gap-1.5">
                          <ContractBadges doc={doc} />
                          <span className="ml-auto text-[11.5px] text-muted">{last ? `heard ${ago(last, now)}` : "not heard yet"}</span>
                        </div>
                      </div>
                    </div>
                  </Link>
                );
              })}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

export function TopicGuide() {
  const { slug = "" } = useParams();
  const { state } = useConsole();
  const now = useNow();
  const doc = docBySlug(slug);

  if (!doc) {
    return (
      <div className="mx-auto max-w-[900px] px-4 py-10 sm:px-8">
        <Empty icon={BookOpen} title="No such topic">
          <Link to="/topics" className="text-glacier hover:underline">Back to the topic guide</Link>
        </Empty>
      </div>
    );
  }
  const rows = (state?.topics ?? []).filter((row) => matches(doc.pattern, row.topic)).sort((a, b) => a.topic.localeCompare(b.topic));
  const Icon = iconFor(rows[0]?.topic ?? doc.pattern.replace(/\+/g, "temp_01"));
  const index = TOPIC_DOCS.indexOf(doc);
  const previous = TOPIC_DOCS[index - 1];
  const next = TOPIC_DOCS[index + 1];

  return (
    <div className="mx-auto max-w-[1100px] px-4 py-6 sm:px-8">
      <Link to="/topics" className="mb-4 inline-flex items-center gap-1.5 text-[12.5px] text-muted hover:text-fg"><ArrowLeft className="h-3.5 w-3.5" /> Topic guide</Link>

      <div className="mb-6 flex items-start gap-4">
        <span className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-raised text-glacier"><Icon className="h-5 w-5" /></span>
        <div className="min-w-0">
          <div className="text-[12px] font-medium uppercase tracking-[0.08em] text-muted">{doc.group}</div>
          <h1 className="text-[24px] font-semibold tracking-tight text-fg">{doc.name}</h1>
          <div className="mt-0.5 font-mono text-[13px] text-muted">{doc.pattern}</div>
          <p className="mt-2 max-w-[70ch] text-[15px] leading-relaxed text-fg/90">{doc.short}</p>
        </div>
      </div>

      {/* Writers, the topic, readers: the flow in one picture. */}
      <Card className="mb-4">
        <div className="grid items-center gap-3 md:grid-cols-[1fr_auto_1.1fr_auto_1fr]">
          <div className="grid gap-1.5">
            <Label>Written by</Label>
            {doc.producers.map((producer) => (
              <div key={producer.who + producer.when} className="rounded-xl bg-raised/40 px-3 py-2">
                <ActorChip who={producer.who} />
                <div className="mt-1 text-[12px] text-muted">{producer.when}</div>
              </div>
            ))}
          </div>
          <ArrowRight className="mx-auto hidden h-5 w-5 text-muted md:block" />
          <div className="rounded-xl border border-glacier/30 bg-glacier/5 px-4 py-3 text-center">
            <Radio className="mx-auto h-4 w-4 text-glacier" />
            <div className="mt-1 break-all font-mono text-[13px] text-fg">{doc.pattern}</div>
            <div className="mt-2 flex justify-center"><ContractBadges doc={doc} /></div>
          </div>
          <ArrowRight className="mx-auto hidden h-5 w-5 text-muted md:block" />
          <div className="grid gap-1.5">
            <Label>Read by</Label>
            {doc.consumers.map((consumer) => <div key={consumer.who} className="rounded-xl bg-raised/40 px-3 py-2"><ActorChip who={consumer.who} /></div>)}
          </div>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="grid content-start gap-4">
          <Card title="What it means" icon={BookOpen}>
            <div className="grid gap-3 text-[14px] leading-relaxed text-fg/90">
              {doc.long.map((paragraph) => <p key={paragraph.slice(0, 40)}>{paragraph}</p>)}
            </div>
          </Card>
          <Card title="Who reads it, and why">
            <ul className="grid gap-2.5">
              {doc.consumers.map((consumer) => (
                <li key={consumer.who} className="grid gap-1 sm:grid-cols-[150px_1fr] sm:gap-3">
                  <span><ActorChip who={consumer.who} /></span>
                  <span className="text-[13.5px] text-fg/85">{consumer.why}</span>
                </li>
              ))}
            </ul>
          </Card>
          <Card title={`Fields · ${doc.schema}`}>
            <dl className="grid grid-cols-[minmax(120px,auto)_1fr] gap-x-4 gap-y-2 text-[13px]">
              {doc.fields.map((field) => (
                <div key={field.name} className="contents">
                  <dt className="font-mono text-[12.5px] text-glacier">{field.name}</dt>
                  <dd className="text-fg/85">{field.meaning}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-4 flex flex-wrap items-center gap-1.5 border-t hairline pt-3">
              <span className="mr-1 text-[12px] text-muted">Requirements</span>
              {doc.requirements.map((id) => <Badge key={id}>{id}</Badge>)}
            </div>
          </Card>
        </div>

        <div className="grid content-start gap-4">
          <Card title="Right now" icon={Radio} action={rows[0] && <Link to={`/mqtt?topic=${encodeURIComponent(rows[0].topic)}`} className="flex items-center gap-1 text-[12px] text-muted hover:text-fg">On the board <ArrowRight className="h-3 w-3" /></Link>}>
            {rows.length === 0 ? (
              <Empty icon={Radio} title="Not heard yet">{doc.retained ? "Nothing has been published on it since the broker started." : "It carries events, so it is quiet until one happens."}</Empty>
            ) : (
              <div className="grid gap-4">
                {rows.slice(0, 6).map((row) => (
                  <div key={row.topic} className="grid gap-2">
                    <div className="flex items-center gap-2 text-[12px]">
                      <Link to={`/mqtt?topic=${encodeURIComponent(row.topic)}`} className="font-mono text-fg/80 hover:text-glacier">{row.topic}</Link>
                      <span className="ml-auto text-muted">{ago(row.last_ts, now)}</span>
                    </div>
                    {rows.length === 1
                      ? safely(() => rendererFor(row.topic).detail(row.payload as Record<string, unknown>, row.topic, now), <Json value={row.payload} />)
                      : <div className="truncate text-[12.5px]">{safely(() => rendererFor(row.topic).summary(row.payload as Record<string, unknown>, row.topic), null)}</div>}
                  </div>
                ))}
                {rows.length > 6 && <div className="text-[12px] text-muted">and {rows.length - 6} more on the board</div>}
              </div>
            )}
          </Card>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap justify-between gap-3 border-t hairline pt-4 text-[13px]">
        {previous ? <Link to={`/topics/${previous.slug}`} className="flex items-center gap-1.5 text-muted hover:text-fg"><ArrowLeft className="h-3.5 w-3.5" /> {previous.name}</Link> : <span />}
        {next && <Link to={`/topics/${next.slug}`} className="flex items-center gap-1.5 text-muted hover:text-fg">{next.name} <ArrowRight className="h-3.5 w-3.5" /></Link>}
      </div>
    </div>
  );
}
