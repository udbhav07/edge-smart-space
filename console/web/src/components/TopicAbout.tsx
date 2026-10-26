import { ArrowRight, BookOpen, ChevronDown } from "lucide-react";
import { Link } from "react-router-dom";
import { ACTOR_NAMES, docFor, type Actor, type TopicDoc } from "@/lib/docs";
import { cn } from "@/lib/format";

// The short answer to "what is this topic?", where the topic is shown: one
// sentence, who writes it and who reads it, and a way to the full guide.

/** A service, linking to its place in the architecture; the CLI tools have none. */
export function ActorChip({ who, className }: { who: Actor; className?: string }) {
  const chip = "inline-flex items-center gap-1 whitespace-nowrap rounded-md border hairline bg-raised/50 px-2 py-0.5 text-[12px] text-fg/90";
  if (who === "cli") return <span className={cn(chip, "text-muted", className)}>{ACTOR_NAMES[who]}</span>;
  return (
    <Link to={`/architecture#svc-${who === "devices" ? "simulator" : who}`} className={cn(chip, "transition-colors hover:border-glacier/40 hover:text-glacier", className)}>
      {ACTOR_NAMES[who]}
    </Link>
  );
}

export function Badge({ children, tone = "muted" }: { children: React.ReactNode; tone?: "muted" | "info" }) {
  return <span className={cn("whitespace-nowrap rounded px-1.5 py-px text-[10.5px] font-medium", tone === "info" ? "bg-glacier/10 text-glacier" : "bg-raised text-muted")}>{children}</span>;
}

/** Retention, delivery and rate: the topic's contract in three badges. */
export function ContractBadges({ doc }: { doc: TopicDoc }) {
  return (
    <span className="inline-flex flex-wrap gap-1.5">
      <Badge tone={doc.retained ? "info" : "muted"}>{doc.retained ? "retained" : "not retained"}</Badge>
      <Badge>{doc.qos === 1 ? "QoS 1 · never lost" : "QoS 0 · next one replaces it"}</Badge>
      <Badge>{doc.cadence}</Badge>
    </span>
  );
}

export function TopicAbout({ topic, open, onToggle }: { topic: string; open: boolean; onToggle: () => void }) {
  const doc = docFor(topic);
  if (!doc) return null;
  const producers = [...new Set(doc.producers.map((producer) => producer.who))];
  const consumers = [...new Set(doc.consumers.map((consumer) => consumer.who))];
  return (
    <div className="min-w-0 rounded-xl border hairline bg-raised/25">
      <button onClick={onToggle} aria-expanded={open} className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left">
        <BookOpen className="h-3.5 w-3.5 shrink-0 text-glacier" />
        <span className="shrink-0 text-[12.5px] font-medium text-fg">About this topic</span>
        {!open && <span className="min-w-0 flex-1 truncate text-[12.5px] text-muted">{doc.short}</span>}
        <ChevronDown className={cn("ml-auto h-4 w-4 shrink-0 text-muted transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div className="grid gap-3 px-3.5 pb-3.5">
          <p className="text-[13.5px] leading-relaxed text-fg/90">{doc.short}</p>
          <div className="grid gap-2 text-[12.5px] sm:grid-cols-[auto_1fr] sm:items-baseline sm:gap-x-3">
            <span className="text-muted">Written by</span>
            <span className="flex flex-wrap gap-1.5">{producers.map((who) => <ActorChip key={who} who={who} />)}</span>
            <span className="text-muted">Read by</span>
            <span className="flex flex-wrap gap-1.5">{consumers.map((who) => <ActorChip key={who} who={who} />)}</span>
            <span className="text-muted">Contract</span>
            <ContractBadges doc={doc} />
          </div>
          <Link to={`/topics/${doc.slug}`} className="inline-flex w-fit items-center gap-1.5 text-[12.5px] font-medium text-glacier hover:underline">
            Full guide: why each reader needs it, every field, the live message <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      )}
    </div>
  );
}
