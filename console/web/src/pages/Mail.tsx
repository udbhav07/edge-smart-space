import { Inbox, Mail as MailIcon, PenLine, Send } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { ago, clock, cn } from "@/lib/format";
import type { InboxItem, SentMail } from "@/lib/types";
import { markMailSeen, useConsole, useNow } from "@/lib/useConsole";
import { Button, Card, Dot, Empty, Field, PageHeader, inputClass, type Health } from "@/components/ui";
import { Pending } from "@/components/Pending";

// Mail, as this node can honestly do it (NFR-06: nothing leaves it). The
// inbox is what the room tells the occupant -- fault explanations, booking
// confirmations, calendar changes. Sending reaches a mock outbox: like a
// booking it waits for a yes, and nothing is ever actually sent.

type Box = "inbox" | "sent" | "compose";
const toneHealth: Record<string, Health> = { good: "good", warn: "warn", alert: "alert", info: "idle", neutral: "idle" };

function Compose({ onSent }: { onSent: () => void }) {
  const [to, setTo] = useState("");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    try {
      await api.sendMail(to, subject, body);
      toast.success("Written; confirm it above to send the mock");
      setTo(""); setSubject(""); setBody("");
      onSent();
    } catch (error) {
      toast.error((error as Refused).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <form onSubmit={submit} className="grid gap-3">
      <Field label="To"><input id="mail-to" required value={to} onChange={(event) => setTo(event.target.value)} placeholder="Anu, or anu@example.com" className={inputClass} /></Field>
      <Field label="Subject"><input id="mail-subject" value={subject} onChange={(event) => setSubject(event.target.value)} placeholder="Running late" className={inputClass} /></Field>
      <Field label="Message"><textarea id="mail-body" rows={8} value={body} onChange={(event) => setBody(event.target.value)} placeholder="I'll be ten minutes late. Sorry!" className={cn(inputClass, "resize-y")} /></Field>
      <div className="flex flex-wrap items-center gap-3">
        <Button type="submit" variant="primary" icon={Send} disabled={busy || !to.trim() || (!subject.trim() && !body.trim())}>Send</Button>
        <span className="text-[12px] text-muted">It waits for your confirmation, then goes to a mock outbox. Nothing leaves this node.</span>
      </div>
    </form>
  );
}

function Reading({ item }: { item: InboxItem | SentMail | null }) {
  if (!item) return <Empty icon={MailIcon} title="Nothing selected">Pick a message to read it here.</Empty>;
  const sent = "reference" in item;
  return (
    <article className="grid gap-3">
      <div>
        <h2 className="text-balance text-[18px] font-semibold text-fg">{item.subject || "(no subject)"}</h2>
        <div className="mt-1 text-[12.5px] text-muted">
          {sent ? `To ${(item as SentMail).to} · ${(item as SentMail).reference} · mock, not sent` : `From ${(item as InboxItem).sender}`} · {clock(item.ts)}
        </div>
      </div>
      <p className="whitespace-pre-wrap text-[14.5px] leading-relaxed text-fg/90">{item.body || "(empty)"}</p>
    </article>
  );
}

export function Mail() {
  const { state } = useConsole();
  const now = useNow();
  const [box, setBox] = useState<Box>("inbox");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  useEffect(() => markMailSeen(), [state?.inbox.length]);
  const inbox = useMemo(() => [...(state?.inbox ?? [])].reverse(), [state?.inbox]);
  const sent = useMemo(() => [...(state?.sent ?? [])].reverse(), [state?.sent]);
  if (!state) return null;
  const waiting = state.waiting.filter((item) => item.tool === "send_mail");
  const list: (InboxItem | SentMail)[] = box === "sent" ? sent : inbox;
  const key = (item: InboxItem | SentMail) => ("id" in item ? item.id : `${item.reference}-${item.ts}`);
  const selected = list.find((item) => key(item) === selectedId) ?? list[0] ?? null;

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Mail"
        subtitle="What the room tells you, and mail you send through Jarvis. This node never connects to the internet, so sending reaches a mock outbox: like a booking, it waits for your yes and nothing actually leaves."
        actions={<Button variant="primary" icon={PenLine} onClick={() => setBox("compose")}>Write</Button>}
      />
      <div className="grid gap-4">
        <Pending items={waiting} now={now} noun="mail" />
        <div className="grid gap-4 lg:grid-cols-[200px_minmax(0,1fr)]">
          <nav className="flex gap-1 lg:flex-col" aria-label="Mailboxes">
            {([["inbox", "Inbox", Inbox, inbox.length], ["sent", "Sent (mock)", Send, sent.length], ["compose", "Write", PenLine, 0]] as const).map(([value, label, Icon, count]) => (
              <button key={value} onClick={() => setBox(value)} className={cn("flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13.5px]", box === value ? "bg-raised text-fg" : "text-muted hover:bg-raised/60 hover:text-fg")}>
                <Icon className="h-4 w-4" />{label}
                {count > 0 && <span className="ml-auto text-[11.5px] text-muted tabular">{count}</span>}
              </button>
            ))}
          </nav>
          {box === "compose" ? (
            <Card title="New mail" icon={PenLine}><Compose onSent={() => setBox("sent")} /></Card>
          ) : (
            <div className="grid min-h-[480px] gap-4 xl:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
              <Card padded={false} className="overflow-hidden">
                {list.length === 0 ? (
                  <Empty icon={box === "sent" ? Send : Inbox} title={box === "sent" ? "Nothing sent" : "Inbox is empty"}>
                    {box === "sent" ? "Ask Jarvis to email someone, or write one here." : "The room writes here when something happens: a fault, a booking, a calendar change."}
                  </Empty>
                ) : (
                  <ul className="scroll-thin max-h-[640px] divide-y divide-[rgb(var(--line)/var(--line-alpha))] overflow-y-auto">
                    {list.map((item) => {
                      const id = key(item);
                      const isSent = "reference" in item;
                      return (
                        <li key={id}>
                          <button onClick={() => setSelectedId(id)} className={cn("grid w-full gap-0.5 px-4 py-3 text-left hover:bg-raised/40", selected && key(selected) === id && "bg-raised")}>
                            <span className="flex items-center gap-2 text-[13px]">
                              {!isSent && <Dot health={toneHealth[(item as InboxItem).tone] ?? "idle"} />}
                              <span className="font-medium text-fg">{isSent ? `To ${(item as SentMail).to}` : (item as InboxItem).sender}</span>
                              <span className="ml-auto text-[11.5px] text-muted">{ago(item.ts, now)}</span>
                            </span>
                            <span className="truncate text-[13px] text-fg/90">{item.subject || "(no subject)"}</span>
                            <span className="truncate text-[12px] text-muted">{item.body}</span>
                          </button>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </Card>
              <Card><Reading item={selected} /></Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
