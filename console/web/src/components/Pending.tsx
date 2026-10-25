import { Check, Clock, X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import type { Waiting } from "@/lib/types";
import { Button, Card } from "@/components/ui";

/** Commitments waiting for the occupant's yes (FR-54): nothing has happened yet. */
export function Pending({ items, now, noun }: { items: Waiting[]; now: number; noun: string }) {
  const [busy, setBusy] = useState<string | null>(null);
  if (!items.length) return null;
  const answer = async (id: string, yes: boolean) => {
    setBusy(id);
    try {
      await api.answer(id, yes);
      toast.success(yes ? "Confirmed" : `Declined; no ${noun} was made`);
    } catch (error) {
      toast.error((error as Refused).message);
    } finally {
      setBusy(null);
    }
  };
  return (
    <Card title={`Waiting for your yes · ${items.length}`} icon={Clock} className="border-glacier/30">
      <ul className="grid gap-3 md:grid-cols-2">
        {items.map((item) => {
          const expired = item.expires_ts < now;
          return (
            <li key={item.invocation_id} className="rounded-xl border border-glacier/25 bg-glacier/[0.04] p-4">
              <div className="text-[14.5px] font-medium text-fg">{item.summary}</div>
              <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-4 gap-y-0.5 text-[12.5px]">
                {Object.entries(item.arguments).filter(([, value]) => value !== null && value !== "").map(([key, value]) => (
                  <div key={key} className="contents">
                    <dt className="text-muted">{key.replace(/_/g, " ")}</dt>
                    <dd className="truncate font-mono text-fg/85">{String(value)}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-2 text-[11.5px] text-muted">
                A mock: nothing real happens. {expired ? "This request has expired." : `Expires in ${Math.max(1, Math.round((item.expires_ts - now) / 60))} min.`}
              </p>
              <div className="mt-3 flex gap-2">
                <Button variant="primary" icon={Check} disabled={busy !== null || expired} onClick={() => void answer(item.invocation_id, true)}>Confirm</Button>
                <Button icon={X} disabled={busy !== null} onClick={() => void answer(item.invocation_id, false)}>Decline</Button>
              </div>
            </li>
          );
        })}
      </ul>
    </Card>
  );
}
