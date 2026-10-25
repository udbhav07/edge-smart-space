import { Check, ChevronDown, Ear, X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { cn } from "@/lib/format";
import type { Profile } from "@/lib/types";

// Facts overheard about someone, waiting for them (FR-78). Nothing here is
// remembered until its person keeps it; dismissing deletes it. Shown above
// the conversation, out of the way, and only while something is waiting.

export function Suggestions({ profiles }: { profiles: Profile[] }) {
  const [open, setOpen] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const waiting = profiles.flatMap((profile) =>
    (profile.suggestions ?? []).map((fact) => ({ profile, fact, key: `${profile.profile_id}:${fact}` })),
  );
  if (waiting.length === 0) return null;

  const settle = async (profile: Profile, fact: string, keep: boolean, key: string) => {
    setBusy(key);
    try {
      await api.settleFact(profile.profile_id, fact, keep);
      toast.success(keep ? `Remembering that about ${profile.display_name || profile.profile_id}` : "Dismissed; it is not kept");
    } catch (error) {
      toast.error((error as Refused).message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="hairline-b bg-ember/[0.04] px-4 py-2.5 sm:px-8">
      <button onClick={() => setOpen(!open)} aria-expanded={open} className="flex w-full items-center gap-2 text-left">
        <Ear className="h-3.5 w-3.5 text-ember" />
        <span className="text-[13px] font-medium text-fg">Overheard · {waiting.length} to keep or dismiss</span>
        <span className="hidden text-[12px] text-muted sm:inline">Jarvis remembers these only if you say so.</span>
        <ChevronDown className={cn("ml-auto h-4 w-4 text-muted transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <ul className="mt-2 grid gap-1.5 pb-1">
          {waiting.map(({ profile, fact, key }) => (
            <li key={key} className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-xl border border-dashed border-ember/40 bg-surface/70 px-3 py-2">
              <span className="rounded-md bg-raised px-1.5 py-0.5 text-[11.5px] text-muted">{profile.display_name || profile.profile_id}</span>
              <span className="min-w-0 flex-1 text-[13.5px] text-fg">{fact}</span>
              <span className="flex gap-1.5">
                <button
                  disabled={busy === key}
                  onClick={() => void settle(profile, fact, true, key)}
                  className="inline-flex items-center gap-1 rounded-lg bg-glacier px-2.5 py-1 text-[12px] font-medium text-[rgb(var(--ink))] hover:brightness-110 disabled:opacity-50"
                >
                  <Check className="h-3.5 w-3.5" /> Keep
                </button>
                <button
                  disabled={busy === key}
                  onClick={() => void settle(profile, fact, false, key)}
                  className="inline-flex items-center gap-1 rounded-lg px-2.5 py-1 text-[12px] font-medium text-muted hover:bg-raised hover:text-fg disabled:opacity-50"
                >
                  <X className="h-3.5 w-3.5" /> Dismiss
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
