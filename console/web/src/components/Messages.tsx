import { AlertTriangle, Check, CheckCheck, Home as HomeIcon, MessageCircle, Play, RotateCcw, Volume2 } from "lucide-react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import type { HomeMessage } from "@/lib/types";
import { Card } from "@/components/ui";

// Messages home (FR-95): someone away speaks, the room plays it in their own
// voice and listens for an answer. The sender sees each step as ticks; the
// room keeps the last few to hear again.

let player: HTMLAudioElement | null = null;

/** Play a kept recording on this device -- one at a time. */
export function hear(url: string) {
  player?.pause();
  player = new Audio(url);
  void player.play().catch(() => toast.error("That recording could not be played here."));
}

const seconds = (value: number | null | undefined) => (value ? `${Math.max(1, Math.round(value))} s` : "");

/** Sent ✓, played ✓✓, answered — as a messaging app shows it. */
export function MessageTicks({ message }: { message: HomeMessage }) {
  const { state } = message;
  if (state === "failed")
    return (
      <span className="flex items-center gap-1 text-[12px] text-alert">
        <AlertTriangle className="h-3.5 w-3.5" /> Not played{message.detail ? ` — ${message.detail}` : ""}
      </span>
    );
  const label = { sent: "Sent", playing: "Playing at home…", played: "Played at home", replied: "Played · answered" }[state];
  const Icon = state === "sent" ? Check : CheckCheck;
  return (
    <span className={cn("flex items-center gap-1 text-[12px]", state === "sent" ? "text-muted" : "text-glacier")}>
      <Icon className={cn("h-3.5 w-3.5", state === "playing" && "animate-pulse")} /> {label}
    </span>
  );
}

function PlayButton({ label, onClick, icon: Icon = Play }: { label: string; onClick: () => void; icon?: typeof Play }) {
  return (
    <button type="button" aria-label={label} title={label} onClick={onClick} className="rounded-full p-1.5 text-muted hover:bg-raised hover:text-fg">
      <Icon className="h-4 w-4" />
    </button>
  );
}

/** The reply from home, in their own voice. */
export function ReplyBubble({ message }: { message: HomeMessage }) {
  if (message.state !== "replied") return null;
  return (
    <div className="flex max-w-[85%] items-start gap-1.5 justify-self-start">
      <div className="rounded-2xl bg-raised px-3 py-1.5 text-fg">
        <div className="mb-0.5 flex items-center gap-1 text-[11.5px] font-semibold text-muted"><HomeIcon className="h-3 w-3" /> Home · {seconds(message.reply_duration_s)}</div>
        {message.reply_transcript || "…"}
      </div>
      <PlayButton label="Hear the answer" icon={Volume2} onClick={() => hear(api.replyAudio(message.message_id))} />
    </div>
  );
}

/** The messages this device sent, and what home said back. */
export function SentMessages({ messages }: { messages: HomeMessage[] }) {
  if (!messages.length) return null;
  return (
    <ul className="mt-3 grid gap-3 border-t hairline pt-4">
      {messages.slice(-5).map((message) => (
        <li key={message.message_id} className="grid gap-1.5 text-[14px]">
          <div className="flex max-w-[85%] items-start gap-1.5 justify-self-end">
            <PlayButton label="Hear your message" onClick={() => hear(api.messageAudio(message.message_id))} />
            <div className="rounded-2xl bg-glacier px-3 py-1.5 text-white">{message.transcript || `A ${seconds(message.duration_s)} message`}</div>
          </div>
          <div className="justify-self-end"><MessageTicks message={message} /></div>
          <ReplyBubble message={message} />
        </li>
      ))}
    </ul>
  );
}

/** The room's view: who sent what, hear it again, here or aloud in the room. */
export function MessagesCard({ messages }: { messages: HomeMessage[] }) {
  const now = Date.now() / 1000;
  const again = async (message: HomeMessage) => {
    try {
      await api.replayMessage(message.message_id);
      toast.success(`Playing ${message.sender}'s message in the room`);
    } catch (refused) {
      toast.error((refused as Refused).message);
    }
  };
  return (
    <Card title="Messages" icon={MessageCircle}>
      {!messages.length ? (
        <p className="text-[13.5px] text-muted">When someone away sends a message from the Talk page, it plays here in their voice, and you can answer straight after.</p>
      ) : (
        <ul className="grid divide-y divide-[rgb(var(--line)/var(--line-alpha))]">
          {messages.slice(0, 5).map((message) => (
            <li key={message.message_id} className="grid gap-1 py-2.5 first:pt-0 last:pb-0">
              <div className="flex items-baseline gap-2">
                <span className="text-[14.5px] font-semibold text-fg">{message.sender}</span>
                <span className="text-[12px] text-muted">{ago(message.sent_ts, now)} · {seconds(message.duration_s)}</span>
                <span className="ml-auto flex items-center">
                  <PlayButton label="Hear it here" onClick={() => hear(api.messageAudio(message.message_id))} />
                  <PlayButton label="Play it in the room again" icon={RotateCcw} onClick={() => void again(message)} />
                </span>
              </div>
              {message.transcript && <p className="text-[13.5px] text-fg/85">“{message.transcript}”</p>}
              {message.state === "replied" ? (
                <div className="flex items-center gap-1 text-[13px] text-muted">
                  <span>You answered: <span className="text-fg/85">“{message.reply_transcript}”</span></span>
                  <PlayButton label="Hear your answer" icon={Volume2} onClick={() => hear(api.replyAudio(message.message_id))} />
                </div>
              ) : (
                <MessageTicks message={message} />
              )}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}
