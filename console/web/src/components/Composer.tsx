import * as Popover from "@radix-ui/react-popover";
import * as Slider from "@radix-ui/react-slider";
import { ArrowUp, Check, ChevronDown, Mic, Thermometer, UserRound } from "lucide-react";
import { forwardRef, useEffect, useState } from "react";
import type { Profile } from "@/lib/types";
import { celsius, cn } from "@/lib/format";

const BOUNDS = { low: 18, high: 30 };

export const SUGGESTIONS = [
  "It's a bit stuffy in here",
  "What's on my calendar today?",
  "Put a meeting with Anu tomorrow at 3pm",
  "Book a hotel in Pune for 2 nights from Saturday",
  "What do you know about me?",
];

function SpeakerPicker({ profiles, value, onChange }: { profiles: Profile[]; value: string; onChange: (id: string) => void }) {
  const current = profiles.find((profile) => profile.profile_id === value);
  const options = [{ profile_id: "", display_name: "Guest" }, ...profiles.filter((p) => p.profile_id !== "guest")];
  return (
    <Popover.Root>
      <Popover.Trigger className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12.5px] text-muted hover:bg-raised hover:text-fg">
        <UserRound className="h-3.5 w-3.5" />
        {current?.display_name || "Guest"}
        <ChevronDown className="h-3 w-3" />
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content side="top" align="start" sideOffset={8} className="z-50 w-64 rounded-xl border hairline bg-surface p-1.5 shadow-2xl">
          <div className="px-2.5 pb-1.5 pt-1 text-[11px] uppercase tracking-[0.08em] text-muted">Speaking as</div>
          {options.map((option) => (
            <Popover.Close
              key={option.profile_id || "guest"}
              onClick={() => onChange(option.profile_id)}
              className="flex w-full items-center justify-between rounded-lg px-2.5 py-1.5 text-left text-[13.5px] text-fg hover:bg-raised"
            >
              {option.display_name || option.profile_id}
              {option.profile_id === value && <Check className="h-3.5 w-3.5 text-glacier" />}
            </Popover.Close>
          ))}
          <p className="px-2.5 pb-1 pt-2 text-[11.5px] leading-snug text-muted">
            Personalises replies only. It unlocks nothing (FR-52).
          </p>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function ComfortDial({ current, held, onSet }: { current: number | null; held: number | null; onSet: (value: number) => void }) {
  const [value, setValue] = useState(current ?? 24);
  useEffect(() => {
    if (current !== null) setValue(current);
  }, [current]);
  return (
    <Popover.Root>
      <Popover.Trigger className="flex items-center gap-1.5 rounded-lg px-2 py-1 text-[12.5px] text-muted hover:bg-raised hover:text-fg">
        <Thermometer className="h-3.5 w-3.5" />
        {current !== null ? celsius(current) : "Comfort"}
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content side="top" align="start" sideOffset={8} className="z-50 w-72 rounded-xl border hairline bg-surface p-4 shadow-2xl">
          <div className="flex items-baseline justify-between">
            <span className="text-[11px] uppercase tracking-[0.08em] text-muted">Comfort</span>
            <span className="font-mono text-2xl text-fg tabular">{value.toFixed(1)}°</span>
          </div>
          <Slider.Root
            className="relative mt-4 flex h-5 w-full touch-none select-none items-center"
            min={BOUNDS.low}
            max={BOUNDS.high}
            step={0.5}
            value={[value]}
            onValueChange={([next]) => setValue(next)}
            aria-label="Target temperature"
          >
            <Slider.Track className="relative h-1.5 grow overflow-hidden rounded-full bg-raised">
              <Slider.Range className="absolute h-full rounded-full bg-gradient-to-r from-glacier to-ember" />
            </Slider.Track>
            <Slider.Thumb className="block h-4 w-4 rounded-full border-2 border-[rgb(var(--ink))] bg-fg shadow focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-glacier" />
          </Slider.Root>
          <div className="mt-1 flex justify-between font-mono text-[11px] text-muted">
            <span>{BOUNDS.low}°</span>
            <span>{BOUNDS.high}°</span>
          </div>
          {held !== null && <p className="mt-3 text-[12px] text-muted">The room is being held at {celsius(held)} right now.</p>}
          <p className="mt-1 text-[12px] leading-snug text-muted">
            Sent as a preference. The safety gate decides what is applied, exactly as for a spoken request.
          </p>
          <Popover.Close
            onClick={() => onSet(value)}
            className="mt-3 w-full rounded-lg bg-glacier py-1.5 text-[13px] font-medium text-[rgb(var(--ink))] hover:brightness-110"
          >
            Set {value.toFixed(1)}°
          </Popover.Close>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

export const Composer = forwardRef<
  HTMLTextAreaElement,
  {
    disabled: boolean;
    disabledReason: string;
    profiles: Profile[];
    speaker: string;
    onSpeaker: (id: string) => void;
    onSend: (text: string) => Promise<boolean>;
    onTarget: (value: number) => void;
    goal: number | null;
    held: number | null;
    voiceHint: string;
    showSuggestions: boolean;
  }
>(function Composer(props, ref) {
  const [text, setText] = useState("");
  const [sending, setSending] = useState(false);

  const send = async (value = text) => {
    const words = value.trim();
    if (!words || props.disabled || sending) return;
    setSending(true);
    const ok = await props.onSend(words);
    setSending(false);
    if (ok) setText("");
  };

  return (
    <div className="mx-auto w-full max-w-[760px] px-4 pb-4 sm:px-6 sm:pb-6">
      {props.showSuggestions && (
        <div className="mb-3 flex flex-wrap gap-2">
          {SUGGESTIONS.map((suggestion) => (
            <button
              key={suggestion}
              onClick={() => void send(suggestion)}
              disabled={props.disabled}
              className="rounded-full border hairline bg-surface/60 px-3 py-1 text-[12.5px] text-muted hover:border-glacier/40 hover:text-fg disabled:opacity-40"
            >
              {suggestion}
            </button>
          ))}
        </div>
      )}
      <div
        className={cn(
          "glass rounded-2xl border hairline shadow-[0_8px_40px_-12px_rgb(0_0_0/0.5)] transition-colors focus-within:border-glacier/40",
          props.disabled && "opacity-70",
        )}
      >
        <textarea
          ref={ref}
          id="composer"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void send();
            }
          }}
          rows={1}
          maxLength={500}
          placeholder={props.disabled ? props.disabledReason : "Ask Jarvis anything about the room, your calendar or a trip…"}
          disabled={props.disabled}
          className="block max-h-40 min-h-[52px] w-full resize-none bg-transparent px-4 pt-3.5 text-[15px] text-fg placeholder:text-muted/80 focus:outline-none"
        />
        <div className="flex items-center gap-1 px-2 pb-2">
          <SpeakerPicker profiles={props.profiles} value={props.speaker} onChange={props.onSpeaker} />
          <ComfortDial current={props.goal} held={props.held} onSet={props.onTarget} />
          <span className="ml-1 hidden items-center gap-1.5 text-[12px] text-muted sm:flex">
            <Mic className="h-3.5 w-3.5" />
            {props.voiceHint}
          </span>
          <button
            onClick={() => void send()}
            disabled={!text.trim() || props.disabled || sending}
            aria-label="Send"
            className="ml-auto flex h-8 w-8 items-center justify-center rounded-lg bg-fg text-[rgb(var(--ink))] transition hover:bg-glacier disabled:bg-raised disabled:text-muted"
          >
            <ArrowUp className="h-4 w-4" />
          </button>
        </div>
      </div>
      <p className="mt-2 text-center text-[11.5px] text-muted">
        Jarvis proposes; the safety gate decides. Nothing here can command the air conditioner directly.
      </p>
    </div>
  );
});
