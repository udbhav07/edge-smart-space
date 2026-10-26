import { Music, Pause, Play, Search, SkipForward, Square, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { cn } from "@/lib/format";
import type { Device } from "@/lib/types";
import { Button, Switch, inputClass } from "@/components/ui";
import { Slider, boundedNumber, capability, has } from "./parts";
import type { RemoteProps, RemoteSpec } from "./types";

// A music player: what is playing, play and pause, a search, a file of your
// own, and the volume. A file is uploaded and played by name; no audio
// crosses the board. A song the library lacks is found online and streamed
// (FR-96); the search takes a few seconds, and the page says so.

function MusicRemote({ device, size, call }: RemoteProps) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [searching, setSearching] = useState(false);
  const file = useRef<HTMLInputElement>(null);
  const state = device.state as {
    playing?: boolean; paused?: boolean; starting?: boolean; track?: string; source?: string;
    volume?: number; tracks?: number; autoplay?: boolean; up_next?: string;
  };
  const canSkip = has(device, "next");
  const canAutoplay = has(device, "set_autoplay");
  const status = state.starting ? "Starting after Jarvis…" : state.playing ? "Playing" : state.paused ? "Paused" : "Stopped";
  const online = state.source === "youtube";
  const search = async () => {
    setSearching(true);
    try {
      await call("play", { query: query.trim() });
      setQuery("");
    } finally {
      setSearching(false);
    }
  };
  const off = !device.online;
  const volume = boundedNumber(capability(device, "set_volume"));
  const upload = async (chosen: File | undefined) => {
    if (!chosen) return;
    setBusy(true);
    try {
      await api.uploadMusic(chosen);
      toast.success(`Uploaded ${chosen.name}; playing it`);
    } catch (error) {
      toast.error((error as Refused).message);
    } finally {
      setBusy(false);
      if (file.current) file.current.value = "";
    }
  };
  return (
    <div className="grid gap-3">
      <div className={cn("flex items-center gap-3", size === "full" && "flex-col py-4 text-center")}>
        <span className={cn("flex items-center justify-center rounded-2xl", size === "full" ? "h-28 w-28" : "h-11 w-11", state.playing ? "bg-glacier text-white" : "bg-raised text-muted")}>
          <Music className={size === "full" ? "h-12 w-12" : "h-5 w-5"} />
        </span>
        <div className="min-w-0 flex-1">
          <div className={cn("truncate font-medium text-fg", size === "full" ? "text-[22px]" : "text-[15px]")}>{state.track || "Nothing playing"}</div>
          <div className="text-[12px] text-muted">{status}{online ? " · from YouTube" : ` · ${state.tracks ?? 0} ${state.tracks === 1 ? "track" : "tracks"} in your library`}</div>
        </div>
        <div className="flex gap-1.5">
          {state.playing
            ? <Button icon={Pause} disabled={off} onClick={() => void call("pause")}>Pause</Button>
            : <Button icon={Play} variant="primary" disabled={off} onClick={() => void call(state.paused ? "resume" : "play")}>Play</Button>}
          {canSkip && <Button icon={SkipForward} variant="ghost" disabled={off || !state.track} onClick={() => void call("next")}>Next</Button>}
          <Button icon={Square} variant="ghost" disabled={off || (!state.playing && !state.paused)} onClick={() => void call("stop")}>Stop</Button>
        </div>
      </div>
      {state.up_next && <div className={cn("truncate text-[12.5px] text-muted", size === "full" && "text-center")}>Up next: <span className="text-fg/85">{state.up_next}</span></div>}
      {size === "full" && <form onSubmit={(event) => { event.preventDefault(); if (query.trim()) void search(); }} className="flex gap-2">
        <input value={query} disabled={off || searching} onChange={(event) => setQuery(event.target.value)} placeholder="Play something… your music first, then YouTube" aria-label="What to play" className={cn(inputClass, "py-1.5")} />
        <Button type="submit" icon={Search} disabled={off || searching || !query.trim()}>{searching ? "Finding…" : "Play"}</Button>
        <input ref={file} type="file" accept=".mp3,.flac,.wav,.ogg,audio/*" className="hidden" onChange={(event) => void upload(event.target.files?.[0])} />
        <Button icon={Upload} disabled={off || busy} onClick={() => file.current?.click()}>{busy ? "Uploading…" : "Upload"}</Button>
      </form>}
      {size === "full" && canAutoplay && (
        <Switch
          checked={state.autoplay ?? false}
          disabled={off}
          onChange={(on) => void call("set_autoplay", { on })}
          label="Autoplay"
          hint="When a song ends, play the next: what YouTube would play after it, or another of yours."
        />
      )}
      {volume && <Slider title="Volume" parameter={volume} value={state.volume ?? Number(volume.maximum) / 2} disabled={off} onSet={(value) => void call("set_volume", { [volume.name]: value })} />}
    </div>
  );
}

export const musicRemote: RemoteSpec = {
  component: MusicRemote,
  fits: (device: Device) => has(device, "play", "pause", "stop"),
};
