import { CalendarDays, Link2, Mic, Thermometer } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { ago, cn } from "@/lib/format";
import { DEFAULT_SPEECH_SETTINGS, type SpeechSettings } from "@/lib/types";
import { useConsole, useNow } from "@/lib/useConsole";
import { Button, Card, Field, PageHeader, Pill, Switch, inputClass } from "@/components/ui";

// What the occupant chooses once and leaves: the Google account, how the room
// listens and speaks, and the presets Home offers (FR-79, FR-82, FR-84).

const guard = (promise: Promise<unknown>, done?: string) =>
  promise.then(() => { if (done) toast.success(done); }).catch((error: Refused) => toast.error(error.message));

function Google() {
  const { state } = useConsole();
  const now = useNow();
  const google = state?.google;
  const disconnect = () => {
    if (!window.confirm("Disconnect Google? Calendar entries stay on this node; nothing more is sent to Google.")) return;
    void guard(api.disconnectGoogle(), "Disconnected from Google");
  };
  return (
    <Card title="Google Calendar and Gmail" icon={Link2}>
      {google?.connected ? (
        <div className="grid gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <Pill health={google.online ? "good" : "warn"}>{google.online ? "Connected" : "Connected · offline"}</Pill>
            <span className="text-[14px] text-fg">{google.account || "your Google account"}</span>
          </div>
          <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1 text-[13px]">
            <dt className="text-muted">Waiting to send</dt><dd className="text-fg tabular">{google.outbox === 0 ? "Nothing" : `${google.outbox} (sent when the system is back online)`}</dd>
            <dt className="text-muted">Last synced</dt><dd className="text-fg">{google.last_sync_ts ? ago(google.last_sync_ts, now) : "not yet"}</dd>
            {google.last_error && <><dt className="text-muted">Last problem</dt><dd className="text-ember">{google.last_error}</dd></>}
          </dl>
          <p className="text-[12.5px] text-muted">Your calendar is kept on this node and mirrored to Google both ways. Mail you confirm is sent from your Gmail; bookings add a calendar entry and email you the itinerary.</p>
          <div className="flex gap-2"><Button variant="danger" onClick={disconnect}>Disconnect</Button></div>
        </div>
      ) : google?.needs_reconnect ? (
        <div className="grid gap-3">
          <Pill health="warn">Sign-in expired</Pill>
          <p className="text-[13.5px] text-fg/90">Google stopped accepting the sign-in (in testing mode it lasts seven days). Nothing is lost: what was waiting is still queued.</p>
          <div><a href="/api/google/connect" className="inline-flex rounded-full bg-glacier px-4 py-[7px] text-[14px] font-medium text-white hover:brightness-110">Reconnect Google</a></div>
        </div>
      ) : google?.configured ? (
        <div className="grid gap-3">
          <p className="text-[13.5px] text-fg/90">Connect your Google account to put your calendar in Google Calendar and send confirmed mail from Gmail. Until then everything stays on this node, and mail goes to a mock outbox.</p>
          <div><a href="/api/google/connect" className="inline-flex rounded-full bg-glacier px-4 py-[7px] text-[14px] font-medium text-white hover:brightness-110">Connect Google</a></div>
          <p className="text-[12px] text-muted">The room asks only to manage calendar events and to send mail, never to read your inbox. The sign-in is kept in a file on this node and never shared.</p>
        </div>
      ) : (
        <div className="grid gap-2 text-[13.5px] text-fg/90">
          <p>Google isn't set up on this node yet. Once, on a computer:</p>
          <ol className="ml-5 list-decimal space-y-1 text-[13px]">
            <li>Create a project at console.cloud.google.com and enable the Google Calendar API and the Gmail API.</li>
            <li>Set up the OAuth consent screen as External, in Testing, and add your Gmail as a test user.</li>
            <li>Create an OAuth client ID of type Web application, with the redirect URI <code className="rounded bg-raised px-1 font-mono text-[12px]">http://localhost:8077/api/google/callback</code>.</li>
            <li>Download its JSON to <code className="rounded bg-raised px-1 font-mono text-[12px]">state/google/client_secret.json</code> and restart the system.</li>
          </ol>
          <p className="text-[12px] text-muted">Open this console at http://localhost:8077 when you connect, so Google can send you back. The README has the details.</p>
        </div>
      )}
    </Card>
  );
}

function Voice() {
  const { state } = useConsole();
  const settings = state?.speech_settings ?? DEFAULT_SPEECH_SETTINGS;
  const [volume, setVolume] = useState(settings.volume);
  const [quiet, setQuiet] = useState({ from: settings.quiet_from, to: settings.quiet_to });
  useEffect(() => setVolume(settings.volume), [settings.volume]);
  useEffect(() => setQuiet({ from: settings.quiet_from, to: settings.quiet_to }), [settings.quiet_from, settings.quiet_to]);
  const change = (changes: Partial<SpeechSettings>, done?: string) => void guard(api.speechSettings(changes), done);
  const voice = state?.speech?.voice;
  return (
    <Card title="Voice" icon={Mic}>
      <div className="grid divide-y divide-[rgb(var(--line)/var(--line-alpha))]">
        <Switch checked={settings.listening} onChange={(listening) => change({ listening })} label={settings.listening ? "Microphone on" : "Microphone muted"} hint="Muted means nothing is captured at all, not captured and ignored." />
        <Switch checked={settings.speak_replies} onChange={(speak_replies) => change({ speak_replies })} label="Speak replies to what you say" />
        <Switch checked={settings.speak_typed} onChange={(speak_typed) => change({ speak_typed })} label="Also speak replies to what you type" />
        <Switch checked={settings.speak_alerts} onChange={(speak_alerts) => change({ speak_alerts })} label="Say alerts aloud" hint="When the room changes mode, for example when the air conditioner is put on hold." />
      </div>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Field label={`Volume · ${Math.round(volume * 100)}%`}>
          <input type="range" min={0} max={1} step={0.05} value={volume} onChange={(event) => setVolume(Number(event.target.value))} onPointerUp={() => change({ volume })} onKeyUp={() => change({ volume })} className="w-full accent-[rgb(var(--glacier))]" aria-label="Volume" />
        </Field>
        <Field label="Quiet hours" hint="No alerts are spoken between these times; replies still are.">
          <div className="flex items-center gap-2">
            <input type="time" aria-label="Quiet from" value={quiet.from} onChange={(event) => setQuiet({ ...quiet, from: event.target.value })} className={cn(inputClass, "py-1.5")} />
            <span className="text-muted">to</span>
            <input type="time" aria-label="Quiet until" value={quiet.to} onChange={(event) => setQuiet({ ...quiet, to: event.target.value })} className={cn(inputClass, "py-1.5")} />
          </div>
        </Field>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <Button disabled={!!quiet.from !== !!quiet.to} onClick={() => change({ quiet_from: quiet.from, quiet_to: quiet.to }, quiet.from ? `Quiet from ${quiet.from} to ${quiet.to}` : "No quiet hours")}>Save quiet hours</Button>
        {(settings.quiet_from || quiet.from) && <Button variant="ghost" onClick={() => { setQuiet({ from: "", to: "" }); change({ quiet_from: "", quiet_to: "" }, "No quiet hours"); }}>Clear</Button>}
        <span className="ml-auto text-[12px] text-muted">{voice ? `Speaking with ${voice}` : state?.speech ? "No voice is available on this node" : "The speech service is not running"}</span>
      </div>
    </Card>
  );
}

function Climate() {
  const { state } = useConsole();
  if (!state) return null;
  const [low, high] = state.bounds_c ?? [18, 30];
  return (
    <Card title="Climate" icon={Thermometer}>
      <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-1.5 text-[13.5px]">
        {Object.entries(state.presets ?? {}).map(([name, value]) => (
          <div key={name} className="contents"><dt className="text-muted">{name}</dt><dd className="text-fg tabular">{value}°</dd></div>
        ))}
        <dt className="text-muted">Cooling off</dt><dd className="text-fg">Off unless the room passes {state.cooling_off_c}°</dd>
        <dt className="text-muted">Allowed range</dt><dd className="text-fg tabular">{low}° to {high}°</dd>
      </dl>
      <p className="mt-3 text-[12px] text-muted">Every target goes through the room's safety gate, which may move toward it in steps. The presets are set in config/default.yaml.</p>
    </Card>
  );
}

export function Settings() {
  return (
    <div className="mx-auto max-w-[900px] px-4 py-6 sm:px-8">
      <PageHeader title="Settings" subtitle="Your Google account, how the room listens and speaks, and the comfort presets." />
      <div className="grid grid-cols-[minmax(0,1fr)] gap-4">
        <Google />
        <Voice />
        <Climate />
        <Card title="Privacy" icon={CalendarDays}>
          <p className="text-[13.5px] text-fg/90">Speech is recognised on this node; nothing you say leaves it. Only once Google is connected does the room reach the internet, and only Google Calendar and Gmail.</p>
        </Card>
      </div>
    </div>
  );
}
