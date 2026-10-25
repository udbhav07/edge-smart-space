import { ArrowRight, BookOpen, Cable, Cpu, EyeOff, HardDrive, Layers, Lock, Radio, ShieldCheck, Terminal, Unplug } from "lucide-react";
import { useEffect } from "react";
import { Link, useLocation } from "react-router-dom";
import { ACTOR_NAMES, SERVICE_DOCS, docBySlug, topicsOfActor, type Actor } from "@/lib/docs";
import { ago, cn } from "@/lib/format";
import { SERVICES, judge } from "@/lib/services";
import { useConsole, useNow } from "@/lib/useConsole";
import { Card, Dot, Label, PageHeader, Pill, type Health } from "@/components/ui";

// How the system works, for someone seeing it for the first time: the layers,
// the shared board they talk through, four walks through what happens, and
// each service with what it writes and reads. Statuses are live.

const LAYERS: { name: string; role: string; services: Actor[]; note: string }[] = [
  { name: "Layer 4 · Reasoning", role: "Thinks and talks", services: ["reasoning"], note: "A local language model plans setpoints, answers people and explains faults. It can only propose; it commands nothing." },
  { name: "Layer 3 · Supervision", role: "Watches for trouble", services: ["detectors"], note: "Five detectors and the mode manager decide what can be trusted and how cautiously to run." },
  { name: "Layer 2 · Estimation & control", role: "Understands and regulates", services: ["estimator", "control"], note: "Learns the room's physics and holds it at the setpoint, every command through the safety gate." },
  { name: "Layer 1 · The world", role: "Senses, acts, listens", services: ["simulator", "speech", "assistance", "console"], note: "Sensors and the air conditioner (simulated or real), the microphone, the tools, and this console." },
];

interface Step {
  title: string;
  text: string;
  topic?: string;
}

const FLOWS: { id: string; title: string; intro: string; steps: Step[] }[] = [
  {
    id: "loop",
    title: "Every 5 seconds: the control loop",
    intro: "This runs without any language model, any network beyond the broker, or this console. It is what keeps the room comfortable.",
    steps: [
      { title: "Sensors report", text: "Indoor temperature, humidity, presence and the AC's power draw are published.", topic: "sensor-state" },
      { title: "The model predicts", text: "The estimator compares the reading with what its learned model expected and refines the model.", topic: "estimate-thermal" },
      { title: "The controller decides", text: "Above the setpoint by more than 0.5 °C: cool. Below by more than 0.5 °C: switch off. Otherwise: maintain." },
      { title: "The safety gate rules", text: "Range, step size, compressor rest time, command spacing, the mode. Every ruling is published.", topic: "audit-validation" },
      { title: "The AC is told", text: "The command goes out, and is re-sent every 60 s because infrared can be lost.", topic: "actuator-command" },
      { title: "The meter confirms", text: "A draw above 200 W means the compressor is really running: that is the acknowledgement.", topic: "actuator-state" },
    ],
  },
  {
    id: "voice",
    title: "When you speak, or type in Chat",
    intro: "Your words become a preference, never a command. The same gate rules on it as on everything else.",
    steps: [
      { title: "Wake word", text: "The speech service hears “Hey Jarvis”, records what follows, and transcribes it here on this machine.", topic: "speech-status" },
      { title: "Understood", text: "A small model call extracts the intent: warmer, cooler, a named temperature, a question, a task.", topic: "context-preference" },
      { title: "Weighed", text: "The goal manager turns a comfort wish into a setpoint and puts it to the safety gate." },
      { title: "Ruled on", text: "Ask for 10 °C and it is clamped to 18 °C, and the audit says so.", topic: "audit-validation" },
      { title: "Answered", text: "The assistant replies, saying what was actually applied.", topic: "context-reply" },
    ],
  },
  {
    id: "fault",
    title: "When something breaks",
    intro: "Protection comes first and fast; explanation follows. A slow model never delays the response.",
    steps: [
      { title: "A detector confirms", text: "A sensor goes silent, freezes, reads the impossible, drifts, or the AC ignores its commands.", topic: "fault" },
      { title: "The mode changes within 2 s", text: "DEGRADED_SENSOR or DEGRADED_ACTUATOR, published before any model is asked.", topic: "system-mode" },
      { title: "The system adapts", text: "The estimator stops learning; without the thermometer, the controller regulates on the model's prediction." },
      { title: "It is explained", text: "The language model writes what is probably wrong and what it recommends.", topic: "diagnosis" },
      { title: "Too long, it holds", text: "Degraded for 30 minutes: SAFE_HOLD, until a person checks the room and resets it.", topic: "system-reset" },
    ],
  },
  {
    id: "task",
    title: "When you ask for a task",
    intro: "Calendar, travel and mail go through a tool executor with three levels of trust.",
    steps: [
      { title: "The assistant asks", text: "“Book me a hotel in Pune for Friday” becomes a tool request on the board.", topic: "assist-proposed" },
      { title: "The executor checks", text: "Unknown tools and bad arguments are refused. Reads run at once; calendar writes run at once because they can be undone." },
      { title: "Commits wait for you", text: "Bookings and mail are held until you say yes, by voice or on this console, and lapse if you don't.", topic: "assist-confirmed" },
      { title: "The result comes back", text: "Every request gets exactly one outcome. Bookings and mail are mock: nothing real is ever booked or sent.", topic: "assist-result" },
    ],
  },
];

const PRINCIPLES = [
  { icon: ShieldCheck, title: "The model never touches the AC", text: "Language models propose. A deterministic validator decides, and every decision is published with its reason." },
  { icon: Radio, title: "If it is not on the board, it did not happen", text: "Every decision, fault, verdict and model call is a message anyone can read, which is how this console sees everything." },
  { icon: Unplug, title: "Everything degrades, nothing stops", text: "Kill the model, speech or this console and the room is still regulated. Each service is its own process." },
  { icon: Cable, title: "Hardware is a configuration change", text: "Layers 2–4 never import Layer 1. The simulator and the real sensors publish the same messages on the same topics." },
  { icon: Lock, title: "Local only", text: "Speech, transcription and the language model run on this machine. Audio is never published; nothing goes to the internet." },
  { icon: EyeOff, title: "A voice unlocks nothing", text: "Recognising a speaker personalises replies. Anything that commits needs an explicit yes." },
];

function useHealth(id: Actor): { health: Health; word: string; last: number | null } {
  const { state } = useConsole();
  const now = useNow();
  const spec = SERVICES.find((service) => service.id === (id === "devices" ? "simulator" : id));
  if (!state || !spec) return { health: "idle", word: "unknown", last: null };
  const judged = judge(spec, state.topics, now);
  return { health: judged.health, word: judged.word, last: judged.last_ts };
}

function ServiceBox({ id }: { id: Actor }) {
  const { health } = useHealth(id);
  return (
    <a href={`#svc-${id}`} className="flex min-w-0 items-center gap-2 rounded-xl border hairline bg-surface px-3 py-2 text-[13px] text-fg transition-colors hover:border-glacier/40">
      <Dot health={health} />
      <span className="truncate">{ACTOR_NAMES[id]}</span>
    </a>
  );
}

function TopicLink({ slug }: { slug: string }) {
  const doc = docBySlug(slug);
  if (!doc) return null;
  return (
    <Link to={`/topics/${slug}`} className="inline-flex max-w-full items-center gap-1 truncate rounded-md bg-glacier/10 px-1.5 py-0.5 font-mono text-[11.5px] text-glacier hover:bg-glacier/20">
      {doc.pattern}
    </Link>
  );
}

function Flow({ title, intro, steps }: { title: string; intro: string; steps: Step[] }) {
  return (
    <Card>
      <h3 className="text-[15px] font-semibold text-fg">{title}</h3>
      <p className="mt-1 text-[13px] text-muted">{intro}</p>
      <ol className="mt-4 grid gap-0">
        {steps.map((step, index) => (
          <li key={step.title} className="relative grid grid-cols-[28px_1fr] gap-3 pb-4 last:pb-0">
            {index < steps.length - 1 && <span className="absolute left-[13.5px] top-7 h-[calc(100%-24px)] w-px bg-[rgb(var(--line)/0.15)]" />}
            <span className="grid h-7 w-7 place-items-center rounded-full border border-glacier/40 bg-glacier/10 font-mono text-[12px] text-glacier">{index + 1}</span>
            <div className="min-w-0 pt-0.5">
              <div className="text-[13.5px] font-medium text-fg">{step.title}</div>
              <p className="mt-0.5 text-[13px] leading-relaxed text-fg/75">{step.text}</p>
              {step.topic && <div className="mt-1.5"><TopicLink slug={step.topic} /></div>}
            </div>
          </li>
        ))}
      </ol>
    </Card>
  );
}

function ServiceSection({ id }: { id: Actor }) {
  const doc = SERVICE_DOCS.find((service) => service.id === id)!;
  const spec = SERVICES.find((service) => service.id === id);
  const now = useNow();
  const { health, word, last } = useHealth(id);
  const { produces, consumes } = topicsOfActor(id);
  return (
    <section id={`svc-${id}`} className="scroll-mt-20 rounded-2xl border hairline bg-surface/80 p-5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <h3 className="text-[16px] font-semibold text-fg">{ACTOR_NAMES[id]}</h3>
        {spec && <span className="text-[12px] text-muted">{spec.layer}</span>}
        <span className="ml-auto flex items-center gap-2">
          <span className="text-[12px] text-muted">{last ? `last heard ${ago(last, now)}` : ""}</span>
          <Pill health={health}>{word}</Pill>
        </span>
      </div>
      <div className="mt-0.5 font-mono text-[12px] text-muted">python -m {doc.module}</div>
      <p className="mt-3 text-[14px] font-medium text-fg">{doc.summary}</p>
      <div className="mt-1.5 grid gap-2 text-[13.5px] leading-relaxed text-fg/80">
        {doc.details.map((paragraph) => <p key={paragraph.slice(0, 40)}>{paragraph}</p>)}
      </div>
      <div className="mt-4 grid gap-3 border-t hairline pt-3 md:grid-cols-2">
        <div>
          <Label>Writes</Label>
          <div className="mt-1.5 flex flex-wrap gap-1.5">{produces.length ? produces.map((topic) => <TopicLink key={topic.slug} slug={topic.slug} />) : <span className="text-[12.5px] text-muted">nothing</span>}</div>
        </div>
        <div>
          <Label>Reads</Label>
          <div className="mt-1.5 flex flex-wrap gap-1.5">{consumes.length ? consumes.map((topic) => <TopicLink key={topic.slug} slug={topic.slug} />) : <span className="text-[12.5px] text-muted">nothing</span>}</div>
        </div>
      </div>
    </section>
  );
}

export function Architecture() {
  const { hash } = useLocation();
  useEffect(() => {
    if (!hash) return;
    // The sections render with the page, so the target exists by the next frame.
    const frame = requestAnimationFrame(() => document.getElementById(hash.slice(1))?.scrollIntoView({ behavior: "smooth", block: "start" }));
    return () => cancelAnimationFrame(frame);
  }, [hash]);

  return (
    <div className="mx-auto max-w-[1200px] px-4 py-6 sm:px-8">
      <PageHeader
        title="How it works"
        subtitle="An edge controller for one room: it learns how the room heats and cools, holds it where you want it, notices when a sensor or the air conditioner fails, and lets you talk to it. Everything runs on this machine."
        actions={<Link to="/topics" className="inline-flex items-center gap-1.5 rounded-lg border hairline bg-raised/60 px-3 py-1.5 text-[13px] font-medium text-fg hover:bg-raised"><BookOpen className="h-3.5 w-3.5" /> Topic guide</Link>}
      />

      <nav className="mb-6 flex flex-wrap gap-1.5 text-[12.5px]">
        {[["layers", "The layers"], ["flows", "What happens when"], ["services", "The services"], ["principles", "Design rules"], ["running", "Running it"]].map(([id, label]) => (
          <a key={id} href={`#${id}`} className="rounded-full border hairline px-3 py-1 text-muted transition-colors hover:border-glacier/40 hover:text-fg">{label}</a>
        ))}
      </nav>

      {/* --- layers and the board ---------------------------------------------- */}
      <section id="layers" className="scroll-mt-20">
        <Card title="Four layers, one shared board" icon={Layers}>
          <p className="mb-4 max-w-[80ch] text-[13.5px] leading-relaxed text-fg/80">
            The system is eight separate programs. None of them calls another: each one publishes what it knows to an MQTT broker and subscribes to what it needs. That broker is the <em>blackboard</em>. It is the only wiring, so anything on it can be watched, recorded and replayed, and any program can be restarted on its own. Higher layers depend only on the messages, never on the code below them.
          </p>
          <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_64px]">
            <div className="grid gap-2.5">
              {LAYERS.map((layer) => (
                <div key={layer.name} className="grid gap-3 rounded-xl bg-raised/30 p-3.5 lg:grid-cols-[220px_minmax(0,1fr)]">
                  <div>
                    <div className="text-[13px] font-semibold text-fg">{layer.name}</div>
                    <div className="text-[12px] text-glacier">{layer.role}</div>
                    <p className="mt-1 text-[12.5px] leading-relaxed text-muted">{layer.note}</p>
                  </div>
                  <div className="grid content-center gap-2 sm:grid-cols-2 xl:grid-cols-4">
                    {layer.services.map((id) => <ServiceBox key={id} id={id} />)}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex items-center justify-center gap-2 rounded-xl border border-glacier/30 bg-glacier/5 px-3 py-3 md:flex-col">
              <Radio className="h-4 w-4 text-glacier" />
              <span className="text-[12px] font-medium text-glacier md:[writing-mode:vertical-rl]">MQTT blackboard · space/#</span>
            </div>
          </div>
          <p className="mt-3 text-[12.5px] text-muted">The dots are live: each service is judged by whether it is still publishing at its own pace. Click one to read about it.</p>
        </Card>
      </section>

      {/* --- flows ---------------------------------------------------------------- */}
      <section id="flows" className="mt-8 scroll-mt-20">
        <h2 className="mb-3 text-[18px] font-semibold text-fg">What happens when…</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          {FLOWS.map((flow) => <Flow key={flow.id} {...flow} />)}
        </div>
      </section>

      {/* --- services --------------------------------------------------------------- */}
      <section id="services" className="mt-8 scroll-mt-20">
        <h2 className="text-[18px] font-semibold text-fg">The services</h2>
        <p className="mb-3 mt-1 max-w-[80ch] text-[13.5px] text-muted">
          Each runs as its own process. “Writes” and “Reads” are taken from the code: every topic it publishes and every one it subscribes to. Click a topic to see what it means and what it says now.
        </p>
        <div className="grid gap-3">
          {SERVICE_DOCS.map((service) => <ServiceSection key={service.id} id={service.id} />)}
        </div>
      </section>

      {/* --- principles --------------------------------------------------------------- */}
      <section id="principles" className="mt-8 scroll-mt-20">
        <h2 className="mb-3 text-[18px] font-semibold text-fg">Design rules</h2>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {PRINCIPLES.map(({ icon: Icon, title, text }) => (
            <div key={title} className="rounded-2xl border hairline bg-surface/80 p-4">
              <Icon className="h-4 w-4 text-glacier" />
              <div className="mt-2 text-[14px] font-medium text-fg">{title}</div>
              <p className="mt-1 text-[13px] leading-relaxed text-fg/75">{text}</p>
            </div>
          ))}
        </div>
      </section>

      {/* --- running ------------------------------------------------------------------- */}
      <section id="running" className="mt-8 scroll-mt-20">
        <Card title="Running it" icon={Terminal}>
          <div className="grid gap-4 md:grid-cols-3">
            {[
              { icon: Radio, title: "The broker", text: "Mosquitto on port 1883. Every service connects to it and nothing else.", code: "mosquitto" },
              { icon: Cpu, title: "The language model", text: "Ollama serving a small model locally. Optional: without it, reasoning degrades and control carries on.", code: "ollama serve" },
              { icon: HardDrive, title: "Everything else", text: "Starts every service as its own process, simulator or real devices according to the configuration.", code: "python start.py" },
            ].map(({ icon: Icon, title, text, code }) => (
              <div key={title} className="rounded-xl bg-raised/30 p-3.5">
                <div className="flex items-center gap-2"><Icon className="h-3.5 w-3.5 text-muted" /><span className="text-[13.5px] font-medium text-fg">{title}</span></div>
                <p className="mt-1 text-[12.5px] leading-relaxed text-fg/75">{text}</p>
                <code className="mt-2 block rounded-lg bg-ink/70 px-2.5 py-1.5 font-mono text-[12px] text-fg/90">{code}</code>
              </div>
            ))}
          </div>
          <p className={cn("mt-4 text-[12.5px] text-muted")}>
            Faults can be injected from <Link to="/faults" className="text-glacier hover:underline">Faults & control</Link> or with <code className="font-mono">python -m tools.inject</code>; everything on the board can be watched live on the <Link to="/mqtt" className="inline-flex items-center gap-0.5 text-glacier hover:underline">MQTT board <ArrowRight className="h-3 w-3" /></Link>
          </p>
        </Card>
      </section>
    </div>
  );
}
