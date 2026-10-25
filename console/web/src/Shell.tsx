import { Activity, Command as CommandIcon, Menu, Mic, Wifi, WifiOff, X } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { Toaster, toast } from "sonner";
import { api, Refused } from "@/lib/api";
import { celsius, cn, modeLabel, speechAlive } from "@/lib/format";
import { NAV } from "@/lib/nav";
import { useConsole, useMailSeen, useNow } from "@/lib/useConsole";
import { CommandPalette } from "@/components/CommandPalette";
import { Orb } from "@/components/Orb";

function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const { state, link } = useConsole();
  const now = useNow();
  const mailSeen = useMailSeen();
  const unreadMail = (state?.inbox ?? []).filter((item) => item.ts > mailSeen).length;
  const waiting = state?.waiting.length ?? 0;
  const faults = state?.faults.length ?? 0;
  const badge: Record<string, number> = { "/bookings": waiting, "/faults": faults };
  const voice = speechAlive(state?.speech ?? null, now);
  return (
    <nav className="flex h-full flex-col gap-6 px-3 py-4" aria-label="Pages">
      <div className="flex items-center gap-2.5 px-2">
        <Orb presence={link === "offline" ? "offline" : "idle"} size={28} />
        <div className="leading-tight">
          <div className="text-[14px] font-semibold tracking-tight text-fg">Smart Space</div>
          <div className="text-[11.5px] text-muted">Room 1 · edge node</div>
        </div>
      </div>
      {NAV.map((section) => (
        <div key={section.group} className="grid gap-0.5">
          <div className="px-2.5 pb-1 text-[11px] font-medium uppercase tracking-[0.08em] text-muted/80">{section.group}</div>
          {section.items.map((item) => {
            const Icon = item.icon;
            const count = item.to === "/mail" ? unreadMail : badge[item.to] ?? 0;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end ?? false}
                onClick={onNavigate}
                className={({ isActive }) =>
                  cn(
                    "group flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-[13.5px] transition",
                    isActive ? "bg-raised text-fg" : "text-muted hover:bg-raised/60 hover:text-fg",
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {item.label}
                {count > 0 && (
                  <span className={cn("ml-auto rounded-full px-1.5 text-[11px] font-medium tabular", item.to === "/faults" ? "bg-alert/15 text-alert" : "bg-glacier/15 text-glacier")}>
                    {count}
                  </span>
                )}
              </NavLink>
            );
          })}
        </div>
      ))}
      <div className="mt-auto grid gap-2 rounded-xl border hairline p-3 text-[12px]">
        <div className="flex items-center gap-2 text-muted">
          {link === "offline" ? <WifiOff className="h-3.5 w-3.5 text-alert" /> : <Wifi className={cn("h-3.5 w-3.5", link === "live" ? "text-good" : "text-ember")} />}
          {link === "live" ? "Live updates" : link === "polling" ? "Reconnecting · polling" : link === "offline" ? "Console offline" : "Connecting…"}
        </div>
        <div className="flex items-center gap-2 text-muted">
          <Activity className={cn("h-3.5 w-3.5", state?.broker ? "text-good" : "text-alert")} />
          {state?.broker ? "Blackboard connected" : "Blackboard unreachable"}
        </div>
        <div className="flex items-center gap-2 text-muted">
          <Mic className={cn("h-3.5 w-3.5", voice ? "text-good" : "text-muted/60")} />
          {!voice ? "Voice off" : state?.speech?.listen_mode === "continuous" && state.speech.activity === "waiting" ? "Always listening" : `Voice: ${state?.speech?.activity}`}
        </div>
      </div>
    </nav>
  );
}

function Topbar({ onMenu, onPalette }: { onMenu: () => void; onPalette: () => void }) {
  const { state } = useConsole();
  const room = state?.room;
  const mode = room?.mode?.mode;
  const indoor = room?.health.temp_01?.quality === "faulted" ? room?.thermal?.t_pred : room?.sensors.temp_01?.value;
  const location = useLocation();
  const title = NAV.flatMap((section) => section.items).find((item) => (item.to === "/" ? location.pathname === "/" : location.pathname.startsWith(item.to)))?.label ?? "";
  return (
    <header className="hairline-b glass sticky top-0 z-30 flex items-center gap-3 px-4 py-2.5 sm:px-6">
      <button onClick={onMenu} className="rounded-lg p-1.5 text-muted hover:bg-raised hover:text-fg lg:hidden" aria-label="Open the menu">
        <Menu className="h-5 w-5" />
      </button>
      <span className="text-[13.5px] font-medium text-fg lg:hidden">{title}</span>
      <div className="ml-auto flex items-center gap-2">
        {room && (
          <span className="hidden items-center gap-2 rounded-full border hairline px-3 py-1 text-[12.5px] sm:flex">
            <span className="text-muted">Room</span>
            <span className="font-mono text-fg tabular">{celsius(indoor)}</span>
            <span className="text-muted">holding</span>
            <span className="font-mono text-glacier tabular">{celsius(room.tracking?.tracked_c)}</span>
          </span>
        )}
        {mode && (
          <span className={cn("flex items-center gap-1.5 rounded-full px-3 py-1 text-[12.5px] font-medium", mode === "NORMAL" ? "bg-good/10 text-good" : "bg-ember/10 text-ember")}>
            <span className={cn("h-1.5 w-1.5 rounded-full", mode === "NORMAL" ? "bg-good" : "animate-pulse bg-ember")} />
            {modeLabel[mode]}
          </span>
        )}
        <button onClick={onPalette} className="flex items-center gap-1.5 rounded-lg border hairline px-2.5 py-1 text-[12px] text-muted hover:text-fg" aria-label="Commands">
          <CommandIcon className="h-3.5 w-3.5" /> K
        </button>
      </div>
    </header>
  );
}

export function Shell() {
  const { state } = useConsole();
  const [menu, setMenu] = useState(false);
  const [palette, setPalette] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => setMenu(false), [location.pathname]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPalette((open) => !open);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const guard = (promise: Promise<unknown>, done?: string) =>
    promise.then(() => { if (done) toast.success(done); }).catch((error: Refused) => toast.error(error.message));

  return (
    <div className="atmosphere flex h-full">
      <aside className="hidden w-[248px] shrink-0 border-r hairline lg:block">
        <Sidebar />
      </aside>
      {menu && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/50" onClick={() => setMenu(false)} />
          <div className="absolute inset-y-0 left-0 w-[264px] bg-surface shadow-2xl">
            <button onClick={() => setMenu(false)} className="absolute right-3 top-3 rounded-lg p-1.5 text-muted hover:bg-raised" aria-label="Close the menu"><X className="h-4 w-4" /></button>
            <Sidebar onNavigate={() => setMenu(false)} />
          </div>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <Topbar onMenu={() => setMenu(true)} onPalette={() => setPalette(true)} />
        <main className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          {state ? <Outlet /> : <Connecting />}
        </main>
      </div>
      <CommandPalette
        open={palette}
        onOpenChange={setPalette}
        operator={!!state?.operator_controls}
        actions={{
          ask: (text) => { navigate("/chat"); void guard(api.say(text, "")); },
          focusComposer: () => navigate("/chat"),
          setTarget: (value) => void guard(api.setTarget(value, ""), `Asked for ${value}°`),
          go: (to) => navigate(to),
          refreshCalendar: () => void guard(api.refreshCalendar()),
          inject: (subject, fault, magnitude) => void guard(api.inject(subject, fault, magnitude), fault === "NONE" ? `Cleared ${subject}` : `Injected at ${subject}`),
          reset: () => void guard(api.reset("checked from the console"), "Reset sent"),
        }}
      />
      <Toaster theme="system" position="bottom-right" toastOptions={{ className: "!bg-surface !text-fg !border-[rgb(var(--line)/0.08)]" }} />
    </div>
  );
}

function Connecting() {
  const { link } = useConsole();
  return (
    <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
      <Orb presence={link === "offline" ? "offline" : "thinking"} size={72} />
      <div className="text-[15px] text-fg">{link === "offline" ? "Can't reach the console server" : "Connecting to the room…"}</div>
      <p className="max-w-[44ch] text-[13px] text-muted">
        {link === "offline" ? "The room keeps running on its own. Start it with python start.py, or the console alone with python -m src.console." : "Waiting for the first snapshot."}
      </p>
    </div>
  );
}
