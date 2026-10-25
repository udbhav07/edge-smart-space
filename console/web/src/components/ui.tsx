import { cn } from "@/lib/format";
import type { LucideIcon } from "lucide-react";

// The few building blocks every page shares. Cards are used by role -- a
// page's sections, not every number -- so the hierarchy stays readable.

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 pb-6">
      <div className="min-w-0">
        <h1 className="text-balance text-[22px] font-semibold tracking-tight text-fg">{title}</h1>
        {subtitle && <p className="mt-1 max-w-[72ch] text-[13.5px] text-muted">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </div>
  );
}

export function Card({ title, icon: Icon, action, children, className, padded = true }: {
  title?: React.ReactNode;
  icon?: LucideIcon;
  action?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <section className={cn("min-w-0 rounded-2xl border hairline bg-surface/80", className)}>
      {title && (
        <header className="flex items-center gap-2 px-5 pb-1 pt-4">
          {Icon && <Icon className="h-4 w-4 text-muted" />}
          <h2 className="text-[13px] font-medium text-fg">{title}</h2>
          {action && <div className="ml-auto">{action}</div>}
        </header>
      )}
      <div className={cn(padded && "px-5 pb-5 pt-3")}>{children}</div>
    </section>
  );
}

export function Label({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("text-[11px] font-medium uppercase tracking-[0.08em] text-muted", className)}>{children}</div>;
}

export type Health = "good" | "warn" | "alert" | "idle";

const healthDot: Record<Health, string> = { good: "bg-good", warn: "bg-ember", alert: "bg-alert", idle: "bg-muted/50" };
const healthText: Record<Health, string> = { good: "text-good", warn: "text-ember", alert: "text-alert", idle: "text-muted" };
const healthBg: Record<Health, string> = { good: "bg-good/10", warn: "bg-ember/10", alert: "bg-alert/10", idle: "bg-raised" };

/** A state: dot, word, and colour together -- never colour alone. */
export function Pill({ health, children, pulse }: { health: Health; children: React.ReactNode; pulse?: boolean }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-0.5 text-[12px] font-medium", healthBg[health], healthText[health])}>
      <span className={cn("h-1.5 w-1.5 rounded-full", healthDot[health], pulse && "animate-pulse")} />
      {children}
    </span>
  );
}

export function Dot({ health }: { health: Health }) {
  return <span className={cn("inline-block h-2 w-2 shrink-0 rounded-full", healthDot[health])} />;
}

export function Stat({ label, value, unit, note, health, icon: Icon }: {
  label: string;
  value: React.ReactNode;
  unit?: string;
  note?: React.ReactNode;
  health?: Health;
  icon?: LucideIcon;
}) {
  return (
    <div className="min-w-0 rounded-2xl border hairline bg-surface/80 px-4 py-3.5">
      <div className="flex items-center gap-2">
        {Icon && <Icon className="h-3.5 w-3.5 text-muted" />}
        <Label>{label}</Label>
        {health && <span className="ml-auto"><Dot health={health} /></span>}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span className="font-mono text-[28px] font-light leading-none text-fg tabular">{value}</span>
        {unit && <span className="text-[13px] text-muted">{unit}</span>}
      </div>
      {note && <div className="mt-1.5 truncate text-[12px] text-muted">{note}</div>}
    </div>
  );
}

export function Button({ children, onClick, variant = "secondary", disabled, type = "button", className, icon: Icon }: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "primary" | "secondary" | "ghost" | "danger";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
  icon?: LucideIcon;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-lg px-3 py-1.5 text-[13px] font-medium transition disabled:cursor-not-allowed disabled:opacity-50",
        variant === "primary" && "bg-glacier text-[rgb(var(--ink))] hover:brightness-110",
        variant === "secondary" && "border hairline bg-raised/60 text-fg hover:bg-raised",
        variant === "ghost" && "text-muted hover:bg-raised hover:text-fg",
        variant === "danger" && "border border-alert/30 text-alert hover:bg-alert/10",
        className,
      )}
    >
      {Icon && <Icon className="h-3.5 w-3.5" />}
      {children}
    </button>
  );
}

export function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-[12px] font-medium text-fg/85">{label}</span>
      {children}
      {hint && <span className="text-[11.5px] text-muted">{hint}</span>}
    </label>
  );
}

export const inputClass =
  "w-full rounded-lg border hairline bg-ink/60 px-3 py-2 text-[14px] text-fg placeholder:text-muted/70 focus:border-glacier/50 focus:outline-none";

export function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      <span className="flex h-10 w-10 items-center justify-center rounded-full bg-raised"><Icon className="h-5 w-5 text-muted" /></span>
      <div className="text-[14px] font-medium text-fg">{title}</div>
      {children && <div className="max-w-[46ch] text-[13px] text-muted">{children}</div>}
    </div>
  );
}

export function Json({ value }: { value: unknown }) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return (
    <pre className="scroll-thin max-h-[420px] overflow-auto rounded-xl bg-ink/70 p-3 font-mono text-[12px] leading-relaxed text-fg/90">{text}</pre>
  );
}
