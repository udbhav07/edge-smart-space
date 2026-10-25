import { cn } from "@/lib/format";
import type { LucideIcon } from "lucide-react";

// The few building blocks every page shares. Cards are used by role -- a
// page's sections, not every number -- so the hierarchy stays readable.

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-4 pb-8 pt-2">
      <div className="min-w-0">
        <h1 className="display text-balance text-[34px] font-bold leading-[1.1] text-fg sm:text-[40px]">{title}</h1>
        {subtitle && <p className="mt-2 max-w-[68ch] text-[15px] leading-relaxed text-muted">{subtitle}</p>}
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
    <section className={cn("card min-w-0 rounded-[22px]", className)}>
      {title && (
        <header className="flex items-center gap-2 px-6 pb-1 pt-5">
          {Icon && <Icon className="h-[18px] w-[18px] text-muted" />}
          <h2 className="text-[17px] font-semibold tracking-[-0.02em] text-fg">{title}</h2>
          {action && <div className="ml-auto">{action}</div>}
        </header>
      )}
      <div className={cn(padded && "px-6 pb-6 pt-3")}>{children}</div>
    </section>
  );
}

export function Label({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("text-[12px] font-medium tracking-[-0.005em] text-muted", className)}>{children}</div>;
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
    <div className="card min-w-0 rounded-[22px] px-5 py-4">
      <div className="flex items-center gap-2">
        {Icon && <Icon className="h-3.5 w-3.5 text-muted" />}
        <Label>{label}</Label>
        {health && <span className="ml-auto"><Dot health={health} /></span>}
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span className="display text-[32px] font-semibold leading-none text-fg tabular">{value}</span>
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
        "inline-flex items-center justify-center gap-1.5 rounded-full px-4 py-[7px] text-[14px] font-medium transition active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40",
        variant === "primary" && "bg-glacier text-white hover:brightness-110",
        variant === "secondary" && "bg-raised text-fg hover:bg-raised/70",
        variant === "ghost" && "text-glacier hover:bg-glacier/10",
        variant === "danger" && "bg-alert/10 text-alert hover:bg-alert/15",
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
  "w-full rounded-xl border hairline bg-raised/60 px-3.5 py-2.5 text-[15px] text-fg placeholder:text-muted/70 focus:border-glacier focus:bg-surface focus:outline-none focus:ring-4 focus:ring-glacier/15 transition";

export function Empty({ icon: Icon, title, children }: { icon: LucideIcon; title: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      <span className="flex h-10 w-10 items-center justify-center rounded-full bg-raised"><Icon className="h-5 w-5 text-muted" /></span>
      <div className="text-[17px] font-semibold tracking-[-0.02em] text-fg">{title}</div>
      {children && <div className="max-w-[46ch] text-[13px] text-muted">{children}</div>}
    </div>
  );
}

export function Json({ value }: { value: unknown }) {
  const text = typeof value === "string" ? value : JSON.stringify(value, null, 2);
  return (
    <pre className="scroll-thin max-h-[420px] overflow-auto rounded-2xl bg-raised/70 p-4 font-mono text-[12px] leading-relaxed text-fg/90">{text}</pre>
  );
}
