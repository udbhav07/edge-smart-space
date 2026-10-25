import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipProps,
} from "recharts";
import { clockSeconds } from "@/lib/format";

// One y-axis, at most three series in the validated order, 2 px lines, a
// recessive grid, and a crosshair tooltip. A single series needs no legend
// (the card title names it); two or more always get one.

export interface Series {
  key: string;
  label: string;
  points: [number, number][];
  /** 1, 2 or 3: the categorical slot. */
  slot: 1 | 2 | 3;
  dashed?: boolean;
  step?: boolean;
}

type Row = { t: number } & Record<string, number | undefined>;

/** Points from different publishers, merged on the nearest second. */
function merge(series: Series[]): Row[] {
  const rows = new Map<number, Row>();
  for (const one of series) {
    for (const [ts, value] of one.points) {
      const t = Math.round(ts);
      const row = rows.get(t) ?? { t };
      row[one.key] = value;
      rows.set(t, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.t - b.t);
}

function Tip({ active, payload, label, unit, digits, labels }: TooltipProps<number, string> & { unit: string; digits: number; labels: Record<string, string> }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border hairline bg-surface px-3 py-2 text-[12px] shadow-xl">
      <div className="mb-1 font-mono text-muted">{clockSeconds(Number(label))}</div>
      {payload.map((item) => (
        <div key={String(item.dataKey)} className="flex items-center gap-2">
          <span className="h-0.5 w-3 rounded" style={{ background: item.color }} />
          <span className="text-fg/80">{labels[String(item.dataKey)]}</span>
          <span className="ml-auto pl-4 font-mono text-fg tabular">{Number(item.value).toFixed(digits)}{unit}</span>
        </div>
      ))}
    </div>
  );
}

export function TimeChart({ series, unit = "", digits = 1, height = 220, domain }: {
  series: Series[];
  unit?: string;
  digits?: number;
  height?: number;
  domain?: [number | "auto" | ((value: number) => number), number | "auto" | ((value: number) => number)];
}) {
  const data = useMemo(() => merge(series), [series]);
  // Axis decimals follow the range shown, so a narrow band never repeats a
  // label ("44, 44, 43, 43") and a wide one carries no false precision.
  const axisDigits = useMemo(() => {
    const values = series.flatMap((one) => one.points.map(([, value]) => value));
    if (!values.length) return digits;
    const range = Math.max(...values) - Math.min(...values);
    return range >= 20 ? 0 : range >= 2 ? 1 : range >= 0.2 ? 2 : range >= 0.02 ? 3 : 4;
  }, [series, digits]);
  const labels = Object.fromEntries(series.map((one) => [one.key, one.label]));
  if (data.length < 2) {
    return (
      <div className="flex items-center justify-center rounded-xl bg-raised/30 text-[13px] text-muted" style={{ height }}>
        Collecting readings…
      </div>
    );
  }
  return (
    <div className="min-w-0">
      {series.length > 1 && (
        <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-fg/80">
          {series.map((one) => (
            <span key={one.key} className="flex items-center gap-1.5">
              <svg width="16" height="6" aria-hidden><line x1="0" y1="3" x2="16" y2="3" stroke={`var(--series-${one.slot})`} strokeWidth="2" strokeDasharray={one.dashed ? "4 3" : undefined} /></svg>
              {one.label}
            </span>
          ))}
        </div>
      )}
      <div style={{ height }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="rgb(var(--line) / 0.06)" vertical={false} />
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t) => clockSeconds(t).slice(0, 5)}
              stroke="rgb(var(--muted))"
              tick={{ fontSize: 11, fill: "rgb(var(--muted))" }}
              tickLine={false}
              axisLine={{ stroke: "rgb(var(--line) / 0.1)" }}
              minTickGap={48}
            />
            <YAxis
              width={44}
              domain={domain ?? ["auto", "auto"]}
              tickFormatter={(v: number) => v.toFixed(axisDigits)}
              tick={{ fontSize: 11, fill: "rgb(var(--muted))" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              content={<Tip unit={unit} digits={digits} labels={labels} />}
              cursor={{ stroke: "rgb(var(--muted) / 0.6)", strokeWidth: 1 }}
              isAnimationActive={false}
            />
            {series.map((one) => (
              <Line
                key={one.key}
                dataKey={one.key}
                type={one.step ? "stepAfter" : "monotone"}
                stroke={`var(--series-${one.slot})`}
                strokeWidth={2}
                strokeDasharray={one.dashed ? "5 4" : undefined}
                dot={false}
                activeDot={{ r: 4, strokeWidth: 2, stroke: "rgb(var(--surface))" }}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
