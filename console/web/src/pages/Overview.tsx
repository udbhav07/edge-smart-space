import { AlertTriangle, ArrowRight, Droplets, Gauge, Sun, Thermometer, UserRound, Wind, Zap } from "lucide-react";
import { Link } from "react-router-dom";
import { ago, celsius, clock, cn, sensorLabel } from "@/lib/format";
import { SERVICES, judge } from "@/lib/services";
import { useConsole, useNow } from "@/lib/useConsole";
import { Card, Dot, Empty, Label, PageHeader, Stat, type Health } from "@/components/ui";
import { TimeChart } from "@/components/TimeChart";

const toneHealth: Record<string, Health> = { good: "good", warn: "warn", alert: "alert", info: "idle", neutral: "idle" };

export function Overview() {
  const { state } = useConsole();
  const now = useNow();
  if (!state) return null;
  const room = state.room;
  const series = state.series;
  const indoorFaulted = room.health.temp_01?.quality === "faulted";
  const indoor = indoorFaulted ? room.thermal?.t_pred : room.sensors.temp_01?.value;
  const occupied = (room.sensors.pir_01?.value ?? 0) >= 0.5;
  const power = room.sensors.pwr_01?.value;
  const health = (id: string): Health => {
    const quality = room.health[id]?.quality ?? room.sensors[id]?.quality;
    if (!room.sensors[id]) return "idle";
    return quality === "faulted" ? "alert" : quality === "suspect" ? "warn" : "good";
  };

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Overview"
        subtitle="Room 1, live. Every figure here is read from the blackboard as the services publish it; nothing is computed in the browser but the chart."
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat
          label="Indoor"
          icon={Thermometer}
          value={celsius(indoor)}
          health={health("temp_01")}
          note={indoorFaulted ? <span className="text-ember">from the model: sensor faulted</span> : `holding ${celsius(room.tracking?.tracked_c)}`}
        />
        <Stat label="Humidity" icon={Droplets} value={room.sensors.hum_01 ? Math.round(room.sensors.hum_01.value) : "--"} unit="%RH" health={health("hum_01")} note={room.sensors.hum_01 ? ago(room.sensors.hum_01.ts, now) : "waiting"} />
        <Stat label="Outside" icon={Sun} value={celsius(room.sensors.outdoor_01?.value)} health={health("outdoor_01")} note="every 60 s" />
        <Stat label="Presence" icon={UserRound} value={room.sensors.pir_01 ? (occupied ? "In" : "Out") : "--"} health={health("pir_01")} note={occupied ? "someone is in the room" : "room empty"} />
        <Stat label="Air conditioner" icon={Zap} value={power !== undefined ? Math.round(power) : "--"} unit="W" health={health("pwr_01")} note={room.actuator ? `${room.actuator.kind} · ack ${room.actuator.ack.toLowerCase()}` : "no state"} />
        <Stat label="Tariff" icon={Wind} value={room.tariff ? (room.tariff.band === "peak" ? "Peak" : "Normal") : "--"} health={room.tariff?.band === "peak" ? "warn" : "good"} note={room.tariff ? `changes ${clock(room.tariff.next_transition_ts)}` : "unknown"} />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card title="Indoor temperature" icon={Thermometer} className="xl:col-span-2" action={<span className="text-[12px] text-muted">last 30 min</span>}>
          <TimeChart
            unit="°"
            height={260}
            series={[
              { key: "t_in", label: "Measured", slot: 1, points: series.t_in ?? [] },
              { key: "t_pred", label: "Model prediction", slot: 2, dashed: true, points: series.t_pred ?? [] },
              { key: "held_c", label: "Setpoint held", slot: 3, step: true, points: extend(series.held_c ?? [], now, series.t_in?.[0]?.[0]) },
            ]}
          />
        </Card>
        <Card title="Active faults" icon={AlertTriangle} action={<Link to="/faults" className="flex items-center gap-1 text-[12px] text-muted hover:text-fg">Faults & control <ArrowRight className="h-3 w-3" /></Link>}>
          {state.faults.length === 0 ? (
            <Empty icon={Gauge} title="No faults">Every detector is satisfied with what it sees.</Empty>
          ) : (
            <ul className="grid gap-2">
              {state.faults.map((fault) => (
                <li key={fault.fault_id} className="rounded-xl border border-alert/25 bg-alert/5 p-3">
                  <div className="text-[13px] font-medium text-alert">{fault.detector.replace(/_/g, " ")} · {sensorLabel[fault.subject] ?? fault.subject}</div>
                  <p className="mt-1 text-[13px] text-fg/90">{state.diagnoses[fault.fault_id]?.user_message ?? "Diagnosing…"}</p>
                  <div className="mt-1 text-[11.5px] text-muted">{ago(fault.detected_ts, now)}</div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Humidity" icon={Droplets}>
          <TimeChart unit="%" digits={0} height={180} series={[{ key: "hum_01", label: "Relative humidity", slot: 1, points: series.hum_01 ?? [] }]} />
        </Card>
        <Card title="Air conditioner power" icon={Zap}>
          <TimeChart unit=" W" digits={0} height={180} domain={[0, "auto"]} series={[{ key: "pwr_01", label: "Electrical power", slot: 1, step: true, points: series.pwr_01 ?? [] }]} />
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card title="Services" icon={Gauge} action={<Link to="/services" className="flex items-center gap-1 text-[12px] text-muted hover:text-fg">Details <ArrowRight className="h-3 w-3" /></Link>}>
          <ul className="divide-y divide-[rgb(var(--line)/var(--line-alpha))]">
            {SERVICES.map((service) => {
              const judged = judge(service, state.topics, now);
              return (
                <li key={service.id} className="flex items-center gap-3 py-2 text-[13.5px]">
                  <Dot health={judged.health} />
                  <span className="text-fg">{service.name}</span>
                  <span className="ml-auto text-[12px] text-muted">{judged.word}{judged.last_ts ? ` · ${ago(judged.last_ts, now)}` : ""}</span>
                </li>
              );
            })}
          </ul>
        </Card>
        <Card title="Latest decisions" icon={Gauge} action={<Link to="/faults" className="flex items-center gap-1 text-[12px] text-muted hover:text-fg">All <ArrowRight className="h-3 w-3" /></Link>}>
          <ul className="grid gap-1.5">
            {[...state.activity].reverse().slice(0, 9).map((step) => (
              <li key={step.id} className="flex items-start gap-2.5 text-[13px]">
                <span className="mt-1.5"><Dot health={toneHealth[step.tone] ?? "idle"} /></span>
                <span className="min-w-0 flex-1 truncate text-fg/90">{step.text}</span>
                <span className="shrink-0 font-mono text-[11.5px] text-muted tabular">{clock(step.ts)}</span>
              </li>
            ))}
            {state.activity.length === 0 && <li className="text-[13px] text-muted">Nothing decided yet.</li>}
          </ul>
          <Label className={cn("mt-3")}>Published only on change; the line is held until the next one.</Label>
        </Card>
      </div>
    </div>
  );
}

/** A step series published only on change, drawn up to now. */
function extend(points: [number, number][], now: number, from?: number): [number, number][] {
  if (!points.length) return points;
  // Published once at start and then only on change: drawn from where the
  // readings begin, or one old point would stretch the whole time axis.
  let shown = points;
  if (from !== undefined && points[0][0] < from) {
    const before = points.filter(([ts]) => ts <= from);
    shown = [[from, before[before.length - 1][1]], ...points.filter(([ts]) => ts > from)];
  }
  return [...shown, [now, shown[shown.length - 1][1]]];
}
