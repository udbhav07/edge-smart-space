import { Activity, Brain, Gauge, Sigma } from "lucide-react";
import { useMemo } from "react";
import { useConsole } from "@/lib/useConsole";
import { Card, Label, PageHeader, Pill, Stat } from "@/components/ui";
import { TimeChart } from "@/components/TimeChart";

// The thermal model (DESIGN.md section 5.2): what recursive least squares has
// learned about this room, what the coefficients mean physically, and how
// well the model predicts the next reading.

const STEP_S = 5;

export function Estimator() {
  const { state } = useConsole();
  const series = state?.series ?? {};
  const bounds = useMemo(() => {
    const sigma = series.residual_sigma ?? [];
    return {
      upper: sigma.map(([ts, value]) => [ts, 2 * value] as [number, number]),
      lower: sigma.map(([ts, value]) => [ts, -2 * value] as [number, number]),
    };
  }, [series.residual_sigma]);
  if (!state) return null;
  const c = state.room.coefficients;
  const thermal = state.room.thermal;
  const tauMin = c && c.a2 > 0 ? STEP_S / c.a2 / 60 : null;
  const coolingPerMin = c ? c.a3 * (60 / STEP_S) : null;
  const occupantPerMin = c ? c.a4 * (60 / STEP_S) : null;

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6 sm:px-8">
      <PageHeader
        title="Thermal model"
        subtitle={
          <>
            What the room has taught itself, by recursive least squares, every 5 s. The model is{" "}
            <span className="font-mono text-fg/90">T[k+1] − T[k] = a2·(T_out − T) + a3·u + a4·o</span>, with a1 = 1 − a2 by construction, so the room settles to the
            outside temperature when nothing acts on it.
          </>
        }
        actions={thermal && <Pill health={thermal.adaptation === "active" ? "good" : "warn"}>{thermal.adaptation === "active" ? "Learning" : "Learning frozen"}</Pill>}
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <Stat label="a2 · coupling" icon={Gauge} value={c ? c.a2.toFixed(5) : "--"} note={tauMin ? `time constant ≈ ${tauMin.toFixed(0)} min` : "not yet identified"} />
        <Stat label="a3 · cooling" icon={Gauge} value={c ? c.a3.toFixed(4) : "--"} note={coolingPerMin !== null ? `${coolingPerMin.toFixed(2)} °C/min at full power` : "—"} />
        <Stat label="a4 · occupant" icon={Gauge} value={c ? c.a4.toFixed(4) : "--"} note={occupantPerMin !== null ? `${occupantPerMin >= 0 ? "+" : ""}${occupantPerMin.toFixed(2)} °C/min when occupied` : "—"} />
        <Stat label="a1 · retained" icon={Gauge} value={c ? c.a1.toFixed(5) : "--"} note="fraction kept per step" />
        <Stat label="trace(P)" icon={Sigma} value={c ? c.trace_p.toFixed(4) : "--"} note="uncertainty; smaller is surer" />
        <Stat label="Samples" icon={Activity} value={c ? c.samples_since_reset.toLocaleString() : "--"} note={thermal ? `confidence ${(thermal.model_confidence * 100).toFixed(0)}%` : "—"} />
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-2">
        <Card title="Prediction against the measurement" icon={Brain} action={<span className="text-[12px] text-muted">one step ahead</span>}>
          <TimeChart
            unit="°"
            height={230}
            digits={2}
            series={[
              { key: "t_in", label: "Measured", slot: 1, points: series.t_in ?? [] },
              { key: "t_pred", label: "Predicted", slot: 2, dashed: true, points: series.t_pred ?? [] },
            ]}
          />
        </Card>
        <Card title="Residual" icon={Sigma} action={<span className="text-[12px] text-muted">measured − predicted</span>}>
          <TimeChart
            unit="°"
            height={230}
            digits={3}
            series={[
              { key: "residual", label: "Residual", slot: 1, points: series.residual ?? [] },
              { key: "upper", label: "+2σ", slot: 2, dashed: true, points: bounds.upper },
              { key: "lower", label: "−2σ", slot: 3, dashed: true, points: bounds.lower },
            ]}
          />
          <p className="mt-2 text-[12px] text-muted">A residual that stays outside ±2σ is what the drift detector (D4) accumulates.</p>
        </Card>
      </div>

      <h2 className="mb-3 mt-8 text-[14px] font-medium text-fg">Coefficients over time</h2>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[
          { key: "a2", title: "a2 · ambient coupling", digits: 5, points: series.a2 },
          { key: "a3", title: "a3 · cooling authority", digits: 4, points: series.a3 },
          { key: "a4", title: "a4 · occupancy gain", digits: 4, points: series.a4 },
          { key: "trace_p", title: "trace(P) · uncertainty", digits: 4, points: series.trace_p },
        ].map((chart) => (
          <Card key={chart.key} title={chart.title}>
            <TimeChart height={150} digits={chart.digits} series={[{ key: chart.key, label: chart.title, slot: 1, points: chart.points ?? [] }]} />
          </Card>
        ))}
      </div>

      <Card className="mt-4" title="Reading the coefficients">
        <div className="grid gap-4 text-[13px] text-fg/85 md:grid-cols-3">
          <div><Label>Coupling</Label><p className="mt-1">How strongly the room is pulled toward the outside temperature each step. Its reciprocal is the room's time constant.</p></div>
          <div><Label>Authority</Label><p className="mt-1">How much one step of full cooling lowers the room. It must stay negative: a positive estimate would mean the unit heats the room, which is a fault, not physics.</p></div>
          <div><Label>Plausibility</Label><p className="mt-1">An update outside the physical bounds is rejected and the previous estimate kept (FR-06). Learning freezes while a sensor feeding the model is faulted (FR-29).</p></div>
        </div>
      </Card>
    </div>
  );
}
