import { useEffect, useState } from "react";
import { getBenchmark, getTrackRecord, type Benchmark, type TrackRecord as TR } from "../api";
import { fmtNum, fmtPct } from "../format";
import { XYChart } from "./XYChart";

const MODEL_NAMES: Record<string, string> = {
  leveledge_v2_stacked: "LevelEdge v2",
  xgboost_classifier: "v1 (per-level XGBoost)",
  v1: "v1 (per-level XGBoost)",
  gauss_rw: "Gaussian random walk",
  v2_base: "v2 volatility baseline",
  v2: "LevelEdge v2",
};
const label = (k: string) => MODEL_NAMES[k] ?? k;
const isV2 = (k: string) => k === "leveledge_v2_stacked" || k === "v2";

const RANGES = [14, 30, 60, 90];

export function TrackRecord() {
  const [days, setDays] = useState(60);
  const [tr, setTr] = useState<TR | null>(null);
  const [bench, setBench] = useState<Benchmark | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setErr(null);
    getTrackRecord(days).then((r) => live && setTr(r)).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [days]);
  useEffect(() => { getBenchmark().then(setBench).catch(() => setBench(null)); }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
      {tr?.status === "ok" && <div className="segmented" role="group" aria-label="Track record window" style={{ alignSelf: "flex-start" }}>
        {RANGES.map((d) => (
          <button key={d} type="button" aria-pressed={days === d} onClick={() => setDays(d)}>Last {d} days</button>
        ))}
      </div>}

      <section className="card" aria-labelledby="live-h">
        <div className="card-head">
          <div>
            <h2 id="live-h">Live forward test</h2>
            <p className="card-sub">Forecasts logged at 10:00, 12:00 and 14:00 ET for the 4:00 PM close, scored once the close is known</p>
          </div>
        </div>
        {err && <p className="err" role="alert">{err}</p>}
        {!tr && !err && <div className="skeleton" style={{ height: 180 }} aria-label="Loading track record" />}
        {tr && tr.status !== "ok" && (
          <div className="empty">
            <h3>{tr.status === "not_configured" ? "Connect the forward-test log" : tr.status === "empty" ? "No resolved forecasts yet" : "Couldn't load the track record"}</h3>
            {tr.status !== "not_configured" && <p>{tr.message}</p>}
            {tr.status === "not_configured" && (
              <p>Add <code>SUPABASE_URL</code> and <code>SUPABASE_KEY</code> as secrets on the HuggingFace Space (or in <code>.env</code> locally). The GitHub Action in <code>run_forwardtest.yml</code> fills the <code>forwardtest_logs</code> table on weekdays.</p>
            )}
          </div>
        )}
        {tr && tr.status === "ok" && (
          <>
            <table className="table">
              <thead>
                <tr>
                  <th scope="col">Model</th><th scope="col" className="r">Forecasts</th><th scope="col" className="r">Brier</th>
                  <th scope="col" className="r">Log loss</th><th scope="col" className="r">Calibration error</th><th scope="col" className="r">Skill vs base rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(tr.by_model).sort(([a], [b]) => Number(isV2(b)) - Number(isV2(a))).map(([k, s]) => (
                  <tr key={k} data-emph={isV2(k) ? "true" : undefined}>
                    <td>{label(k)}</td>
                    <td className="r num">{s.n.toLocaleString()}</td>
                    <td className="r num">{fmtNum(s.brier)}</td>
                    <td className="r num">{fmtNum(s.log_loss)}</td>
                    <td className="r num">{fmtPct(s.ece, 1)}</td>
                    <td className="r num">{fmtPct(s.brier_skill_vs_base_rate, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="grid-2" style={{ marginTop: 24 }}>
              <div>
                <h3 style={{ fontSize: "var(--t-body)", fontWeight: 600, margin: "0 0 8px" }}>Calibration</h3>
                <XYChart
                  series={Object.entries(tr.by_model).map(([k, s]) => ({
                    name: label(k), color: isV2(k) ? "var(--accent)" : "var(--neutral-series)", kind: "dots" as const, hollow: !isV2(k),
                    points: s.reliability.map((r) => ({ x: r.forecast, y: r.observed })),
                  }))}
                  xLabel={(v) => fmtPct(v)} yLabel={(v) => fmtPct(v)} xDomain={[0, 1]} yDomain={[0, 1]} diagonal
                  xTitle="Predicted chance" yTitle="Observed frequency"
                  summary="Live calibration by model. Points on the diagonal are perfectly calibrated."
                />
              </div>
              <div>
                <h3 style={{ fontSize: "var(--t-body)", fontWeight: 600, margin: "0 0 8px" }}>Daily Brier score (lower is better)</h3>
                <DailyBrier tr={tr} />
              </div>
            </div>
            <p className="footnote">{tr.rows.toLocaleString()} resolved forecasts from {tr.first} to {tr.last}. Skill is measured against always predicting the window's base rate.</p>
          </>
        )}
      </section>

      <section className="card" aria-labelledby="bench-h">
        <div className="card-head">
          <div>
            <h2 id="bench-h">Offline benchmark</h2>
            <p className="card-sub">Every model saw only bars closed by the decision time and forecast the 4:00 PM close</p>
          </div>
        </div>
        {!bench && <div className="skeleton" style={{ height: 120 }} />}
        {bench && bench.status === "missing" && <p className="help">{bench.message}</p>}
        {bench && bench.status === "ok" && <BenchmarkTable b={bench} />}
      </section>
    </div>
  );
}

function DailyBrier({ tr }: { tr: Extract<TR, { status: "ok" }> }) {
  const days = [...new Set(tr.daily.map((d) => d.day))].sort();
  const idx = new Map(days.map((d, i) => [d, i]));
  const models = [...new Set(tr.daily.map((d) => d.model_type))];
  if (days.length < 2) return <p className="help">Needs at least two trading days.</p>;
  return (
    <XYChart
      series={models.map((mt) => ({
        name: label(mt), color: isV2(mt) ? "var(--accent)" : "var(--neutral-series)", dashed: !isV2(mt),
        points: tr.daily.filter((d) => d.model_type === mt).map((d) => ({ x: idx.get(d.day)!, y: d.brier })),
      }))}
      xLabel={(v) => days[Math.round(v)]?.slice(5) ?? ""} yLabel={(v) => v.toFixed(2)} yDomain={[0, 0.4]}
      xDomain={[0, days.length - 1]}
      summary={`Daily Brier score over ${days.length} trading days.`}
    />
  );
}

function BenchmarkTable({ b }: { b: Extract<Benchmark, { status: "ok" }> }) {
  const order = ["v2", "v2_base", "gauss_rw", "v1"];
  const diff = b["brier_diff_v2_minus_v1"] as { point: number; ci95: [number, number] } | undefined;
  return (
    <>
      <table className="table">
        <thead>
          <tr><th scope="col">Model</th><th scope="col" className="r">Brier</th><th scope="col" className="r">Log loss</th><th scope="col" className="r">AUC</th><th scope="col" className="r">Calibration error</th><th scope="col" className="r">Brier skill vs v1</th></tr>
        </thead>
        <tbody>
          {order.filter((k) => b.overall[k]).map((k) => {
            const s = b.overall[k];
            return (
              <tr key={k} data-emph={k === "v2" ? "true" : undefined}>
                <td>{label(k)}</td>
                <td className="r num">{fmtNum(s.brier)}</td>
                <td className="r num">{fmtNum(s.log_loss)}</td>
                <td className="r num">{fmtNum(s.auc)}</td>
                <td className="r num">{fmtPct(s.ece, 1)}</td>
                <td className="r num">{k === "v1" ? "—" : fmtPct(s.brier_skill_vs_v1, 0)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="footnote">
        {b.forecasts.toLocaleString()} forecasts, {b.days[0]} to {b.days[1]}.
        {diff && ` Brier difference v2 − v1: ${diff.point.toFixed(4)} (95% CI ${diff.ci95[0].toFixed(4)} to ${diff.ci95[1].toFixed(4)}, resampling whole ticker-days).`}
      </p>
    </>
  );
}
