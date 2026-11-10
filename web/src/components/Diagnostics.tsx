import type { ForecastResponse, Score } from "../api";
import { fmtNum, fmtPct, fmtPrice, fmtDay } from "../format";
import { XYChart } from "./XYChart";

const FEATURE_NAMES: Record<string, string> = {
  "level (k)": "Level distance (σ)", crossed: "Sessions crossed", log_scale: "Horizon volatility",
  minute_of_day: "Time of day", dow: "Day of week", diurnal_factor: "Intraday vol profile",
  ret_since_open: "Move since open", vwap_dist: "Distance from VWAP", pos_in_day: "Position in day range",
  dist_prev_high: "Distance to prior high", dist_prev_low: "Distance to prior low", ret_vs_prev_close: "Move vs prior close",
  rsi_14: "RSI (14)", macd_pct: "MACD", macd_hist_pct: "MACD histogram", bb_pos: "Bollinger position",
  bb_width_pct: "Bollinger width", zscore_20: "Z-score (20 bars)", pos_in_range_20: "Position in 20-bar range",
  atr_pct: "ATR", range_vs_atr: "Bar range vs ATR", ret_in_atr: "Last move in ATRs", parkinson_20: "Range volatility",
  pk_vs_cc: "Range vs close volatility", rvol_ratio_5_50: "Vol ratio 5/50 bars", rvol_ratio_20_50: "Vol ratio 20/50 bars",
  sma_5_20: "SMA 5 vs 20", sma_20_50: "SMA 20 vs 50", body_ratio: "Candle body", upper_wick: "Upper wick",
  lower_wick: "Lower wick", gap: "Gap", streak: "Up/down streak", accel: "Momentum change", vol_ratio: "Volume vs average",
  vol_ratio_5: "5-bar volume vs average", signed_vol_20: "Volume pressure (20 bars)", vol_x_ret: "Volume-weighted move",
  dist_52w_high: "Distance to 52-week high",
};
const PATTERNS: [RegExp, (m: RegExpMatchArray) => string][] = [
  [/^ret_lag_(\d+)$/, (m) => `Return ${m[1]} bar${m[1] === "1" ? "" : "s"} back`],
  [/^ret_(\d+)$/, (m) => `Return over ${m[1]} bars`],
  [/^rvol_(\d+)$/, (m) => `Volatility (${m[1]} bars)`],
  [/^dist_sma_(\d+)$/, (m) => `Distance to SMA ${m[1]}`],
  [/^dist_ema_(\d+)$/, (m) => `Distance to EMA ${m[1]}`],
];
const nice = (n: string) => {
  if (FEATURE_NAMES[n]) return FEATURE_NAMES[n];
  for (const [re, f] of PATTERNS) { const m = n.match(re); if (m) return f(m); }
  return n.replace(/_/g, " ");
};

function ScoreTable({ rows }: { rows: [string, Score | undefined, boolean?][] }) {
  return (
    <table className="table">
      <thead>
        <tr><th scope="col">Forecast</th><th scope="col" className="r">Log loss</th><th scope="col" className="r">Brier</th><th scope="col" className="r">AUC</th><th scope="col" className="r">Samples</th></tr>
      </thead>
      <tbody>
        {rows.map(([name, s, emph]) => s && (
          <tr key={name} data-emph={emph ? "true" : undefined}>
            <td>{name}</td>
            <td className="r num">{fmtNum(s.log_loss)}</td>
            <td className="r num">{fmtNum(s.brier)}</td>
            <td className="r num">{fmtNum(s.auc)}</td>
            <td className="r num">{s.n.toLocaleString()}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function Diagnostics({ data }: { data: ForecastResponse }) {
  const d = data.diagnostics;
  const spot = data.meta.spot;
  const curve = data.curve;
  const rel = d.reliability;
  const hl = d.vol_half_life_bars;

  return (
    <div className="grid-2">
      <section className="card" aria-labelledby="curve-h">
        <div className="card-head">
          <div>
            <h2 id="curve-h">Probability across levels</h2>
            <p className="card-sub">P(close above) for every level, and the volatility baseline it starts from</p>
          </div>
        </div>
        <XYChart
          series={[
            { name: "LevelEdge forecast", color: "var(--accent)", points: curve.map((c) => ({ x: c.level, y: c.p_above })) },
            { name: "Volatility baseline", color: "var(--neutral-series)", dashed: true, points: curve.map((c) => ({ x: c.level, y: c.p_baseline })) },
          ]}
          xLabel={fmtPrice} yLabel={(v) => fmtPct(v)} yDomain={[0, 1]}
          vline={{ x: spot, label: `Spot ${fmtPrice(spot)}` }} xTitle="Level" yTitle="P(close above)"
          summary={`Probability curve. At the current price ${fmtPrice(spot)} the forecast gives ${fmtPct(
            curve.reduce((a, b) => (Math.abs(b.level - spot) < Math.abs(a.level - spot) ? b : a)).p_above)} for closing higher.`}
        />
      </section>

      <section className="card" aria-labelledby="rel-h">
        <div className="card-head">
          <div>
            <h2 id="rel-h">Calibration</h2>
            <p className="card-sub">Out-of-sample forecasts grouped by predicted chance, against how often they happened</p>
          </div>
        </div>
        {rel.length ? (
          <XYChart
            series={[{ name: "Forecast groups", color: "var(--accent)", kind: "dots", points: rel.map((r) => ({ x: r.forecast, y: r.observed })) }]}
            xLabel={(v) => fmtPct(v)} yLabel={(v) => fmtPct(v)} xDomain={[0, 1]} yDomain={[0, 1]} diagonal
            xTitle="Predicted chance" yTitle="Observed frequency"
            summary={`Reliability diagram with ${rel.length} groups. Points near the diagonal mean predicted chances matched observed frequencies.`}
          />
        ) : <p className="help">Not enough history to validate this horizon.</p>}
        <p className="help">Points on the diagonal line are perfectly calibrated.</p>
      </section>

      <section className="card" aria-labelledby="scores-h">
        <div className="card-head">
          <div>
            <h2 id="scores-h">Out-of-sample scores</h2>
            <p className="card-sub">Purged walk-forward validation. Lower log loss and Brier are better.</p>
          </div>
        </div>
        <ScoreTable rows={[
          ["LevelEdge (stacked)", d.metrics.blend, true],
          ["Volatility baseline", d.metrics.baseline],
          ["Pattern model alone", d.metrics.xgb],
        ]} />
        {d.regime_metrics && Object.keys(d.regime_metrics).length > 0 && (
          <>
            <h3 style={{ fontSize: "var(--t-body)", fontWeight: 600, margin: "24px 0 8px" }}>By volatility regime</h3>
            <ScoreTable rows={Object.entries(d.regime_metrics).flatMap(([regime, r]) => [
              [`${regime === "high_vol" ? "High" : "Low"} vol · LevelEdge`, r.blend, true] as [string, Score, boolean],
              [`${regime === "high_vol" ? "High" : "Low"} vol · baseline`, r.baseline] as [string, Score],
            ])} />
          </>
        )}
      </section>

      <section className="card" aria-labelledby="model-h">
        <div className="card-head">
          <div>
            <h2 id="model-h">What the model uses</h2>
            <p className="card-sub">Share of the pattern model's total gain by feature. Bar length is relative to the top feature.</p>
          </div>
        </div>
        {d.importance.length ? (
          <div className="bars" role="list">
            {d.importance.map((f) => (
              <div className="bar-row" role="listitem" key={f.name}>
                <span className="name" title={nice(f.name)}>{nice(f.name)}</span>
                <span className="track" aria-hidden="true"><i style={{ width: `${(f.share / d.importance[0].share) * 100}%` }} /></span>
                <span className="num" style={{ textAlign: "right" }}>{fmtPct(f.share)}</span>
              </div>
            ))}
          </div>
        ) : <p className="help">The pattern model is off for this horizon.</p>}
        <dl className="kv" style={{ marginTop: 24 }}>
          <dt>Stack weights (pattern · baseline)</dt>
          <dd className="num">{d.weights.xgb.toFixed(2)} · {d.weights.baseline.toFixed(2)}</dd>
          <dt>Volatility memory</dt>
          <dd className="num">{hl == null ? "No mean reversion" : `${Math.round(hl)}-bar half-life`}</dd>
          <dt>Variance calibration factor</dt>
          <dd className="num">{d.vol_inflation.toFixed(2)}×</dd>
          <dt>Training history</dt>
          <dd className="num">{fmtDay(d.history_start)} to {fmtDay(d.history_end)}</dd>
        </dl>
      </section>
    </div>
  );
}
