import { AlertTriangle, CircleDot, CircleHelp, SlidersHorizontal } from "lucide-react";
import type { ForecastResponse } from "../api";
import { fmtPct } from "../format";

const VERDICT = {
  model_adds_skill: { icon: SlidersHorizontal, title: "Pattern model is adjusting the forecast", plain: "In testing, the pattern-recognition model beat the simpler volatility model on data it hadn't seen, so it gets a say." },
  baseline: { icon: CircleDot, title: "Using the simpler volatility model alone", plain: "The pattern-recognition model didn't beat the simpler volatility model in testing, so it gets no say. That is the safer choice, and the usual one." },
  unvalidated: { icon: CircleHelp, title: "Not enough history to test", plain: "There wasn't enough past data to test this forecast, so treat it with extra caution." },
} as const;

export function TrustPanel({ data }: { data: ForecastResponse }) {
  const { trust, levels, diagnostics } = data;
  const v = VERDICT[trust.verdict];
  const Icon = v.icon;
  const cov = trust.coverage80;
  const covNote = cov == null ? "—" : `${fmtPct(cov)} of outcomes`;
  const informative = levels.filter((l) => l.backtest?.brier_skill != null);
  const best = informative.length
    ? informative.reduce((a, b) => ((a.backtest!.brier_skill ?? -9) > (b.backtest!.brier_skill ?? -9) ? a : b))
    : null;

  const covWarn = cov == null ? null
    : cov < 0.72 ? `Recent 8-in-10 ranges held only ${fmtPct(cov)} of the time, so real outcomes may land outside the range more often than it suggests.`
    : cov > 0.88 ? `Recent 8-in-10 ranges held ${fmtPct(cov)} of the time, so the range is probably wider than it needs to be.`
    : null;

  return (
    <section className="card" aria-labelledby="trust-h">
      <div className="card-head">
        <h2 id="trust-h">How far to trust this forecast</h2>
      </div>
      <dl className="kv">
        <dt>How often its 8-in-10 ranges held in testing</dt>
        <dd className="num">{covNote}</dd>
        <dt>Pattern model's share of the forecast</dt>
        <dd className="num">{fmtPct(trust.xgb_weight)}</dd>
        <dt>Tested on</dt>
        <dd className="num">{diagnostics.rows.toLocaleString()} past price bars</dd>
        {best && (
          <>
            <dt>Best gain over a naive guess</dt>
            <dd className="num">{(best.backtest!.brier_skill! * 100).toFixed(0)}% at {best.input}</dd>
          </>
        )}
      </dl>
      {covWarn && (
        <p className="trust-note"><AlertTriangle size={16} aria-hidden="true" />{covWarn}</p>
      )}
      <div className="trust-head">
        <Icon size={20} aria-hidden="true" color="var(--ink-3)" />
        <p><strong>{v.title}</strong>{v.plain}</p>
      </div>
      <p className="help">
        Every number here comes from testing the model on past prices it never saw while learning. An honest 8-in-10 range
        should hold about 80% of the time; much less means the forecast is overconfident.
      </p>
    </section>
  );
}
