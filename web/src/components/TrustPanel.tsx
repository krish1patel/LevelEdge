import { AlertTriangle, CircleDot, CircleHelp, SlidersHorizontal } from "lucide-react";
import type { ForecastResponse } from "../api";
import { fmtPct } from "../format";

const VERDICT = {
  model_adds_skill: { icon: SlidersHorizontal, title: "Pattern adjustment on" },
  baseline: { icon: CircleDot, title: "Volatility baseline only" },
  unvalidated: { icon: CircleHelp, title: "Not validated" },
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
    : cov < 0.72 ? `Recent 80% ranges held only ${fmtPct(cov)} of the time, so real outcomes may land outside the range more often than it suggests.`
    : cov > 0.88 ? `Recent 80% ranges held ${fmtPct(cov)} of the time, so the range is probably wider than it needs to be.`
    : null;

  return (
    <section className="card" aria-labelledby="trust-h">
      <div className="card-head">
        <h2 id="trust-h">How far to trust this</h2>
      </div>
      <dl className="kv">
        <dt>80% range held, out of sample</dt>
        <dd className="num">{covNote}</dd>
        <dt>Weight on pattern model</dt>
        <dd className="num">{fmtPct(trust.xgb_weight)}</dd>
        <dt>Validated on</dt>
        <dd className="num">{diagnostics.rows.toLocaleString()} bars</dd>
        {best && (
          <>
            <dt>Best level skill vs base rate</dt>
            <dd className="num">{(best.backtest!.brier_skill! * 100).toFixed(0)}% at {best.input}</dd>
          </>
        )}
      </dl>
      {covWarn && (
        <p className="trust-note"><AlertTriangle size={16} aria-hidden="true" />{covWarn}</p>
      )}
      <div className="trust-head">
        <Icon size={20} aria-hidden="true" color="var(--ink-3)" />
        <p><strong>{v.title}</strong>{trust.detail}</p>
      </div>
      <p className="help">
        Out-of-sample means walk-forward tests on held-out bars the model never trained on. A calibrated 80% range
        should hold about 80% of the time.
      </p>
    </section>
  );
}
