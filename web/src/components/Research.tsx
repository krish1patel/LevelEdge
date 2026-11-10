import { useEffect, useState, type ReactNode } from "react";
import { AlertCircle, CircleCheck, CircleMinus, CircleX } from "lucide-react";
import { getResearch, type ResearchResponse, type TradeStats } from "../api";
import { fmtNum, fmtRate } from "../format";
import { XYChart } from "./XYChart";

type Verdict = "adopted" | "rejected" | "neutral";
const VERDICT_ICON = { adopted: CircleCheck, rejected: CircleX, neutral: CircleMinus };

function Experiment({ id, title, question, verdict, verdictText, children, method }: {
  id: string; title: string; question: string; verdict: Verdict; verdictText: string; children: ReactNode; method: string;
}) {
  const Icon = VERDICT_ICON[verdict];
  return (
    <section className="card experiment" aria-labelledby={`${id}-h`}>
      <div className="exp-text">
        <h2 id={`${id}-h`}>{title}</h2>
        <p className="card-sub">{question}</p>
        <p className="verdict" data-verdict={verdict}><Icon size={18} aria-hidden="true" />{verdictText}</p>
        <p className="footnote">{method}</p>
      </div>
      <div className="exp-data">{children}</div>
    </section>
  );
}

const cents = (v: number | undefined) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}¢`);
const minus = (x: number, d: number) => `${x < 0 ? "−" : ""}${Math.abs(x).toFixed(d)}`;
const ci = (c: [number, number] | undefined, f = (x: number) => minus(x, 4)) => (c ? `${f(c[0])} to ${f(c[1])}` : "—");
const signed = (v: number, d = 4) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`;

function TradeRow({ label, s }: { label: string; s: TradeStats }) {
  return (
    <>
      <dt>{label}</dt>
      <dd className="num">{cents(s.avg_pnl_cents)} per contract</dd>
      <dt>95% interval</dt>
      <dd className="num">{ci(s.avg_pnl_cents_ci95, (x) => `${minus(x, 1)}¢`)}</dd>
    </>
  );
}

export function Research() {
  const [r, setR] = useState<ResearchResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getResearch().then(setR).catch((e) => setErr(e.message)); }, []);

  if (err) {
    return <div className="banner" data-tone="error" role="alert"><AlertCircle size={18} aria-hidden="true" /><p><strong>Couldn't load the research log.</strong> {err}</p></div>;
  }

  return (
    <>
      <section className="summary">
        <h1>What we tested</h1>
        <p className="lede">
          Every idea had to beat a simple alternative on data it hadn't seen, after costs. Most didn't. The failures are listed
          here with the same detail as the successes, because they are why the paper book looks the way it does.
        </p>
      </section>

      {!r && <div className="card"><div className="skeleton" style={{ height: 240 }} aria-label="Loading research log" /></div>}

      {r && (
        <div className="research">
          {r.forecaster && (
            <Experiment id="fc" title="A calibrated forecaster" verdict="adopted" verdictText="Adopted: v2 replaced v1"
              question="Can the probability of closing above a level be forecast better than the original per-level model?"
              method={`${r.forecaster.forecasts.toLocaleString()} held-out forecasts on SPY, QQQ, NVDA, TSLA and BTC-USD, scored on the realised close. Lower is better.`}>
              <dl className="kv">
                <dt>Brier score, v2</dt><dd className="num">{fmtNum(r.forecaster.overall.v2?.brier)}</dd>
                <dt>Brier score, v1</dt><dd className="num">{fmtNum(r.forecaster.overall.v1?.brier)}</dd>
                <dt>Brier score, random walk</dt><dd className="num">{fmtNum(r.forecaster.overall.gauss_rw?.brier)}</dd>
                {r.forecaster.brier_diff_v2_minus_v1 && (<><dt>v2 minus v1, 95% interval</dt><dd className="num">{ci(r.forecaster.brier_diff_v2_minus_v1.ci95)}</dd></>)}
              </dl>
            </Experiment>
          )}

          {r.kalshi_taker && (
            <Experiment id="kt" title="Trading Kalshi at the market price" verdict="rejected" verdictText="No edge: the market was better calibrated"
              question="Do LevelEdge's probabilities beat Kalshi's hourly S&P 500, Nasdaq-100 and Bitcoin prices 15 to 60 minutes before settlement?"
              method={`${r.kalshi_taker.rows.toLocaleString()} quotes on ${r.kalshi_taker.events.toLocaleString()} events, ${r.kalshi_taker.period[0]} to ${r.kalshi_taker.period[1]}. Liquid quotes have a spread of 5¢ or less. Trades buy at the ask and pay Kalshi's 7% fee.`}>
              <dl className="kv">
                <dt>Brier, Kalshi price (liquid)</dt><dd className="num">{fmtNum(r.kalshi_taker.liquid.calibration.mid?.brier, 4)}</dd>
                <dt>Brier, LevelEdge (liquid)</dt><dd className="num">{fmtNum(r.kalshi_taker.liquid.calibration.p_v2?.brier, 4)}</dd>
                <dt>LevelEdge minus market</dt><dd className="num">{signed(r.kalshi_taker.liquid.brier_v2_minus_market.diff)}</dd>
                <TradeRow label="Trading the difference" s={r.kalshi_taker.liquid.primary} />
              </dl>
            </Experiment>
          )}

          {r.kalshi_maker && <MakerCard m={r.kalshi_maker} />}

          {r.kalshi_horizon && (
            <Experiment id="kh" title="Kalshi from the morning, over three years" verdict="rejected" verdictText="No edge at longer horizons either"
              question="Forecasting at 10:30, 12:30 and 14:30 for the 4 PM settle, where volatility modelling should matter more than speed."
              method={`${r.kalshi_horizon.rows.toLocaleString()} quotes on ${r.kalshi_horizon.events.toLocaleString()} S&P 500 and Nasdaq-100 events, ${r.kalshi_horizon.period[0]} to ${r.kalshi_horizon.period[1]}, hourly bars.`}>
              <dl className="kv">
                <dt>Brier, Kalshi price (liquid)</dt><dd className="num">{fmtNum(r.kalshi_horizon.liquid.calibration.mid?.brier, 4)}</dd>
                <dt>Brier, LevelEdge (liquid)</dt><dd className="num">{fmtNum(r.kalshi_horizon.liquid.calibration.p_v2?.brier, 4)}</dd>
                <dt>LevelEdge minus market, 95% interval</dt><dd className="num">{ci(r.kalshi_horizon.liquid.brier_v2_minus_market.ci95)}</dd>
                <TradeRow label="Trading the difference" s={r.kalshi_horizon.liquid.primary} />
              </dl>
            </Experiment>
          )}

          {r.etf_trend && (
            <Experiment id="et" title="Trend following across ETFs" verdict="neutral" verdictText="Real signal, no extra return: kept as a crash-protection sleeve"
              question="Does 12-month time-series momentum across 11 ETFs, sized to a 10% volatility target, beat simple portfolios?"
              method={`Rules fixed from the published research before running. ${r.etf_trend.period[0]} to ${r.etf_trend.period[1]}, next-open fills, 5 bp costs. Random pickers use the same sizing with random long/flat signals.`}>
              <div className="table-scroll">
                <table className="table">
                  <thead><tr><th scope="col">Portfolio</th><th scope="col" className="r">Return</th><th scope="col" className="r">Sharpe</th><th scope="col" className="r">Worst fall</th></tr></thead>
                  <tbody>
                    {([["strategy", "Trend strategy"], ["sixty_forty", "60/40"], ["spy_buy_hold", "SPY"]] as const).map(([k, l]) => (
                      <tr key={k} data-emph={k === "strategy" ? "true" : undefined}>
                        <th scope="row" className="row-h">{l}</th>
                        <td className="r num">{fmtRate(r.etf_trend!.metrics[k].cagr)}</td>
                        <td className="r num">{r.etf_trend!.metrics[k].sharpe.toFixed(2)}</td>
                        <td className="r num">{fmtRate(r.etf_trend!.metrics[k].max_drawdown)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="help">Beat {Math.round((1 - r.etf_trend.random_picker.p_value_strategy_sharpe) * 100)}% of {r.etf_trend.random_picker.sims} random pickers (their average Sharpe: {r.etf_trend.random_picker.sharpe_mean.toFixed(2)}).</p>
            </Experiment>
          )}

          {r.etf_search && r.etf_search_points && <SearchCard s={r.etf_search} pts={r.etf_search_points} />}

          {r.etf_trend && (
            <Experiment id="vol" title="LevelEdge's volatility model on daily data" verdict="neutral" verdictText="A tie: the standard method is used instead"
              question="Does LevelEdge's mean-reverting volatility model forecast next week's volatility better than the standard EWMA?"
              method="QLIKE loss on next-week realised variance for 11 ETFs, out of sample. Lower is better. The simpler method wins ties.">
              <dl className="kv">
                <dt>QLIKE, EWMA</dt><dd className="num">{r.etf_trend.vol_engine.ewma.toFixed(4)}</dd>
                <dt>QLIKE, LevelEdge</dt><dd className="num">{r.etf_trend.vol_engine.leveledge.toFixed(4)}</dd>
              </dl>
            </Experiment>
          )}
        </div>
      )}
    </>
  );
}

function MakerCard({ m }: { m: NonNullable<ResearchResponse["kalshi_maker"]> }) {
  const best = [...m.results].sort((a, b) => b.model_avg_pnl_cents - a.model_avg_pnl_cents)[0];
  const allLose = m.results.every((x) => x.model_ci95[1] < 0);
  return (
    <Experiment id="km" title="Market-making on Kalshi" verdict="rejected" verdictText={`Loses money in all ${m.variants} variants`}
      question="Instead of paying the spread, post limit orders inside it around LevelEdge's fair value and earn the spread."
      method={`Fills simulated from Kalshi's real trade tape: an order fills only if a real trade crossed its price while it was live. Margins of 1 to 3¢, orders live 5 minutes to the close, maker fee 0 or 1.75%. ${m.rows.toLocaleString()} quotes.`}>
      <dl className="kv">
        <dt>Best variant</dt><dd className="num">{cents(best.model_avg_pnl_cents)} per fill</dd>
        <dt>Its 95% interval</dt><dd className="num">{ci(best.model_ci95, (x) => `${minus(x, 1)}¢`)}</dd>
        <dt>Same rules around the market price</dt><dd className="num">{cents(best.mid_avg_pnl_cents)} per fill</dd>
        <dt>Every interval below zero</dt><dd>{allLose ? "Yes" : "No"}</dd>
      </dl>
      <p className="help">Orders only fill when a better-informed trader wants the other side. The model loses less than quoting around the market price, so it carries some information, just not enough.</p>
    </Experiment>
  );
}

function SearchCard({ s, pts }: { s: NonNullable<ResearchResponse["etf_search"]>; pts: NonNullable<ResearchResponse["etf_search_points"]> }) {
  const sel = pts.find((p) => p.variant === s.selected);
  const others = pts.filter((p) => p.variant !== s.selected);
  const vs = s.holdout_sharpe_vs.sixty_forty;
  return (
    <section className="card experiment experiment-wide" aria-labelledby="es-h">
      <div className="exp-text">
        <h2 id="es-h">Searching 80 strategy variants</h2>
        <p className="card-sub">Pick the best of 80 published variants on {s.period.dev[0].slice(0, 4)}–2019, then test it once on 2020 onward.</p>
        <p className="verdict" data-verdict="rejected"><CircleX size={18} aria-hidden="true" />Failed the held-out test: the best before 2020 was among the worst after</p>
        <dl className="kv">
          <dt>Selected variant, Sharpe before 2020</dt><dd className="num">{s.deflated_sharpe_dev.sharpe_annual.toFixed(2)}</dd>
          <dt>Chance it wasn't luck (deflated Sharpe)</dt><dd className="num">{fmtRate(s.deflated_sharpe_dev.deflated_sharpe_prob, 0)}</dd>
          <dt>Sharpe after 2020</dt><dd className="num">{s.holdout.selected.sharpe.toFixed(2)}</dd>
          <dt>Minus 60/40 after 2020, 95% interval</dt><dd className="num">{ci(vs.ci95, (x) => minus(x, 3))}</dd>
          <dt>Rank correlation, before vs after</dt><dd className="num">{signed(s.rank_stability.spearman_dev_vs_holdout, 2)}</dd>
        </dl>
        <p className="footnote">The variants that looked best leaned on Treasury bonds, which only rose while interest rates fell from 2007 to 2019. When rates climbed after 2021, they fell hardest. The paper book keeps the original rules, chosen before the search.</p>
      </div>
      <div className="exp-data">
        <XYChart
          series={[
            { name: "Other variants", color: "var(--neutral-series)", kind: "dots", hollow: true, connect: false, points: others.map((p) => ({ x: p.dev, y: p.hold })) },
            ...(sel ? [{ name: "Selected before 2020", color: "var(--accent)", kind: "dots" as const, connect: false, points: [{ x: sel.dev, y: sel.hold }] }] : []),
          ]}
          xLabel={(v) => v.toFixed(1)} yLabel={(v) => v.toFixed(1)} height={300} xTitle="Sharpe before 2020" yTitle="Sharpe from 2020"
          summary={`Each dot is one of ${pts.length} variants. Rank correlation between the two periods is ${s.rank_stability.spearman_dev_vs_holdout.toFixed(2)}: variants that did best before 2020 tended to do worst after.`}
        />
      </div>
    </section>
  );
}
