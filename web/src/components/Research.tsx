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

          {r.return_models && <ModelsCard m={r.return_models} />}
          {r.construction && <ConstructionCard c={r.construction} />}
          {r.construction && <FrontierCard c={r.construction} />}
          {r.reward && <RewardCard w={r.reward} />}
          {r.stock_factors && <FactorsCard f={r.stock_factors} />}
          {r.literature && <LiteratureCard l={r.literature} />}
          {r.literature && <BandsCard l={r.literature} />}
        </div>
      )}
    </>
  );
}

const sh = (v: number | undefined) => (v == null ? "—" : minus(v, 2));
const rate = (v: number | undefined, d = 1) => (v == null ? "—" : `${minus(v * 100, d)}%`);

function PerfTable({ rows, cols }: { rows: { label: string; emph?: boolean; cells: string[] }[]; cols: string[] }) {
  return (
    <div className="table-scroll">
      <table className="table">
        <thead><tr><th scope="col">{cols[0]}</th>{cols.slice(1).map((c) => <th key={c} scope="col" className="r">{c}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} data-emph={r.emph ? "true" : undefined}>
              <th scope="row" className="row-h">{r.label}</th>
              {r.cells.map((c, i) => <td key={i} className="r num">{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ModelsCard({ m }: { m: NonNullable<ResearchResponse["return_models"]> }) {
  const names: [string, string][] = [["rule", "Trend rule"], ["ridge", "Pooled ridge"], ["spline", "Splines"], ["mixed", "Mixed effects"], ["xgb", "Boosted trees"]];
  const rw = m.reward_vs_levered_sixty_forty;
  return (
    <Experiment id="rm" title="Richer return models" verdict="rejected" verdictText="No gain over the simple trend rule"
      question="Do ridge regression, splines, a mixed-effects model (per-ETF random slopes) or boosted trees forecast next week's ETF returns better than the 12-month trend rule?"
      method="Weekly, walked forward year by year from 2011 with penalties chosen on earlier data only; the same sizing, costs and engine for every model. 'Fresh' is 18 sector and country ETFs never used in any earlier test.">
      <PerfTable cols={["Model", "Forecast R²", "Sharpe", "Sharpe, fresh ETFs"]}
        rows={names.map(([k, l]) => ({ label: l, emph: k === "rule",
          cells: [minus(m.main.forecast[k]?.oos_r2 ?? 0, 3), sh(m.main.metrics[k]?.sharpe), sh(m.fresh?.metrics[k]?.sharpe)] }))} />
      <p className="help">Out-of-sample R² is at or below zero for every model: next week's ETF returns are close to unpredictable from these features. The random effects shrank to zero, so the ETFs don't differ enough to need their own slopes.
        {rw && ` Levering the best model to ${rate(rw.vol, 0)} volatility returned ${rate(rw.auto_reward.cagr)} a year; 60/40 levered to the same volatility returned ${rate(rw.sixty_forty_levered.cagr)}.`}</p>
    </Experiment>
  );
}

function ConstructionCard({ c }: { c: NonNullable<ResearchResponse["construction"]> }) {
  const names: [string, string][] = [["vt6040", "Vol-targeted 60/40"], ["rp", "Risk parity"], ["rp_trend", "Risk parity + trend"],
    ["vt6040_trend", "Vol-targeted 60/40 + trend"], ["current_book", "Unlevered book"], ["sixty_forty", "60/40"]];
  const vs = c.holdout_sharpe_vs.sixty_forty;
  return (
    <Experiment id="cn" title="Risk parity and volatility targeting" verdict="rejected" verdictText="Nothing beats plain 60/40 reliably"
      question="Do risk-balanced or volatility-targeted portfolios earn more per unit of risk than 60/40, with leverage paid for?"
      method="Weekly, next-open fills, 5 bp costs, borrowing at T-bills + 2%, gross exposure up to 150%. Best Sharpe before 2020 selected, then 2020 onward read once.">
      <PerfTable cols={["Design", "Sharpe before 2020", "Sharpe from 2020"]}
        rows={names.map(([k, l]) => ({ label: l, emph: k === c.selected, cells: [sh(c.dev[k]?.sharpe), sh(c.holdout[k]?.sharpe)] }))} />
      <p className="help">Vol-targeted 60/40 won before 2020, then trailed 60/40 after it (difference {sh(vs.diff)}, 95% interval {sh(vs.ci95[0])} to {sh(vs.ci95[1])}): it cut exposure after volatility spikes and missed the fast rebounds.</p>
    </Experiment>
  );
}

function FrontierCard({ c }: { c: NonNullable<ResearchResponse["construction"]> }) {
  const book = c.frontier.filter((f) => f.family === "current_book_levered");
  return (
    <Experiment id="fr" title="How much return each unit of risk buys" verdict="neutral" verdictText="Leverage adds return at a steep risk cost; 1.5× was chosen"
      question="Holding the book at more than 100% gross, financed by margin, how do return and the worst fall scale?"
      method="The unlevered book's own weights scaled by a fixed multiple, borrowing at T-bills + 2%, 2007 onward. SPY alone: about 10.7% a year with a −55% worst fall.">
      <PerfTable cols={["Gross", "Return", "Worst fall", "Worst year", "Sharpe"]}
        rows={book.map((f) => ({ label: `${f.knob}×`, emph: f.knob === 1.5, cells: [rate(f.cagr), rate(f.max_drawdown, 0), rate(f.worst_year, 0), sh(f.sharpe)] }))} />
      <p className="help">At 2× the book nearly matches SPY's return with a shallower worst fall, but borrowing costs pull Sharpe down as leverage rises.</p>
    </Experiment>
  );
}

function RewardCard({ w }: { w: NonNullable<ResearchResponse["reward"]> }) {
  const h = w.holdout;
  return (
    <Experiment id="rw" title="A bigger trend sleeve with leverage" verdict="rejected" verdictText="Failed the held-out test"
      question={`Most return before 2020 with the worst fall no deeper than ${rate(w.dd_limit, 0)}: trend share 20–50%, leverage 1–2×.`}
      method={`20 books, selected on 2007–2019 (picked: ${w.selected.replace("s=", "trend ").replace("|L=", ", ")}×), then 2020 onward read once.`}>
      <PerfTable cols={["From 2020", "Return", "Volatility", "Worst fall"]}
        rows={[{ label: "Selected book", emph: true, cells: [rate(h.selected.cagr), rate(h.selected.vol), rate(h.selected.max_drawdown, 0)] },
               { label: "Unlevered current book", cells: [rate(h.current_book.cagr), rate(h.current_book.vol), rate(h.current_book.max_drawdown, 0)] },
               { label: "SPY", cells: [rate(h.spy_buy_hold.cagr), rate(h.spy_buy_hold.vol), rate(h.spy_buy_hold.max_drawdown, 0)] }]} />
      <p className="help">The leverage was spent on a trend sleeve that lagged after 2020: more trend lowered returns at every leverage level.</p>
    </Experiment>
  );
}

function FactorsCard({ f }: { f: NonNullable<ResearchResponse["stock_factors"]> }) {
  const names: [string, string][] = [["mom_12_1", "Momentum (12-1)"], ["mom_lowvol", "Momentum + low vol"], ["lowvol", "Low volatility"], ["rev_1m", "1-month reversal"]];
  return (
    <Experiment id="sf" title="Stock factors on historical S&P 500 members" verdict="rejected" verdictText="No factor beat an equal-weight portfolio of the same stocks"
      question="Do momentum, low volatility or short-term reversal, holding the top 20% of S&P 500 members each month, beat simply equal-weighting the members?"
      method={`Point-in-time membership, so the strategy only holds stocks in the index on that date. Yahoo lacks most failed, acquired and renamed members: coverage is ${rate(f.coverage.share_2006, 0)} of members in 2006, ${rate(f.coverage.share_2019, 0)} in 2019 and ${rate(f.coverage.share_2025, 0)} in 2025, so each factor is compared with the same covered stocks. Monthly, 10 bp costs.`}>
      <PerfTable cols={["Factor", "vs equal weight, before 2020", "after 2020", "Sharpe after 2020"]}
        rows={names.map(([k, l]) => ({ label: l, cells: [rate(f.dev_vs_ew[k].excess_cagr), rate(f.holdout_vs_ew[k].excess_cagr), sh(f.holdout[k].sharpe)] }))} />
      <p className="help">Low volatility had the best risk-adjusted record before 2020 and the worst after. A real stock-level test needs survivorship-free data (paid).</p>
    </Experiment>
  );
}

function LiteratureCard({ l }: { l: NonNullable<ResearchResponse["literature"]> }) {
  const books: [string, string][] = [["book_orig_trend", "Current sleeve (12-month trend)"], ["book_mtp", "Fast/slow blend (Goulding et al. 2023)"],
    ["book_daa_g12", "Defensive Asset Allocation (Keller & Keuning 2018)"], ["book_gtaa5", "10-month average (Faber 2007)"]];
  const vs = l.holdout_vs_current_book;
  return (
    <Experiment id="lit" title="Published strategies as the book's sleeve" verdict="neutral"
      verdictText="Defensive Asset Allocation leads but isn't proven: tracked as a shadow book"
      question="Do strategies from published papers, implemented exactly as specified, beat the current trend sleeve inside the 1.5× book?"
      method={`${l.period[0]} onward, borrowing at T-bills + 2%, 5 bp costs. Selected on Sharpe before 2020, then 2020 onward read once (fresh for all of them). Corrected for ${l.deflated.trials} strategies tried, there is a ${rate(l.deflated.deflated_sharpe_prob, 0)} chance the winner is not luck; 95% is the usual bar.`}>
      <PerfTable cols={["Sleeve", "Return", "Sharpe before 2020", "Sharpe from 2020"]}
        rows={books.map(([k, n]) => ({ label: n, emph: k === l.selected, cells: [rate(l.full[k].cagr), sh(l.dev[k].sharpe), sh(l.holdout[k].sharpe)] }))} />
      <p className="help">From 2020 the Defensive Asset Allocation book beat the current one by {sh(vs.diff)} Sharpe (95% interval {sh(vs.ci95[0])} to {sh(vs.ci95[1])}). It is followed forward from public prices, untraded, and replaces the live sleeve only on months of evidence.
        {" "}Turn-of-the-month and overnight-only SPY were rejected: the turn-of-month days earned {l.per_day_bp.tom_window.toFixed(1)} bp a day against {l.per_day_bp.spy_all_days.toFixed(1)} bp on any day, and overnight returns fall from {l.per_day_bp.overnight_0bp.toFixed(1)} to {l.per_day_bp.overnight_1bp.toFixed(1)} bp a day at 1 bp of cost per side.</p>
    </Experiment>
  );
}

function BandsCard({ l }: { l: NonNullable<ResearchResponse["literature"]> }) {
  const b0 = l.bands["0.00"], bs = l.bands[l.band_selected];
  return (
    <Experiment id="bd" title="Rebalancing bands: fewer, larger trades" verdict="adopted" verdictText={`Adopted: trade an ETF only when it drifts ${(Number(l.band_selected) * 100).toFixed(0)} points from target`}
      question="How much of the trading is worth its cost? Rebalance weekly only the ETFs that drifted past a band."
      method="Pre-declared choice: the widest band whose tracking error against weekly full rebalancing stays below 0.5% a year. Consistent with Vanguard's threshold-rebalancing research (about 5–21 bp a year).">
      <PerfTable cols={["Band", "Orders a year", "Costs a year", "Tracking error"]}
        rows={Object.entries(l.bands).map(([k, v]) => ({ label: k === "0.00" ? "None (weekly)" : `${(Number(k) * 100).toFixed(0)} point${Number(k) === 0.01 ? "" : "s"}`,
          emph: k === l.band_selected, cells: [v.orders_per_year.toFixed(0), `${v.cost_drag_bp_per_year.toFixed(0)} bp`, rate(v.tracking_error_vs_full, 2)] }))} />
      <p className="help">Orders fall from {b0.orders_per_year.toFixed(0)} to {bs.orders_per_year.toFixed(0)} a year and costs from {b0.cost_drag_bp_per_year.toFixed(0)} to {bs.cost_drag_bp_per_year.toFixed(0)} bp, so no money is spent on trades too small to matter.</p>
    </Experiment>
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
