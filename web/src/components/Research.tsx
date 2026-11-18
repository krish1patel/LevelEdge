import { useEffect, useState, type ReactNode } from "react";
import { AlertCircle, CircleCheck, CircleMinus, CircleX } from "lucide-react";
import { getResearch, type Brief, type ResearchExtra, type ResearchResponse, type TradeStats, type TrustReport } from "../api";
import { fmtNum, fmtRate } from "../format";
import { XYChart } from "./XYChart";
import { Term } from "./Term";

type Verdict = "adopted" | "rejected" | "neutral";
const VERDICT_ICON = { adopted: CircleCheck, rejected: CircleX, neutral: CircleMinus };

function Experiment({ id, title, question, verdict, verdictText, children, method }: {
  id: string; title: string; question: ReactNode; verdict: Verdict; verdictText: string; children: ReactNode; method: string;
}) {
  const Icon = VERDICT_ICON[verdict];
  return (
    <section className="card experiment" aria-labelledby={`${id}-h`}>
      <div className="exp-text">
        <h2 id={`${id}-h`}>{title}</h2>
        <p className="card-sub">{question}</p>
        <p className="verdict" data-verdict={verdict}><Icon size={18} aria-hidden="true" />{verdictText}</p>
        <details className="disclosure"><summary>How it was tested</summary><p className="footnote">{method}</p></details>
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
  const [r, setR] = useState<(ResearchResponse & ResearchExtra) | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { getResearch().then(setR).catch((e) => setErr(e.message)); }, []);

  if (err) {
    return <div className="banner" data-tone="error" role="alert"><AlertCircle size={18} aria-hidden="true" /><p><strong>Couldn't load the research log.</strong> {err}</p></div>;
  }

  const go = (id: string) => {
    const el = document.getElementById(`${id}-h`);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
    el?.setAttribute("tabindex", "-1");
    el?.focus({ preventScroll: true });
  };

  return (
    <>
      <section className="page-intro">
        <h1>What we tested</h1>
        <p>
          Most trading ideas sound convincing and fail when tested honestly. Every idea here was tested the same way: write the rules
          first, test them on years of data they never saw (a <Term k="holdout">held-out test</Term>), include trading costs, and publish the
          result either way. The failures matter as much as the successes: they're why the practice portfolio looks the way it does.
        </p>
      </section>

      {!r && <div className="card"><div className="skeleton" style={{ height: 240 }} aria-label="Loading the research log" /></div>}

      {r && <Scoreboard go={go} />}
      {r?.integrity && <TrustScores t={r.integrity} />}

      {r && (
        <div className="research">
          <h2 className="research-group">Can a model predict where prices close?</h2>
          {r.forecaster && (
            <Experiment id="fc" title="A better forecasting model" verdict="adopted" verdictText="Adopted: the new model replaced the original"
              question="Can the chance of a price closing above a level be forecast more accurately than the original model managed?"
              method={`${r.forecaster.forecasts.toLocaleString()} forecasts on SPY, QQQ, NVDA, TSLA and BTC-USD that the model never saw, each scored against the actual close.`}>
              <dl className="kv">
                <dt><Term k="brier">Brier score</Term>, new model (lower is better)</dt><dd className="num">{fmtNum(r.forecaster.overall.v2?.brier)}</dd>
                <dt>Brier score, original model</dt><dd className="num">{fmtNum(r.forecaster.overall.v1?.brier)}</dd>
                <dt>Brier score, a simple coin-flip-style model</dt><dd className="num">{fmtNum(r.forecaster.overall.gauss_rw?.brier)}</dd>
                {r.forecaster.brier_diff_v2_minus_v1 && (<><dt>Improvement, <Term k="interval95">95% interval</Term></dt><dd className="num">{ci(r.forecaster.brier_diff_v2_minus_v1.ci95)}</dd></>)}
              </dl>
            </Experiment>
          )}
          {r.etf_trend && (
            <Experiment id="vol" title="A custom volatility model for the portfolio" verdict="neutral" verdictText="A tie, so the simpler standard method is used"
              question="Does LevelEdge's own model forecast next week's swings in 11 markets better than the standard method?"
              method="Out-of-sample forecasts of next week's realised variance for 11 ETFs, scored with QLIKE loss (lower is better). Ties go to the simpler method.">
              <dl className="kv">
                <dt>Error, standard method (EWMA)</dt><dd className="num">{r.etf_trend.vol_engine.ewma.toFixed(4)}</dd>
                <dt>Error, LevelEdge model</dt><dd className="num">{r.etf_trend.vol_engine.leveledge.toFixed(4)}</dd>
              </dl>
            </Experiment>
          )}

          <h2 className="research-group">Can it beat a prediction market?</h2>
          {r.kalshi_taker && (
            <Experiment id="kt" title="Trading Kalshi at the market price" verdict="rejected" verdictText="No edge: the market's prices were more accurate"
              question={<>Do LevelEdge's probabilities beat the prices on <Term k="kalshi">Kalshi</Term>, where people bet on where the S&P 500, Nasdaq-100 and Bitcoin close, 15 to 60 minutes before settlement?</>}
              method={`${r.kalshi_taker.rows.toLocaleString()} quotes on ${r.kalshi_taker.events.toLocaleString()} events, ${r.kalshi_taker.period[0]} to ${r.kalshi_taker.period[1]}. Liquid quotes have a spread of 5¢ or less. Trades buy at the ask and pay Kalshi's 7% fee.`}>
              <dl className="kv">
                <dt>Brier score, Kalshi's price (lower is better)</dt><dd className="num">{fmtNum(r.kalshi_taker.liquid.calibration.mid?.brier, 4)}</dd>
                <dt>Brier score, LevelEdge</dt><dd className="num">{fmtNum(r.kalshi_taker.liquid.calibration.p_v2?.brier, 4)}</dd>
                <dt>LevelEdge minus market</dt><dd className="num">{signed(r.kalshi_taker.liquid.brier_v2_minus_market.diff)}</dd>
                <TradeRow label="Profit from trading the difference" s={r.kalshi_taker.liquid.primary} />
              </dl>
            </Experiment>
          )}
          {r.kalshi_horizon && (
            <Experiment id="kh" title="Kalshi from the morning, over three years" verdict="rejected" verdictText="No edge at longer horizons either"
              question="Forecasting in the morning and early afternoon for the 4 PM close, where careful modelling should matter more than speed."
              method={`${r.kalshi_horizon.rows.toLocaleString()} quotes on ${r.kalshi_horizon.events.toLocaleString()} S&P 500 and Nasdaq-100 events, ${r.kalshi_horizon.period[0]} to ${r.kalshi_horizon.period[1]}, hourly bars.`}>
              <dl className="kv">
                <dt>Brier score, Kalshi's price</dt><dd className="num">{fmtNum(r.kalshi_horizon.liquid.calibration.mid?.brier, 4)}</dd>
                <dt>Brier score, LevelEdge</dt><dd className="num">{fmtNum(r.kalshi_horizon.liquid.calibration.p_v2?.brier, 4)}</dd>
                <dt>LevelEdge minus market, 95% interval</dt><dd className="num">{ci(r.kalshi_horizon.liquid.brier_v2_minus_market.ci95)}</dd>
                <TradeRow label="Profit from trading the difference" s={r.kalshi_horizon.liquid.primary} />
              </dl>
            </Experiment>
          )}
          {r.kalshi_maker && <MakerCard m={r.kalshi_maker} />}

          <h2 className="research-group">Building the portfolio</h2>
          {r.etf_trend && (
            <Experiment id="et" title="Following trends across 11 markets" verdict="neutral" verdictText="A real signal but no extra return: kept for crash protection"
              question={<>Does the <Term k="trend">trend rule</Term> across 11 markets beat simple portfolios?</>}
              method={`Rules fixed from published research before running. ${r.etf_trend.period[0]} to ${r.etf_trend.period[1]}, trades at the next open, 5 bp costs. Random pickers use the same sizing with random in/out signals.`}>
              <div className="table-scroll">
                <table className="table">
                  <thead><tr><th scope="col">Portfolio</th><th scope="col" className="r">Return a year</th><th scope="col" className="r"><Term k="sharpe">Sharpe</Term></th><th scope="col" className="r"><Term k="drawdown">Worst fall</Term></th></tr></thead>
                  <tbody>
                    {([["strategy", "Trend rule"], ["sixty_forty", "60/40 mix"], ["spy_buy_hold", "S&P 500"]] as const).map(([k, l]) => (
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
              <p className="help">It beat {Math.round((1 - r.etf_trend.random_picker.p_value_strategy_sharpe) * 100)}% of {r.etf_trend.random_picker.sims} portfolios that picked markets at random, so the signal isn't luck.</p>
            </Experiment>
          )}
          {r.etf_search && r.etf_search_points && <SearchCard s={r.etf_search} pts={r.etf_search_points} />}
          {r.return_models && <ModelsCard m={r.return_models} />}
          {r.construction && <ConstructionCard c={r.construction} />}
          {r.literature && <LiteratureCard l={r.literature} />}
          {r.literature && <BandsCard l={r.literature} />}
          {r.stock_factors && <FactorsCard f={r.stock_factors} />}

          <h2 className="research-group">Reading the news</h2>
          {r.news && <NewsCard n={r.news} grade={r.integrity?.strategies.find((x) => x.id === "news_sentiment_spy")?.grade} />}

          <h2 className="research-group">Protecting against crashes</h2>
          {r.hedge && <HedgeCard h={r.hedge} />}
          {r.put_hedge && <PutCard p={r.put_hedge} />}
          {r.regime && <TimingCard g={r.regime} />}

          <h2 className="research-group">Chasing more return</h2>
          {r.construction && <FrontierCard c={r.construction} />}
          {r.leveraged && <LeveragedCard v={r.leveraged} />}
          {r.reward && <RewardCard w={r.reward} />}
          {r.dip && <DipCard d={r.dip} />}
          {r.intraday && <IntradayCard i={r.intraday} />}
        </div>
      )}
    </>
  );
}

type Row = { id: string; idea: string; verdict: Verdict; label: string; plain: string };
const SCORE: { group: string; rows: Row[] }[] = [
  { group: "Predicting prices", rows: [
    { id: "fc", idea: "A better forecasting model", verdict: "adopted", label: "Adopted", plain: "Its probabilities are 32% more accurate than the original model's on 4,800 forecasts it never saw." },
    { id: "vol", idea: "A custom volatility model", verdict: "neutral", label: "Tie", plain: "No better than the standard method, so the simpler one is used." },
  ] },
  { group: "Beating a prediction market", rows: [
    { id: "kt", idea: "Trading Kalshi at market prices", verdict: "rejected", label: "No edge", plain: "The market was slightly more accurate. Trading the gap lost about 2¢ per contract after fees." },
    { id: "kh", idea: "Kalshi, forecasting from the morning", verdict: "rejected", label: "No edge", plain: "Three years of data, same answer." },
    { id: "km", idea: "Market-making on Kalshi", verdict: "rejected", label: "Loses", plain: "Lost 6–10¢ per trade in all 18 versions: orders only filled when better-informed traders wanted the other side." },
  ] },
  { group: "Building the portfolio", rows: [
    { id: "et", idea: "Following trends across 11 markets", verdict: "neutral", label: "Kept", plain: "A real signal that cuts crash losses, but no extra return. It's the 30% trend part of the portfolio." },
    { id: "es", idea: "Searching 80 variations for better rules", verdict: "rejected", label: "Failed", plain: "The best rules before 2020 were among the worst after. The original rules stayed." },
    { id: "rm", idea: "Machine-learning return forecasts", verdict: "rejected", label: "No gain", plain: "Five model types, none better than the simple trend rule." },
    { id: "cn", idea: "Risk parity and volatility targeting", verdict: "rejected", label: "No gain", plain: "Nothing beat a plain 60/40 mix reliably." },
    { id: "lit", idea: "Published strategies from academic papers", verdict: "neutral", label: "Watching", plain: "One leads but isn't proven; it's tracked without trading it." },
    { id: "bd", idea: "Trading only when holdings drift", verdict: "adopted", label: "Adopted", plain: "Cut trades from 419 to 64 a year with almost no change in results." },
    { id: "sf", idea: "Stock-picking factors", verdict: "rejected", label: "No gain", plain: "No factor beat simply owning every S&P 500 stock equally." },
  ] },
  { group: "Reading the news", rows: [
    { id: "ns", idea: "Timing the market with news sentiment", verdict: "rejected", label: "Failed", plain: "Strong before 2020 (Sharpe 0.71 vs 0.48 for holding), then 0.22 vs 0.65 after. As a crash warning, barely better than a coin flip." },
  ] },
  { group: "Protecting against crashes", rows: [
    { id: "hg", idea: "Selling on warning signs", verdict: "neutral", label: "1 of 6 passed", plain: "Only the 12-month trend filter cut losses without costing too much. It's tracked, not yet traded." },
    { id: "ph", idea: "Insurance with options", verdict: "rejected", label: "Too costly", plain: "The cost of the insurance ate most of the protection." },
    { id: "rg", idea: "Predicting crashes in advance", verdict: "rejected", label: "Can't be done", plain: "No method predicted 5% drops reliably. How big moves get can be forecast; when they come can't." },
  ] },
  { group: "Chasing more return", rows: [
    { id: "fr", idea: "Borrowing to invest more (margin)", verdict: "rejected", label: "Loses", plain: "At a retail broker's ~11.6% interest rate, borrowing lowered returns and deepened falls." },
    { id: "ag", idea: "2× and 3× leveraged funds", verdict: "neutral", label: "Watching", plain: "More return, but past falls of 59% to 90% even with a safety rule. Tracked without trading." },
    { id: "rw", idea: "A bigger trend share with leverage", verdict: "rejected", label: "Failed", plain: "Looked good before 2020, failed after." },
    { id: "dp", idea: "Buying the dip", verdict: "rejected", label: "Doesn't work", plain: "A stock that fell 20% in a month beat the average stock only 49% of the time over the next 3 months." },
    { id: "id", idea: "Day-trading strategies", verdict: "rejected", label: "Loses", plain: "Published day-trading rules lost money or trailed simply holding, after costs." },
  ] },
];

function Scoreboard({ go }: { go: (id: string) => void }) {
  const all = SCORE.flatMap((g) => g.rows);
  const n = (v: Verdict) => all.filter((r) => r.verdict === v).length;
  return (
    <section className="card scoreboard" aria-labelledby="score-h">
      <h2 id="score-h" className="takeaway">{all.length} ideas tested: {n("adopted")} adopted, {n("neutral")} kept or watched, {n("rejected")} rejected</h2>
      <p className="card-sub">Select an idea to read the evidence below.</p>
      <div style={{ marginTop: 16 }}>
        {SCORE.map((g) => (
          <div className="score-group" key={g.group}>
            <h3>{g.group}</h3>
            {g.rows.map((row) => {
              const Icon = VERDICT_ICON[row.verdict];
              return (
                <div className="score-row" key={row.id}>
                  <a href={`#${row.id}`} onClick={(e) => { e.preventDefault(); go(row.id); }}>{row.idea}</a>
                  <p className="verdict" data-verdict={row.verdict}><Icon size={16} aria-hidden="true" />{row.label}</p>
                  <p>{row.plain}</p>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </section>
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
    <Experiment id="rm" title="Machine-learning return forecasts" verdict="rejected" verdictText="No gain over the simple trend rule"
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
    <Experiment id="fr" title="Borrowing to invest more (margin)" verdict="rejected" verdictText="At a retail broker's interest rate, borrowing lowers returns and deepens falls"
      question="If the portfolio borrowed to hold more than 100% of its value, how would return and the worst fall change?"
      method="The portfolio's own weights scaled up by a fixed multiple, borrowing at T-bills + 2%, 2007 onward. SPY alone: about 10.7% a year with a −55% worst fall.">
      <PerfTable cols={["Gross", "Return", "Worst fall", "Worst year", "Sharpe"]}
        rows={book.map((f) => ({ label: `${f.knob}×`, emph: f.knob === 1.5, cells: [rate(f.cagr), rate(f.max_drawdown, 0), rate(f.worst_year, 0), sh(f.sharpe)] }))} />
      <p className="help">These rows assume borrowing almost as cheaply as a big institution (T-bills + 2%), where 2× nearly matches the S&P 500's return with a shallower worst fall. A retail account pays far more (Fidelity charged about 11.6% in November 2026, roughly T-bills + 7.5%), and at that rate 1.5× returned 6.6% a year with a −37% worst fall, against 7.3% and −24% with no borrowing. The practice portfolio borrows nothing.</p>
    </Experiment>
  );
}

function RewardCard({ w }: { w: NonNullable<ResearchResponse["reward"]> }) {
  const h = w.holdout;
  return (
    <Experiment id="rw" title="A bigger trend share with leverage" verdict="rejected" verdictText="Failed the held-out test"
      question={`Most return before 2020 with the worst fall no deeper than ${rate(w.dd_limit, 0)}: trend share 20–50%, leverage 1–2×.`}
      method={`20 books, selected on 2007–2019 (picked: ${w.selected.replace("s=", "trend ").replace("|L=", ", ")}×), then 2020 onward read once.`}>
      <PerfTable cols={["From 2020", "Return", "Volatility", "Worst fall"]}
        rows={[{ label: "Best choice before 2020", emph: true, cells: [rate(h.selected.cagr), rate(h.selected.vol), rate(h.selected.max_drawdown, 0)] },
               { label: "Current portfolio", cells: [rate(h.current_book.cagr), rate(h.current_book.vol), rate(h.current_book.max_drawdown, 0)] },
               { label: "SPY", cells: [rate(h.spy_buy_hold.cagr), rate(h.spy_buy_hold.vol), rate(h.spy_buy_hold.max_drawdown, 0)] }]} />
      <p className="help">The borrowing was spent on more trend-following, which lagged after 2020: more trend lowered returns at every leverage level.</p>
    </Experiment>
  );
}

function FactorsCard({ f }: { f: NonNullable<ResearchResponse["stock_factors"]> }) {
  const names: [string, string][] = [["mom_12_1", "Momentum (12-1)"], ["mom_lowvol", "Momentum + low vol"], ["lowvol", "Low volatility"], ["rev_1m", "1-month reversal"]];
  return (
    <Experiment id="sf" title="Stock-picking factors" verdict="rejected" verdictText="No factor beat an equal-weight portfolio of the same stocks"
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
    <Experiment id="lit" title="Published strategies from academic papers" verdict="neutral"
      verdictText="One leads but isn't proven, so it's tracked without trading"
      question="Do strategies from published papers, implemented exactly as specified, beat the current trend rule as the portfolio's 30% trend part?"
      method={`${l.period[0]} onward, borrowing at T-bills + 2%, 5 bp costs. Selected on Sharpe before 2020, then 2020 onward read once (fresh for all of them). Corrected for ${l.deflated.trials} strategies tried, there is a ${rate(l.deflated.deflated_sharpe_prob, 0)} chance the winner is not luck; 95% is the usual bar.`}>
      <PerfTable cols={["Sleeve", "Return", "Sharpe before 2020", "Sharpe from 2020"]}
        rows={books.map(([k, n]) => ({ label: n, emph: k === l.selected, cells: [rate(l.full[k].cagr), sh(l.dev[k].sharpe), sh(l.holdout[k].sharpe)] }))} />
      <p className="help">From 2020 the Defensive Asset Allocation version beat the current one by {sh(vs.diff)} Sharpe (95% interval {sh(vs.ci95[0])} to {sh(vs.ci95[1])}). It is followed forward from public prices, untraded, and would replace the current trend rule only after months of evidence.
        {" "}Turn-of-the-month and overnight-only SPY were rejected: the turn-of-month days earned {l.per_day_bp.tom_window.toFixed(1)} bp a day against {l.per_day_bp.spy_all_days.toFixed(1)} bp on any day, and overnight returns fall from {l.per_day_bp.overnight_0bp.toFixed(1)} to {l.per_day_bp.overnight_1bp.toFixed(1)} bp a day at 1 bp of cost per side.</p>
    </Experiment>
  );
}

function BandsCard({ l }: { l: NonNullable<ResearchResponse["literature"]> }) {
  const b0 = l.bands["0.00"], bs = l.bands[l.band_selected];
  return (
    <Experiment id="bd" title="Trading only when holdings drift" verdict="adopted" verdictText={`Adopted: trade an ETF only when it drifts ${(Number(l.band_selected) * 100).toFixed(0)} points from target`}
      question="How much of the trading is worth its cost? Each week, trade only the investments that have drifted past a set distance from their target."
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
        <h2 id="es-h">Searching 80 variations for better rules</h2>
        <p className="card-sub">Pick the best of 80 published variants on {s.period.dev[0].slice(0, 4)}–2019, then test it once on 2020 onward.</p>
        <p className="verdict" data-verdict="rejected"><CircleX size={18} aria-hidden="true" />Failed the held-out test: the best before 2020 was among the worst after</p>
        <dl className="kv">
          <dt>Best variation, <Term k="sharpe">Sharpe</Term> before 2020</dt><dd className="num">{s.deflated_sharpe_dev.sharpe_annual.toFixed(2)}</dd>
          <dt>Chance it wasn't luck</dt><dd className="num">{fmtRate(s.deflated_sharpe_dev.deflated_sharpe_prob, 0)}</dd>
          <dt>Sharpe after 2020</dt><dd className="num">{s.holdout.selected.sharpe.toFixed(2)}</dd>
          <dt>Minus 60/40 after 2020, 95% interval</dt><dd className="num">{ci(vs.ci95, (x) => minus(x, 3))}</dd>
          <dt>Did good-before stay good-after? (1 = yes, −1 = reversed)</dt><dd className="num">{signed(s.rank_stability.spearman_dev_vs_holdout, 2)}</dd>
        </dl>
        <p className="footnote">The variants that looked best leaned on Treasury bonds, which only rose while interest rates fell from 2007 to 2019. When rates climbed after 2021, they fell hardest. The practice portfolio keeps the original rules, chosen before the search.</p>
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

// ---- newer studies ------------------------------------------------------------------------
const b = (m: Brief | undefined) => [rate(m?.cagr), rate(m?.max_drawdown, 0), sh(m?.sharpe)];

function HedgeCard({ h }: { h: NonNullable<ResearchExtra["hedge"]> }) {
  const names: [string, string][] = [["current", "No overlay (the portfolio as traded then)"], ["core_tsmom12", "12-month trend filter"],
    ["core_sma10", "10-month average"], ["vix30", "Sell when the VIX is above 30"], ["curve_invert", "Sell when the yield curve inverts"],
    ["drawdown_cut", "Sell after a 10% fall"], ["vol_managed", "Scale down when swings rise"]];
  return (
    <Experiment id="hg" title="Selling on warning signs" verdict="neutral" verdictText="Only the 12-month trend filter passed; it's tracked, not yet traded"
      question="Can a warning signal sell the stock-and-bond part of the portfolio before a crash, and buy back afterwards, without giving up too much return?"
      method="Six signals, each with fixed rules. To pass, a signal had to cut the worst fall by at least 5 points in both 2007–2019 and 2020+, without lowering return per unit of risk, and cost no more than 1 point a year. Tested on the earlier 1.5× version of the portfolio.">
      <PerfTable cols={["Signal", "Return a year", "Worst fall", "Sharpe"]}
        rows={names.filter(([k]) => h.variants[k]).map(([k, l]) => ({ label: l, emph: h.adopted.includes(k), cells: b(h.variants[k].full) }))} />
      <p className="help">Most signals sold too late, after the damage, or too often, missing the rebounds. The trend filter cut the worst fall from 35% to 21% for about 0.9 points a year less return. The 10-month average fell least but gave up about 2 points a year, so it failed the cost rule.</p>
    </Experiment>
  );
}

function PutCard({ p }: { p: NonNullable<ResearchExtra["put_hedge"]> }) {
  const names: [string, string][] = [["current", "No insurance"], ["put5", "Monthly insurance, 5% below the price"], ["put10", "Monthly insurance, 10% below"],
    ["collar5", "Insurance paid for by giving up gains above 5%"], ["core_tsmom12 (reference)", "12-month trend filter, for comparison"]];
  return (
    <Experiment id="ph" title="Insurance with options" verdict="rejected" verdictText="The insurance cost more than it protected"
      question="Do monthly S&P 500 put options, which pay out if the market falls past a level, protect the portfolio for a fair price?"
      method="Monthly options on the portfolio's S&P 500 holding, priced from the VIX with a realistic skew and a 3% spread, held to expiry and rolled. 2007–2026.">
      <PerfTable cols={["Approach", "Return a year", "Worst fall", "Sharpe"]}
        rows={names.filter(([k]) => p.variants[k]).map(([k, l]) => ({ label: l, cells: b(p.variants[k].full) }))} />
      <p className="help">Insurance 5% below the price cost about 4.7% of the portfolio a year, which ate most of the protection. The trend filter protected more and cost less.</p>
    </Experiment>
  );
}

function TimingCard({ g }: { g: NonNullable<ResearchExtra["regime"]> }) {
  const names: [string, string][] = [["vix", "The VIX"], ["har", "Recent daily ranges (HAR)"], ["hmm", "Hidden Markov model"], ["jump_model", "Statistical jump model"],
    ["turbulence", "Turbulence index"], ["trend12_off", "12-month trend turned off"]];
  const m1 = g.range_holdout["21"];
  return (
    <Experiment id="rg" title="Predicting crashes in advance" verdict="rejected" verdictText="How big moves get can be forecast; when they come can't"
      question="Can any signal tell, a month ahead, when the S&P 500 is about to fall 5% or more?"
      method={`Every Friday since 2000, each signal forecast the chance of a 5%+ fall over the next month (it happened ${rate(g.base_rate, 0)} of the time in the test years). Scored by AUC: 0.5 is a coin flip, 1.0 is perfect. Range forecasts scored by how often their ranges held.`}>
      <PerfTable cols={["Signal", "Score before 2020", "Score since 2020"]}
        rows={names.filter(([k]) => g.danger_auc[k]).map(([k, l]) => ({ label: l, cells: [g.danger_auc[k].dev?.toFixed(2) ?? "—", g.danger_auc[k].holdout?.toFixed(2) ?? "—"] }))} />
      <p className="help">No signal scored much above a coin flip (0.5), and two scored below it, meaning worse than guessing. Forecasting the size of moves did work: since 2020 the 1-month likely range held {m1 ? rate(m1.cover80, 0) : "about 80%"} of the time, close to the 80% it promises. That forecast powers the market-outlook card on the Portfolio page.</p>
    </Experiment>
  );
}

function LeveragedCard({ v }: { v: NonNullable<ResearchExtra["leveraged"]> }) {
  const rows: [string, string][] = [["GSPCx1_none", "S&P 500, held"], ["GSPCx2_none", "2× S&P 500, held"], ["GSPCx2_sma200", "2× S&P 500 with a safety rule"],
    ["GSPCx3_sma200", "3× S&P 500 with a safety rule"], ["NDXx1_none", "Nasdaq-100, held"], ["NDXx2_sma200", "2× Nasdaq-100 with a safety rule"], ["NDXx3_sma200", "3× Nasdaq-100 with a safety rule"]];
  return (
    <Experiment id="ag" title="2× and 3× leveraged funds" verdict="neutral" verdictText="More return, far bigger falls: tracked without trading"
      question={<>How much more could <Term k="leveraged_etf">leveraged funds</Term> earn without borrowing, and what happens in a crash?</>}
      method="Leveraged funds modelled from daily index returns with financing and fees, checked against the real SSO, UPRO, QLD and TQQQ (within 0.3% a year). The safety rule holds the fund only while the index is above its 200-day average, cash otherwise. S&P 500 from 1960, Nasdaq-100 from 1985, price only (dividends excluded).">
      <PerfTable cols={["Approach", "Return a year", "Worst fall", "Dot-com crash"]}
        rows={rows.filter(([k]) => v[k]).map(([k, l]) => ({ label: l, cells: [rate(v[k].cagr), rate(v[k].max_dd, 0), v[k].windows["Dot-com crash"] != null ? rate(v[k].windows["Dot-com crash"], 0) : "—"] }))} />
      <p className="help">Held without a safety rule, 2× and 3× funds lost 91–100% at some point. With the rule they survived, but their worst falls of 59% to 90% are far deeper than most people can sit through. They're tracked in the weekly report without trading.</p>
    </Experiment>
  );
}

function DipCard({ d }: { d: NonNullable<ResearchExtra["dip"]> }) {
  const f = d.fell_20_1m;
  const st = d.strategy.holdout;
  return (
    <Experiment id="dp" title="Buying the dip" verdict="rejected" verdictText="Doesn't work: a fallen stock is a coin flip against the average stock"
      question={`After an S&P 500 stock falls 20% or more in a month, does buying it and holding for 3 months pay off?`}
      method={`${f.n.toLocaleString()} falls in historical S&P 500 members, 2006–2026, bought at the next open and held 3 months, against the average member bought the same day. Data for failed companies is missing, which flatters dip buying, so the real result is worse.`}>
      <dl className="kv">
        <dt>Fallen stocks that were higher 3 months later</dt><dd className="num">{rate(f.p_profit, 0)}</dd>
        <dt>Average stock bought the same day</dt><dd className="num">{rate(f.bench_p_profit, 0)}</dd>
        <dt>Fallen stock beat the average stock</dt><dd className="num">{rate(f.p_beat_bench, 0)} of the time</dd>
        <dt>Worst 1 in 10 outcomes</dt><dd className="num">{rate(f.p10_ret, 0)} or worse</dd>
        {st && <><dt>As a strategy since 2020, return a year vs the S&P 500</dt><dd className="num">{rate(st.dips.cagr)} vs {rate(st.spy.cagr)}</dd></>}
      </dl>
      <p className="help">Big falls cluster in market sell-offs, and markets usually recover, so the fallen stock often rose, just not by more than any other stock bought that day. It's a bet on the market, with far more risk.</p>
    </Experiment>
  );
}

function IntradayCard({ i }: { i: NonNullable<ResearchExtra["intraday"]> }) {
  return (
    <Experiment id="id" title="Day-trading strategies" verdict="rejected" verdictText="Lost money or trailed simply holding, after costs"
      question="Do published day-trading rules on the Nasdaq-100 make money once realistic trading costs are paid?"
      method={`10 years of 5-minute bars, ${i.period[0]} to ${i.period[1]}, 1 basis point of cost per trade. Rules implemented as their papers specify.`}>
      <PerfTable cols={["Approach", "Return a year", "Worst fall", "Sharpe"]}
        rows={[{ label: "Opening-range breakout", cells: b(i.orb_1bp) }, { label: "Intraday momentum", cells: b(i.intramom_1bp) },
               { label: "Just holding the Nasdaq-100 (QQQ)", emph: true, cells: b(i.qqq) }]} />
      <p className="help">Small edges that look good before costs disappear once every trade pays a little to get in and out.</p>
    </Experiment>
  );
}

// ---- trust scores (research-integrity engine) ---------------------------------------------
const CHECK_LABEL: Record<string, string> = {
  holdout: "Still works on unseen years", benchmark: "Beats a simple alternative", luck: "Timing beats chance",
  multiple: "Survives counting every try", stability: "Variants keep their order", lookahead: "Never sees the future",
  costs: "Survives higher costs", concentration: "Not one lucky year",
};
const STATUS_ICON = { pass: CircleCheck, warn: CircleMinus, fail: CircleX, "n/a": CircleMinus } as const;

function TrustScores({ t }: { t: NonNullable<ResearchExtra["integrity"]> }) {
  const top = t.strategies[0];
  const searched = t.strategies.find((s) => s.id === "best_of_80");
  return (
    <section className="card trust-scores" aria-labelledby="trust-h">
      <h2 id="trust-h" className="takeaway">
        Every strategy graded for self-deception: {top.name.toLowerCase()} scores {top.grade}
        {searched ? `; the strategy found by searching 80 variants scores ${searched.grade}` : ""}
      </h2>
      <p className="card-sub">
        Eight checks, each asking one way a backtest can fool its author. The grade combines them (0–100). It ran as {t.tasks} independent
        tasks{t.executor === "aws" ? " spread across cloud workers" : " in parallel"}, about {Math.round(t.compute_seconds / 60)} minutes of computing in total.
      </p>
      <ol className="trust-rows">
        {t.strategies.map((s) => <TrustRow key={s.id} s={s} />)}
      </ol>
      <details className="disclosure">
        <summary>What each check asks, and how much it counts</summary>
        <dl className="check-legend">
          {Object.entries(t.weights).map(([k, w]) => (
            <div key={k}><dt>{CHECK_LABEL[k]} <span className="move">{w} points</span></dt><dd>{CHECK_HELP[k]}</dd></div>
          ))}
        </dl>
        <p className="help">A strategy that fails the look-ahead check is capped at 20 points: if a backtest saw the future, nothing else in it counts.</p>
      </details>
    </section>
  );
}

const CHECK_HELP: Record<string, string> = {
  holdout: "Compares its record after 2020, which played no part in designing it, with its record before and with the benchmark's after.",
  benchmark: "Is its edge over a simple alternative bigger than luck? Measured with a 95% interval that resamples whole months.",
  luck: "Shuffles its own holdings in time 200 times. Real timing skill should beat almost every shuffled copy.",
  multiple: "The more versions you try, the better the best one looks by chance. This discounts for every version tried.",
  stability: "If the versions that looked best before 2020 did worst after, the search was fitting noise.",
  lookahead: "Recomputes its decisions with the data cut off at past dates. Any difference means it used information it couldn't have had.",
  costs: "Charges four times the normal trading costs. A real edge shouldn't vanish.",
  concentration: "Removes its single best year. An edge that disappears depended on one event.",
};

function TrustRow({ s }: { s: TrustReport }) {
  const tone = s.grade <= "B" ? "good" : s.grade === "C" ? "warn" : "bad";
  return (
    <li className="trust-row">
      <span className="grade" data-tone={tone} aria-label={`Grade ${s.grade}`}>{s.grade}</span>
      <div className="trust-main">
        <div className="trust-title">
          <strong>{s.name}</strong>
          <span className="trust-score num">{s.score.toFixed(0)}<span className="move"> / 100 · {s.verdict}</span></span>
        </div>
        <span className="pbar trust-bar" aria-hidden="true"><i style={{ width: `${s.score}%` }} /></span>
        <p>{s.findings[0]}</p>
        <details className="disclosure">
          <summary>See all 8 checks</summary>
          <ul className="check-list">
            {Object.entries(s.checks).map(([k, c]) => {
              const Icon = STATUS_ICON[c.status];
              return (
                <li key={k} data-status={c.status}>
                  <Icon size={16} aria-hidden="true" />
                  <div><strong>{CHECK_LABEL[k]}</strong> <span className="move">{c.status === "n/a" ? "not applicable" : c.status}</span><p>{c.detail}</p></div>
                </li>
              );
            })}
          </ul>
          {s.findings.length > 1 && <><p className="help"><strong>How it might be fooling us</strong></p><ul className="help">{s.findings.map((f) => <li key={f}>{f}</li>)}</ul></>}
        </details>
      </div>
    </li>
  );
}

function NewsCard({ n, grade }: { n: NonNullable<ResearchExtra["news"]>; grade?: string }) {
  const rows: [string, string][] = [["spy_buy_hold", "Just holding the S&P 500"], ["level", "Hold while sentiment is above its 1-year average"],
    ["momentum", "Hold while sentiment is rising"], ["book", "The portfolio"], ["book_news_filter", "The portfolio, selling its core on bad news"],
    ["book_trend_filter", "The portfolio with the trend filter, for comparison"]];
  const lv = n.crash_auc.level_vs_avg;
  return (
    <Experiment id="ns" title="Timing the market with news sentiment" verdict="rejected" verdictText="Worked before 2020, failed after: rejected"
      question="Does the tone of economic news, as scored by the San Francisco Fed across 24 major newspapers, tell you when to be in the stock market?"
      method={`${n.source}, daily since 1980. A value counts only ${n.lag_days} days after the date it covers, because the data is published weekly. Weekly decisions, trades at the next open, 5 bp costs; ${n.period[0]} to ${n.period[1]}; rules fixed before running.`}>
      <PerfTable cols={["Approach", "Return a year", "Worst fall", "Sharpe before 2020", "Sharpe since 2020"]}
        rows={rows.filter(([k]) => n.runs[k]).map(([k, l]) => ({ label: l, emph: k === "level",
          cells: [rate(n.runs[k].full.cagr), rate(n.runs[k].full.max_drawdown, 0), sh(n.runs[k].dev.sharpe), sh(n.runs[k].holdout.sharpe)] }))} />
      <p className="help">The level rule beat holding the S&P 500 before 2020, then trailed it badly after. As a crash warning, news sentiment scored {lv.dev.toFixed(2)} before 2020
        and {lv.holdout.toFixed(2)} since, where 0.5 is a coin flip. {grade ? ` The trust engine above grades it ${grade}.` : ""}</p>
    </Experiment>
  );
}
