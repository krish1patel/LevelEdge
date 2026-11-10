import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Info, ShieldCheck } from "lucide-react";
import { getPortfolio, type LivePaper, type PortfolioBacktest, type PortfolioResponse, type SeriesKey, type Targets } from "../api";
import { fmtIsoDay, fmtMoney, fmtNum, fmtRate, fmtSignedPct } from "../format";
import { XYChart, type Series } from "./XYChart";

const NAMES: Record<SeriesKey, string> = {
  book: "LevelEdge book",
  sixty_forty: "60/40",
  spy_buy_hold: "SPY",
  trend_sleeve_alone: "Trend sleeve alone",
};
// One voice: the book is the accent; comparisons are grey, told apart by line style.
const STYLE: Record<"book" | "sixty_forty" | "spy_buy_hold", Pick<Series, "color" | "dashed">> = {
  book: { color: "var(--accent)" },
  sixty_forty: { color: "var(--neutral-series)" },
  spy_buy_hold: { color: "var(--neutral-series)", dashed: true },
};
const CHART_KEYS = ["book", "sixty_forty", "spy_buy_hold"] as const;

export function Portfolio() {
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    getPortfolio().then((d) => live && setData(d)).catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, []);

  if (err) {
    return (
      <div className="banner" data-tone="error" role="alert">
        <AlertCircle size={18} aria-hidden="true" />
        <p><strong>Couldn't load the portfolio.</strong> {err}</p>
      </div>
    );
  }

  const t = data?.targets ?? null;
  const live = data?.live;
  const bt = data?.backtest ?? null;

  return (
    <>
      <section className="summary" aria-live="polite">
        <h1>
          <span className={live?.status === "ok" ? "sym" : undefined}>Paper book</span>
          {live?.status === "ok" && <span className="spot num">{fmtMoney(live.account.equity)}</span>}
        </h1>
        {data ? (
          <dl className="facts">
            <div><dt>Design</dt><dd>{t?.leverage && t.leverage !== 1 ? `${t.leverage}× of ` : ""}70% 60/40 + 30% trend</dd></div>
            {t && t.gross > 1 && <div><dt>Gross exposure</dt><dd className="num">{fmtRate(t.gross, 0)} (margin)</dd></div>}
            {t && <div><dt>Targets from</dt><dd>Close of {fmtIsoDay(t.as_of_close)}</dd></div>}
            {t && <div><dt>Next rebalance</dt><dd>{fmtIsoDay(t.next_rebalance)}, after the close</dd></div>}
            {t && <div><dt>Trend sleeve invested</dt><dd className="num">{fmtRate(t.sleeve_gross, 0)}</dd></div>}
            <div><dt>Paper account</dt><dd>{live?.status === "ok" ? (live.paper ? "Connected (paper)" : "Connected") : "Not connected"}</dd></div>
          </dl>
        ) : (
          <div className="skeleton" style={{ width: 420, maxWidth: "100%", height: 40 }} />
        )}
      </section>

      {data && data.targets_status !== "ok" && (
        <div className="banner" data-tone="error" role="alert">
          <AlertCircle size={18} aria-hidden="true" />
          <p><strong>This week's targets aren't available.</strong> {data.targets_status}</p>
        </div>
      )}

      <div className="grid-main">
        <div className="side main-col">
          <LiveAccount live={live} />
          {bt ? <BacktestCards bt={bt} /> : data ? (
            <section className="card"><p className="help">Run <code>python benchmarks/paper_portfolio.py</code> to generate the backtest.</p></section>
          ) : (
            <section className="card"><div className="skeleton skeleton-chart" aria-label="Loading backtest" /></section>
          )}
        </div>
        <div className="side">
          <TargetsCard t={t} loading={!data} />
          {bt && <TrustCard bt={bt} live={live} />}
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
function LiveAccount({ live }: { live: LivePaper | undefined }) {
  return (
    <section className="card" aria-labelledby="live-acct-h">
      <div className="card-head">
        <div>
          <h2 id="live-acct-h">Live paper account</h2>
          <p className="card-sub">Alpaca paper trading: real prices and order handling, simulated money</p>
        </div>
      </div>
      {!live && <div className="skeleton" style={{ height: 160 }} aria-label="Loading paper account" />}
      {live?.status === "error" && <p className="err" role="alert">{live.message}</p>}
      {live?.status === "not_configured" && (
        <div className="empty">
          <h3>Connect a paper account to start the live record</h3>
          <ol className="steps">
            <li>Create a free account at alpaca.markets and open the paper-trading dashboard.</li>
            <li>Generate paper API keys and add them as <code>ALPACA_API_KEY_ID</code> and <code>ALPACA_API_SECRET_KEY</code> in <code>.env</code>, the HuggingFace Space secrets and the GitHub repository secrets.</li>
            <li>The <code>paper_trade.yml</code> job rebalances after the close on the last trading day of each week. Orders fill at the next open, as in the backtest.</li>
          </ol>
          <p className="help">Until then, everything below is a backtest. LevelEdge refuses live-money endpoints unless you set <code>LEVELEDGE_ALLOW_LIVE_TRADING=1</code>.</p>
        </div>
      )}
      {live?.status === "ok" && <LiveBody live={live} />}
    </section>
  );
}

function LiveBody({ live }: { live: Extract<LivePaper, { status: "ok" }> }) {
  const day = live.account.equity - live.account.last_equity;
  const curve = live.curve;
  const first = curve[0], last = curve[curve.length - 1];
  return (
    <>
      <dl className="kv kv-inline">
        <dt>Equity</dt><dd className="num">{fmtMoney(live.account.equity)}</dd>
        <dt>Today</dt><dd className="num">{fmtMoney(day)} ({fmtSignedPct(day / live.account.last_equity, 2)})</dd>
        <dt>Cash</dt><dd className="num">{fmtMoney(live.account.cash)}</dd>
        {curve.length >= 2 && (
          <>
            <dt>Since {fmtIsoDay(first.date)}</dt><dd className="num">{fmtSignedPct(last.book - 1, 2)}</dd>
            {last.sixty_forty != null && (<><dt>60/40 over the same days</dt><dd className="num">{fmtSignedPct(last.sixty_forty - 1, 2)}</dd></>)}
          </>
        )}
      </dl>
      {curve.length >= 2 ? (
        <XYChart
          series={CHART_KEYS.map((k) => ({
            name: NAMES[k], ...STYLE[k],
            points: curve.flatMap((p, i) => (p[k] == null ? [] : [{ x: i, y: (p[k] as number) - 1 }])),
          }))}
          xLabel={(v) => fmtIsoDay(curve[Math.round(v)]?.date ?? first.date)}
          yLabel={(v) => fmtSignedPct(v, 1)} hline={0} xDomain={[0, curve.length - 1]} height={220}
          summary={`Paper book return since ${first.date}: ${fmtSignedPct(last.book - 1, 2)}.`}
        />
      ) : (
        <p className="help">The live chart appears after the first rebalance has filled and a second day of equity is recorded.</p>
      )}
      {live.positions.length > 0 && (
        <table className="table" style={{ marginTop: 16 }}>
          <caption className="visually-hidden">Current positions</caption>
          <thead><tr><th scope="col">ETF</th><th scope="col" className="r">Weight</th><th scope="col" className="r">Value</th><th scope="col" className="r">Unrealized</th></tr></thead>
          <tbody>
            {[...live.positions].sort((a, b) => b.market_value - a.market_value).map((p) => (
              <tr key={p.symbol}>
                <td>{p.symbol}</td>
                <td className="r num">{fmtRate(p.weight, 1)}</td>
                <td className="r num">{fmtMoney(p.market_value, 0)}</td>
                <td className="r num">{fmtMoney(p.unrealized_pl, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {live.orders.length > 0 && (
        <details className="disclosure">
          <summary>Recent orders ({live.orders.length})</summary>
          <table className="table" style={{ marginTop: 8 }}>
            <thead><tr><th scope="col">Submitted</th><th scope="col">Order</th><th scope="col">Status</th><th scope="col" className="r">Fill price</th></tr></thead>
            <tbody>
              {live.orders.map((o, i) => (
                <tr key={i}>
                  <td className="num">{o.submitted_at?.slice(0, 10)}</td>
                  <td>{o.side === "buy" ? "Buy" : "Sell"} {o.symbol} {o.notional != null ? fmtMoney(o.notional, 0) : `${o.qty} sh`}</td>
                  <td>{o.status.replace(/_/g, " ")}</td>
                  <td className="r num">{o.filled_avg_price != null ? fmtMoney(o.filled_avg_price) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
function yearTicks(dates: string[]) {
  const out: number[] = [];
  dates.forEach((d, i) => { if (i === 0 || d.slice(0, 4) !== dates[i - 1].slice(0, 4)) out.push(i); });
  return out.slice(1);
}

function BacktestCards({ bt }: { bt: PortfolioBacktest }) {
  const [period, setPeriod] = useState<"full" | "holdout">("full");
  const dd = bt.drawdown_weekly;
  const eq = bt.equity_weekly;
  const ddDates = useMemo(() => dd.map((p) => p.date), [dd]);
  const eqDates = useMemo(() => eq.map((p) => p.date), [eq]);
  const worst = (k: SeriesKey) => bt.metrics[k].max_drawdown;
  const m = period === "full" ? bt.metrics : bt.holdout;
  const ci = period === "full" ? bt.sharpe_vs_sixty_forty.full : bt.sharpe_vs_sixty_forty.holdout;
  const startYear = bt.period[0].slice(0, 4);

  return (
    <>
      <section className="card" aria-labelledby="dd-h">
        <div className="card-head">
          <div>
            <h2 id="dd-h">How deep the falls were</h2>
            <p className="card-sub">Decline from the previous peak, each week since {startYear}. Backtest with next-open fills and {bt.cost_bps} bp trading costs.</p>
          </div>
        </div>
        <XYChart
          series={CHART_KEYS.map((k) => ({ name: NAMES[k], ...STYLE[k], points: dd.map((p, i) => ({ x: i, y: p[k] })) }))}
          xLabel={(v) => ddDates[Math.round(v)]?.slice(0, 4) ?? ""} xTip={(v) => fmtIsoDay(ddDates[Math.round(v)] ?? ddDates[0])}
          xTicks={yearTicks(ddDates)} yLabel={(v) => fmtRate(v, 0)} xDomain={[0, dd.length - 1]} height={280}
          summary={`Worst decline from a peak: LevelEdge book ${fmtRate(worst("book"))}, 60/40 ${fmtRate(worst("sixty_forty"))}, SPY ${fmtRate(worst("spy_buy_hold"))}.`}
        />
      </section>

      <section className="card" aria-labelledby="eq-h">
        <div className="card-head">
          <div>
            <h2 id="eq-h">Growth of $10,000</h2>
            <p className="card-sub">Same backtest. SPY grows most and falls hardest; the book trades some growth for shallower falls.</p>
          </div>
        </div>
        <XYChart
          series={CHART_KEYS.map((k) => ({ name: NAMES[k], ...STYLE[k], points: eq.map((p, i) => ({ x: i, y: p[k] * 10000 })) }))}
          xLabel={(v) => eqDates[Math.round(v)]?.slice(0, 4) ?? ""} xTip={(v) => fmtIsoDay(eqDates[Math.round(v)] ?? eqDates[0])}
          xTicks={yearTicks(eqDates)} yLabel={(v) => fmtMoney(v, 0)} xDomain={[0, eq.length - 1]} height={280}
          summary={`$10,000 from ${bt.period[0]} to ${bt.period[1]}: LevelEdge book ${fmtMoney(eq[eq.length - 1].book * 10000, 0)}, 60/40 ${fmtMoney(eq[eq.length - 1].sixty_forty * 10000, 0)}, SPY ${fmtMoney(eq[eq.length - 1].spy_buy_hold * 10000, 0)}.`}
        />
      </section>

      <section className="card" aria-labelledby="perf-h">
        <div className="card-head">
          <div>
            <h2 id="perf-h">Performance</h2>
            <p className="card-sub">{period === "full" ? `${bt.period[0]} to ${bt.period[1]}` : `Since 2020: the period held out of the strategy search`}</p>
          </div>
          <div className="segmented" role="group" aria-label="Period">
            <button type="button" aria-pressed={period === "full"} onClick={() => setPeriod("full")}>Since {startYear}</button>
            <button type="button" aria-pressed={period === "holdout"} onClick={() => setPeriod("holdout")}>Since 2020</button>
          </div>
        </div>
        <div className="table-scroll">
          <table className="table">
            <thead>
              <tr>
                <th scope="col">Portfolio</th><th scope="col" className="r">Annual return</th><th scope="col" className="r">Volatility</th>
                <th scope="col" className="r">Sharpe</th><th scope="col" className="r">Worst fall</th><th scope="col" className="r">Worst month</th>
              </tr>
            </thead>
            <tbody>
              {(["book", "sixty_forty", "spy_buy_hold", "trend_sleeve_alone"] as SeriesKey[]).map((k) => (
                <tr key={k} data-emph={k === "book" ? "true" : undefined}>
                  <th scope="row" className="row-h">{NAMES[k]}</th>
                  <td className="r num">{fmtRate(m[k].cagr)}</td>
                  <td className="r num">{fmtRate(m[k].vol)}</td>
                  <td className="r num">{m[k].sharpe.toFixed(2)}</td>
                  <td className="r num">{fmtRate(m[k].max_drawdown)}</td>
                  <td className="r num">{fmtRate(m[k].worst_month)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="footnote">
          Sharpe is return above T-bills per unit of volatility. Book minus 60/40: {ci.diff >= 0 ? "+" : "−"}{Math.abs(ci.diff).toFixed(2)} (95% CI {fmtNum(ci.ci95[0], 2)} to {fmtNum(ci.ci95[1], 2)}, block bootstrap), so the two are statistically indistinguishable.
        </p>
      </section>

      <section className="card" aria-labelledby="crash-h">
        <div className="card-head">
          <div>
            <h2 id="crash-h">Through the crashes</h2>
            <p className="card-sub">Total return from each market peak to its trough</p>
          </div>
        </div>
        <div className="table-scroll">
          <table className="table">
            <thead><tr><th scope="col">Episode</th><th scope="col" className="r">LevelEdge book</th><th scope="col" className="r">60/40</th><th scope="col" className="r">SPY</th></tr></thead>
            <tbody>
              {bt.crisis_windows.map((c) => (
                <tr key={c.label}>
                  <th scope="row" className="row-h">{c.label}<span className="move">{fmtIsoDay(c.start)} to {fmtIsoDay(c.end)}</span></th>
                  <td className="r num strong">{fmtSignedPct(c.book)}</td>
                  <td className="r num">{fmtSignedPct(c.sixty_forty)}</td>
                  <td className="r num">{fmtSignedPct(c.spy_buy_hold)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}

// ---------------------------------------------------------------------------
function TargetsCard({ t, loading }: { t: Targets | null; loading: boolean }) {
  return (
    <section className="card" aria-labelledby="tgt-h">
      <div className="card-head">
        <div>
          <h2 id="tgt-h">This week's targets</h2>
          <p className="card-sub">{t ? `From the close of ${fmtIsoDay(t.as_of_close)}` : "Share of the account per ETF"}</p>
        </div>
      </div>
      {loading && <div className="skeleton" style={{ height: 320 }} aria-label="Loading targets" />}
      {t && (
        <>
          <table className="levels targets">
            <thead><tr><th scope="col">ETF</th><th scope="col">12-mo trend</th><th scope="col" className="r">Weight</th></tr></thead>
            <tbody>
              {[...t.assets].sort((a, b) => b.weight - a.weight).map((a) => (
                <tr key={a.symbol}>
                  <td className="level">{a.symbol}<span className="move">{a.name}</span></td>
                  <td className="trend-cell">
                    <span className="trend-state">{a.trend_on ? "On" : "Off"}</span>
                    <span className="move num">{fmtSignedPct(a.return_12m, 0)}</span>
                  </td>
                  <td className="r">
                    <div className="pcell">
                      <span className="pbar" aria-hidden="true"><i style={{ width: `${Math.min(100, a.weight / 0.75 * 100)}%` }} /></span>
                      <span className="pval num">{fmtRate(a.weight, 1)}</span>
                    </div>
                  </td>
                </tr>
              ))}
              {t.cash_weight >= 0 ? (
                <tr>
                  <td className="level">Cash<span className="move">T-bill rate</span></td>
                  <td />
                  <td className="r">
                    <div className="pcell">
                      <span className="pbar" aria-hidden="true"><i style={{ width: `${Math.min(100, t.cash_weight / 0.75 * 100)}%` }} /></span>
                      <span className="pval num">{fmtRate(t.cash_weight, 1)}</span>
                    </div>
                  </td>
                </tr>
              ) : (
                <tr>
                  <td className="level">Borrowed<span className="move">On margin</span></td>
                  <td />
                  <td className="r"><span className="pval num">{fmtRate(-t.cash_weight, 1)}</span></td>
                </tr>
              )}
            </tbody>
          </table>
          <p className="help">SPY and IEF carry the fixed 60/40 core. The trend sleeve holds an ETF only while its 12-month return beats cash, sized to a 10% volatility target.{t.leverage > 1 ? ` The whole book is held at ${t.leverage}×, so about ${fmtRate(t.gross - 1, 0)} of the account is borrowed on margin.` : ""} Bars are drawn to 75% of the account.</p>
        </>
      )}
    </section>
  );
}

function TrustCard({ bt, live }: { bt: PortfolioBacktest; live: LivePaper | undefined }) {
  const full = bt.sharpe_vs_sixty_forty.full;
  const beats = Math.round((1 - bt.random_sleeve.p_value_book_sharpe) * bt.random_sleeve.sims);
  const m = bt.metrics;
  const h = bt.holdout;
  const startYear = bt.period[0].slice(0, 4);
  const worseThanSpy = bt.crisis_windows.filter((c) => c.book < c.spy_buy_hold);
  const levered = bt.labels.book.includes("1.5x");
  return (
    <section className="card" aria-labelledby="trust-p-h">
      <div className="card-head"><h2 id="trust-p-h">How far to trust this</h2></div>
      <div className="trust-head" style={{ marginBottom: 16 }}>
        <ShieldCheck size={18} aria-hidden="true" style={{ color: "var(--accent)" }} />
        <p><strong>{levered
            ? (m.book.cagr > m.sixty_forty.cagr ? "More return than 60/40 over the whole backtest, a smaller fall than SPY." : "No more return than 60/40 over the whole backtest.")
            : "Built for smaller crashes, not more return."}</strong>
          Since {startYear}: {fmtRate(m.book.cagr)} a year against {fmtRate(m.sixty_forty.cagr)} for 60/40 and {fmtRate(m.spy_buy_hold.cagr)} for SPY.
          Since 2020: {fmtRate(h.book.cagr)} against {fmtRate(h.sixty_forty.cagr)} for 60/40{h.book.max_drawdown < h.sixty_forty.max_drawdown ? ", with deeper falls" : ", with shallower falls"}.{" "}
          {levered ? "Borrowing costs mean it does not beat 60/40 on return per unit of risk." : "It has matched 60/40's return per unit of risk while falling less; it has not beaten it."}</p>
      </div>
      <dl className="kv">
        <dt>Worst fall, book</dt><dd className="num">{fmtRate(m.book.max_drawdown, 0)}</dd>
        <dt>Worst fall, 60/40</dt><dd className="num">{fmtRate(m.sixty_forty.max_drawdown, 0)}</dd>
        <dt>Worst fall, SPY</dt><dd className="num">{fmtRate(m.spy_buy_hold.max_drawdown, 0)}</dd>
        <dt>Worst fall since 2020, book vs 60/40</dt><dd className="num">{fmtRate(h.book.max_drawdown, 0)} vs {fmtRate(h.sixty_forty.max_drawdown, 0)}</dd>
        <dt>Worst month, book</dt><dd className="num">{fmtRate(m.book.worst_month, 1)}</dd>
        <dt>Sharpe, book minus 60/40</dt><dd className="num">{full.diff >= 0 ? "+" : "−"}{Math.abs(full.diff).toFixed(2)}</dd>
        <dt>95% interval</dt><dd className="num">{fmtNum(full.ci95[0], 2)} to {fmtNum(full.ci95[1], 2)}</dd>
        <dt>Beats random trend sleeves</dt><dd className="num">{beats} of {bt.random_sleeve.sims}</dd>
        <dt>Rules chosen</dt><dd>Before any search</dd>
        <dt>Leverage</dt><dd className="num">{levered ? "1.5×" : "None"}</dd>
      </dl>
      {worseThanSpy.length > 0 && (
        <p className="trust-note"><Info size={14} aria-hidden="true" />
          Leverage cuts both ways: in the {worseThanSpy.map((c) => c.label.toLowerCase()).join(" and ")} the book fell further than SPY,
          because stocks and the bonds it holds on margin fell together.</p>
      )}
      {live?.status !== "ok" && (
        <p className="trust-note"><Info size={14} aria-hidden="true" />Every number here is a backtest. The live record starts once a paper account is connected and needs months before it says much.</p>
      )}
      <p className="help">Backtests use total-return ETF prices from Yahoo Finance, fills at the next open{levered ? ", and margin interest at the T-bill rate plus 2%" : ""}. They can't include taxes.{levered ? " Leverage magnifies losses; a margin call is possible after a very large fall." : " The book runs without margin because brokers only allow it from $2,000 of equity."}</p>
    </section>
  );
}
