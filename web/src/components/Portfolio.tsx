import { useEffect, useMemo, useState } from "react";
import { AlertCircle } from "lucide-react";
import { getPortfolio, type LivePaper, type PortfolioBacktest, type PortfolioResponse, type SeriesKey, type Targets } from "../api";
import { fmtIsoDay, fmtMoney, fmtNum, fmtRate, fmtSignedPct } from "../format";
import { XYChart, type Series } from "./XYChart";
import { OutlookCard } from "./Outlook";
import { Term } from "./Term";

const NAMES: Record<SeriesKey, string> = {
  book: "This portfolio",
  sixty_forty: "60/40 mix",
  spy_buy_hold: "S&P 500",
  trend_sleeve_alone: "Trend rule alone",
};
// The portfolio is the accent; the S&P 500 is plain ink, the market as fact; 60/40 is grey and dashed.
const STYLE: Record<"book" | "sixty_forty" | "spy_buy_hold", Pick<Series, "color" | "dashed">> = {
  book: { color: "var(--accent)" },
  sixty_forty: { color: "var(--neutral-series)", dashed: true },
  spy_buy_hold: { color: "var(--price-line)" },
};
const CHART_KEYS = ["book", "spy_buy_hold", "sixty_forty"] as const;

/** Plain names for the 11 ETFs, and the asset class each belongs to. */
const ETF: Record<string, { plain: string; short: string; cls: Cls }> = {
  SPY: { plain: "S&P 500: large US companies", short: "Large US companies", cls: "us" },
  QQQ: { plain: "Nasdaq-100: large tech-heavy companies", short: "Big tech-heavy US companies", cls: "us" },
  IWM: { plain: "Small US companies", short: "Small US companies", cls: "us" },
  EFA: { plain: "Developed markets outside the US", short: "Other developed markets", cls: "intl" },
  EEM: { plain: "Emerging markets", short: "Emerging markets", cls: "intl" },
  IEF: { plain: "Mid-term US government bonds", short: "US government bonds", cls: "bonds" },
  TLT: { plain: "Long-term US government bonds", short: "Long-term US bonds", cls: "bonds" },
  LQD: { plain: "High-grade corporate bonds", short: "Corporate bonds", cls: "bonds" },
  VNQ: { plain: "US real estate", short: "US real estate", cls: "real" },
  DBC: { plain: "Commodities: oil, metals, crops", short: "Commodities", cls: "comm" },
  GLD: { plain: "Gold", short: "Gold", cls: "comm" },
};
type Cls = "us" | "intl" | "bonds" | "real" | "comm" | "cash";
const CLASSES: { k: Cls; label: string; what: string }[] = [
  { k: "us", label: "US stocks", what: "SPY, QQQ, IWM" },
  { k: "intl", label: "International stocks", what: "EFA, EEM" },
  { k: "bonds", label: "Bonds", what: "IEF, TLT, LQD" },
  { k: "real", label: "Real estate", what: "VNQ" },
  { k: "comm", label: "Commodities and gold", what: "DBC, GLD" },
  { k: "cash", label: "Cash", what: "Earns the T-bill rate" },
];

const pct0 = (v: number) => `${Math.abs(v * 100).toFixed(0)}%`;

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
        <p><strong>Couldn't load the portfolio.</strong> {err} Reload the page to try again.</p>
      </div>
    );
  }

  const t = data?.targets ?? null;
  const live = data?.live;
  const bt = data?.backtest ?? null;

  return (
    <>
      <section className="page-intro" aria-live="polite">
        <h1>Practice portfolio{live?.status === "ok" && <span className="spot">{fmtMoney(live.account.equity, 0)}</span>}</h1>
        <p>
          LevelEdge runs a $100,000 portfolio with <Term k="paper">simulated money at real prices</Term>. Every week it follows the
          same fixed rules: keep 70% in a classic <Term k="sixty_forty">stock-and-bond mix</Term>, and put the other 30% into whichever
          of 11 markets are in an <Term k="trend">uptrend</Term>. The aim is to fall less than the market in a crash, not to beat it in a boom.
        </p>
      </section>

      {data && data.targets_status !== "ok" && (
        <div className="banner" data-tone="error" role="alert">
          <AlertCircle size={18} aria-hidden="true" />
          <p><strong>This week's targets aren't available.</strong> {data.targets_status}</p>
        </div>
      )}

      {data ? <KeyPoints t={t} live={live} bt={bt} /> : <section className="card keypoints"><div className="skeleton" style={{ height: 96 }} aria-label="Loading key points" /></section>}

      <div className="grid-main">
        <div className="side main-col">
          {t ? <Allocation t={t} /> : <section className="card"><div className="skeleton" style={{ height: 160 }} aria-label="Loading holdings" /></section>}
          <LiveAccount live={live} />
        </div>
        <div className="side"><HoldingsCard t={t} loading={!data} /></div>
      </div>

      <h2 className="chapter">How the rules held up, 2007 to today</h2>
      <div className="grid-main">
        <div className="side main-col">
          {bt ? <BacktestCards bt={bt} /> : data ? (
            <section className="card"><p className="help">The backtest hasn't been generated. Run <code>python benchmarks/paper_portfolio.py</code>.</p></section>
          ) : (
            <section className="card"><div className="skeleton skeleton-chart" aria-label="Loading backtest" /></section>
          )}
        </div>
        <div className="side">{bt && <TrustCard bt={bt} live={live} />}</div>
      </div>

      <h2 className="chapter">Looking ahead</h2>
      <div className="side">
        <OutlookCard live={live} />
        {t && <TriggersCard t={t} live={live} />}
        {bt && <PerformanceCard bt={bt} />}
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
function KeyPoints({ t, live, bt }: { t: Targets | null; live: LivePaper | undefined; bt: PortfolioBacktest | null }) {
  const points: { lead: string; text: string }[] = [];
  if (live?.status === "ok" && live.curve.length >= 2) {
    const first = live.curve[0], last = live.curve[live.curve.length - 1];
    const ret = last.book - 1, sf = last.sixty_forty != null ? last.sixty_forty - 1 : null;
    points.push({ lead: `${fmtSignedPct(ret, 1)} since ${fmtIsoDay(first.date)}`,
      text: `Live, with real orders and simulated money.${sf != null ? ` A plain 60/40 mix did ${fmtSignedPct(sf, 1)} over the same days.` : ""}` });
  }
  if (bt) {
    const gfc = bt.crisis_windows.find((c) => c.label.startsWith("2008")) ?? bt.crisis_windows[0];
    if (gfc) points.push({ lead: `−${pct0(gfc.book)} in the 2008 crash`,
      text: `The S&P 500 fell ${pct0(gfc.spy_buy_hold)} in the same crash, and a 60/40 mix fell ${pct0(gfc.sixty_forty)}.` });
    const m = bt.metrics;
    points.push({ lead: `${fmtRate(m.book.cagr)} a year since ${bt.period[0].slice(0, 4)}`,
      text: `Less than the S&P 500's ${fmtRate(m.spy_buy_hold.cagr)}. That's the price of falling much less in crashes.` });
  }
  if (t) {
    const moving = t.assets.filter((a) => Math.abs(a.flip_move) < 0.03).map((a) => a.symbol);
    points.push({ lead: `Next check: ${fmtIsoDay(t.next_rebalance)}`,
      text: `After the market closes.${moving.length ? ` Closest to changing: ${moving.join(", ")}, within 3% of a trend flip.` : " No market is close to a trend flip."}` });
  }
  if (!points.length) return null;
  return (
    <section className="card keypoints" aria-labelledby="kp-h">
      <h2 id="kp-h" className="visually-hidden">Key points</h2>
      <ul>{points.map((p) => <li key={p.lead}><strong>{p.lead}</strong>{p.text}</li>)}</ul>
    </section>
  );
}

// ---------------------------------------------------------------------------
function Allocation({ t }: { t: Targets }) {
  const by: Record<Cls, number> = { us: 0, intl: 0, bonds: 0, real: 0, comm: 0, cash: Math.max(0, t.cash_weight) };
  t.assets.forEach((a) => { by[ETF[a.symbol]?.cls ?? "us"] += a.weight; });
  const stocks = by.us + by.intl;
  const rest = by.real + by.comm;
  return (
    <section className="card" aria-labelledby="alloc-h">
      <h2 id="alloc-h" className="takeaway">
        This week: {pct0(stocks)} in stocks, {pct0(by.bonds)} in bonds, {pct0(by.cash)} in cash{rest > 0.005 ? `, and ${pct0(rest)} in real estate and commodities` : ""}
      </h2>
      <p className="card-sub">Target share of the account for each kind of investment, set from the close of {fmtIsoDay(t.as_of_close)}</p>
      <div className="alloc-bar" role="img" aria-label={CLASSES.map((c) => `${c.label} ${fmtRate(by[c.k], 0)}`).join(", ")}>
        {CLASSES.filter((c) => by[c.k] > 0.001).map((c) => (
          <span key={c.k} style={{ flexGrow: by[c.k], background: `var(--c-${c.k})` }} />
        ))}
      </div>
      <dl className="alloc-legend">
        {CLASSES.map((c) => (
          <div key={c.k}>
            <i style={{ background: `var(--c-${c.k})` }} aria-hidden="true" />
            <dt>{c.label}</dt>
            <dd>{fmtRate(by[c.k], 0)}</dd>
            <span className="what">{c.what}</span>
          </div>
        ))}
      </dl>
      <p className="help">
        Stocks and bonds are always held: 42% S&P 500 and 28% government bonds. The rest moves with the trend rule. Markets
        that have fallen over the past year get nothing, and that money waits in cash. This is why the mix shifts toward safety in long downturns.
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
function LiveAccount({ live }: { live: LivePaper | undefined }) {
  const ok = live?.status === "ok" ? live : null;
  const curve = ok?.curve ?? [];
  const last = curve[curve.length - 1];
  const head = ok && curve.length >= 2 && last
    ? (() => {
        const r = last.book - 1, sf = last.sixty_forty != null ? last.sixty_forty - 1 : null;
        const vs = sf == null ? "" : Math.abs(r - sf) < 0.005 ? `, about the same as a 60/40 mix` : r > sf ? `, ahead of a 60/40 mix (${fmtSignedPct(sf, 1)})` : `, behind a 60/40 mix (${fmtSignedPct(sf, 1)})`;
        return `Since live trading began, the account is ${r >= 0 ? "up" : "down"} ${Math.abs(r * 100).toFixed(1)}%${vs}`;
      })()
    : "Live account";
  return (
    <section className="card" aria-labelledby="live-acct-h">
      <h2 id="live-acct-h" className="takeaway">{head}</h2>
      <p className="card-sub">The real record: orders placed with a broker each week at market prices, using simulated money</p>
      {!live && <div className="skeleton" style={{ height: 160, marginTop: 16 }} aria-label="Loading the live account" />}
      {live?.status === "error" && <p className="err" role="alert">{live.message}</p>}
      {live?.status === "not_configured" && (
        <div className="empty">
          <h3>The live account isn't connected here</h3>
          <p>The rest of this page shows the backtest. Connect Alpaca paper keys (<code>ALPACA_API_KEY_ID</code>, <code>ALPACA_API_SECRET_KEY</code>) to show the live record.</p>
        </div>
      )}
      {ok && <LiveBody live={ok} />}
    </section>
  );
}

function LiveBody({ live }: { live: Extract<LivePaper, { status: "ok" }> }) {
  const day = live.account.equity - live.account.last_equity;
  const curve = live.curve;
  const first = curve[0], last = curve[curve.length - 1];
  return (
    <>
      <dl className="kv kv-inline" style={{ marginTop: 16 }}>
        <dt>Account value</dt><dd className="num">{fmtMoney(live.account.equity)}</dd>
        <dt>Today</dt><dd className="num">{fmtMoney(day)} ({fmtSignedPct(day / live.account.last_equity, 2)})</dd>
        <dt>Uninvested cash</dt><dd className="num">{fmtMoney(live.account.cash)}</dd>
      </dl>
      {curve.length >= 2 ? (
        <XYChart
          series={CHART_KEYS.map((k) => ({
            name: NAMES[k], ...STYLE[k],
            points: curve.flatMap((p, i) => (p[k] == null ? [] : [{ x: i, y: (p[k] as number) - 1 }])),
          }))}
          xLabel={(v) => fmtIsoDay(curve[Math.round(v)]?.date ?? first.date).replace(/^\w+, /, "")}
          xTip={(v) => fmtIsoDay(curve[Math.round(v)]?.date ?? first.date)}
          xTicks={curve.flatMap((p, i) => (i === 0 || p.date.slice(0, 10) !== curve[i - 1].date.slice(0, 10) ? [i] : []))}
          yLabel={(v) => fmtSignedPct(v, 1)} hline={0} xDomain={[0, curve.length - 1]} height={220} endLabels
          summary={`Return since ${first.date}: this portfolio ${fmtSignedPct(last.book - 1, 2)}.`}
        />
      ) : (
        <p className="help">The chart starts once the first trades have filled and a second day is recorded.</p>
      )}
      <p className="source">Hover the chart to read any day. Source: Alpaca paper account, updated every few minutes.</p>
      {live.positions.length > 0 && (
        <details className="disclosure">
          <summary>What it holds now ({live.positions.length} investments)</summary>
          <div className="table-scroll">
            <table className="table" style={{ marginTop: 8 }}>
              <thead><tr><th scope="col">Investment</th><th scope="col" className="r">Share of account</th><th scope="col" className="r">Value</th><th scope="col" className="r">Gain or loss so far</th></tr></thead>
              <tbody>
                {[...live.positions].sort((a, b) => b.market_value - a.market_value).map((p) => (
                  <tr key={p.symbol}>
                    <th scope="row" className="row-h">{p.symbol}<span className="move">{ETF[p.symbol]?.plain ?? ""}</span></th>
                    <td className="r num">{fmtRate(p.weight, 1)}</td>
                    <td className="r num">{fmtMoney(p.market_value, 0)}</td>
                    <td className="r num">{fmtMoney(p.unrealized_pl, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
      {live.orders.length > 0 && (
        <details className="disclosure">
          <summary>Recent orders ({live.orders.length})</summary>
          <div className="table-scroll">
            <table className="table" style={{ marginTop: 8 }}>
              <thead><tr><th scope="col">Date</th><th scope="col">Order</th><th scope="col">Status</th><th scope="col" className="r">Price paid</th></tr></thead>
              <tbody>
                {live.orders.map((o, i) => (
                  <tr key={i}>
                    <td className="num">{o.submitted_at?.slice(0, 10)}</td>
                    <td>{o.side === "buy" ? "Buy" : "Sell"} {o.symbol} {o.notional != null ? fmtMoney(o.notional, 0) : `${o.qty} shares`}</td>
                    <td>{o.status.replace(/_/g, " ")}</td>
                    <td className="r num">{o.filled_avg_price != null ? fmtMoney(o.filled_avg_price) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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
  const dd = bt.drawdown_weekly;
  const eq = bt.equity_weekly;
  const ddDates = useMemo(() => dd.map((p) => p.date), [dd]);
  const eqDates = useMemo(() => eq.map((p) => p.date), [eq]);
  const m = bt.metrics;
  const startYear = bt.period[0].slice(0, 4);
  const end = eq[eq.length - 1];
  const fewer = bt.crisis_windows.every((c) => c.book > c.spy_buy_hold);
  const src = `Source: ${startYear}–${bt.period[1].slice(0, 4)} backtest with real ETF prices, trades at the next day's open and ${bt.cost_bps} bp trading costs.`;

  return (
    <>
      <section className="card" aria-labelledby="crash-h">
        <h2 id="crash-h" className="takeaway">
          {fewer ? `In all ${bt.crisis_windows.length} crashes since ${startYear}, the portfolio fell less than the S&P 500` : "How far each portfolio fell in the crashes"}
        </h2>
        <p className="card-sub">Total loss from the market's peak to its bottom in each crash. Shorter bars are better.</p>
        <div className="crashes">
          {bt.crisis_windows.map((c) => {
            const worst = Math.max(...bt.crisis_windows.flatMap((w) => [Math.abs(w.book), Math.abs(w.sixty_forty), Math.abs(w.spy_buy_hold)]));
            return (
              <div className="crash" key={c.label}>
                <h3>{c.label}</h3>
                <p className="when">{fmtIsoDay(c.start)} to {fmtIsoDay(c.end)}</p>
                {(["book", "sixty_forty", "spy_buy_hold"] as const).map((k) => {
                  const w = Math.abs(c[k]) / worst * 82;
                  return (
                    <div className="crash-bar" key={k}>
                      <span className="who">{NAMES[k]}</span>
                      <span className="track" aria-label={`${NAMES[k]} ${fmtSignedPct(c[k], 1)}`} role="img">
                        <span className="fill" style={{ width: `${w}%`, background: k === "book" ? "var(--accent)" : k === "spy_buy_hold" ? "var(--price-line)" : "var(--neutral-series)" }} />
                        <span className="val" style={{ left: `calc(${w}% + 8px)` }}>{fmtSignedPct(c[k], 0)}</span>
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
        <p className="source">{src}</p>
      </section>

      <section className="card" aria-labelledby="eq-h">
        <h2 id="eq-h" className="takeaway">
          $10,000 invested in {startYear} grew to {fmtMoney(end.book * 10000, 0)}. In the S&P 500 it grew to {fmtMoney(end.spy_buy_hold * 10000, 0)}, with much deeper falls along the way
        </h2>
        <p className="card-sub">The value of $10,000 invested at the start, week by week. Hover to read any date.</p>
        <XYChart
          series={CHART_KEYS.map((k) => ({ name: NAMES[k], ...STYLE[k], points: eq.map((p, i) => ({ x: i, y: p[k] * 10000 })) }))}
          xLabel={(v) => eqDates[Math.round(v)]?.slice(0, 4) ?? ""} xTip={(v) => fmtIsoDay(eqDates[Math.round(v)] ?? eqDates[0])}
          xTicks={yearTicks(eqDates)} yLabel={(v) => fmtMoney(v, 0)} xDomain={[0, eq.length - 1]} height={300} endLabels
          summary={`$10,000 from ${bt.period[0]} to ${bt.period[1]}: this portfolio ${fmtMoney(end.book * 10000, 0)}, 60/40 ${fmtMoney(end.sixty_forty * 10000, 0)}, S&P 500 ${fmtMoney(end.spy_buy_hold * 10000, 0)}.`}
        />
        <p className="source">{src}</p>
      </section>

      <section className="card" aria-labelledby="dd-h">
        <h2 id="dd-h" className="takeaway">
          Its worst fall from a high was {pct0(m.book.max_drawdown)}. The S&P 500's was {pct0(m.spy_buy_hold.max_drawdown)}
        </h2>
        <p className="card-sub">How far each portfolio was below its previous high, week by week. The line at 0% means a new high; dips show the falls.</p>
        <XYChart
          series={CHART_KEYS.map((k) => ({ name: NAMES[k], ...STYLE[k], points: dd.map((p, i) => ({ x: i, y: p[k] })) }))}
          xLabel={(v) => ddDates[Math.round(v)]?.slice(0, 4) ?? ""} xTip={(v) => fmtIsoDay(ddDates[Math.round(v)] ?? ddDates[0])}
          xTicks={yearTicks(ddDates)} yLabel={(v) => fmtRate(v, 0)} xDomain={[0, dd.length - 1]} height={280} endLabels
          summary={`Worst fall from a high: this portfolio ${fmtRate(m.book.max_drawdown)}, 60/40 ${fmtRate(m.sixty_forty.max_drawdown)}, S&P 500 ${fmtRate(m.spy_buy_hold.max_drawdown)}.`}
        />
        <p className="source">{src}</p>
      </section>
    </>
  );
}

// ---------------------------------------------------------------------------
function PerformanceCard({ bt }: { bt: PortfolioBacktest }) {
  const [period, setPeriod] = useState<"full" | "holdout">("full");
  const m = period === "full" ? bt.metrics : bt.holdout;
  const ci = period === "full" ? bt.sharpe_vs_sixty_forty.full : bt.sharpe_vs_sixty_forty.holdout;
  const startYear = bt.period[0].slice(0, 4);
  const tie = ci.ci95[0] < 0 && ci.ci95[1] > 0;
  return (
    <section className="card" aria-labelledby="perf-h">
      <div className="card-head">
        <div>
          <h2 id="perf-h" className="takeaway">
            {tie ? "It earned as much per unit of risk as a 60/40 mix, with smaller falls" : ci.diff > 0 ? "It earned more per unit of risk than a 60/40 mix" : "It earned less per unit of risk than a 60/40 mix"}
          </h2>
          <p className="card-sub">{period === "full" ? `${bt.period[0]} to ${bt.period[1]}` : "Since 2020: years kept out of the design, used once as a test"}</p>
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
              <th scope="col">Portfolio</th>
              <th scope="col" className="r">Return a year</th>
              <th scope="col" className="r"><Term k="volatility">Swings</Term></th>
              <th scope="col" className="r"><Term k="sharpe">Sharpe</Term></th>
              <th scope="col" className="r"><Term k="drawdown">Worst fall</Term></th>
              <th scope="col" className="r">Worst month</th>
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
        Sharpe difference against 60/40: {ci.diff >= 0 ? "+" : "−"}{Math.abs(ci.diff).toFixed(2)}, with a{" "}
        <Term k="interval95">95% interval</Term> of {fmtNum(ci.ci95[0], 2)} to {fmtNum(ci.ci95[1], 2)}.
        {tie ? " That range includes zero, so on this measure the two are a statistical tie." : ""}
      </p>
    </section>
  );
}

// ---------------------------------------------------------------------------
const BAND = 0.02;   // live.BAND: an investment trades only when it is more than 2 points off target

function TriggersCard({ t, live }: { t: Targets; live: LivePaper | undefined }) {
  const held = live?.status === "ok" ? Object.fromEntries(live.positions.map((p) => [p.symbol, p.weight])) : null;
  const rows = [...t.assets]
    .map((a) => {
      const now = held ? held[a.symbol] ?? 0 : null;
      const drift = now == null ? null : now - a.weight;
      return { ...a, now, drift, trades: drift != null && Math.abs(drift) > BAND };
    })
    .sort((x, y) => Math.abs(x.flip_move) - Math.abs(y.flip_move));
  const pending = rows.filter((r) => r.trades);
  const near = rows[0];
  return (
    <section className="card" aria-labelledby="trig-h">
      <h2 id="trig-h" className="takeaway">
        {held ? (pending.length ? `${pending.length} investment${pending.length > 1 ? "s" : ""} would trade at the next check` : "Nothing would trade if the check were today") : "What could make it trade"}
        {near ? `. Closest to a change: ${near.symbol}, ${Math.abs(near.flip_move * 100).toFixed(1)}% from a trend flip` : ""}
      </h2>
      <p className="card-sub">
        Every Friday after the close ({fmtIsoDay(t.next_rebalance)} next), the portfolio checks two things. Has any holding drifted more than
        2 points from its target? Has any market's 12-month <Term k="trend">trend</Term> flipped? Orders then go in at the next open.
      </p>
      <div className="table-scroll">
        <table className="table">
          <thead>
            <tr>
              <th scope="col">Investment</th>
              {held && <th scope="col" className="r">Holding now</th>}
              <th scope="col" className="r">Target</th>
              {held && <th scope="col" className="r">Off target by</th>}
              <th scope="col">In an uptrend?</th>
              <th scope="col" className="r">Trend flips if the price crosses</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.symbol} data-emph={r.trades ? "true" : undefined}>
                <th scope="row" className="row-h">{r.symbol}<span className="move">{ETF[r.symbol]?.plain}</span></th>
                {held && <td className="r num">{fmtRate(r.now ?? 0, 1)}</td>}
                <td className="r num">{fmtRate(r.weight, 1)}</td>
                {held && <td className="r num">{r.drift == null ? "—" : `${fmtSignedPct(r.drift, 1)}${r.trades ? " · trades" : ""}`}</td>}
                <td>{r.trend_on ? "Yes: sold if it falls below" : "No: bought if it rises above"}</td>
                <td className="r num">${r.flip_price.toFixed(2)} <span className="move">{fmtSignedPct(r.flip_move, 1)} from ${r.last_close.toFixed(2)}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="help">
        A trend flip moves only the trend-following 30%; SPY and IEF also carry the fixed stock-and-bond mix. The flip price is
        the price a year ago plus what cash earned since, so it moves slowly. Checking daily instead of weekly doubled trading costs with no gain, so the portfolio acts once a week.
      </p>
    </section>
  );
}

function HoldingsCard({ t, loading }: { t: Targets | null; loading: boolean }) {
  return (
    <section className="card" aria-labelledby="tgt-h">
      <h2 id="tgt-h">The 11 markets it can own</h2>
      <p className="card-sub">{t ? `Target for each, from the close of ${fmtIsoDay(t.as_of_close)}` : "Share of the account in each"}</p>
      {loading && <div className="skeleton" style={{ height: 320, marginTop: 16 }} aria-label="Loading targets" />}
      {t && (
        <>
          <table className="levels targets" style={{ marginTop: 16 }}>
            <thead><tr><th scope="col">Market</th><th scope="col">Past year</th><th scope="col" className="r">Target</th></tr></thead>
            <tbody>
              {[...t.assets].sort((a, b) => b.weight - a.weight).map((a) => (
                <tr key={a.symbol}>
                  <td className="level">{a.symbol}<span className="move">{ETF[a.symbol]?.short ?? a.name}</span></td>
                  <td className="trend-cell">
                    <span className="trend-state">{a.trend_on ? "Uptrend" : "Downtrend"}</span>
                    <span className="move num">{fmtSignedPct(a.return_12m, 0)}</span>
                  </td>
                  <td className="r">
                    <div className="pcell">
                      <span className="pbar" aria-hidden="true"><i style={{ width: `${Math.min(100, a.weight / 0.5 * 100)}%` }} /></span>
                      <span className="pval num">{fmtRate(a.weight, 1)}</span>
                    </div>
                  </td>
                </tr>
              ))}
              <tr>
                <td className="level">Cash<span className="move">Earns the <Term k="tbill">T-bill</Term> rate</span></td>
                <td />
                <td className="r">
                  <div className="pcell">
                    <span className="pbar" aria-hidden="true"><i style={{ width: `${Math.min(100, Math.max(0, t.cash_weight) / 0.5 * 100)}%` }} /></span>
                    <span className="pval num">{fmtRate(Math.max(0, t.cash_weight), 1)}</span>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
          <p className="help">A market in a downtrend gets nothing from the trend-following 30%. Each one in an uptrend gets a share sized so the whole portfolio swings about 10% a year. Bars are drawn to 50% of the account.</p>
        </>
      )}
    </section>
  );
}

function TrustCard({ bt, live }: { bt: PortfolioBacktest; live: LivePaper | undefined }) {
  const beats = Math.round((1 - bt.random_sleeve.p_value_book_sharpe) * bt.random_sleeve.sims);
  const startYear = bt.period[0].slice(0, 4);
  const liveSince = live?.status === "ok" && live.curve.length ? fmtIsoDay(live.curve[0].date) : null;
  return (
    <section className="card" aria-labelledby="trust-p-h">
      <h2 id="trust-p-h">How much to trust these numbers</h2>
      <ul className="trust-list">
        <li><strong>Mostly a backtest.</strong> The <Term k="backtest">backtest</Term> replays the rules on {startYear}–{bt.period[1].slice(0, 4)} prices, including trading costs.
          {liveSince ? ` Live trading began ${liveSince}, too recently to judge.` : " Live trading hasn't started on this deployment."}</li>
        <li><strong>Rules written first.</strong> The rules were fixed before testing. A search through 80 variations for better rules failed its <Term k="holdout">held-out test</Term>, so the original rules stayed.</li>
        <li><strong>The signal is real.</strong> The trend rule beat {beats} of {bt.random_sleeve.sims} look-alike portfolios that picked at random.</li>
        <li><strong>A real account can copy it.</strong> No <Term k="margin">margin</Term>, whole shares only, and orders at the market open, so any brokerage account can make the same trades.</li>
        <li><strong>Not included.</strong> Taxes, and anything markets do that hasn't happened since {startYear}.</li>
      </ul>
    </section>
  );
}
