import { useEffect, useMemo, useState } from "react";
import { AlertCircle, ArrowDown, ArrowUp } from "lucide-react";
import { getScan, type Scan, type ScanRow } from "../api";
import { fmtIsoDay, fmtSignedPct } from "../format";
import { Term } from "./Term";

type Ok = Extract<Scan, { status: "ok" }>;
type SortKey = "ticker" | "ret_1d" | "swing" | "p_up" | "vol_pct" | "vs_sma200" | "from_high";

const money = (v: number) => `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
const pct0 = (v: number) => `${Math.round(v * 100)}%`;

/** Link into the forecaster for this ticker at the scan's one-week target. */
function forecastHref(r: ScanRow) {
  return `${window.location.pathname}?ticker=${encodeURIComponent(r.ticker)}&interval=1d&target=${r.target.slice(0, 16)}#forecast`;
}

export function Scanner() {
  const [d, setD] = useState<Scan | null>(null);
  useEffect(() => {
    let on = true;
    getScan().then((x) => on && setD(x)).catch((e) => on && setD({ status: "error", message: e.message }));
    return () => { on = false; };
  }, []);

  return (
    <>
      <section className="page-intro">
        <h1>Daily market scan</h1>
        <p>
          Every weekday after the market closes, LevelEdge runs its forecasting model on 100 major US stocks and funds, in parallel in
          the cloud. Each one gets a one-week forecast and a plain-English note on anything unusual, so nobody has to check
          100 charts by hand.
        </p>
      </section>
      {!d && <section className="card"><div className="skeleton" style={{ height: 220 }} aria-label="Loading the market scan" /></section>}
      {d && d.status !== "ok" && (
        <div className="banner" data-tone={d.status === "error" ? "error" : undefined} role={d.status === "error" ? "alert" : "status"}>
          <AlertCircle size={18} aria-hidden="true" />
          <p><strong>{d.status === "error" ? "Couldn't load the scan." : "No scan yet."}</strong> {d.message}</p>
        </div>
      )}
      {d?.status === "ok" && <ScanBody d={d} />}
    </>
  );
}

function ScanBody({ d }: { d: Ok }) {
  const s = d.summary;
  const big = d.rows.find((r) => r.ticker === s.biggest_swing);
  const running = s.analyzed < d.total;
  return (
    <>
      <p className="scan-status">
        Scan of the {fmtIsoDay(d.scan_date)} close · {s.analyzed} of {d.total} analyzed{running ? " (still running)" : ""} · runs automatically
        on weekdays at 5:45 PM ET
      </p>
      <section className="card keypoints" aria-labelledby="scan-kp">
        <h2 id="scan-kp" className="visually-hidden">Key points</h2>
        <ul>
          <li><strong>{s.in_uptrend} of {s.analyzed}</strong>are above their 200-day average, the usual sign of a long-term <Term k="trend">uptrend</Term>.</li>
          <li><strong>{s.volatile} swinging hard</strong>Their recent moves are bigger than 85% of the past five years.</li>
          {s.median_swing != null && <li><strong>±{(s.median_swing * 100).toFixed(1)}% this week</strong>The typical expected move for these names over the next {d.horizon_days} trading days.</li>}
          {big && <li><strong>{big.ticker}: ±{(big.swing * 100).toFixed(1)}%</strong>The biggest expected move this week ({big.name}).</li>}
        </ul>
      </section>

      <div className="grid-main">
        <section className="card main-col" aria-labelledby="notable-h">
          <h2 id="notable-h" className="takeaway">What stands out today</h2>
          <p className="card-sub">The most unusual reading for each name, ranked. Select a ticker to open its full forecast.</p>
          <ol className="notable">
            {d.notable.map((n) => {
              const r = d.rows.find((x) => x.ticker === n.ticker)!;
              return (
                <li key={n.ticker}>
                  <a href={forecastHref(r)}>{n.ticker}</a>
                  <span className="who">{n.name}</span>
                  <p>{n.text}</p>
                </li>
              );
            })}
          </ol>
        </section>
        <section className="card" aria-labelledby="how-h">
          <h2 id="how-h">How the scan works</h2>
          <ol className="how-list">
            <li><strong>5:45 PM ET:</strong> a scheduler starts the scan and splits the 100 tickers into batches of 5.</li>
            <li><strong>In parallel:</strong> up to 5 cloud workers each take a batch from a queue. A batch that fails is retried, up to 3 times.</li>
            <li><strong>Per ticker:</strong> the forecasting model is fitted on that ticker's full daily history, tested on years it never saw, and asked where the price will be in a week.</li>
            <li><strong>Results:</strong> each forecast is saved once (a retry never overwrites one) and turned into the notes on this page.</li>
          </ol>
          <p className="help">A whole scan takes a few minutes and costs nothing within the cloud's free tier. The model forecasts the <em>size</em> of moves well; it rarely has a strong view on direction.</p>
        </section>
      </div>

      <ScanTable rows={d.rows} />
    </>
  );
}

function ScanTable({ rows }: { rows: ScanRow[] }) {
  const [q, setQ] = useState("");
  const [group, setGroup] = useState("All");
  const [sort, setSort] = useState<{ k: SortKey; dir: 1 | -1 }>({ k: "swing", dir: -1 });
  const groups = useMemo(() => ["All", ...Array.from(new Set(rows.map((r) => r.group)))], [rows]);
  const shown = useMemo(() => {
    const needle = q.trim().toUpperCase();
    const val = (r: ScanRow) => (sort.k === "ticker" ? r.ticker : (r[sort.k] as number | null) ?? -Infinity);
    return rows
      .filter((r) => (group === "All" || r.group === group) && (!needle || r.ticker.includes(needle) || r.name.toUpperCase().includes(needle)))
      .sort((a, b) => {
        const x = val(a), y = val(b);
        return (x < y ? -1 : x > y ? 1 : 0) * sort.dir;
      });
  }, [rows, q, group, sort]);

  const Head = ({ k, children, right = true }: { k: SortKey; children: React.ReactNode; right?: boolean }) => (
    <th scope="col" className={right ? "r" : undefined} aria-sort={sort.k === k ? (sort.dir === 1 ? "ascending" : "descending") : "none"}>
      <button type="button" className="sort" onClick={() => setSort((s) => ({ k, dir: s.k === k ? (s.dir === 1 ? -1 : 1) : k === "ticker" ? 1 : -1 }))}>
        {children}{sort.k === k && (sort.dir === 1 ? <ArrowUp size={12} aria-hidden="true" /> : <ArrowDown size={12} aria-hidden="true" />)}
      </button>
    </th>
  );

  return (
    <section className="card" aria-labelledby="all-h" style={{ marginTop: 24 }}>
      <div className="card-head">
        <div>
          <h2 id="all-h">All {rows.length} names</h2>
          <p className="card-sub">Select a column to sort. The expected range is where the price should be in a week, 8 times out of 10.</p>
        </div>
        <div className="scan-filters">
          <input className="input" type="search" placeholder="Search a ticker or name" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search tickers" />
          <select className="select" value={group} onChange={(e) => setGroup(e.target.value)} aria-label="Group">
            {groups.map((g) => <option key={g}>{g}</option>)}
          </select>
        </div>
      </div>
      <div className="table-scroll">
        <table className="table scan-table">
          <thead>
            <tr>
              <Head k="ticker" right={false}>Name</Head>
              <th scope="col" className="r">Price</th>
              <Head k="ret_1d">Last day</Head>
              <th scope="col" className="r"><Term k="range80">Expected range</Term>, 1 week</th>
              <Head k="swing">Expected move</Head>
              <Head k="p_up">Chance higher</Head>
              <Head k="vol_pct">Swings vs normal</Head>
              <Head k="vs_sma200">Vs 200-day average</Head>
              <Head k="from_high">From 1-year high</Head>
            </tr>
          </thead>
          <tbody>
            {shown.map((r) => (
              <tr key={r.ticker}>
                <th scope="row" className="row-h"><a href={forecastHref(r)}>{r.ticker}</a><span className="move">{r.name}</span></th>
                <td className="r num">{money(r.price)}</td>
                <td className="r num">{r.ret_1d == null ? "—" : fmtSignedPct(r.ret_1d, 1)}</td>
                <td className="r num">{money(r.range80[0])} – {money(r.range80[1])}</td>
                <td className="r num">±{(r.swing * 100).toFixed(1)}%</td>
                <td className="r num">{pct0(r.p_up)}</td>
                <td className="r">
                  {r.vol_pct == null ? "—" : (
                    <span className="pcell"><span className="pbar" aria-hidden="true"><i style={{ width: `${r.vol_pct * 100}%` }} /></span>
                      <span className="pval num">{r.vol_pct >= 0.85 ? "High" : r.vol_pct <= 0.15 ? "Low" : "Normal"}</span></span>
                  )}
                </td>
                <td className="r num">{r.vs_sma200 == null ? "—" : `${Math.abs(r.vs_sma200 * 100).toFixed(1)}% ${r.vs_sma200 >= 0 ? "above" : "below"}`}</td>
                <td className="r num">{r.from_high == null ? "—" : fmtSignedPct(r.from_high, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {shown.length === 0 && <p className="help">No names match. Clear the search or choose another group.</p>}
      <p className="source">Source: LevelEdge's forecasting model on each name's full daily price history from Yahoo Finance, refitted every weekday.
        &ldquo;Swings vs normal&rdquo; compares the past month's moves with the past five years.</p>
    </section>
  );
}
