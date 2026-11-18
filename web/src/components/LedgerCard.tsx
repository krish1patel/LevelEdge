import { useEffect, useState } from "react";
import { ShieldAlert, ShieldCheck } from "lucide-react";
import { getLedger, type Ledger } from "../api";
import { fmtNum } from "../format";
import { XYChart } from "./XYChart";

const INTERVAL_ORDER = ["2m", "5m", "15m", "30m", "60m", "1h"];

function when(iso: string) {
  return new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZone: "America/New_York" }) + " ET";
}

/** The public, hash-chained forecast ledger, verified from scratch on every load. */
export function LedgerCard() {
  const [l, setL] = useState<Ledger | null>(null);
  useEffect(() => {
    let on = true;
    getLedger().then((d) => on && setL(d)).catch((e) => on && setL({ status: "error", message: e.message }));
    return () => { on = false; };
  }, []);

  return (
    <section className="card" aria-labelledby="ledger-h" aria-busy={!l}>
      <div className="card-head">
        <div>
          <h2 id="ledger-h">Verifiable record</h2>
          <p className="card-sub">Every live forecast is chained, timestamped by an outside authority before its outcome, and scored without editing</p>
        </div>
        {l?.status === "ok" && (
          <span className="badge" data-tone={l.ok ? "good" : "bad"}>
            {l.ok ? <ShieldCheck size={14} aria-hidden="true" /> : <ShieldAlert size={14} aria-hidden="true" />}
            {l.ok ? "Chain verified" : "Verification failed"}
          </span>
        )}
      </div>
      {!l && <div className="skeleton" style={{ height: 120 }} aria-label="Checking the ledger" />}
      {l && l.status !== "ok" && (
        <div className="empty">
          <h3>{l.status === "error" ? "Couldn't check the ledger" : "The ledger starts with the next forward test"}</h3>
          <p>{l.status === "error" ? l.message : "Forecasts made at 10:00, 12:00 and 14:00 ET on trading days are appended to a public branch and timestamped before the 4:00 PM close."}</p>
        </div>
      )}
      {l?.status === "ok" && <LedgerBody l={l} />}
    </section>
  );
}

function LedgerBody({ l }: { l: Extract<Ledger, { status: "ok" }> }) {
  const rows = Object.entries(l.by_interval).sort(([a], [b]) => INTERVAL_ORDER.indexOf(a) - INTERVAL_ORDER.indexOf(b));
  return (
    <>
      {!l.ok && <p className="err" role="alert">{l.problems[0]}</p>}
      <dl className="facts">
        <div><dt>Forecasts committed</dt><dd className="num">{l.forecasts.toLocaleString()}</dd></div>
        <div><dt>Scored on time</dt><dd className="num">{l.scored_on_time.toLocaleString()}{l.scored > l.scored_on_time ? ` of ${l.scored.toLocaleString()}` : ""}</dd></div>
        <div><dt>Awaiting outcome</dt><dd className="num">{l.pending.toLocaleString()}</dd></div>
        <div><dt>Brier (lower is better)</dt><dd className="num">{l.brier_on_time == null ? "—" : fmtNum(l.brier_on_time)}</dd></div>
      </dl>
      {rows.length > 0 && (
        <div className="grid-2" style={{ marginTop: 24 }}>
          <table className="table">
            <thead><tr><th scope="col">Bar interval</th><th scope="col" className="r">Scored</th><th scope="col" className="r">Brier</th></tr></thead>
            <tbody>
              {rows.map(([k, v]) => (
                <tr key={k}><td>{k}</td><td className="r num">{v.n.toLocaleString()}</td><td className="r num">{fmtNum(v.brier)}</td></tr>
              ))}
            </tbody>
          </table>
          {l.reliability.length >= 3 ? (
            <XYChart
              series={[{ name: "Ledger", color: "var(--accent)", kind: "dots" as const, points: l.reliability.map((r) => ({ x: r.forecast, y: r.observed })) }]}
              xLabel={(v) => `${Math.round(v * 100)}%`} yLabel={(v) => `${Math.round(v * 100)}%`} xDomain={[0, 1]} yDomain={[0, 1]} diagonal
              xTitle="Predicted chance" yTitle="Observed frequency"
              summary={`Calibration of ${l.levels_scored.toLocaleString()} scored levels from the verified ledger.`}
            />
          ) : <p className="help">The calibration chart appears once enough levels are scored.</p>}
        </div>
      )}
      <p className="footnote">
        {l.records.toLocaleString()} records since {l.first ?? "the first forecast"}, head <code>{l.head.slice(0, 12)}</code>
        {l.last_anchor ? `, last timestamped ${when(l.last_anchor)} by freetsa.org` : ", not yet timestamped"}.
        {" "}A forecast counts only if a timestamp covering it predates its target. Check it yourself: <code>{l.verify_cmd}</code>{" "}
        (<a href={l.view_url} target="_blank" rel="noreferrer">history on GitHub</a>). This proves the record was never edited, not that the forecasts are good.
      </p>
    </>
  );
}
