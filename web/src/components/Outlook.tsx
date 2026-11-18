import { useEffect, useMemo, useRef, useState } from "react";
import { scaleLinear } from "d3-scale";
import { area, line, curveMonotoneX } from "d3-shape";
import { Activity } from "lucide-react";
import { getOutlook, type LivePaper, type Outlook, type Quantiles } from "../api";
import { fmtIsoDay, fmtMoney, fmtPct, fmtRate, fmtSignedPct } from "../format";
import { useWidth } from "./useWidth";
import { Term } from "./Term";

type Ok = Extract<Outlook, { status: "ok" }>;

const RISK_COPY: Record<Ok["risk"]["label"], string> = {
  calm: "Calm",
  normal: "Normal",
  elevated: "Elevated",
  stressed: "Stressed",
};

/** Where the S&P 500 could be over the next 3 months, with calibrated ranges, and what it means for the account. */
export function OutlookCard({ live }: { live: LivePaper | undefined }) {
  const [data, setData] = useState<Outlook | null>(null);
  useEffect(() => {
    let on = true;
    getOutlook().then((d) => on && setData(d)).catch((e) => on && setData({ status: "error", message: e.message }));
    return () => { on = false; };
  }, []);
  const equity = live?.status === "ok" ? live.account.equity : null;
  const m1 = data?.status === "ok" ? data.horizons.find((h) => h.h === 21) : undefined;

  return (
    <section className="card" aria-labelledby="outlook-h" aria-busy={!data}>
      <div className="card-head">
        <div>
          <h2 id="outlook-h" className="takeaway">
            {m1 ? `8 times out of 10, the S&P 500 ends the next month between ${fmtSignedPct(m1.ret.q100, 1)} and ${fmtSignedPct(m1.ret.q900, 1)} from today`
              : "Where the stock market could go"}
          </h2>
          <p className="card-sub">
            {data?.status === "ok"
              ? `From the ${fmtIsoDay(data.as_of_close)} close of ${data.spot.toLocaleString("en-US", { maximumFractionDigits: 0 })}. The darkest band holds half of the likely paths, the lighter bands 80% and 95%. Hover the chart to read any point.`
              : "Likely ranges for the S&P 500 over the next week, month and quarter."}
          </p>
        </div>
        {data?.status === "ok" && (
          <span className="badge" title={`Expected swings: about ${fmtRate(data.risk.vol_forecast_annual, 0)} a year`}>
            <Activity size={14} aria-hidden="true" />
            Market mood: {RISK_COPY[data.risk.label]} · <Term k="vix">VIX</Term> {data.risk.vix.toFixed(1)}
          </span>
        )}
      </div>
      {!data && <p className="help">Fitting three volatility models to 35 years of S&P 500 history. This takes a few seconds.</p>}
      {data?.status === "error" && <p className="help">{data.message}</p>}
      {data?.status === "ok" && <OutlookBody o={data} equity={equity} />}
    </section>
  );
}

function OutlookBody({ o, equity }: { o: Ok; equity: number | null }) {
  const rec = o.record?.["21"]?.holdout;
  const book = o.book?.horizons;
  return (
    <>
      <OutlookChart o={o} />
      <div className="table-scroll">
        <table className="table outlook-table">
          <thead>
            <tr>
              <th scope="col">Over the next</th>
              <th scope="col" className="r"><Term k="outlook_range">Likely range</Term> (8 in 10)</th>
              <th scope="col" className="r">Wide range (19 in 20)</th>
              <th scope="col" className="r">Chance it's higher</th>
              <th scope="col" className="r">Chance of a 5%+ drop</th>
              {book && <th scope="col" className="r">{equity ? "This account, likely range" : "Portfolio, likely range"}</th>}
            </tr>
          </thead>
          <tbody>
            {o.horizons.map((h) => {
              const b = book?.[h.label];
              return (
                <tr key={h.h}>
                  <th scope="row" className="row-h">{h.label}</th>
                  <td className="r num">{range(h.ret, "q100", "q900")}</td>
                  <td className="r num">{range(h.ret, "q025", "q975")}</td>
                  <td className="r num">{fmtPct(h.p_up)}</td>
                  <td className="r num">{fmtPct(h.p_down5)}</td>
                  {book && (
                    <td className="r num">
                      {b ? (equity ? `${fmtMoney(b.q100 * equity, 0)} to ${fmtMoney(b.q900 * equity, 0)}` : range(b, "q100", "q900")) : "—"}
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <ForwardLine o={o} />
      <p className="help">
        <strong>How to read this.</strong> This forecasts how big the market's moves may be, not which way they'll go. It averages three
        standard ways of measuring expected swings, including the <Term k="vix">VIX</Term>. Tested on every Friday since 2000 using only what was
        known that day, its ranges held about as often as promised{rec ? `: since 2020, the 1-month likely range held ${fmtRate(rec.cover80, 0)} of the time` : ""}.
        {" "}Predicting <em>when</em> a fall will come didn't work with any method tested, so this card isn't a buy or sell signal.
        {o.book ? ` The account column assumes the portfolio moves about ${o.book.beta.toFixed(2)}× as much as the S&P 500, as its current holdings did over the past year.` : ""}
      </p>
    </>
  );
}

/** The live record: forecasts committed to a public git branch before their outcomes, then scored. */
function ForwardLine({ o }: { o: Ok }) {
  const f = o.forward;
  if (!f) return null;
  if (f.status !== "ok") {
    return <p className="help">Live record: the daily forecast log starts after the next close; each forecast is committed publicly before its outcome is known.</p>;
  }
  const matured = Object.values(f.horizons).filter((h) => h.n > 0);
  return (
    <p className="help">
      <strong>Live record.</strong>{" "}
      {matured.length
        ? matured.map((h) => `${h.label}: the 80% range held ${Math.round((h.cover80 ?? 0) * h.n)} of ${h.n}`).join("; ") + "."
        : `${f.logged} forecast${f.logged === 1 ? "" : "s"} logged since ${f.first ? fmtIsoDay(f.first) : "the first close"}; none has resolved yet.`}
      {f.next_result ? ` Next result on ${fmtIsoDay(f.next_result)}.` : ""}{" "}
      Each day's forecast is committed to a <a href={f.view_url} target="_blank" rel="noreferrer">public log</a> before the outcome exists.
      {matured.some((h) => h.n < 30) ? " Under 30 results is too few to judge." : ""}
    </p>
  );
}

function range(q: Quantiles, lo: keyof Quantiles, hi: keyof Quantiles) {
  return `${fmtSignedPct(q[lo], 1)} to ${fmtSignedPct(q[hi], 1)}`;
}

function OutlookChart({ o }: { o: Ok }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const narrow = width < 560;
  const height = narrow ? 240 : 320;
  const m = { top: 12, right: narrow ? 64 : 88, bottom: 28, left: 52 };
  const iw = Math.max(10, width - m.left - m.right);
  const ih = height - m.top - m.bottom;
  const hist = narrow ? o.history.slice(-63) : o.history;
  const nH = hist.length;
  const fan = useMemo(() => {
    const s = o.spot;
    const start: { step: number } & Quantiles = { step: 0, q025: s, q050: s, q100: s, q250: s, q500: s, q750: s, q900: s, q950: s, q975: s };
    return [start, ...o.fan];
  }, [o]);
  const last = fan[fan.length - 1];
  const x = scaleLinear().domain([-(nH - 1), last.step]).range([0, iw]);
  const lo = Math.min(...hist.map((d) => d.close), last.q025);
  const hi = Math.max(...hist.map((d) => d.close), last.q975);
  const y = scaleLinear().domain([lo, hi]).range([ih, 0]).nice(5);
  const band = (a: keyof Quantiles, b: keyof Quantiles) =>
    area<(typeof fan)[number]>().x((d) => x(d.step)).y0((d) => y(d[a])).y1((d) => y(d[b])).curve(curveMonotoneX)(fan) ?? "";
  const mid = line<(typeof fan)[number]>().x((d) => x(d.step)).y((d) => y(d.q500)).curve(curveMonotoneX)(fan) ?? "";
  const price = line<(typeof hist)[number]>().x((_, i) => x(i - (nH - 1))).y((d) => y(d.close))(hist) ?? "";
  const marks = [5, 21, 63].filter((s) => s <= last.step);
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  function onMove(e: React.PointerEvent) {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r) return;
    const v = x.invert(e.clientX - r.left - m.left);
    if (v <= 0) setHover(Math.max(-(nH - 1), Math.round(v)));
    else setHover(fan.reduce((a, b) => (Math.abs(b.step - v) < Math.abs(a.step - v) ? b : a), fan[0]).step);
  }
  const ahead = (st: number) => st === 5 ? "In 1 week" : st === 21 ? "In 1 month" : st === 63 ? "In 3 months" : `In ${st} trading day${st === 1 ? "" : "s"}`;
  const hp = hover == null ? null : hover <= 0 ? { kind: "hist" as const, d: hist[hover + nH - 1] } : { kind: "fan" as const, f: fan.find((f) => f.step === hover)! };
  const tipLeft = hover == null ? 0 : m.left + x(hover);
  const q63 = o.horizons.find((h) => h.h === 63);
  const summary = q63
    ? `S&P 500 now ${o.spot.toFixed(0)}. In 3 months, 80% likely between ${last.q100.toFixed(0)} and ${last.q900.toFixed(0)}, 95% between ${last.q025.toFixed(0)} and ${last.q975.toFixed(0)}.`
    : "S&P 500 forecast fan.";

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <svg ref={svgRef} className="chart" width={width} height={height} role="img" aria-label={summary}
           onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        <g transform={`translate(${m.left},${m.top})`}>
          {y.ticks(5).map((t) => (
            <g key={t} transform={`translate(0,${y(t)})`}>
              <line className="gridline" x1={0} x2={iw} />
              <text x={-8} dy="0.32em" textAnchor="end">{t.toLocaleString("en-US")}</text>
            </g>
          ))}
          <path d={band("q025", "q975")} fill="var(--fan-outer)" opacity={0.6} />
          <path d={band("q100", "q900")} fill="var(--fan-outer)" />
          <path d={band("q250", "q750")} fill="var(--fan-inner)" />
          <path d={mid} fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeDasharray="4 3" />
          <path d={price} fill="none" stroke="var(--price-line)" strokeWidth={1.5} />
          <line className="axis-line" x1={x(0)} x2={x(0)} y1={0} y2={ih} />
          <text x={x(0) - 4} y={ih + 20} textAnchor="end">Today</text>
          {marks.map((s) => (
            <g key={s}>
              <line className="gridline" x1={x(s)} x2={x(s)} y1={0} y2={ih} />
              {x(s) - x(0) >= 28 && (
                <text x={x(s)} y={ih + 20} textAnchor={s === last.step ? "end" : "middle"}>{s === 5 ? "1w" : s === 21 ? "1m" : "3m"}</text>
              )}
            </g>
          ))}
          {hover != null && (
            <g pointerEvents="none">
              <line x1={x(hover)} x2={x(hover)} y1={0} y2={ih} stroke="var(--ink-3)" />
              {hp?.kind === "hist" && hp.d && <circle cx={x(hover)} cy={y(hp.d.close)} r={4} fill="var(--price-line)" stroke="var(--surface)" strokeWidth={2} />}
              {hp?.kind === "fan" && hp.f && <circle cx={x(hover)} cy={y(hp.f.q500)} r={4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />}
            </g>
          )}
          <rect width={iw} height={ih} fill="transparent" />
          <text x={iw + 6} y={y(last.q975)} dy="0.32em" className="label-mid">{last.q975.toFixed(0)}</text>
          <text x={iw + 6} y={y(last.q500)} dy="0.32em" className="label-strong">{last.q500.toFixed(0)}</text>
          <text x={iw + 6} y={y(last.q025)} dy="0.32em" className="label-mid">{last.q025.toFixed(0)}</text>
        </g>
      </svg>
      {hp && (
        <div className="tooltip" aria-hidden="true" style={{ left: tipLeft > width - 230 ? tipLeft - 220 : tipLeft + 12, top: 8 }}>
          {hp.kind === "hist" && hp.d ? (
            <>
              <div className="tt-time">{fmtIsoDay(hp.d.date)}</div>
              <div className="tt-row"><span>S&amp;P 500 closed at</span><b>{hp.d.close.toLocaleString("en-US", { maximumFractionDigits: 0 })}</b></div>
            </>
          ) : hp.kind === "fan" && hp.f ? (
            <>
              <div className="tt-time">{ahead(hp.f.step)}</div>
              <div className="tt-row"><span>Middle forecast</span><b>{hp.f.q500.toFixed(0)}</b></div>
              <div className="tt-row"><span>8 in 10 between</span><b>{hp.f.q100.toFixed(0)}–{hp.f.q900.toFixed(0)}</b></div>
              <div className="tt-row"><span>19 in 20 between</span><b>{hp.f.q025.toFixed(0)}–{hp.f.q975.toFixed(0)}</b></div>
            </>
          ) : null}
        </div>
      )}
    </div>
  );
}
