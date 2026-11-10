import { useMemo, useRef, useState } from "react";
import { scaleLinear } from "d3-scale";
import { area, line, curveMonotoneX } from "d3-shape";
import type { ForecastResponse } from "../api";
import { fmtDay, fmtPct, fmtPrice, fmtTime } from "../format";
import { useWidth } from "./useWidth";

/**
 * Signature chart: recent price flowing into the forecast fan.
 * The x-axis is TRADING time (one slot per bar), so overnight and weekend gaps
 * take no width and the fan's length is the real number of bars left.
 */
export function FanChart({ data }: { data: ForecastResponse }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const height = width < 560 ? 300 : 500;
  const m = { top: 16, right: width < 560 ? 96 : 132, bottom: 32, left: 56 };
  const iw = Math.max(10, width - m.left - m.right);
  const ih = height - m.top - m.bottom;

  const { fan, levels, meta } = data;
  // Keep enough history for context but let the fan own a real share of the plot
  // (>= ~1/3 on desktop, >= ~40% on phones).
  const history = useMemo(() => {
    const h = data.history;
    const keep = width >= 560 ? Math.max(24, fan.length * 2) : Math.max(12, Math.round(fan.length * 1.4));
    return h.slice(-keep);
  }, [data.history, fan.length, width]);
  const nH = history.length;
  const nF = fan.length;

  const model = useMemo(() => {
    const prices = [
      ...history.map((b) => b.c), ...fan.flatMap((f) => [f.q10, f.q90]),
      ...levels.map((l) => l.level),
    ];
    let lo = Math.min(...prices);
    let hi = Math.max(...prices);
    // keep far-away levels from flattening the chart: clamp to 1.6x the fan span
    const fanLo = Math.min(...fan.map((f) => f.q10), ...history.map((b) => b.c));
    const fanHi = Math.max(...fan.map((f) => f.q90), ...history.map((b) => b.c));
    const span = fanHi - fanLo;
    lo = Math.max(lo, fanLo - 0.6 * span);
    hi = Math.min(hi, fanHi + 0.6 * span);
    const pad = (hi - lo) * 0.06;
    const x = scaleLinear().domain([-(nH - 1), nF]).range([0, iw]);
    const y = scaleLinear().domain([lo - pad, hi + pad]).range([ih, 0]).nice(6);
    return { x, y };
  }, [history, fan, levels, iw, ih, nH, nF]);
  const { x, y } = model;

  const spot = meta.spot;
  const fanPts = [{ step: 0, q10: spot, q25: spot, q50: spot, q75: spot, q90: spot }, ...fan];
  const outer = area<(typeof fanPts)[number]>().x((d) => x(d.step)).y0((d) => y(d.q10)).y1((d) => y(d.q90)).curve(curveMonotoneX);
  const inner = area<(typeof fanPts)[number]>().x((d) => x(d.step)).y0((d) => y(d.q25)).y1((d) => y(d.q75)).curve(curveMonotoneX);
  const median = line<(typeof fanPts)[number]>().x((d) => x(d.step)).y((d) => y(d.q50)).curve(curveMonotoneX);
  const price = line<(typeof history)[number]>().x((_, i) => x(i - (nH - 1))).y((d) => y(d.c));

  // x ticks: first bar of each session (or every N bars), labelled by day or time
  const ticks = useMemo(() => {
    const all = [
      ...history.map((b, i) => ({ step: i - (nH - 1), t: b.t })),
      ...fan.map((f) => ({ step: f.step, t: f.time })),
    ];
    const daily = meta.interval.endsWith("d");
    const sessions: { step: number; label: string }[] = [];
    const times: { step: number; label: string }[] = [];
    let lastDay = "";
    const maxTicks = Math.max(3, Math.floor(iw / 90));
    const every = Math.max(1, Math.ceil(all.length / maxTicks));
    // prefer whole-hour times (bar starts for history, bar closes for the fan)
    const onHour = (iso: string) => new Date(iso).getUTCMinutes() === 0;
    const hourly = !daily && all.filter((p) => onHour(p.t)).length >= 2;
    all.forEach((p, i) => {
      const day = fmtDay(p.t);
      if (!daily && day !== lastDay) sessions.push({ step: p.step, label: day });
      else if (hourly ? onHour(p.t) : i % every === 0) times.push({ step: p.step, label: daily ? day : fmtTime(p.t) });
      lastDay = day;
    });
    // session labels win; time labels fill the gaps between them
    const keep = (step: number, taken: number[]) => taken.every((s) => Math.abs(x(s) - x(step)) > 72);
    const out: { step: number; label: string; session: boolean }[] = [];
    for (const t of sessions) if (keep(t.step, out.map((o) => o.step))) out.push({ ...t, session: true });
    for (const t of times) if (keep(t.step, out.map((o) => o.step))) out.push({ ...t, session: false });
    return { labels: out.sort((a, b) => a.step - b.step), boundaries: sessions.slice(1).map((s) => s.step) };
  }, [history, fan, nH, iw, x, meta.interval]);

  const eventMarks = useMemo(() => (data.events ?? []).map((ev) => {
    const f = fan.find((b) => b.time.slice(0, 10) >= ev.date);
    return f ? { step: f.step, label: `Earnings ${fmtDay(f.time)}` } : null;
  }).filter((e): e is { step: number; label: string } => e !== null), [data.events, fan]);

  // level labels on the right, de-collided with leader lines
  const levelLabels = useMemo(() => {
    const items = [...levels]
      .map((l) => ({ ...l, y0: y(l.level), clipped: l.level < y.domain()[0] || l.level > y.domain()[1] }))
      .sort((a, b) => a.y0 - b.y0);
    const gap = 18;
    let prev = -Infinity;
    for (const it of items) {
      const yy = Math.min(Math.max(it.y0, 8), ih - 4);
      (it as typeof it & { ly: number }).ly = Math.max(yy, prev + gap);
      prev = (it as typeof it & { ly: number }).ly;
    }
    return items as (typeof items[number] & { ly: number })[];
  }, [levels, y, ih]);

  const yTicks = y.ticks(6);
  const hoverInfo = hover == null ? null : hover <= 0
    ? { kind: "hist" as const, bar: history[hover + nH - 1] }
    : { kind: "fan" as const, f: fan[hover - 1] };

  function onMove(e: React.PointerEvent) {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left - m.left;
    const step = Math.round(x.invert(px));
    setHover(Math.max(-(nH - 1), Math.min(nF, step)));
  }

  const q = data.quantiles;
  const summary = `${meta.ticker} at ${fmtPrice(spot)}. Forecast for ${fmtDay(meta.effective_target)} ${fmtTime(meta.effective_target)}: ` +
    `median ${fmtPrice(q.q50)}, 80% range ${fmtPrice(q.q10)} to ${fmtPrice(q.q90)}. ` +
    levels.map((l) => `${fmtPrice(l.level)}: ${fmtPct(l.p_above)} chance of closing above`).join("; ") + ".";

  const tipLeft = hover == null ? 0 : m.left + x(hover);
  const tipFlip = tipLeft > width - 200;

  return (
    <figure ref={ref} style={{ margin: 0, position: "relative" }} aria-label={summary}>
      <svg ref={svgRef} className="chart" width={width} height={height} role="img" aria-label={summary}
           onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
        <g transform={`translate(${m.left},${m.top})`} aria-hidden="true">
          {yTicks.map((t) => (
            <g key={t} className="tick">
              <line x1={0} x2={iw} y1={y(t)} y2={y(t)} />
              <text x={-10} y={y(t)} dy="0.32em" textAnchor="end" className="num">{fmtPrice(t)}</text>
            </g>
          ))}
          {ticks.boundaries.map((b) => (
            <line key={`b${b}`} x1={x(b - 0.5)} x2={x(b - 0.5)} y1={0} y2={ih} stroke="var(--grid)" />
          ))}
          {ticks.labels.map((t) => (
            <text key={`x${t.step}`} x={x(t.step)} y={ih + 22} textAnchor="middle"
                  className={t.session ? "label-mid" : undefined}>{t.label}</text>
          ))}
          <line className="axis-line" x1={0} x2={iw} y1={ih} y2={ih} />

          {/* forecast region */}
          <rect x={x(0)} y={0} width={Math.max(0, x(nF) - x(0))} height={ih} fill="var(--accent-wash)" opacity={0.35} />
          <path d={outer(fanPts) ?? ""} fill="var(--fan-outer)" />
          <path d={inner(fanPts) ?? ""} fill="var(--fan-inner)" />
          <path d={median(fanPts) ?? ""} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinecap="round" />
          {eventMarks.map((ev) => (
            <g key={ev.step}>
              <line x1={x(ev.step - 0.5)} x2={x(ev.step - 0.5)} y1={18} y2={ih} stroke="var(--warn)" strokeWidth={1.5} />
              <text x={x(ev.step - 0.5) + 6} y={30} className="label-mid">{ev.label}</text>
            </g>
          ))}

          {/* levels */}
          {levelLabels.map((l) => (
            <g key={l.input}>
              {!l.clipped && (
                <line x1={x(-(nH - 1))} x2={iw} y1={l.y0} y2={l.y0} stroke="var(--ink-3)" strokeWidth={1}
                      strokeDasharray="4 4" opacity={0.7} />
              )}
              {!l.clipped && <circle cx={x(nF)} cy={l.y0} r={4} fill="var(--ink)" stroke="var(--surface)" strokeWidth={2} />}
              {Math.abs(l.ly - l.y0) > 1 && !l.clipped && (
                <path d={`M${iw + 4},${l.y0} L${iw + 10},${l.ly}`} stroke="var(--ink-3)" strokeWidth={1} fill="none" />
              )}
              {l.clipped && (
                <path d={l.level > spot ? `M${iw + 4},${l.ly + 3} l4,-7 l4,7 z` : `M${iw + 4},${l.ly - 3} l4,7 l4,-7 z`}
                      fill="var(--ink-3)" />
              )}
              <text x={iw + (l.clipped ? 16 : 12)} y={l.ly} dy="0.32em" className="num">
                <tspan className="label-strong">{fmtPct(l.p_above)}</tspan>
                <tspan className="label-mid" dx={6}>{fmtPrice(l.level)}</tspan>
              </text>
            </g>
          ))}

          {/* price */}
          <path d={price(history) ?? ""} fill="none" stroke="var(--price-line)" strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
          <line x1={x(0)} x2={x(0)} y1={0} y2={ih} stroke="var(--hairline-strong)" />
          <circle cx={x(0)} cy={y(spot)} r={4.5} fill="var(--price-line)" stroke="var(--surface)" strokeWidth={2} />
          <text x={x(0) - 6} y={12} textAnchor="end">Now</text>
          <text x={x(nF)} y={12} textAnchor="end">{fmtTime(meta.effective_target)}</text>

          {hover != null && (
            <g>
              <line x1={x(hover)} x2={x(hover)} y1={0} y2={ih} stroke="var(--ink-3)" strokeWidth={1} />
              {hoverInfo?.kind === "hist" && hoverInfo.bar && (
                <circle cx={x(hover)} cy={y(hoverInfo.bar.c)} r={4} fill="var(--price-line)" stroke="var(--surface)" strokeWidth={2} />
              )}
              {hoverInfo?.kind === "fan" && hoverInfo.f && (
                <circle cx={x(hover)} cy={y(hoverInfo.f.q50)} r={4} fill="var(--accent)" stroke="var(--surface)" strokeWidth={2} />
              )}
            </g>
          )}
          <rect x={0} y={0} width={iw} height={ih} fill="transparent" />
        </g>
      </svg>
      {hoverInfo && (
        <div className="tooltip" style={{ left: tipFlip ? tipLeft - 170 : tipLeft + 12, top: m.top + 8 }} aria-hidden="true">
          {hoverInfo.kind === "hist" && hoverInfo.bar && (
            <>
              <div className="tt-time">{fmtDay(hoverInfo.bar.t)}, {fmtTime(hoverInfo.bar.t)}</div>
              <div className="tt-row"><span><i className="key" style={{ background: "var(--price-line)" }} />Close</span><b>{fmtPrice(hoverInfo.bar.c)}</b></div>
            </>
          )}
          {hoverInfo.kind === "fan" && hoverInfo.f && (
            <>
              <div className="tt-time">By {fmtTime(hoverInfo.f.time)} · bar {hoverInfo.f.step} of {nF}</div>
              <div className="tt-row"><span>90th pct</span><b>{fmtPrice(hoverInfo.f.q90)}</b></div>
              <div className="tt-row"><span>75th</span><b>{fmtPrice(hoverInfo.f.q75)}</b></div>
              <div className="tt-row"><span><i className="key" style={{ background: "var(--accent)" }} />Median</span><b>{fmtPrice(hoverInfo.f.q50)}</b></div>
              <div className="tt-row"><span>25th</span><b>{fmtPrice(hoverInfo.f.q25)}</b></div>
              <div className="tt-row"><span>10th</span><b>{fmtPrice(hoverInfo.f.q10)}</b></div>
            </>
          )}
        </div>
      )}
      <figcaption className="legend">
        <span><i style={{ background: "var(--price-line)" }} />Price</span>
        <span><i style={{ background: "var(--accent)" }} />Median forecast</span>
        <span><i className="band" style={{ background: "var(--fan-inner)" }} />50% range</span>
        <span><i className="band" style={{ background: "var(--fan-outer)" }} />80% range</span>
        <span><i style={{ background: "transparent", borderTop: "1px dashed var(--ink-3)" }} />Your levels, P(close above)</span>
      </figcaption>
    </figure>
  );
}
