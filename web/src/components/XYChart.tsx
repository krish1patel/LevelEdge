import { useRef, useState } from "react";
import { scaleLinear } from "d3-scale";
import { line } from "d3-shape";
import { useWidth } from "./useWidth";

export interface Series {
  name: string;
  color: string;           // CSS var
  points: { x: number; y: number }[];
  kind?: "line" | "dots";
  dashed?: boolean;
  hollow?: boolean;        // dots drawn as rings: a second, non-colour cue
}

interface Props {
  series: Series[];
  xLabel: (v: number) => string;
  yLabel: (v: number) => string;
  xDomain?: [number, number];
  yDomain?: [number, number];
  height?: number;
  diagonal?: boolean;     // y = x reference (reliability diagrams)
  vline?: { x: number; label: string };
  summary: string;
  xTitle?: string;
  yTitle?: string;
}

export function XYChart({ series, xLabel, yLabel, xDomain, yDomain, height = 260, diagonal, vline, summary, xTitle, yTitle }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>(480);
  const svgRef = useRef<SVGSVGElement>(null);
  const [hx, setHx] = useState<number | null>(null);
  const m = { top: 12, right: 16, bottom: xTitle ? 44 : 30, left: yTitle ? 60 : 48 };
  const iw = Math.max(10, width - m.left - m.right);
  const ih = height - m.top - m.bottom;
  const all = series.flatMap((s) => s.points);
  const xd = xDomain ?? [Math.min(...all.map((p) => p.x)), Math.max(...all.map((p) => p.x))];
  const yd = yDomain ?? [Math.min(...all.map((p) => p.y)), Math.max(...all.map((p) => p.y))];
  const x = scaleLinear().domain(xd).range([0, iw]);
  const y = scaleLinear().domain(yd).range([ih, 0]).nice(5);
  const path = line<{ x: number; y: number }>().x((d) => x(d.x)).y((d) => y(d.y));

  const nearest = (s: Series, vx: number) =>
    s.points.reduce((a, b) => (Math.abs(b.x - vx) < Math.abs(a.x - vx) ? b : a), s.points[0]);

  function onMove(e: React.PointerEvent) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const vx = x.invert(e.clientX - rect.left - m.left);
    setHx(Math.min(xd[1], Math.max(xd[0], vx)));
  }

  const tipLeft = hx == null ? 0 : m.left + x(hx);

  return (
    <figure ref={ref} style={{ margin: 0, position: "relative" }}>
      <svg ref={svgRef} className="chart" width={width} height={height} role="img" aria-label={summary}
           onPointerMove={onMove} onPointerLeave={() => setHx(null)}>
        <g transform={`translate(${m.left},${m.top})`} aria-hidden="true">
          {y.ticks(5).map((t) => (
            <g key={t} className="tick">
              <line x1={0} x2={iw} y1={y(t)} y2={y(t)} />
              <text x={-8} y={y(t)} dy="0.32em" textAnchor="end" className="num">{yLabel(t)}</text>
            </g>
          ))}
          {x.ticks(Math.max(3, Math.floor(iw / 80))).map((t) => (
            <text key={t} x={x(t)} y={ih + 18} textAnchor="middle" className="num">{xLabel(t)}</text>
          ))}
          <line className="axis-line" x1={0} x2={iw} y1={ih} y2={ih} />
          {xTitle && <text x={iw / 2} y={ih + 38} textAnchor="middle">{xTitle}</text>}
          {yTitle && <text transform={`translate(${-46},${ih / 2}) rotate(-90)`} textAnchor="middle">{yTitle}</text>}
          {diagonal && <line x1={x(xd[0])} y1={y(xd[0])} x2={x(xd[1])} y2={y(xd[1])} stroke="var(--hairline-strong)" strokeWidth={1} />}
          {vline && (
            <g>
              <line x1={x(vline.x)} x2={x(vline.x)} y1={0} y2={ih} stroke="var(--hairline-strong)" />
              <text x={x(vline.x) + 6} y={10}>{vline.label}</text>
            </g>
          )}
          {series.map((s) =>
            s.kind === "dots" ? (
              <g key={s.name}>
                {s.points.length > 1 && (
                  <path d={path(s.points) ?? ""} fill="none" stroke={s.color} strokeWidth={1} opacity={0.5} />
                )}
                {s.points.map((p, i) => (
                  <circle key={i} cx={x(p.x)} cy={y(p.y)} r={4.5} fill={s.hollow ? "var(--surface)" : s.color}
                          stroke={s.hollow ? s.color : "var(--surface)"} strokeWidth={2} />
                ))}
              </g>
            ) : (
              <path key={s.name} d={path(s.points) ?? ""} fill="none" stroke={s.color} strokeWidth={2}
                    strokeDasharray={s.dashed ? "5 4" : undefined} strokeLinecap="round" strokeLinejoin="round" />
            ),
          )}
          {hx != null && (
            <g>
              <line x1={x(hx)} x2={x(hx)} y1={0} y2={ih} stroke="var(--ink-3)" />
              {series.map((s) => {
                const p = nearest(s, hx);
                return p ? <circle key={s.name} cx={x(p.x)} cy={y(p.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} /> : null;
              })}
            </g>
          )}
          <rect width={iw} height={ih} fill="transparent" />
        </g>
      </svg>
      {hx != null && (
        <div className="tooltip" style={{ left: tipLeft > width - 190 ? tipLeft - 180 : tipLeft + 12, top: 8 }} aria-hidden="true">
          <div className="tt-time">{xLabel(nearest(series[0], hx).x)}</div>
          {series.map((s) => {
            const p = nearest(s, hx);
            return (
              <div className="tt-row" key={s.name}>
                <span><i className="key" style={{ background: s.color }} />{s.name}</span>
                <b>{yLabel(p.y)}</b>
              </div>
            );
          })}
        </div>
      )}
      {series.length > 1 && (
        <figcaption className="legend">
          {series.map((s) => (
            <span key={s.name}>
              <i style={s.hollow ? { width: 10, height: 10, borderRadius: "50%", border: `2px solid ${s.color}`, background: "transparent" }
                : s.dashed ? { background: "transparent", borderTop: `2px dashed ${s.color}` } : { background: s.color }} />
              {s.name}
            </span>
          ))}
        </figcaption>
      )}
    </figure>
  );
}
