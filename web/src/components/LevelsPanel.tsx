import { useState } from "react";
import { ArrowDown, ArrowUp, Plus, X } from "lucide-react";
import type { LevelRow } from "../api";
import { fmtPct, fmtPrice, fmtSigned } from "../format";

const LEVEL_RE = /^\s*[+-]?\$?\d+(\.\d+)?\s*%?\s*$/;

interface Props {
  rows: LevelRow[] | null;
  inputs: string[];
  previous: Record<string, number>;
  flash: Set<string>;
  onChange: (inputs: string[]) => void;
  busy: boolean;
}

export function LevelsPanel({ rows, inputs, previous, flash, onChange, busy }: Props) {
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const byInput = new Map(rows?.map((r) => [r.input, r]) ?? []);
  const sorted = [...inputs].sort((a, b) => (byInput.get(b)?.level ?? 0) - (byInput.get(a)?.level ?? 0));

  function add(e: React.FormEvent) {
    e.preventDefault();
    const parts = draft.split(",").map((s) => s.trim()).filter(Boolean);
    if (!parts.length) return;
    const bad = parts.find((p) => !LEVEL_RE.test(p));
    if (bad) {
      setError(`“${bad}” isn't a level. Use a price like 665 or a move like +1%.`);
      return;
    }
    const next = [...inputs, ...parts.filter((p) => !inputs.includes(p))].slice(0, 24);
    setError(null);
    setDraft("");
    onChange(next);
  }

  return (
    <section className="card" aria-labelledby="levels-h">
      <div className="card-head">
        <div>
          <h2 id="levels-h">Levels</h2>
          <p className="card-sub">Chance the price closes above each level</p>
        </div>
      </div>
      {inputs.length === 0 ? (
        <div className="empty" style={{ padding: 0 }}>
          <p>Add a level to see its probability. Type a price (665) or a move from the current price (+1%, −0.5%).</p>
        </div>
      ) : (
        <table className="levels">
          <thead>
            <tr>
              <th scope="col">Level</th>
              <th scope="col" className="r">P(above)</th>
              <th scope="col" className="r"><span className="visually-hidden">Change since last run</span></th>
              <th scope="col"><span className="visually-hidden">Remove</span></th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((inp) => {
              const r = byInput.get(inp);
              const prev = previous[inp];
              const delta = r && prev != null ? Math.round((r.p_above - prev) * 100) : null;
              return (
                <tr key={inp} className={flash.has(inp) ? "row-flash" : undefined}>
                  <td className="level num">
                    {r ? fmtPrice(r.level) : inp}
                    <span className="move">{r ? fmtSigned(r.move_pct) : busy ? "Calculating…" : "Not run yet"}</span>
                  </td>
                  <td className="r">
                    {r ? (
                      <div className="pcell">
                        <div className="pbar" aria-hidden="true"><i style={{ width: `${r.p_above * 100}%` }} /></div>
                        <span className="pval num">{fmtPct(r.p_above)}</span>
                      </div>
                    ) : <span className="pval">—</span>}
                  </td>
                  <td className="r">
                    {delta != null && delta !== 0 ? (
                      <span className="delta num" aria-label={`${delta > 0 ? "up" : "down"} ${Math.abs(delta)} points since last run`}>
                        {delta > 0 ? <ArrowUp size={11} aria-hidden="true" /> : <ArrowDown size={11} aria-hidden="true" />}
                        {Math.abs(delta)}
                      </span>
                    ) : <span className="delta" />}
                  </td>
                  <td className="r" style={{ width: 44 }}>
                    <button type="button" className="btn btn-quiet btn-remove"
                            aria-label={`Remove level ${r ? fmtPrice(r.level) : inp}`}
                            onClick={() => onChange(inputs.filter((x) => x !== inp))}>
                      <X size={16} aria-hidden="true" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <form className="add-level" onSubmit={add}>
        <label className="visually-hidden" htmlFor="add-level">Add levels</label>
        <input id="add-level" className="input num" placeholder="Add a level: 665 or +1%" value={draft}
               aria-invalid={error ? "true" : undefined} aria-describedby={error ? "add-level-err" : undefined}
               onChange={(e) => { setDraft(e.target.value); setError(null); }} />
        <button className="btn btn-ghost" type="submit" disabled={!draft.trim()}>
          <Plus size={16} aria-hidden="true" />Add
        </button>
      </form>
      {error ? <p id="add-level-err" className="err" role="alert">{error}</p>
        : <p className="help">Separate several with commas. Percent moves are measured from the last completed bar.</p>}
    </section>
  );
}
