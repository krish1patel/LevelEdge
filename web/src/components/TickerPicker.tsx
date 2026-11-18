import { useEffect, useId, useMemo, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

const GROUPS: { label: string; items: [string, string][] }[] = [
  { label: "Index funds", items: [["SPY", "S&P 500: the 500 largest US companies"], ["QQQ", "Nasdaq-100: the largest tech-heavy companies"],
    ["IWM", "Russell 2000: smaller US companies"], ["DIA", "Dow Jones: 30 blue-chip companies"]] },
  { label: "Stocks", items: [["NVDA", "Nvidia"], ["AAPL", "Apple"], ["MSFT", "Microsoft"], ["TSLA", "Tesla"], ["AMZN", "Amazon"], ["META", "Meta"]] },
  { label: "Other markets", items: [["GLD", "Gold"], ["TLT", "Long-term US government bonds"], ["BTC-USD", "Bitcoin (trades around the clock)"], ["ETH-USD", "Ether (trades around the clock)"]] },
];
const ALL = GROUPS.flatMap((g) => g.items.map(([sym, name]) => ({ sym, name, group: g.label })));

export function describeTicker(sym: string) {
  return ALL.find((o) => o.sym === sym)?.name;
}

/** Ticker combobox: shows every suggestion on focus, filters as you type, and accepts any symbol. */
export function TickerPicker({ value, onPick }: { value: string; onPick: (sym: string) => void }) {
  const [text, setText] = useState(value);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [typed, setTyped] = useState(false);
  const id = useId();
  const box = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => setText(value), [value]);

  const q = text.trim().toUpperCase();
  const options = useMemo(() => {
    const hits = typed && q ? ALL.filter((o) => o.sym.includes(q) || o.name.toUpperCase().includes(q)) : ALL;
    const custom = typed && q && !ALL.some((o) => o.sym === q) && /^[A-Z0-9.\-^=]{1,15}$/.test(q)
      ? [{ sym: q, name: "Use this ticker", group: "Your ticker" }] : [];
    return [...custom, ...hits];
  }, [q, typed]);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => { if (box.current && !box.current.contains(e.target as Node)) close(); };
    window.addEventListener("pointerdown", onDown);
    return () => window.removeEventListener("pointerdown", onDown);
  });

  function close() { setOpen(false); setTyped(false); setText(value); }
  function pick(sym: string) { setOpen(false); setTyped(false); setText(sym); onPick(sym); }
  function show() { setOpen(true); setTyped(false); setActive(Math.max(0, ALL.findIndex((o) => o.sym === value))); }

  const optId = (i: number) => `${id}-opt-${i}`;
  let lastGroup = "";

  return (
    <div className="picker" ref={box}>
      <div className="picker-field">
        <input ref={input} className="input ticker-input" role="combobox" aria-expanded={open} aria-controls={`${id}-list`}
               aria-autocomplete="list" aria-activedescendant={open && options[active] ? optId(active) : undefined}
               aria-label="Ticker" value={text} autoComplete="off" spellCheck={false}
               onFocus={(e) => { e.target.select(); show(); }} onClick={() => { if (!open) show(); }}
               onChange={(e) => { setText(e.target.value.toUpperCase()); setTyped(true); setOpen(true); setActive(0); }}
               onKeyDown={(e) => {
                 if (e.key === "ArrowDown") { e.preventDefault(); if (!open) show(); else setActive((a) => Math.min(options.length - 1, a + 1)); }
                 else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
                 else if (e.key === "Enter") { e.preventDefault(); if (open && options[active]) pick(options[active].sym); else if (q) pick(q); }
                 else if (e.key === "Escape") { close(); }
               }} />
        <button type="button" className="picker-toggle" tabIndex={-1} aria-label="Show tickers"
                onClick={() => { if (open) close(); else { input.current?.focus(); show(); } }}>
          <ChevronDown size={16} aria-hidden="true" />
        </button>
      </div>
      {open && (
        <ul className="picker-list" role="listbox" id={`${id}-list`} aria-label="Tickers">
          {options.length === 0 && <li className="picker-empty">No match. Type a full ticker, such as AMD, and press Enter.</li>}
          {options.map((o, i) => {
            const head = o.group !== lastGroup ? (lastGroup = o.group) : null;
            return (
              <li key={`${o.group}-${o.sym}`} role="presentation">
                {head && <div className="picker-group" aria-hidden="true">{head}</div>}
                <div role="option" id={optId(i)} aria-selected={i === active} className="picker-opt"
                     onPointerEnter={() => setActive(i)} onPointerDown={(e) => { e.preventDefault(); pick(o.sym); }}>
                  <span className="picker-sym">{o.sym}</span>
                  <span className="picker-name">{o.name}</span>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
