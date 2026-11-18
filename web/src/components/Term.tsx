import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { GLOSSARY, type TermKey } from "../glossary";

const WIDTH = 300;

/** A jargon word with its plain-English definition on hover, keyboard focus or tap.
 *  The trigger is a real button, so it works without a mouse; Escape closes it.
 *  The definition is positioned against the window, so table scroll areas can't clip it. */
export function Term({ k, children }: { k: TermKey; children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const id = useId();
  const btn = useRef<HTMLButtonElement>(null);
  const wrap = useRef<HTMLSpanElement>(null);
  const [title, body] = GLOSSARY[k];

  useLayoutEffect(() => {
    if (!open || !btn.current) { setPos(null); return; }
    const r = btn.current.getBoundingClientRect();
    const w = Math.min(WIDTH, window.innerWidth * 0.8);
    setPos({ top: r.bottom + 6, left: Math.max(12, Math.min(r.left, window.innerWidth - w - 12)) });
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") close(); };
    const onDown = (e: PointerEvent) => { if (wrap.current && !wrap.current.contains(e.target as Node)) close(); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("scroll", close, true);       // a fixed popover would drift away from its word
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  return (
    <span className="term-wrap" ref={wrap} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button type="button" ref={btn} className="term" aria-describedby={open ? id : undefined} aria-expanded={open}
              onFocus={() => setOpen(true)} onBlur={() => setOpen(false)} onClick={() => setOpen((o) => !o)}>
        {children ?? title}
      </button>
      {open && pos && (
        <span className="term-pop" role="tooltip" id={id} style={{ top: pos.top, left: pos.left }}>
          <strong>{title}</strong>
          {body}
        </span>
      )}
    </span>
  );
}
