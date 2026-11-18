import { useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { GLOSSARY, type TermKey } from "../glossary";

/** A jargon word with its plain-English definition on hover, keyboard focus or tap.
 *  The trigger is a real button, so it works without a mouse; Escape closes it. */
export function Term({ k, children }: { k: TermKey; children?: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [shift, setShift] = useState(0);
  const id = useId();
  const pop = useRef<HTMLSpanElement>(null);
  const wrap = useRef<HTMLSpanElement>(null);
  const [title, body] = GLOSSARY[k];

  useLayoutEffect(() => {
    if (!open || !pop.current) return;
    const r = pop.current.getBoundingClientRect();
    const over = r.right - (window.innerWidth - 12);
    const under = 12 - r.left;
    setShift(over > 0 ? -over : under > 0 ? under : 0);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    const onDown = (e: PointerEvent) => { if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("pointerdown", onDown); };
  }, [open]);

  return (
    <span className="term-wrap" ref={wrap} onMouseEnter={() => setOpen(true)} onMouseLeave={() => { setOpen(false); setShift(0); }}>
      <button type="button" className="term" aria-describedby={open ? id : undefined} aria-expanded={open}
              onFocus={() => setOpen(true)} onBlur={() => { setOpen(false); setShift(0); }} onClick={() => setOpen((o) => !o)}>
        {children ?? title}
      </button>
      {open && (
        <span className="term-pop" role="tooltip" id={id} ref={pop} style={{ transform: `translateX(${shift}px)` }}>
          <strong>{title}</strong>
          {body}
        </span>
      )}
    </span>
  );
}
