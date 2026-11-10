const ET = "America/New_York";

export const fmtPrice = (v: number) =>
  v >= 1000 ? v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })
    : v.toFixed(2);

export const fmtPct = (p: number, digits = 0) => {
  const v = p * 100;
  if (v > 0 && v < 1) return "<1%";
  if (v < 100 && v > 99) return ">99%";
  return `${v.toFixed(digits)}%`;
};

export const fmtSigned = (v: number, digits = 2, suffix = "%") =>
  `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}${suffix}`;

export const fmtNum = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "—" : `${v < 0 ? "−" : ""}${Math.abs(v).toFixed(digits)}`;

export const fmtTime = (iso: string) =>
  new Date(iso).toLocaleTimeString("en-US", { timeZone: ET, hour: "numeric", minute: "2-digit" });

export const fmtDay = (iso: string) =>
  new Date(iso).toLocaleDateString("en-US", { timeZone: ET, weekday: "short", month: "short", day: "numeric" });

// non-breaking spaces keep "4:00 PM ET" together when the line wraps
export const fmtDayTime = (iso: string) => `${fmtDay(iso)}, ${fmtTime(iso).replace(" ", "\u00a0")}\u00a0ET`;

/** Wall-clock parts of a Date in US/Eastern. */
export function etParts(d: Date) {
  const parts = Object.fromEntries(
    new Intl.DateTimeFormat("en-US", {
      timeZone: ET, year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", weekday: "short", hourCycle: "h23",
    }).formatToParts(d).map((p) => [p.type, p.value]),
  );
  return {
    y: +parts.year, m: +parts.month, d: +parts.day, hh: +parts.hour, mm: +parts.minute,
    weekday: parts.weekday as string,
  };
}

const pad = (n: number) => String(n).padStart(2, "0");
export const etLocalString = (y: number, m: number, d: number, hh: number, mm: number) =>
  `${y}-${pad(m)}-${pad(d)}T${pad(hh)}:${pad(mm)}`;

export function nowEtLocal(): string {
  const p = etParts(new Date());
  return etLocalString(p.y, p.m, p.d, p.hh, p.mm);
}

/** Next weekday 16:00 ET at or after today (holidays are resolved by the server). */
export function closeTarget(daysAhead: number, crypto: boolean): string {
  const p = etParts(new Date());
  const base = new Date(Date.UTC(p.y, p.m - 1, p.d));
  let date = base;
  const pastClose = p.hh * 60 + p.mm >= 16 * 60 - 1;
  let remaining = daysAhead + (pastClose ? 1 : 0);
  const isWeekend = (dt: Date) => !crypto && (dt.getUTCDay() === 0 || dt.getUTCDay() === 6);
  while (isWeekend(date)) date = new Date(date.getTime() + 864e5);
  while (remaining > 0) {
    date = new Date(date.getTime() + 864e5);
    if (!isWeekend(date)) remaining--;
  }
  return etLocalString(date.getUTCFullYear(), date.getUTCMonth() + 1, date.getUTCDate(), 16, 0);
}

export function plusHours(h: number): string {
  const p = etParts(new Date(Date.now() + h * 3600e3));
  return etLocalString(p.y, p.m, p.d, p.hh, Math.floor(p.mm / 5) * 5);
}

export const isCryptoTicker = (t: string) => t.includes("-");

export function horizonPhrase(bars: number, interval: string): string {
  if (interval.endsWith("d")) return `${bars} trading day${bars === 1 ? "" : "s"}`;
  const mins = interval.endsWith("h") ? 60 * parseInt(interval) : parseInt(interval);
  const total = bars * mins;
  const span = total >= 60 ? `${+(total / 60).toFixed(1)} h` : `${total} min`;
  return `${bars} × ${mins}-min bars (${span} of trading)`;
}

export const fmtMoney = (v: number, digits = 2) => {
  const r = Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v;   // no "−$0" for values that round to zero
  return `${r < 0 ? "−" : ""}$${Math.abs(r).toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits })}`;
};

/** Signed percent from a fraction: 0.0734 -> "+7.3%", -0.2366 -> "−23.7%". */
export const fmtSignedPct = (f: number, digits = 1) => fmtSigned(f * 100, digits);

/** Plain percent from a fraction without the <1% / >99% clamping used for probabilities. */
export const fmtRate = (f: number, digits = 1) => `${f < 0 ? "−" : ""}${Math.abs(f * 100).toFixed(digits)}%`;

/** "2026-11-13" -> "Fri, Nov 13". */
export const fmtIsoDay = (iso: string) => fmtEventDate(iso.slice(0, 10));

export function fmtEventDate(iso: string): string {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US", { timeZone: "UTC", weekday: "short", month: "short", day: "numeric" });
}
