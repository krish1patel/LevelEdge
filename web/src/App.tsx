import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, AlertTriangle, History, Loader2, Monitor, Moon, Play, Sun } from "lucide-react";
import { getForecast, type ForecastResponse } from "./api";
import { closeTarget, fmtDayTime, fmtEventDate, fmtPrice, fmtSigned, horizonPhrase, isCryptoTicker, nowEtLocal, plusHours } from "./format";
import { FanChart } from "./components/FanChart";
import { LevelsPanel } from "./components/LevelsPanel";
import { TrustPanel } from "./components/TrustPanel";
import { Diagnostics } from "./components/Diagnostics";
import { TrackRecord } from "./components/TrackRecord";
import { Portfolio } from "./components/Portfolio";
import { Research } from "./components/Research";

const INTERVALS = ["1m", "2m", "5m", "15m", "30m", "1h", "90m", "1d"];
const SUGGESTED = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "IWM", "BTC-USD", "ETH-USD"];
type Preset = "close" | "next" | "hour" | "custom";
type Tab = "forecast" | "diagnostics" | "track";
type Section = "portfolio" | "forecaster" | "research";
const SECTIONS: [Section, string][] = [["portfolio", "Portfolio"], ["forecaster", "Forecaster"], ["research", "Research"]];

/** Route from the URL: #research, forecaster tabs (#forecast, #diagnostics, #track), else the portfolio.
 *  Links that carry ?ticker= (the old forecaster URLs) still open the forecaster. */
function routeFromUrl(): { section: Section; tab: Tab } {
  const h = window.location.hash.replace("#", "");
  if (h === "research") return { section: "research", tab: "forecast" };
  if (h === "forecast" || h === "diagnostics" || h === "track") return { section: "forecaster", tab: h };
  if (!h && new URLSearchParams(window.location.search).has("ticker")) return { section: "forecaster", tab: "forecast" };
  return { section: "portfolio", tab: "forecast" };
}
type Theme = "system" | "light" | "dark";

const store = {
  get: (k: string) => { try { return localStorage.getItem(k); } catch { return null; } },
  set: (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

function targetFor(preset: Preset, custom: string, crypto: boolean) {
  if (preset === "close") return closeTarget(0, crypto);
  if (preset === "next") return closeTarget(1, crypto);
  if (preset === "hour") return plusHours(1);
  return custom;
}

function closeLabel(daysAhead: number, ticker: string) {
  const t = closeTarget(daysAhead, isCryptoTicker(ticker));
  const [y, m, d] = t.slice(0, 10).split("-").map(Number);
  const wd = new Date(Date.UTC(y, m - 1, d)).toLocaleDateString("en-US", { weekday: "short", timeZone: "UTC" });
  const today = nowEtLocal().slice(0, 10) === t.slice(0, 10);
  return today ? "Close today" : `Close ${wd}`;
}

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const [ticker, setTicker] = useState(() => (params.get("ticker") ?? store.get("le.ticker") ?? "SPY").toUpperCase());
  const [interval, setIntervalV] = useState(() => {
    const iv = params.get("interval") ?? store.get("le.interval") ?? "15m";
    return INTERVALS.includes(iv) ? iv : "15m";
  });
  const [preset, setPreset] = useState<Preset>(() => {
    const t = params.get("target");
    if (t && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(t)) return "custom";
    return t === "next" || t === "hour" ? t : "close";
  });
  const [custom, setCustom] = useState(() => {
    const t = params.get("target");
    return t && /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/.test(t) ? t : closeTarget(0, false);
  });
  const [backtest, setBacktest] = useState(false);
  const [asOf, setAsOf] = useState(() => nowEtLocal());
  const [levels, setLevels] = useState<string[]>(() => {
    const saved = store.get("le.levels");
    return saved ? JSON.parse(saved) : ["-1%", "-0.5%", "+0.5%", "+1%"];
  });
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [prev, setPrev] = useState<Record<string, number>>({});
  const [flash, setFlash] = useState<Set<string>>(new Set());
  const [tab, setTabState] = useState<Tab>(() => routeFromUrl().tab);
  const [section, setSectionState] = useState<Section>(() => routeFromUrl().section);
  const go = (sec: Section, t: Tab = tab) => {
    setSectionState(sec);
    setTabState(t);
    const hash = sec === "portfolio" ? "" : sec === "research" ? "#research" : `#${t}`;
    try { history.replaceState(null, "", `${window.location.pathname}${window.location.search}${hash}`); } catch { /* sandboxed */ }
  };
  const setTab = (t: Tab) => go("forecaster", t);
  const [theme, setTheme] = useState<Theme>(() => (store.get("le.theme") as Theme) ?? "system");
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    const onHash = () => {
      const r = routeFromUrl();
      setSectionState(r.section);
      setTabState(r.tab);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    store.set("le.theme", theme);
  }, [theme]);

  const run = useCallback(async (lv: string[] = levels) => {
    const sym = ticker.trim().toUpperCase();
    if (!sym) { setError("Enter a ticker, for example SPY or BTC-USD."); return; }
    const target = targetFor(preset, custom, isCryptoTicker(sym));
    abort.current?.abort();
    const ctl = new AbortController();
    abort.current = ctl;
    setBusy(true);
    setError(null);
    try {
      const res = await getForecast({ ticker: sym, target, interval, levels: lv, as_of: backtest ? asOf : null }, ctl.signal);
      const sameContext = data && data.meta.ticker === res.meta.ticker && data.meta.effective_target === res.meta.effective_target
        && data.meta.interval === res.meta.interval;
      const before: Record<string, number> = sameContext ? Object.fromEntries(data!.levels.map((l) => [l.input, l.p_above])) : {};
      setPrev(before);
      setFlash(new Set(res.levels.filter((l) => before[l.input] != null && Math.abs(before[l.input] - l.p_above) >= 0.01).map((l) => l.input)));
      setData(res);
      store.set("le.ticker", sym);
      store.set("le.interval", interval);
      store.set("le.levels", JSON.stringify(lv));
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError((e as Error).message);
    } finally {
      if (abort.current === ctl) setBusy(false);
    }
  }, [ticker, preset, custom, interval, levels, backtest, asOf, data]);

  // first visit to the forecaster: run the default forecast so the screen teaches itself
  const ran = useRef(false);
  useEffect(() => { if (section === "forecaster" && !ran.current) { ran.current = true; run(); } }, [run, section]);
  useEffect(() => {
    if (section === "portfolio") document.title = "Paper book · LevelEdge";
    else if (section === "research") document.title = "Research · LevelEdge";
    else if (data) document.title = `${data.meta.ticker} ${fmtPrice(data.meta.spot)} · LevelEdge`;
    else document.title = "Forecaster · LevelEdge";
  }, [section, data]);

  function onLevels(next: string[]) {
    setLevels(next);
    if (data) run(next);
  }

  const meta = data?.meta;
  const q = data?.quantiles;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M3 17 L10 11 L14 13 L21 6" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M14 13 L21 9 L21 17 Z" fill="currentColor" opacity="0.28" />
            </svg>
            LevelEdge
          </div>
          <nav className="section-nav" aria-label="Sections">
            {SECTIONS.map(([k, l]) => (
              <a key={k} href={k === "portfolio" ? window.location.pathname : k === "research" ? "#research" : "#forecast"}
                 aria-current={section === k ? "page" : undefined}
                 onClick={(e) => { e.preventDefault(); go(k, k === "forecaster" ? tab : "forecast"); }}>{l}</a>
            ))}
          </nav>
          <div className="topbar-spacer" />
          <button type="button" className="btn btn-ghost btn-icon" onClick={() => setTheme(theme === "system" ? "light" : theme === "light" ? "dark" : "system")}
                  aria-label={`Theme: ${theme}. Change theme`} title={`Theme: ${theme}`}>
            {theme === "system" ? <Monitor size={16} /> : theme === "light" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          {section === "forecaster" && <form className="controls" onSubmit={(e) => { e.preventDefault(); run(); }} aria-label="Forecast settings">
            <label className="field">
              <span>Ticker</span>
              <input className="input ticker-input" value={ticker} list="tickers" autoComplete="off" spellCheck={false}
                     onChange={(e) => setTicker(e.target.value.toUpperCase())} aria-label="Ticker" />
              <datalist id="tickers">{SUGGESTED.map((t) => <option key={t} value={t} />)}</datalist>
            </label>
            <div className="field">
              <span id="target-l">Target</span>
              <div className="segmented" role="group" aria-labelledby="target-l">
                {([["close", closeLabel(0, ticker)], ["next", closeLabel(1, ticker)], ["hour", "In 1 hour"], ["custom", "Custom"]] as const).map(([k, l]) => (
                  <button key={k} type="button" aria-pressed={preset === k} onClick={() => setPreset(k)}>{l}</button>
                ))}
              </div>
            </div>
            {preset === "custom" && (
              <label className="field">
                <span>Target time (ET)</span>
                <input type="datetime-local" className="input dt-input num" value={custom} onChange={(e) => setCustom(e.target.value)} />
              </label>
            )}
            <label className="field">
              <span>Bars</span>
              <select className="select" value={interval} onChange={(e) => setIntervalV(e.target.value)} aria-label="Bar interval">
                {INTERVALS.map((i) => <option key={i} value={i}>{i}</option>)}
              </select>
            </label>
            {backtest && (
              <label className="field">
                <span>As of (ET)</span>
                <input type="datetime-local" className="input dt-input num" value={asOf} onChange={(e) => setAsOf(e.target.value)} />
              </label>
            )}
            <button type="button" className="btn btn-ghost" aria-pressed={backtest} onClick={() => setBacktest((b) => !b)}
                    title="Forecast as of a past time">
              <History size={16} aria-hidden="true" />{backtest ? "Backtest on" : "Backtest"}
            </button>
            <button className="btn btn-primary" type="submit" disabled={busy}>
              {busy ? <Loader2 size={16} className="spin" aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
              {busy ? "Forecasting" : "Forecast"}
            </button>
          </form>}
        </div>
      </header>

      <main>
        {section === "portfolio" && <Portfolio />}
        {section === "research" && <Research />}
        {section === "forecaster" && <>
        {error && (
          <div className="banner" data-tone="error" role="alert">
            <AlertCircle size={18} aria-hidden="true" />
            <p>{error}</p>
          </div>
        )}

        {meta && q ? (
          <section className="summary" aria-live="polite">
            <h1><span className="sym">{meta.ticker}</span><span className="spot num">{fmtPrice(meta.spot)}</span></h1>
            <dl className="facts">
              <div><dt>Last bar closed</dt><dd>{fmtDayTime(meta.anchor_time)}</dd></div>
              <div><dt>Resolves</dt><dd>{fmtDayTime(meta.effective_target)}</dd></div>
              <div><dt>Horizon</dt><dd className="num">{horizonPhrase(meta.horizon_bars, meta.interval)}</dd></div>
              <div><dt>80% range</dt><dd className="num">{fmtPrice(q.q10)} to {fmtPrice(q.q90)}</dd></div>
              <div><dt>Typical move (1σ)</dt><dd className="num">{fmtSigned(meta.sigma_pct, 2).replace("+", "±")}</dd></div>
            </dl>
            {meta.is_backtest && <span className="badge"><History size={14} aria-hidden="true" />Backtest as of {fmtDayTime(meta.as_of)}</span>}
          </section>
        ) : (
          <div className="summary"><div className="skeleton" style={{ width: 420, maxWidth: "100%", height: 40 }} /></div>
        )}

        {data && data.events.length > 0 && (
          <div className="banner" data-tone="warn" role="status">
            <AlertTriangle size={18} aria-hidden="true" />
            <p>
              <strong>Earnings on {data.events.map((e) => fmtEventDate(e.date)).join(", ")} fall inside this window.</strong>{" "}
              The forecast is built from price history alone and doesn't price the announcement, so expect a wider real range.
            </p>
          </div>
        )}

        <div className="tabs" role="tablist" aria-label="Views" style={{ marginTop: 0 }}>
          {([["forecast", "Forecast"], ["diagnostics", "Diagnostics"], ["track", "Track record"]] as const).map(([k, l]) => (
            <button key={k} role="tab" id={`tab-${k}`} aria-selected={tab === k} aria-controls={`panel-${k}`}
                    tabIndex={tab === k ? 0 : -1} onClick={() => setTab(k)}
                    onKeyDown={(e) => {
                      const order: Tab[] = ["forecast", "diagnostics", "track"];
                      const i = order.indexOf(tab);
                      const next = e.key === "ArrowRight" ? order[(i + 1) % 3] : e.key === "ArrowLeft" ? order[(i + 2) % 3]
                        : e.key === "Home" ? order[0] : e.key === "End" ? order[2] : null;
                      if (next) {
                        e.preventDefault();
                        setTab(next);
                        requestAnimationFrame(() => document.getElementById(`tab-${next}`)?.focus());
                      }
                    }}>{l}</button>
          ))}
        </div>

        {tab === "forecast" && (
          <div id="panel-forecast" role="tabpanel" aria-labelledby="tab-forecast" className="grid-main stale" data-stale={busy && !!data}>
            <section className="card" aria-labelledby="fan-h">
              <div className="card-head">
                <div>
                  <h2 id="fan-h">Where the price can be</h2>
                  <p className="card-sub">Each band is the forecast range by that bar, scaled from the target forecast. Time runs in trading bars, so closed-market hours take no space.</p>
                </div>
              </div>
              {data ? <FanChart data={data} /> : <div className="skeleton skeleton-chart" aria-label="Loading forecast" />}
            </section>
            <div className="side">
              <LevelsPanel rows={data?.levels ?? null} inputs={levels} previous={prev} flash={flash} onChange={onLevels} busy={busy} />
              {data ? <TrustPanel data={data} /> : <div className="card"><div className="skeleton" style={{ height: 160 }} /></div>}
            </div>
          </div>
        )}
        {tab === "diagnostics" && (
          <div id="panel-diagnostics" role="tabpanel" aria-labelledby="tab-diagnostics" className="stale" data-stale={busy && !!data}>
            {data ? <Diagnostics data={data} /> : <div className="card"><div className="skeleton" style={{ height: 260 }} /></div>}
          </div>
        )}
        {tab === "track" && (
          <div id="panel-track" role="tabpanel" aria-labelledby="tab-track"><TrackRecord /></div>
        )}
        </>}
      </main>

      <footer className="site-foot">
        <span>{section === "forecaster"
          ? "Probabilities, not trade advice. Regular-session bars from Yahoo Finance; forecasts resolve on the last bar close at or before your target."
          : "Research and paper trading only, not investment advice. Prices from Yahoo Finance, Kalshi and Alpaca."}</span>
        <span>{section === "forecaster" && meta ? `Fitted in ${meta.elapsed_s.toFixed(1)} s · ` : ""}LevelEdge v2</span>
      </footer>
    </div>
  );
}
