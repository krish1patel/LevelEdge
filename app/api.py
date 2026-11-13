"""
LevelEdge web API + static dashboard host.

Run locally:
    uvicorn app.api:app --reload --port 7860

Endpoints
---------
POST /api/forecast       fit (or reuse) a model and return levels, curve, fan, diagnostics
GET  /api/track-record   live forward-test calibration from Supabase (empty state without creds)
GET  /api/benchmark      offline v1-vs-v2 benchmark summary (benchmarks/results/summary.json)
GET  /api/health
Everything else serves the built frontend from web/dist.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
os.environ.setdefault("LEVELEDGE_DISABLE_LOGGING", "1")

from leveledge import ALLOWED_INTERVALS, US_EASTERN, forecast  # noqa: E402
from leveledge.sessions import DAILY_MIN, interval_minutes  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
try:  # local runs read keys from .env; the Space gets them as secrets
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
DIST = ROOT / "web" / "dist"
BENCHMARK = ROOT / "benchmarks" / "results" / "summary.json"
SUGGESTED = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "MSFT", "IWM", "BTC-USD", "ETH-USD"]
MAX_LEVELS = 24

app = FastAPI(title="LevelEdge", version="2.0.0")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _clean(obj):
    """JSON-safe: NaN/inf -> None, numpy -> python, timestamps -> ISO."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=US_EASTERN) if dt.tzinfo is None else dt.astimezone(US_EASTERN)


_LEVEL_RE = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*(%?)\s*$")


def _parse_level(raw: str, spot: float) -> float:
    m = _LEVEL_RE.match(raw.replace("$", "").replace(",", ""))
    if not m:
        raise ValueError(f"Can't read level “{raw}”. Use a price like 665 or a move like +1%.")
    num, pct = float(m.group(1)), m.group(2)
    level = spot * (1 + num / 100) if pct else num
    if level <= 0:
        raise ValueError(f"Level “{raw}” is not a positive price.")
    return level


def _history(bars: pd.DataFrame, interval_min: int, horizon_bars: int) -> pd.DataFrame:
    """Recent bars to draw left of the fan: about two sessions, or 1.5x the horizon."""
    per_session = 1 if interval_min >= DAILY_MIN else max(1, round(390 / interval_min))
    n = int(min(len(bars), max(2 * per_session, 1.5 * horizon_bars, 40)))
    return bars.iloc[-n:]


def _trust(model, level_rows) -> dict:
    m = model.metrics or {}
    gate = getattr(model, "gate", None)
    cov = model.interval_coverage()
    w_xgb = model.weights[0]
    if not m.get("blend"):
        verdict, detail = "unvalidated", "Not enough history to validate this horizon out of sample."
    elif w_xgb > 0 and gate and gate.get("passed"):
        gain = 1 - gate["holdout_log_loss_blend"] / gate["holdout_log_loss_baseline"]
        verdict, detail = "model_adds_skill", (
            f"The pattern model beat the baseline by {gain:.1%} log loss on the most recent held-out fold, "
            "so it adjusts this forecast. Across the 4,800-forecast benchmark this layer added no measurable "
            "skill on average, so treat its tilt with care.")
    else:
        verdict, detail = "baseline", ("The pattern model didn't beat the baseline on the held-out fold, "
                                       "so this is the calibrated volatility baseline alone.")
    return {"verdict": verdict, "detail": detail, "coverage80": cov, "xgb_weight": w_xgb,
            "baseline_weight": model.weights[1]}


_EVENT_CACHE: dict[str, tuple[float, list]] = {}


def _earnings_dates(ticker: str) -> list:
    """Upcoming earnings dates from yfinance (cached 12 h; [] on any failure)."""
    hit = _EVENT_CACHE.get(ticker)
    if hit and time.time() - hit[0] < 12 * 3600:
        return hit[1]
    dates: list = []
    if "-" not in ticker:
        try:
            import logging
            import yfinance as yf
            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
            cal = yf.Ticker(ticker).calendar
            raw = cal.get("Earnings Date") if isinstance(cal, dict) else None
            dates = [pd.Timestamp(d).date() for d in (raw or [])]
        except Exception:
            dates = []
    _EVENT_CACHE[ticker] = (time.time(), dates)
    return dates


def _events_in_window(ticker: str, start, end) -> list[dict]:
    """Scheduled events between the forecast time and its resolution (inclusive).

    The model is trained on price history alone, so it doesn't price these.
    """
    s, e = pd.Timestamp(start).date(), pd.Timestamp(end).date()
    return [{"kind": "earnings", "date": d.isoformat()} for d in _earnings_dates(ticker) if s <= d <= e]


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------

class ForecastRequest(BaseModel):
    ticker: str = Field(..., examples=["SPY"])
    target: str = Field(..., description="ISO datetime; interpreted as US/Eastern without an offset")
    interval: str = "15m"
    levels: list[str] = Field(default_factory=lambda: ["-1%", "-0.5%", "+0.5%", "+1%"])
    as_of: str | None = None


@app.post("/api/forecast")
def api_forecast(req: ForecastRequest):
    t0 = time.time()
    ticker = req.ticker.upper().strip()
    if not re.fullmatch(r"[A-Z0-9.\-^=]{1,15}", ticker):
        raise HTTPException(422, "Ticker should look like SPY or BTC-USD.")
    if req.interval not in ALLOWED_INTERVALS:
        raise HTTPException(422, f"Interval must be one of {', '.join(ALLOWED_INTERVALS)}.")
    if len(req.levels) > MAX_LEVELS:
        raise HTTPException(422, f"Up to {MAX_LEVELS} levels at a time.")
    try:
        target, as_of = _parse_dt(req.target), _parse_dt(req.as_of)
    except ValueError:
        raise HTTPException(422, "Target and as-of must be dates like 2026-11-10T16:00.")
    try:
        model = forecast(ticker, target, req.interval, as_of=as_of)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:  # data provider failure
        raise HTTPException(502, f"Couldn't load {ticker} data: {e}")

    spot = model.current_price
    try:
        values = [_parse_level(raw, spot) for raw in req.levels]
    except ValueError as e:
        raise HTTPException(422, str(e))

    rows = []
    if values:
        comp = model.components(values)
        for raw, (_, r) in zip(req.levels, comp.iterrows()):
            lb = model.level_backtest(r["level"])
            rows.append({
                "input": raw, "level": r["level"], "move_pct": (r["level"] / spot - 1) * 100,
                "p_above": r["p"], "p_below": 1 - r["p"], "p_baseline": r["p_baseline"],
                "p_model": r["p_xgb"], "k_sigma": r["k_sigma"],
                "backtest": None if not lb else {
                    "brier_skill": lb["brier_skill"], "base_rate": lb["base_rate"],
                    "blend_brier": lb["blend"]["brier"], "climatology_brier": lb["climatology"]["brier"],
                    "baseline_brier": lb["baseline"]["brier"], "n": lb["blend"]["n"]},
            })

    imin = interval_minutes(req.interval)
    hist = _history(model.bars, imin, model.horizon.bars)
    fan = model.fan()
    curve = model.curve(n=161, span_sigma=3.5)
    q = model.quantiles((0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
    rel = model.reliability
    imp = model.feature_importance(10)
    summary = model.summary()
    hz = model.horizon
    return JSONResponse(_clean({
        "meta": {
            "ticker": ticker, "interval": req.interval, "spot": spot,
            "anchor_time": hz.anchor_time, "effective_target": hz.effective_target,
            "requested_target": target, "horizon_bars": hz.bars, "sessions_crossed": hz.sessions_crossed,
            "sigma_pct": (math.exp(model.live_scale) - 1) * 100, "is_backtest": as_of is not None,
            "as_of": as_of or datetime.now(tz=US_EASTERN), "is_crypto": model.is_crypto,
            "elapsed_s": time.time() - t0,
        },
        "levels": rows,
        "events": _events_in_window(ticker, hz.anchor_time, hz.effective_target),
        "quantiles": {f"q{int(round(k * 100)):02d}": v for k, v in q.items()},
        "history": [{"t": t, "o": r.Open, "h": r.High, "l": r.Low, "c": r.Close}
                    for t, r in hist.iterrows()],
        "fan": fan.to_dict(orient="records"),
        "curve": curve[["level", "p_above", "p_baseline"]].to_dict(orient="records"),
        "trust": _trust(model, rows),
        "diagnostics": {
            "metrics": summary["metrics"], "regime_metrics": summary["regime_metrics"],
            "weights": summary["weights"], "gate": summary["gate"],
            "vol_half_life_bars": summary["vol_half_life_bars"],
            "vol_inflation": summary["vol_inflation"], "rows": summary["rows"],
            "history_start": summary["history_start"], "history_end": summary["history_end"],
            "reliability": [] if rel is None else rel.to_dict(orient="records"),
            "importance": [{"name": k, "share": v} for k, v in imp.items()],
        },
    }))


# ---------------------------------------------------------------------------
# track record (Supabase)
# ---------------------------------------------------------------------------

_TR_CACHE: dict = {}
_TR_LOCK = threading.Lock()
_TR_TTL = 600


def _supabase_rows(table: str, since: datetime) -> pd.DataFrame:
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    cols = "logged_at_utc,ticker,interval,target_datetime,price_level,current_price,prediction,model_type,outcome_price"
    out, start = [], 0
    while True:
        res = (sb.table(table).select(cols).not_.is_("outcome_price", "null")
               .gte("logged_at_utc", since.isoformat()).range(start, start + 999).execute())
        out.extend(res.data)
        if len(res.data) < 1000 or len(out) >= 50000:
            break
        start += 1000
    return pd.DataFrame(out)


def _score(df: pd.DataFrame) -> dict:
    y, p = df["y"].to_numpy(float), np.clip(df["prediction"].to_numpy(float), 1e-4, 1 - 1e-4)
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    bins = np.clip((p * 10).astype(int), 0, 9)
    rel = [{"forecast": float(p[bins == b].mean()), "observed": float(y[bins == b].mean()), "n": int((bins == b).sum())}
           for b in range(10) if (bins == b).any()]
    ece = sum(abs(r["forecast"] - r["observed"]) * r["n"] for r in rel) / len(y)
    return {"n": int(len(y)), "brier": brier, "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier_skill_vs_base_rate": 1 - brier / max(base * (1 - base), 1e-9), "ece": float(ece),
            "base_rate": float(base), "reliability": rel}


@app.get("/api/track-record")
def api_track_record(days: int = 60, table: str = "forwardtest_logs"):
    if table not in {"forwardtest_logs", "logs"}:
        raise HTTPException(422, "Unknown table.")
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        return {"status": "not_configured",
                "message": "Set SUPABASE_URL and SUPABASE_KEY (HF Space secrets) to show the live track record."}
    key = (table, days)
    with _TR_LOCK:
        hit = _TR_CACHE.get(key)
        if hit and time.time() - hit[0] < _TR_TTL:
            return hit[1]
    try:
        df = _supabase_rows(table, datetime.now(tz=US_EASTERN) - timedelta(days=days))
    except Exception as e:
        return {"status": "error", "message": f"Couldn't read Supabase: {e}"}
    if df.empty:
        payload = {"status": "empty", "message": "No resolved forecasts in this window yet."}
    else:
        df["model_type"] = df["model_type"].fillna("xgboost_classifier")
        df["y"] = (df["outcome_price"].astype(float) > df["price_level"].astype(float)).astype(int)
        df["day"] = pd.to_datetime(df["target_datetime"], utc=True).dt.tz_convert(US_EASTERN).dt.date.astype(str)
        by_model = {mt: _score(g) for mt, g in df.groupby("model_type")}
        daily = (df.groupby(["model_type", "day"])
                 .apply(lambda g: pd.Series({"brier": float(np.mean((g["prediction"] - g["y"]) ** 2)), "n": len(g)}))
                 .reset_index().to_dict(orient="records"))
        by_interval = [{"model_type": mt, "interval": iv, **{k: v for k, v in _score(g).items() if k != "reliability"}}
                       for (mt, iv), g in df.groupby(["model_type", "interval"])]
        payload = _clean({"status": "ok", "table": table, "days": days, "rows": int(len(df)),
                          "first": df["day"].min(), "last": df["day"].max(),
                          "by_model": by_model, "daily": daily, "by_interval": by_interval})
    with _TR_LOCK:
        _TR_CACHE[key] = (time.time(), payload)
    return payload


@app.get("/api/benchmark")
def api_benchmark():
    if not BENCHMARK.exists():
        return {"status": "missing", "message": "Run python benchmarks/compare.py to generate the offline benchmark."}
    return {"status": "ok", **json.loads(BENCHMARK.read_text())}


# ---------------------------------------------------------------------------
# paper portfolio
# ---------------------------------------------------------------------------
RESULTS = ROOT / "benchmarks" / "results"
_PF_LOCK = threading.Lock()
_PF_CACHE: dict = {}


def _cached(key: str, ttl: float, fn):
    with _PF_LOCK:
        hit = _PF_CACHE.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
    val = fn()
    with _PF_LOCK:
        _PF_CACHE[key] = (time.time(), val)
    return val


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def _live_paper() -> dict:
    from leveledge.broker import AlpacaBroker, credentials

    creds = credentials()
    if not creds:
        return {"status": "not_configured",
                "message": "Add ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (paper keys) to show the live paper account."}
    try:
        b = AlpacaBroker(*creds)
        acct = b.account()
        hist = b.portfolio_history(period="1A", timeframe="1D")
        positions = b.positions()
        orders = b.orders(status="all", limit=25)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"Couldn't read the Alpaca paper account: {e}"}
    eq = pd.Series(hist.get("equity") or [], index=pd.to_datetime(hist.get("timestamp") or [], unit="s"), dtype=float)
    eq = eq[eq > 0]
    curve = []
    if len(eq) >= 2:
        eq.index = eq.index.tz_localize("UTC").tz_convert(US_EASTERN).normalize().tz_localize(None)
        start = eq.index[0]
        try:
            import yfinance as yf
            px = pd.concat({s: yf.Ticker(s).history(start=start - timedelta(days=5), interval="1d", auto_adjust=True)["Close"]
                            for s in ("SPY", "IEF")}, axis=1)
            px.index = px.index.tz_localize(None).normalize()
            px = px.reindex(eq.index).ffill()
            r = px.pct_change().fillna(0.0)
            spy = (1 + r["SPY"]).cumprod()
            sixty = (1 + 0.6 * r["SPY"] + 0.4 * r["IEF"]).cumprod()
        except Exception:  # noqa: BLE001
            spy = sixty = pd.Series(np.nan, index=eq.index)
        book = eq / eq.iloc[0]
        curve = [{"date": d.strftime("%Y-%m-%d"), "book": float(book[d]), "spy_buy_hold": _num(spy.get(d)),
                  "sixty_forty": _num(sixty.get(d))} for d in eq.index]
    return _clean({
        "status": "ok", "paper": b.is_paper,
        "account": {"equity": float(acct["equity"]), "cash": float(acct["cash"]),
                    "last_equity": float(acct.get("last_equity") or acct["equity"])},
        "positions": [{"symbol": p["symbol"], "qty": float(p["qty"]), "market_value": float(p["market_value"]),
                       "unrealized_pl": float(p["unrealized_pl"]), "weight": float(p["market_value"]) / float(acct["equity"])}
                      for p in positions],
        "orders": [{"symbol": o["symbol"], "side": o["side"], "status": o["status"],
                    "notional": _num(o.get("notional")), "qty": _num(o.get("qty")),
                    "filled_avg_price": _num(o.get("filled_avg_price")), "submitted_at": o.get("submitted_at"),
                    "filled_at": o.get("filled_at")} for o in orders],
        "curve": curve,
    })


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


@app.get("/api/portfolio")
def api_portfolio():
    from leveledge import live

    backtest = _read_json(RESULTS / "paper_portfolio" / "summary.json")
    try:
        targets = _cached("targets", 1800, live.current_targets)
        targets_status = "ok"
    except Exception as e:  # noqa: BLE001
        targets, targets_status = None, f"Couldn't compute today's targets: {e}"
    return _clean({"strategy": live.STRATEGY_ID,
                   "design": {"core": live.CORE, "core_share": live.CORE_SHARE, "trend_share": live.TREND_SHARE},
                   "targets": targets, "targets_status": targets_status,
                   "backtest": backtest, "live": _cached("live", 300, _live_paper)})


FORWARD_LOG_URL = os.environ.get(
    "FORWARD_LOG_URL", "https://raw.githubusercontent.com/krish1patel/LevelEdge/forward-log/outlook.jsonl")


def _forward_record() -> dict:
    """Score the public forward log (committed daily before outcomes exist) against S&P 500 closes."""
    import urllib.request

    from leveledge import outlook

    try:
        with urllib.request.urlopen(FORWARD_LOG_URL, timeout=10) as r:
            entries = [json.loads(x) for x in r.read().decode().splitlines() if x.strip()]
    except Exception as e:  # noqa: BLE001
        return {"status": "unavailable", "message": f"Forward log not readable yet ({e.__class__.__name__}).", "url": FORWARD_LOG_URL}
    if not entries:
        return {"status": "empty", "url": FORWARD_LOG_URL}
    import yfinance as yf
    h = yf.Ticker("^GSPC").history(start=entries[0]["as_of_close"], interval="1d", auto_adjust=False)
    closes = pd.Series(h["Close"].to_numpy(), index=pd.DatetimeIndex(h.index).tz_localize(None).normalize())
    closes = closes[closes.index < pd.Timestamp(datetime.now(tz=US_EASTERN).date())] if not _session_closed() else closes
    return {"status": "ok", "url": FORWARD_LOG_URL, "view_url": _github_view(FORWARD_LOG_URL), **outlook.score_forward(entries, closes)}


def _github_view(raw: str) -> str:
    m = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", raw)
    return f"https://github.com/{m[1]}/{m[2]}/commits/{m[3]}/{m[4]}" if m else raw


def _session_closed() -> bool:
    from leveledge.safety import last_completed_session

    now = datetime.now(tz=US_EASTERN)
    return last_completed_session(now) == pd.Timestamp(now.date())


def _outlook() -> dict:
    from leveledge import live, outlook

    try:
        weights = _cached("targets", 1800, live.current_targets)["weights"]
    except Exception:  # noqa: BLE001
        weights = None
    out = outlook.market_outlook(book_weights=weights)
    out["forward"] = _forward_record()
    study = _read_json(RESULTS / "regime" / "summary.json")
    if study:
        a = study["A_distribution"]
        out["record"] = {str(h): {per: {k: a[str(h)][per]["combo"][k] for k in ("cover80", "cover95", "crps_skill_vs_hist")}
                                  | {"n": a[str(h)][per]["n"]} for per in ("dev", "holdout")} for h in (5, 21, 63)}
    return out


@app.get("/api/outlook")
def api_outlook():
    try:
        return _clean({"status": "ok", **_cached("outlook", 1800, _outlook)})
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"Couldn't compute the market outlook: {e}"}


@app.get("/api/research")
def api_research():
    """What was tested and what failed, from the saved benchmark results."""
    k = _read_json(RESULTS / "kalshi" / "summary.json")
    mk = _read_json(RESULTS / "kalshi" / "maker_summary.json")
    kh = _read_json(RESULTS / "kalshi_horizon" / "summary.json")
    er = _read_json(RESULTS / "etf_research" / "summary.json")
    et = _read_json(RESULTS / "etf_trend" / "summary.json")
    v1 = _read_json(BENCHMARK)
    out = {}
    if v1:
        out["forecaster"] = {"forecasts": v1.get("forecasts"), "overall": v1.get("overall"),
                             "brier_diff_v2_minus_v1": v1.get("brier_diff_v2_minus_v1")}
    if k:
        out["kalshi_taker"] = {"rows": k["rows"], "events": k["events"], "period": k["period"],
                               "liquid": {"rows": k["liquid"]["rows"], "calibration": k["liquid"]["calibration"],
                                          "brier_v2_minus_market": k["liquid"]["brier_v2_minus_market"],
                                          "primary": k["liquid"]["strategies"]["p_v2|theta=0.02|fee=0.07"]},
                               "by_series_x_offset": k.get("calibration_by_series_x_offset")}
    if mk:
        res = mk["results"]
        out["kalshi_maker"] = {"variants": len(res), "rows": mk["rows"],
                               "results": [{"variant": key, "model_avg_pnl_cents": v["model"].get("avg_pnl_cents"),
                                            "model_ci95": v["model"].get("avg_pnl_cents_ci95"), "model_fills": v["model"].get("fills"),
                                            "mid_avg_pnl_cents": v["control_mid"].get("avg_pnl_cents")} for key, v in res.items()]}
    if kh:
        out["kalshi_horizon"] = {"rows": kh["rows"], "events": kh["events"], "period": kh["period"],
                                 "liquid": {"calibration": kh["liquid"]["calibration"],
                                            "brier_v2_minus_market": kh["liquid"]["brier_v2_minus_market"],
                                            "primary": kh["liquid"]["strategies"]["p_v2|theta=0.02|fee=0.07"]}}
    if et:
        out["etf_trend"] = {"period": et["period"], "metrics": et["metrics"], "vol_engine": et["vol_engine"]["mean_qlike"],
                            "random_picker": et["random_picker"]}
    var_csv = RESULTS / "etf_research" / "variants.csv"
    if er and var_csv.exists():
        vdf = pd.read_csv(var_csv)
        out["etf_search_points"] = [{"variant": r.variant, "dev": float(r.dev_sharpe), "hold": float(r.hold_sharpe)}
                                    for r in vdf.itertuples()]
    am, af = _read_json(RESULTS / "alpha_research_main" / "summary.json"), _read_json(RESULTS / "alpha_research_fresh" / "summary.json")
    if am:
        pick = lambda r: {"forecast": r["forecast"], "metrics": {k: r["metrics"][k] for k in ("rule", "ridge", "spline", "mixed", "xgb", "auto") if k in r["metrics"]},
                          "sharpe_vs_rule": r["sharpe_vs_rule"]}
        out["return_models"] = {"main": pick(am), "fresh": pick(af) if af else None,
                                "reward_vs_levered_sixty_forty": am.get("reward_vs_levered_sixty_forty")}
    cr = _read_json(RESULTS / "construction" / "summary.json")
    if cr:
        keep = ("vt6040", "rp", "rp_trend", "vt6040_trend", "current_book", "sixty_forty")
        out["construction"] = {"selected": cr["selected"], "dev": {k: cr["dev"][k] for k in keep}, "holdout": {k: cr["holdout"][k] for k in keep},
                               "holdout_sharpe_vs": cr["holdout_sharpe_vs"],
                               "frontier": [f for f in cr["frontier"] if f["borrow_spread"] == 0.02 and f["family"] != "vt6040_targeted"]}
    rw = _read_json(RESULTS / "reward" / "summary.json")
    if rw:
        out["reward"] = {"selected": rw["selected"], "dd_limit": rw["dd_limit"], "holdout": rw["holdout"], "full": rw["full"]}
    sf = _read_json(RESULTS / "stock_factors" / "summary.json")
    if sf:
        out["stock_factors"] = {"period": sf["period"], "coverage": sf["coverage"], "dev_vs_ew": sf["dev_vs_ew"],
                                "holdout_vs_ew": sf["holdout_vs_ew"],
                                "dev": {k: sf["dev"][k] for k in ("mom_12_1", "mom_lowvol", "lowvol", "rev_1m", "ew_members")},
                                "holdout": {k: sf["holdout"][k] for k in ("mom_12_1", "mom_lowvol", "lowvol", "rev_1m", "ew_members")}}
    lit = _read_json(RESULTS / "literature" / "summary.json")
    if lit:
        names = ("book_orig_trend", "book_mtp", "book_daa_g12", "book_gtaa5", "daa_g12", "tom", "overnight_0bp", "overnight_1bp", "sixty_forty")
        out["literature"] = {"period": lit["period"], "selected": lit["selected_sleeve"],
                             "full": {k: lit["full"][k] for k in names}, "dev": {k: lit["dev"][k] for k in names},
                             "holdout": {k: lit["holdout"][k] for k in names},
                             "holdout_vs_current_book": lit["holdout_vs_current_book"], "deflated": lit["deflated_sharpe_selected"],
                             "per_day_bp": lit["per_day_in_market_bp"], "bands": lit["bands"], "band_selected": lit["band_selected"]}
    if er:
        out["etf_search"] = {k2: er[k2] for k2 in ("period", "trials", "selected", "deflated_sharpe_dev", "dev", "holdout",
                                                    "holdout_sharpe_vs", "rank_stability")}
    return _clean(out)


@app.get("/api/health")
def api_health():
    return {"ok": True, "intervals": ALLOWED_INTERVALS, "suggested": SUGGESTED,
            "now": datetime.now(tz=US_EASTERN).isoformat()}


# ---------------------------------------------------------------------------
# static frontend
# ---------------------------------------------------------------------------

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = DIST / path
        if path and f.is_file() and DIST in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")
