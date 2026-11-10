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
