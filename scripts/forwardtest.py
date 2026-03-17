"""
Forward Test: QQQ + SPY
========================
Runs predictions 3x per day (10am, 12pm, 2pm ET) on every interval across
6 strikes per ticker (±1%, ±1.5%, ±2%, ±3% of current price). Target is
always 4pm same day.

Signal rules:
  Bullish : 15m >= 0.75 AND 30m >= 0.75 AND (5m >= 0.75 OR 2m >= 0.75)
  Bearish : 15m <= 0.25 AND 30m <= 0.25 AND (5m <= 0.25 OR 2m <= 0.25)

Trade selection (max 1 per prediction window):
  - Strongest signal = highest average across all qualifying intervals
  - Buy the lowest-strike call (or highest-strike put) that:
      * costs <= $200 per contract
      * has Black-Scholes EV >= $75
  - Stop loss: close if option loses $50 from entry price
  - Max contract cost: $200

Logging:
  - All predictions → Supabase forwardtest_logs
  - Trades → Alpaca paper account

Usage:
  python forward_test.py                    # run now (manual)
  python forward_test.py --dry-run          # predictions only, no trades

Schedule via Task Scheduler or GitHub Actions cron:
  10:00 AM ET  →  python forward_test.py
  12:00 PM ET  →  python forward_test.py
   2:00 PM ET  →  python forward_test.py
"""
from __future__ import annotations

import math
import os
import time
import warnings
from datetime import datetime, date, timedelta
from typing import Optional

warnings.filterwarnings("ignore", message="Precision is ill-defined")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

import requests
import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

from leveledge import Predictor
from leveledge.constants import ALLOWED_INTERVALS, US_EASTERN

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TICKERS = ["QQQ", "SPY"]
STRIKE_PCT = (-3.0, -2.0, -1.5, -1.0, 1.0, 1.5, 2.0, 3.0)   # ±1, 1.5, 2, 3 %
INTERVALS  = [i for i in ALLOWED_INTERVALS if i not in ("1m", "1d", "90m")]

BULLISH_THRESHOLD = 0.75
BEARISH_THRESHOLD = 0.25
REQUIRED_CORE     = ("15m", "30m")          # both must clear threshold
REQUIRED_CONFIRM  = ("5m", "2m")            # at least one must clear

MAX_OPTION_COST   = 200.0   # $ per contract (premium × 100)
MIN_EV            = 75.0    # $ minimum Black-Scholes expected value
STOP_LOSS_DOLLARS = 50.0    # $ loss from entry triggers close

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"
ALPACA_DATA_BASE  = "https://data.alpaca.markets"

TABLE_NAME = "forwardtest_logs"

# ---------------------------------------------------------------------------
# Helpers: Black-Scholes
# ---------------------------------------------------------------------------

def _bs_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. T in years."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price. T in years."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = _bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def expected_value(
    option_type: str,       # "call" or "put"
    model_prob: float,      # probability underlying finishes ITM
    S: float,               # current underlying price
    K: float,               # strike
    T: float,               # time to expiry in years
    r: float,               # risk-free rate
    sigma: float,           # implied volatility
    premium: float,         # option premium paid (per share)
) -> float:
    """
    Simple EV:
      EV = P(ITM) × (intrinsic at expiry assuming underlying just clears K)
         - premium
    For a call, we use S_end = K + (K × 0.005) as a conservative payoff estimate.
    For a put,  we use S_end = K - (K × 0.005).
    """
    if option_type == "call":
        # Conservative: underlying finishes just 0.5% above strike
        payoff_per_share = K * 0.005
    else:
        payoff_per_share = K * 0.005

    ev = model_prob * payoff_per_share * 100 - premium * 100
    return ev


# ---------------------------------------------------------------------------
# Helpers: Alpaca
# ---------------------------------------------------------------------------

def _alpaca_headers(key: str, secret: str) -> dict:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "accept": "application/json",
        "content-type": "application/json",
    }


def get_option_chain(
    ticker: str,
    expiry: date,
    option_type: str,       # "call" or "put"
    key: str,
    secret: str,
) -> list[dict]:
    """Fetch option chain from Alpaca for given ticker, expiry, and type."""
    url = f"{ALPACA_DATA_BASE}/v1beta1/options/snapshots/{ticker}"
    params = {
        "expiration_date": expiry.isoformat(),
        "type": option_type,
        "limit": 100,
        "feed": "indicative",
    }
    resp = requests.get(url, headers=_alpaca_headers(key, secret), params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    snapshots = data.get("snapshots", {})
    results = []
    for symbol, snap in snapshots.items():
        greeks  = snap.get("greeks") or {}
        quote   = snap.get("latestQuote") or {}
        details = snap.get("details") or {}
        ask     = float(quote.get("ap", 0) or 0)
        bid     = float(quote.get("bp", 0) or 0)
        mid     = (ask + bid) / 2 if ask and bid else ask or bid
        iv      = float(greeks.get("impliedVolatility") or greeks.get("iv") or 0)
        strike  = float(details.get("strikePrice") or 0)
        if mid > 0 and strike > 0:
            results.append({
                "symbol": symbol,
                "strike": strike,
                "mid":    mid,
                "iv":     iv,
                "bid":    bid,
                "ask":    ask,
            })
    return sorted(results, key=lambda x: x["strike"])


def place_order(
    option_symbol: str,
    side: str,              # "buy" or "sell"
    qty: int,
    key: str,
    secret: str,
) -> dict:
    """Place a market order on Alpaca paper account."""
    url = f"{ALPACA_PAPER_BASE}/v2/orders"
    body = {
        "symbol":        option_symbol,
        "qty":           str(qty),
        "side":          side,
        "type":          "market",
        "time_in_force": "day",
    }
    resp = requests.post(url, headers=_alpaca_headers(key, secret), json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()


def place_stop_order(
    option_symbol: str,
    stop_price: float,
    key: str,
    secret: str,
) -> dict:
    """Place a stop sell order to close a long option position."""
    url = f"{ALPACA_PAPER_BASE}/v2/orders"
    body = {
        "symbol":        option_symbol,
        "qty":           "1",
        "side":          "sell",
        "type":          "stop",
        "time_in_force": "day",
        "stop_price":    str(round(stop_price, 2)),
    }
    resp = requests.post(url, headers=_alpaca_headers(key, secret), json=body, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_positions(key: str, secret: str) -> list[dict]:
    """Get all open positions from Alpaca paper account."""
    url = f"{ALPACA_PAPER_BASE}/v2/positions"
    resp = requests.get(url, headers=_alpaca_headers(key, secret), timeout=10)
    resp.raise_for_status()
    return resp.json()


def close_position(symbol: str, key: str, secret: str) -> dict:
    """Close a position by symbol."""
    url = f"{ALPACA_PAPER_BASE}/v2/positions/{symbol}"
    resp = requests.delete(url, headers=_alpaca_headers(key, secret), timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Helpers: Supabase
# ---------------------------------------------------------------------------

def log_prediction(
    sb,
    ticker: str,
    interval: str,
    prediction: float,
    current_price: float,
    price_level: float,
    target_datetime: datetime,
    candles_ahead: int,
    signal_direction: Optional[str],  # "bullish", "bearish", or None
    model_auc: float,
    run_time: datetime,
) -> None:
    try:
        sb.table(TABLE_NAME).insert({
            "logged_at_utc":    run_time.isoformat(),
            "ticker":           ticker,
            "interval":         interval,
            "prediction":       float(prediction),
            "current_price":    float(current_price),
            "price_level":      float(price_level),
            "target_datetime":  target_datetime.isoformat(),
            "candles_ahead":    candles_ahead,
            "signal_direction": signal_direction,
            "model_auc":        float(model_auc) if model_auc else None,
        }).execute()
    except Exception as e:
        print(f"  [warn] Supabase log failed: {e}")


def log_trade(
    sb,
    ticker: str,
    option_symbol: str,
    direction: str,
    strike: float,
    premium: float,
    ev: float,
    signal_strength: float,
    alpaca_order_id: str,
    run_time: datetime,
) -> None:
    try:
        sb.table("forwardtest_trades").insert({
            "logged_at_utc":   run_time.isoformat(),
            "ticker":          ticker,
            "option_symbol":   option_symbol,
            "direction":       direction,
            "strike":          strike,
            "premium":         premium,
            "ev":              ev,
            "signal_strength": signal_strength,
            "alpaca_order_id": alpaca_order_id,
        }).execute()
    except Exception as e:
        print(f"  [warn] Supabase trade log failed: {e}")


# ---------------------------------------------------------------------------
# Core logic: run predictions for one ticker
# ---------------------------------------------------------------------------

def run_predictions_for_ticker(
    ticker: str,
    target_dt: datetime,
    end_dt: datetime,
    strike_pct: tuple,
    intervals: list,
    sb,
    run_time: datetime,
) -> dict:
    """
    Run predictions across all intervals and strikes for one ticker.

    Returns a dict with keys:
      current_price, predictions_by_interval, signal_direction, signal_strength
    """

    # Resolve current price from first successful Predictor init
    current_price = None
    for probe_interval in intervals:
        try:
            probe = Predictor(ticker, target_dt, probe_interval, 999.0, end_datetime=end_dt)
            current_price = probe.current_price
            break
        except Exception:
            continue

    if current_price is None:
        print(f"  [{ticker}] Could not resolve current price — skipping")
        return {}

    price_levels = [round(current_price * (1 + pct / 100.0), 2) for pct in strike_pct]
    print(f"  [{ticker}] Current price: ${current_price:.2f}, levels: {price_levels}")

    # predictions_by_interval[interval] = avg prediction across all strikes
    predictions_by_interval: dict[str, float] = {}

    for interval in intervals:
        interval_preds = []
        for price in price_levels:
            try:
                p = Predictor(ticker, target_dt, interval, price, end_datetime=end_dt)
                p.train_xgb()
                pred = p.predict_xgb()
                interval_preds.append(pred)

                # Log to Supabase
                auc = p.xgb_expected_model_metrics[0] if hasattr(p, "xgb_expected_model_metrics") else None
                log_prediction(
                    sb, ticker, interval, pred, current_price, price,
                    target_dt, p.candles_ahead, None, auc, run_time
                )
            except Exception as e:
                print(f"  [{ticker}] {interval} ${price}: {e}")

        if interval_preds:
            predictions_by_interval[interval] = sum(interval_preds) / len(interval_preds)

    # Evaluate signal
    signal_direction, signal_strength = evaluate_signal(predictions_by_interval)

    print(f"  [{ticker}] Interval averages: { {k: f'{v:.3f}' for k, v in predictions_by_interval.items()} }")
    print(f"  [{ticker}] Signal: {signal_direction or 'NONE'}" +
          (f" (strength={signal_strength:.3f})" if signal_direction else ""))

    return {
        "ticker":                   ticker,
        "current_price":            current_price,
        "price_levels":             price_levels,
        "predictions_by_interval":  predictions_by_interval,
        "signal_direction":         signal_direction,
        "signal_strength":          signal_strength,
    }


# ---------------------------------------------------------------------------
# Signal evaluation
# ---------------------------------------------------------------------------

def evaluate_signal(
    predictions_by_interval: dict[str, float]
) -> tuple[Optional[str], float]:
    """
    Returns (direction, strength) or (None, 0.0) if no signal.
    direction: "bullish" | "bearish"
    strength: average of all qualifying interval predictions
    """
    def check(threshold_fn, preds):
        core_ok    = all(threshold_fn(preds.get(i, 0.5)) for i in REQUIRED_CORE)
        confirm_ok = any(threshold_fn(preds.get(i, 0.5)) for i in REQUIRED_CONFIRM)
        return core_ok and confirm_ok

    bullish_fn = lambda p: p >= BULLISH_THRESHOLD
    bearish_fn = lambda p: p <= BEARISH_THRESHOLD

    if check(bullish_fn, predictions_by_interval):
        qualifying = [v for k, v in predictions_by_interval.items()
                      if bullish_fn(v)]
        return "bullish", sum(qualifying) / len(qualifying)

    if check(bearish_fn, predictions_by_interval):
        qualifying = [v for k, v in predictions_by_interval.items()
                      if bearish_fn(v)]
        # Invert so higher = stronger bearish for ranking
        return "bearish", 1.0 - (sum(qualifying) / len(qualifying))

    return None, 0.0


# ---------------------------------------------------------------------------
# Trade selection
# ---------------------------------------------------------------------------

def select_trade(
    ticker_result: dict,
    target_dt: datetime,
    alpaca_key: str,
    alpaca_secret: str,
    risk_free_rate: float = 0.05,
) -> Optional[dict]:
    """
    Given a ticker result with a confirmed signal, find the best option to buy.

    For bullish: buy call at lowest strike costing <= $200 with EV >= $75
    For bearish: buy put at highest strike costing <= $200 with EV >= $75

    Returns trade dict or None if no qualifying option found.
    """
    direction   = ticker_result["signal_direction"]
    S           = ticker_result["current_price"]
    ticker      = ticker_result["ticker"]
    strength    = ticker_result["signal_strength"]

    option_type = "call" if direction == "bullish" else "put"
    expiry      = target_dt.date()

    # Time to expiry in years
    now_et = datetime.now(tz=US_EASTERN)
    T = max((target_dt - now_et).total_seconds() / (365.25 * 24 * 3600), 1 / (365.25 * 24))

    try:
        chain = get_option_chain(ticker, expiry, option_type, alpaca_key, alpaca_secret)
    except Exception as e:
        print(f"  [{ticker}] Could not fetch options chain: {e}")
        return None

    if not chain:
        print(f"  [{ticker}] Empty options chain for {expiry}")
        return None

    # For calls: scan strikes ascending (lowest first)
    # For puts:  scan strikes descending (highest first, closest to ATM)
    candidates = chain if option_type == "call" else list(reversed(chain))

    for option in candidates:
        strike  = option["strike"]
        premium = option["mid"]     # per share
        iv      = option["iv"]
        cost    = premium * 100     # per contract

        if cost > MAX_OPTION_COST:
            continue
        if iv <= 0:
            # Fall back to ATM IV estimate if missing
            iv = 0.20

        # Model probability: use the avg prediction for the interval closest
        # to this strike's distance from current price
        model_prob = ticker_result["signal_strength"]
        # Adjust: for bullish, prob = signal_strength; for bearish, invert back
        if direction == "bearish":
            model_prob = 1.0 - strength

        ev = expected_value(option_type, model_prob, S, strike, T,
                            risk_free_rate, iv, premium)

        print(f"  [{ticker}] {option_type.upper()} K={strike:.2f} "
              f"prem=${premium:.2f} cost=${cost:.2f} IV={iv:.1%} EV=${ev:.2f}")

        if ev >= MIN_EV:
            return {
                "ticker":         ticker,
                "option_symbol":  option["symbol"],
                "option_type":    option_type,
                "direction":      direction,
                "strike":         strike,
                "premium":        premium,
                "cost":           cost,
                "iv":             iv,
                "ev":             ev,
                "signal_strength": strength,
            }

    print(f"  [{ticker}] No qualifying option found (EV or cost filter)")
    return None


# ---------------------------------------------------------------------------
# Stop loss monitor
# ---------------------------------------------------------------------------

def monitor_stop_losses(alpaca_key: str, alpaca_secret: str) -> None:
    """
    Check all open option positions. Close any that have lost >= $50
    from their average entry cost.
    """
    try:
        positions = get_positions(alpaca_key, alpaca_secret)
    except Exception as e:
        print(f"  [stop-loss] Could not fetch positions: {e}")
        return

    for pos in positions:
        symbol      = pos.get("symbol", "")
        unrealized  = float(pos.get("unrealized_pl", 0) or 0)
        asset_class = pos.get("asset_class", "")

        # Only manage option positions opened by this system
        if asset_class != "us_option":
            continue

        if unrealized <= -STOP_LOSS_DOLLARS:
            print(f"  [stop-loss] Closing {symbol} — P&L ${unrealized:.2f}")
            try:
                close_position(symbol, alpaca_key, alpaca_secret)
                print(f"  [stop-loss] Closed {symbol}")
            except Exception as e:
                print(f"  [stop-loss] Failed to close {symbol}: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> None:
    load_dotenv()

    alpaca_key    = os.environ["ALPACA_API_KEY"]
    alpaca_secret = os.environ["ALPACA_SECRET_KEY"]
    supabase_url  = os.environ["SUPABASE_URL"]
    supabase_key  = os.environ["SUPABASE_KEY"]

    sb = create_client(supabase_url, supabase_key)

    now_et     = datetime.now(tz=US_EASTERN)
    run_time   = now_et
    today      = now_et.date()
    target_dt  = datetime(today.year, today.month, today.day, 16, 0, tzinfo=US_EASTERN)
    end_dt     = now_et

    if end_dt >= target_dt:
        print("Market is closed or target time has passed. Exiting.")
        return

    print(f"\n{'='*60}")
    print(f"Forward Test Run — {now_et.strftime('%Y-%m-%d %H:%M ET')}")
    print(f"Target: 4:00 PM ET | Dry run: {dry_run}")
    print(f"{'='*60}\n")

    # Run predictions for all tickers
    ticker_results = []
    for ticker in TICKERS:
        print(f"Running predictions for {ticker}...")
        result = run_predictions_for_ticker(
            ticker, target_dt, end_dt, STRIKE_PCT,
            INTERVALS, sb, run_time
        )
        if result and result.get("signal_direction"):
            ticker_results.append(result)
        print()

    if not ticker_results:
        print("No signals fired this window. Done.")
        return

    # Rank by signal strength, pick strongest
    ticker_results.sort(key=lambda x: x["signal_strength"], reverse=True)
    best = ticker_results[0]

    signal_summary = [(r['ticker'], r['signal_direction'], f"{r['signal_strength']:.3f}") for r in ticker_results]
    print(f"Signal(s) fired: {signal_summary}")
    print(f"Selected: {best['ticker']} ({best['signal_direction']}, strength={best['signal_strength']:.3f})\n")

    # Find qualifying option
    trade = select_trade(best, target_dt, alpaca_key, alpaca_secret)

    if not trade:
        print("No trade placed — no qualifying option passed EV and cost filters.")
        return

    print(f"\nTrade selected:")
    print(f"  Ticker:  {trade['ticker']}")
    print(f"  Symbol:  {trade['option_symbol']}")
    print(f"  Type:    {trade['option_type'].upper()}")
    print(f"  Strike:  ${trade['strike']:.2f}")
    print(f"  Premium: ${trade['premium']:.2f} (cost: ${trade['cost']:.2f})")
    print(f"  IV:      {trade['iv']:.1%}")
    print(f"  EV:      ${trade['ev']:.2f}")

    if dry_run:
        print("\n[DRY RUN] Trade not placed.")
        return

    # Place order on Alpaca paper account
    try:
        order = place_order(trade["option_symbol"], "buy", 1, alpaca_key, alpaca_secret)
        order_id = order.get("id", "unknown")
        print(f"\nOrder placed! Alpaca order ID: {order_id}")

        # Place stop loss order immediately at entry - $0.50/share ($50/contract)
        stop_price = round(trade["premium"] - (STOP_LOSS_DOLLARS / 100), 2)
        if stop_price > 0:
            stop_order = place_stop_order(trade["option_symbol"], stop_price, alpaca_key, alpaca_secret)
            print(f"Stop loss placed at ${stop_price:.2f} (order ID: {stop_order.get('id')})")

        # Log trade to Supabase
        log_trade(
            sb,
            trade["ticker"],
            trade["option_symbol"],
            trade["direction"],
            trade["strike"],
            trade["premium"],
            trade["ev"],
            trade["signal_strength"],
            order_id,
            run_time,
        )

    except Exception as e:
        print(f"\nFailed to place order: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forward test: predictions + paper trading")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run predictions and signal detection only, do not place trades")
    args = parser.parse_args()
    main(dry_run=args.dry_run)