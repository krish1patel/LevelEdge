# Product


## Platform

web

## Stack

New web frontend plus a Python API, replacing the Streamlit UI. It must still ship as the single Docker container that `.github/workflows/sync_to_hf.yml` deploys to the HuggingFace Space on every push to `main`, serving on port 7860. Frontend: React + Vite; FastAPI serves the JSON API and the built static frontend.

## Users

Primary user: an intraday trader picking price levels during the US session, for example choosing option strikes or prediction-market brackets for a same-day or next-day close. They open the dashboard mid-session with a ticker and a target time in mind and need trustworthy probabilities for several levels quickly, then come back later in the day to re-check as price moves.

## Product Purpose

LevelEdge estimates the probability that a stock or crypto price closes above a given level at a given future time. Success: the probabilities are calibrated (a 70% call happens about 70% of the time), the trader can compare many levels at once, and the dashboard says plainly how much to trust today's forecast.

## Positioning

One calibrated forecast distribution per ticker, interval and horizon, rather than a classifier per level. Levels are always consistent (P above a higher level never exceeds P above a lower one). The forecast is a volatility-scaled empirical baseline that a gradient-boosted model may adjust only when purged walk-forward validation shows it helps. It is honest about skill: every forecast ships with its out-of-sample record against simple baselines.

## Operating Context

- Used during US market hours (09:30-16:00 ET) on desktop, often next to a broker or prediction-market window. Crypto (e.g. BTC-USD) is 24/7.
- Data: yfinance OHLCV with per-interval history limits (about 60 days for 2m-90m bars, 730 days for 1h, full history for 1d). Optional Polygon data.
- Forward test: GitHub Actions runs `scripts/forwardtest.py` at 10:00, 12:00 and 14:00 ET on weekdays for QQQ, SPY, NVDA and TSLA, targeting the 16:00 close, and logs to the Supabase `forwardtest_logs` table. `scripts/update_outcomes.py` fills in realised outcomes every hour.
- A backtest mode simulates a forecast as of a past time.

## Capabilities and Constraints

- Inputs: ticker, target datetime (ET), bar interval (1m, 2m, 5m, 15m, 30m, 1h, 90m, 1d), price levels in dollars or as percent moves, optional as-of time.
- Outputs: P(above) for each level, the full probability curve, forecast quantiles (price range), the effective resolution time (the last bar close at or before the target), and diagnostics (out-of-fold log loss, Brier and AUC for blend vs baseline, reliability, 80%-range coverage, skill at the requested level, blend weights, feature importance).
- Model fits take about 1-3 s per (ticker, interval, horizon). Fitted models are cached and every level reuses one fit.
- The track record reads Supabase via the `SUPABASE_URL` and `SUPABASE_KEY` environment variables (HF Space secrets). It must degrade to a clear empty state when the credentials are absent.
- Regular-session bars only for stocks. NYSE early closes are respected.
- Earnings dates inside the window are flagged (from the yfinance calendar). The model itself doesn't price events.
- Not financial advice. The product gives probabilities, not trade recommendations.

## Evidence on Hand

- Offline benchmark harness: `benchmarks/compare.py`, which writes `benchmarks/results/summary.json` comparing the v1 per-level XGBoost, a Gaussian random walk, the v2 baseline and the v2 stacked forecast on held-out sessions.
- The Supabase `forwardtest_logs`, `logs` and `backtest_logs` tables (row contents not visible from the repo).
- No testimonials, users or performance claims beyond the benchmark output exist. Do not fabricate any.

## Product Principles

1. Calibration over excitement. A forecast that sounds confident must have earned it out of sample.
2. Show the trust signal next to the number. Every probability sits beside the evidence for how reliable it is.
3. Many levels, one coherent view. Levels are compared on a single consistent distribution, never as isolated verdicts.
4. Explain the resolution precisely. Say which bar close the forecast resolves on and how far away it is.
5. Fast enough to use mid-session. Re-checking a forecast should take seconds.

## Accessibility & Inclusion

WCAG 2.2 AA contrast, keyboard-operable controls, and no meaning carried by colour alone (above/below and good/bad skill also use text or shape). Charts have text equivalents (level tables).
