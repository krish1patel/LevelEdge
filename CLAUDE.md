# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

LevelEdge has three parts, all in one dashboard:

- **Portfolio (default view):** a paper-traded book, 70% 60/40 (SPY/IEF) + 30% trend sleeve over 11 ETFs, rebalanced weekly in an Alpaca **paper** account (`leveledge.live`, `leveledge.broker`, `scripts/paper_trade.py`, `.github/workflows/paper_trade.yml`).
- **Forecaster:** a **calibrated probability that a stock or crypto price closes above a level K at a target time T**, plus the full forecast distribution (`leveledge` v2 forecaster; live forward test via `scripts/forwardtest.py` → Supabase `forwardtest_logs`).
- **Research:** the evidence log (`/api/research`, built from `benchmarks/results/*`).

`app/api.py` + `web/` serve the JSON API and the React/Vite dashboard from one container (port 7860, HuggingFace Space).

### Evidence that shaped the product (don't re-litigate without new data)

- LevelEdge's probabilities **do not beat Kalshi's prices** after fees: taker (15–60 min, `kalshi_edge.py`), market-making (`kalshi_maker.py`) and morning-to-4 PM over 2024–2026 (`kalshi_horizon.py`) all failed. Don't build trading UI on Kalshi edge.
- The trend sleeve beats random pickers but not 60/40 on Sharpe; its value is smaller drawdowns (`etf_trend.py`).
- An 80-variant search failed its 2020+ holdout with reversed rankings (`etf_research.py`). The traded book uses the original pre-declared rules. Adding variants means re-running the holdout protocol, not picking a winner.
- LevelEdge's `VolScaler` ties EWMA on daily ETF vol; EWMA is used in the portfolio.

## Setup

```bash
uv venv -p 3.13 .venv && uv pip install -p .venv -r requirements.txt && uv pip install -p .venv -e ".[web,dev]"
cd web && npm ci && npm run build      # frontend → web/dist
```

`.env` (optional): `SUPABASE_URL`, `SUPABASE_KEY` (track record and forward-test logging), `POLYGON_API_KEY`, `ALPACA_API_KEY_ID` + `ALPACA_API_SECRET_KEY` (paper keys: live paper account on the Portfolio tab and the paper-trading job).
Set `LEVELEDGE_DISABLE_LOGGING=1` to stop `Predictor.predict_xgb()` writing to Supabase (tests and benchmarks set this).

## Running things

```bash
.venv/bin/uvicorn app.api:app --reload --port 7860   # API + built dashboard at http://127.0.0.1:7860
cd web && npm run dev                               # hot-reload UI at :5173, proxies /api to :7860
.venv/bin/python -m pytest -q tests                 # offline tests (synthetic data, no network)
.venv/bin/python benchmarks/compare.py              # v1-vs-v2 out-of-sample benchmark (~8 min, network)
python scripts/forwardtest.py                       # live forward test (needs Supabase)
python scripts/paper_trade.py --force               # paper rebalance plan (dry run; add --execute to submit)
.venv/bin/python benchmarks/paper_portfolio.py      # backtest of the traded book -> results/paper_portfolio (dashboard data)
.venv/bin/python benchmarks/etf_research.py         # 80-variant search with 2020+ holdout
.venv/bin/python benchmarks/kalshi_edge.py          # model vs Kalshi prices (network; caches in benchmarks/cache/)
```

## Architecture (v2)

```python
from leveledge import forecast
m = forecast("SPY", target_datetime, "15m", as_of=None)   # one fit serves every level (cached)
m.prob_above([600, 605, 610])   # monotone in level by construction
m.quantiles(); m.curve(); m.fan(); m.level_backtest(605); m.interval_coverage()
```

| Module | Responsibility |
|---|---|
| `sessions.py` | Bar end times, NYSE sessions (early closes included), dropping the in-progress bar, `compute_horizon()` → exact bars to the target and the bar close it resolves on |
| `data.py` | yfinance fetch within per-interval history limits; `clean_bars()` (regular session only, completed bars only) applied to every source |
| `features.py` | Stationary features only (returns, ratios, vol units). Never feed raw price levels. `tests/test_features.py` enforces no look-ahead and scale invariance |
| `model.py` | `LevelForecaster`: `VolScaler` (diurnal profile + mean-reverting variance, fitted by likelihood) → z-scores → `EmpiricalBaseline` (kernel survival function + Student-t tail) → XGBoost boosted from the baseline margin with a monotone constraint on the level → non-negative, intercept-free stacker → **held-out-fold gate** (classifier weight 0 unless it beats the baseline on the last fold) |
| `predictor.py` | `forecast()` with an LRU cache; `Predictor` = backward-compatible one-level API used by the scripts |
| `portfolio.py` | ETF data, momentum signals, vol-targeted sizing, daily backtest with next-open fills and costs, metrics, block-bootstrap Sharpe CIs, deflated Sharpe |
| `live.py` | The traded book (`STRATEGY_ID = core6040_trend30_v1`), weekly rebalance day on the NYSE calendar, partial-daily-bar guard, `current_targets()` |
| `broker.py` | Alpaca client locked to paper; `plan_orders()` never borrows and skips drift under the band |
| `kalshi.py` / `edge.py` | Kalshi public market data (live and archived candles); after-fee edge and joint event-level Kelly sizing |

Validation: expanding walk-forward, last 60% of rows in 6 folds, purge `h` bars plus embargo `h` bars. Metrics: log loss, Brier, AUC, reliability, 80%-range coverage, per-level Brier skill vs climatology.

Model decisions are evidence-backed by `benchmarks/compare.py` (results in `benchmarks/results/summary.json`; the pre-gate run is archived in `benchmarks/results/run1_ungated/`). Don't add model complexity without re-running it. Report Brier differences with the day-block bootstrap CI it prints.

## Invariants (don't break)

- Never use `scale_pos_weight` or class reweighting: it destroys calibration (v1's main defect).
- Never fit anything (vol profile, baseline, stacker) on rows that overlap a test fold's labels.
- The stacker has no intercept (an intercept learns the CV window's drift).
- P(above) must be non-increasing in the level.
- Times are US/Eastern; bars are indexed by start time; stocks use regular-session bars only.
- Paper trading only: never remove the `LEVELEDGE_ALLOW_LIVE_TRADING` guard or let `plan_orders` spend more than cash plus sale proceeds.
- The traded book's rules are pre-declared; changing them needs `paper_portfolio.py` and the holdout protocol re-run, and a new `STRATEGY_ID`.
- Stock forecasts are not compared with markets before 09:45 ET (the model has no bar from the session yet).
- Forward-test rows carry `model_type` (`leveledge_v2_stacked`, v1 rows are `xgboost_classifier`) so the dashboard can compare versions.

## Frontend

`web/src/`: `App.tsx` (shell with Portfolio / Forecaster / Research sections; hash routes `#research`, `#forecast`, `#diagnostics`, `#track`, default portfolio; `?ticker=` links open the forecaster), `components/Portfolio.tsx`, `components/Research.tsx`, `components/FanChart.tsx` (signature chart, x-axis in trading bars), `LevelsPanel`, `TrustPanel`, `Diagnostics`, `TrackRecord`, `XYChart`. Hand-built SVG with d3-scale/d3-shape. Tokens live in `styles.css`. Design context: `PRODUCT.md`, `DESIGN.md`, `.impeccable/surfaces/`.

## Deployment

`Dockerfile` (multi-stage: Node builds `web/dist`, Python serves it). `.github/workflows/sync_to_hf.yml` pushes `main` to the HuggingFace Space, so **every push to main deploys**. Add `SUPABASE_URL` and `SUPABASE_KEY` as Space secrets for the track record.

## Legacy

`examples/streamlit_batch_predictor.py` (old UI) and `examples/` scripts still run on the compatible `Predictor` API. `benchmarks/legacy_v1.py` is a frozen copy of v1, used only for benchmarking.
