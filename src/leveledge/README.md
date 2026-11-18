# leveledge

The `leveledge` package holds all of LevelEdge's modelling, trading and verification logic. The API, the scheduled scripts and the research studies are thin layers over it.

## Installing

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

## Using the public API

```python
from datetime import datetime
from leveledge import forecast, Predictor, US_EASTERN, ALLOWED_INTERVALS

m = forecast("SPY", datetime(2026, 11, 20, 16, 0, tzinfo=US_EASTERN), "15m")
m.prob_above([725, 735, 745])
```

The package exports `forecast`, `Predictor`, `LevelForecaster`, `ForecastConfig`, `fetch_polygon`, `ALLOWED_INTERVALS` and `US_EASTERN`. Import other modules directly, for example `from leveledge import ledger`.

## Modules

### Forecaster

| Module | Responsibility |
|---|---|
| `sessions.py` | Bar end times, NYSE sessions including early closes, dropping the in-progress bar, and `compute_horizon()`: the exact bars to the target and the bar close it resolves on |
| `data.py` | Yahoo Finance fetches within per-interval history limits; `clean_bars()` keeps regular-session, completed bars from any source |
| `features.py` | Stationary features only: returns, ratios and quantities in volatility units. Raw price levels are never used. |
| `model.py` | `LevelForecaster`: volatility scaler → z-scores → empirical baseline → gated, monotone XGBoost → non-negative, intercept-free stacker |
| `predictor.py` | `forecast()` with an LRU cache, and `Predictor`, the backward-compatible one-level API |
| `polygon_data.py` | Polygon.io bars in the same schema as Yahoo Finance |
| `constants.py` | Supported intervals and the US/Eastern timezone |

### Portfolio and execution

| Module | Responsibility |
|---|---|
| `portfolio.py` | ETF data, momentum signals, volatility-targeted sizing, a daily backtest with next-open fills and costs, metrics, block-bootstrap Sharpe CIs and deflated Sharpe |
| `live.py` | The traded book (`STRATEGY_ID`), the weekly rebalance day, the partial-daily-bar guard, `current_targets()`, and the shadow books |
| `strategies.py` | Published allocation rules implemented as their papers specify, such as Keller & Keuning DAA-G12 |
| `signals.py` | Return-forecasting models for the ETF universe, used by research only |
| `broker.py` | The Alpaca client, locked to paper unless `LEVELEDGE_ALLOW_LIVE_TRADING=1`; `plan_orders()` never borrows and skips drift under the band |
| `safety.py` | The eight pre-trade checks run before any order |
| `mirror.py` | The Fidelity positions CSV parser and the whole-share, no-margin order ticket |

### Market views and integrity

| Module | Responsibility |
|---|---|
| `outlook.py` | The live S&P 500 outlook: the average of VIX, HAR and GJR-GARCH distributions, 1-day to 3-month ranges, and the book's range through its beta |
| `regime.py` | GJR-GARCH-t with filtered historical simulation, HAR, a two-state HMM, a statistical jump model, turbulence, absorption ratio and CRPS (numpy and scipy only) |
| `ledger.py` | The verifiable forecast ledger: hash chain, RFC 3161 anchors, idempotent scoring and `verify()`. Standard library only; it also runs as a standalone verifier. |
| `kalshi.py` | Kalshi public market data, live and archived |
| `edge.py` | After-fee edge against a market price and joint, event-level Kelly sizing |

## Following the invariants

These rules protect calibration and safety. Tests enforce most of them.

- Never use `scale_pos_weight` or class reweighting; it destroys calibration.
- Never fit anything on rows that overlap a test fold's labels.
- The stacker has no intercept, and P(above) must be non-increasing in the level.
- Times are US/Eastern; bars are indexed by their start time.
- `plan_orders` never spends more than cash plus sale proceeds, and the book keeps `MAX_GROSS = 1.0` and `WHOLE_SHARES = True`.
- Don't change `ledger.canonical()`, `ledger.record_hash()` or the record fields; that breaks every existing hash.

The full list is in [CLAUDE.md](../../CLAUDE.md).

## Testing

```bash
pytest -q tests
```

See [tests/README.md](../../tests/README.md) for what each test file covers.
