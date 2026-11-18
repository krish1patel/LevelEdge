# Forecaster

The forecaster returns the probability that a stock or crypto price closes above each of several levels at a target time, plus the full forecast distribution. This guide shows you how to use it, explains how the v2 model works, and reports its out-of-sample accuracy.

## Making forecasts

### Using the library

```python
from datetime import datetime
from leveledge import forecast, US_EASTERN

target = datetime(2026, 11, 20, 16, 0, tzinfo=US_EASTERN)
m = forecast("SPY", target, "15m")           # fits once; cached for repeat calls

m.prob_above([725, 735, 745])                # array of probabilities, non-increasing in the level
m.quantiles()                                # {0.05: ..., 0.5: ..., 0.95: ...} price levels
m.horizon.effective_target                   # the bar close the forecast resolves on
m.level_backtest(745)                        # out-of-sample skill for this level's move
```

The target must be timezone-aware and in the future. Supported bar intervals are `1m`, `2m`, `5m`, `15m`, `30m`, `1h`, `90m` and `1d`.

To simulate a forecast made at a past time, pass `as_of=datetime(...)`.

### Using the original one-level API

The `Predictor` class keeps the v1 interface, which `scripts/forwardtest.py` and the legacy apps use:

```python
from leveledge import Predictor

p = Predictor("SPY", target, "15m", 745.0)    # add end_datetime=... for a backtest
p.train_xgb()
p.predict_xgb()                               # probability in [0, 1]
```

### Using the dashboard

Open the Forecaster section, enter a ticker, choose a target and an interval, and enter levels as prices (`745`) or moves (`+1%`, `-0.5%`). To share a forecast, use a link such as:

```text
https://leveledge.vercel.app/?ticker=NVDA&interval=1d&target=2026-11-20T16:00#diagnostics
```

`target` also accepts `next` (the next close) and `hour`. The hash opens a tab: `#diagnostics` or `#track`.

When an earnings date falls inside the forecast window, the dashboard warns you and marks the date on the chart. The model learns from price history alone and doesn't price scheduled events.

## How the v2 model works

1. **Clean bars.** For stocks it keeps regular-session bars only, on the NYSE calendar including half-days. It drops the still-forming bar, because that bar's partial volume and shape corrupt the row the forecast is made from.
2. **Exact horizon.** It counts the bars between the last completed bar and the target, and reports the bar close the forecast resolves on.
3. **Volatility units.** It divides each historical move by a horizon volatility. That volatility combines an EWMA of de-seasonalised returns, a time-of-day variance profile (intraday volatility is U-shaped), and GARCH-style mean reversion, all fitted by likelihood.
4. **Baseline.** A kernel-smoothed empirical distribution of those standardised moves, with a Student-t tail. The baseline is fat-tailed and adapts to the volatility regime.
5. **Gated pattern model.** An XGBoost model is boosted from the baseline's log-odds, with a monotone constraint so P(above) can't rise with the level. A non-negative, intercept-free stacker combines it with the baseline. The model keeps a weight only if it also beats the baseline on a final fold the stacker never saw; otherwise the forecast is the baseline alone.
6. **Validation.** Expanding walk-forward over the last 60% of rows in 6 folds, with a purge and an embargo of `h` bars. It reports log loss, Brier, AUC, reliability, 80%-range coverage and per-level skill against climatology.

### What changed from v1

| v1 defect | Effect | v2 fix |
|---|---|---|
| `scale_pos_weight` class reweighting | Probabilities pulled toward 50% | Removed; calibration validated directly |
| Raw price features (`Open`, `SMA_200`, `VWAP`, `OBV`) | Trees can't extrapolate to new price regimes | Stationary features only, tested for scale invariance |
| A missing comma merged `SMA_500` and `EMA_5` | `EMA_5` silently dropped | New, tested feature module |
| Cross-validation folds not purged | Training labels overlapped test windows, inflating AUC | Purge plus embargo of `h` bars |
| The in-progress last bar was used | Corrupted features at prediction time | Completed bars only |
| Pre- and post-market bars; 6.5-hour days assumed | Wrong horizons, wrong on half-days | Regular session on the real NYSE calendar |
| One model per level | P(above 600) could exceed P(above 590); 8 fits for 8 strikes | One monotone distribution for every level |

## Accuracy

`benchmarks/compare.py` scores 4,800 forecasts on SPY, QQQ, NVDA, TSLA and BTC-USD, using 15m and 1h bars, from 2026-10-13 to 2026-11-09. Each forecast is made at 10:00, 12:00 or 14:00 ET for the 4:00 PM close, at strikes from −2% to +2%. Every model sees only bars that closed by the decision time.

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calibration error ↓ |
|---|---|---|---|---|
| **LevelEdge v2** | 0.126 | 0.385 | 0.909 | 4.9% |
| v2 volatility baseline alone | 0.125 | 0.384 | 0.909 | 4.9% |
| Gaussian random walk (EWMA volatility) | 0.136 | 0.419 | 0.901 | 7.8% |
| v1 (per-level XGBoost) | 0.185 | 0.580 | 0.805 | 6.8% |

- **v2 against v1:** Brier −0.0597 (32% lower), 95% CI −0.0701 to −0.0494, from resampling whole ticker-days. v2 is better for every ticker, interval and decision time.
- **v2 against a naive volatility model:** −0.0098, 95% CI −0.0137 to −0.0061.
- **Where the gain comes from:** calibration, meaning no class reweighting, volatility scaling, intraday seasonality and a fat-tailed empirical distribution. The pattern classifier adds no measurable skill on average, which is why it must pass a held-out gate on every fit.
- **History length:** 730 days of 1h history against about 60 days changes Brier by −0.0003 (CI ±0.0025), so a longer history doesn't help at these horizons.

The results are in `benchmarks/results/summary.json`. The pre-gate run is in `benchmarks/results/run1_ungated/`, and the history test is in `benchmarks/results/history60/`.

### Re-running the benchmark

```bash
python benchmarks/compare.py        # needs network; about 8 minutes on 6 processes
```

Don't add model complexity without re-running this benchmark, and report Brier differences with the bootstrap CI it prints.

## Limits

- Yahoo Finance keeps about 60 days of 2m–90m bars, 7 days of 1m bars and 730 days of 1h bars. Longer intraday history needs another source.
- The model doesn't price scheduled events. The dashboard flags earnings, but not macro releases such as FOMC or CPI.
- Crypto (`BTC-USD`) is modelled 24/7. Futures and FX tickers aren't specifically supported.
- Intermediate fan bands are interpolated from the target forecast with the fitted variance term structure, not fitted separately.
- The forecaster does not beat Kalshi's market prices after fees. See the [research log](../benchmarks/README.md).
