---
title: LevelEdge
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# LevelEdge

**A calibrated price-level forecaster, a paper-traded ETF portfolio, and an honest record of what did and didn't beat the market.**

The dashboard has three sections:

- **Portfolio:** a paper-traded book of 70% 60/40 (SPY/IEF) plus a 30% trend-following sleeve across 11 ETFs, rebalanced weekly through an Alpaca paper account, scored against 60/40 and SPY.
- **Forecaster:** give it a ticker, a target time and some levels (`665`, `+1%`, `-0.5%`); it returns P(close above) for each level, the forecast distribution, and the out-of-sample evidence for how far to trust it.
- **Research:** every trading idea that was tested, with the numbers, including the ones that failed.

> Research and paper trading only. Not investment advice.

## What we tested

Every idea had to beat a simple alternative on data it hadn't seen, after costs.

| Idea | Result |
|---|---|
| Forecaster v2 vs the original v1 | **Adopted.** Brier 0.126 vs 0.185 on 4,800 held-out forecasts. |
| Trading Kalshi's hourly S&P 500 / Nasdaq-100 / Bitcoin contracts at the market price | **No edge.** On liquid quotes the market is better calibrated (Brier 0.1040 vs 0.1063); trading the difference loses about 2¢ per contract after fees. |
| Market-making on Kalshi around the model's fair value | **Loses** 6–10¢ per fill in all 18 variants (adverse selection). |
| Kalshi from the morning for the 4 PM settle, Feb 2024 – Nov 2026 | **No edge** (Brier 0.1155 vs market 0.1130). |
| 12-month trend following across 11 ETFs | **Real signal** (beats 99% of random pickers) but no better return per unit of risk than 60/40; much smaller drawdowns. Kept as a sleeve. |
| Best of 80 trend variants chosen on 2007–2019 | **Failed the 2020–2026 holdout.** Dev and holdout rankings were negatively correlated (−0.73). |
| LevelEdge's volatility model vs EWMA on daily ETF data | **Tie**; EWMA is used. |
| Warning signals that sell the 60/40 core: 12-month trend, 10-month average, VIX > 30, yield-curve inversion, drawdown stop, volatility scaling | **Only the 12-month trend filter passes**, narrowly: worst fall −35% → −21%, about 0.9 points a year less return. The others sell too late or too early. |
| Monthly SPY puts and collars on the core (priced from the VIX) | **Fail.** The premium eats most of the protection; the trend filter protects more for less. |
| Forecasting how far the S&P 500 may move (VIX, HAR, GJR-GARCH and their average) | **Works.** 3–6% lower CRPS than past returns at 1 week and 1 month, in 2000–2019 and in 2020+; 80% ranges held 81–84% of outcomes. Shown on the Portfolio tab. |
| Greedier books without margin: SPY or QQQ at 1x/2x/3x through leveraged ETFs (modelled within 0.3%/yr of SSO, UPRO, QLD, TQQQ), with no sell rule, a 200-day-average rule, or the 12-month trend rule | **More return, much bigger falls.** Held without a sell rule, 2x and 3x lost 91–100% at some point since 1960 (S&P) or 1985 (Nasdaq-100). With the 200-day rule: 2x S&P 10.1%/yr vs 7.7% (worst fall −59% vs −57%); 2x Nasdaq-100 19.3% vs 14.2% (−75% vs −83%); 3x Nasdaq-100 22.3% (−90%). Price indexes, dividends excluded. Not traded unless chosen. |
| Buying the dip: S&P 500 members after a 20% monthly fall (or 30% off the high, 12% in a week, a 2-sigma month, a fallen winner), and ~40 ETFs after a 10% fall; hold 3 months | **Fails.** Dipped stocks rose in 63% of cases, but the average stock bought the same day rose in 74%; the dipped stock beat it only 49% of the time, and one in ten lost 23% or more. No rule beat the same-day average in both 2006–2019 and 2020+. As a strategy (SPY plus dip slots): −71% worst fall, and behind SPY after 2020. Survivorship flatters these numbers. |
| The trend sleeve over 36 ETFs (sectors, countries, styles) instead of 11 | **Not adopted.** Sharpe 0.69/0.44 vs 0.72/0.49 (dev/2020+), slightly deeper falls. |
| Holding the book at 1.5x on margin, financed at a retail broker's rate (Fidelity, about T-bill + 7.5%) | **Loses.** 6.6%/yr with a −37% worst fall, against 7.3% and −24% unlevered. Borrowing at T-bill + 2% had made 1.5x look like 9.0%. The book holds no margin. |
| Timing falls: hidden Markov model, statistical jump model, trend, VIX, turbulence index, absorption ratio | **Fail.** Best AUC about 0.65 for a 5% monthly drop; regime switching inside the book lost to the trend filter after 2020. |

The paper book: 2007–2026 backtest 7.3%/yr, Sharpe 0.63, worst fall −24% (60/40: 8.0%, 0.62, −33%). Its goal is smaller crashes, not more return. Details: `benchmarks/results/*/summary.json`.

## Accuracy

<!-- BENCHMARK:START -->
Out-of-sample benchmark (`benchmarks/compare.py`). It covers 4,800 forecasts on SPY, QQQ, NVDA, TSLA and BTC-USD, using 15m and 1h bars, over 2026-10-13 to 2026-11-09. Decisions are made at 10:00, 12:00 and 14:00 ET for the 4:00 PM close, at strikes from −2% to +2%. Every model saw only bars that had closed by the decision time.

| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calibration error ↓ |
|---|---|---|---|---|
| **LevelEdge v2** | 0.126 | 0.385 | 0.909 | 4.9% |
| v2 volatility baseline alone | 0.125 | 0.384 | 0.909 | 4.9% |
| Gaussian random walk (EWMA vol) | 0.136 | 0.419 | 0.901 | 7.8% |
| v1 (per-level XGBoost) | 0.185 | 0.580 | 0.805 | 6.8% |

- **v2 against v1:** Brier -0.0597 (32% lower), 95% CI [-0.0701, -0.0494]. The CI comes from resampling whole ticker-days. v2 is better for every ticker, interval and decision time.
- **v2 against a naive volatility model:** -0.0098, CI [-0.0137, -0.0061].
- **What drives the gain:** calibration (no class reweighting, volatility scaling, intraday seasonality, a fat-tailed empirical distribution). The pattern classifier added no measurable skill on average, which is why it has to pass a held-out gate on every fit.
- **History length:** 730 days of 1h history against about 60 days changed Brier by −0.0003 (CI ±0.0025), so a longer history wouldn't help at these horizons.

Results: `benchmarks/results/summary.json`. The pre-gate run is in `benchmarks/results/run1_ungated/` and the history test in `benchmarks/results/history60/`.
<!-- BENCHMARK:END -->

## How it works (v2)

1. **Clean bars.** Regular-session bars only for stocks, on the NYSE calendar including half-days. The still-forming bar is dropped, because its partial volume and shape corrupt the row the forecast is made from.
2. **Exact horizon.** It counts the bars between the last completed bar and the target, and reports the bar close the forecast resolves on.
3. **Volatility units.** Each historical move is divided by a horizon volatility. That volatility combines an EWMA of de-seasonalised returns, a time-of-day variance profile (intraday volatility is U-shaped), and GARCH-style mean reversion, all fitted by likelihood.
4. **Baseline.** A kernel-smoothed empirical distribution of those standardised moves, with a Student-t tail. It is fat-tailed, adapts to the volatility regime, and is well calibrated.
5. **Pattern model (gated).** An XGBoost model boosted from the baseline's log-odds, with a monotone constraint so P(above) can never rise with the level. A non-negative, intercept-free stacker combines it with the baseline. The model keeps a weight only if it also beats the baseline on a final fold the stacker never saw. Otherwise the forecast is the baseline alone.
6. **Validation.** Expanding walk-forward with a purge and embargo of `h` bars. It reports log loss, Brier, AUC, reliability, 80%-range coverage and per-level skill against climatology.

One fit serves every level, so the levels are always consistent with each other. A fit takes about 1–3 s.

### What changed from v1

| v1 defect | Effect | v2 |
|---|---|---|
| `scale_pos_weight` class reweighting | Probabilities pulled toward 50% | Removed; calibration validated directly |
| Raw price features (`Open`, `SMA_200`, `VWAP`, `OBV`…) | Trees can't extrapolate to new price regimes | Stationary features only (tested for scale invariance) |
| Missing comma merged `SMA_500` and `EMA_5` | `EMA_5` silently dropped | New feature module, tested |
| CV folds not purged | Training labels overlapped test windows, inflating AUC | Purge plus embargo of `h` bars |
| In-progress last bar used | Corrupted features at prediction time | Completed bars only |
| Pre/post-market bars; 6.5 h days assumed | Wrong horizons, wrong on half-days | Regular session on the real NYSE calendar |
| One model per level | P(above 600) could exceed P(above 590); 8 fits for 8 strikes | One monotone distribution for all levels |

## Quick start

```bash
git clone https://github.com/krish1patel/LevelEdge.git && cd LevelEdge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e ".[web,dev]"
```

### Library

```python
from datetime import datetime
from leveledge import forecast, US_EASTERN

m = forecast("SPY", datetime(2026, 11, 10, 16, 0, tzinfo=US_EASTERN), "15m")
m.prob_above([745, 750, 755])      # array of probabilities, non-increasing
m.quantiles()                      # {0.1: …, 0.5: …, 0.9: …}
m.horizon.effective_target         # the bar close it resolves on
m.level_backtest(755)              # out-of-sample skill for this level's move
```

The original API still works (`scripts/forwardtest.py` uses it):

```python
from leveledge import Predictor
p = Predictor("SPY", target, "15m", 755.0)   # end_datetime=... for a backtest
p.train_xgb(); p.predict_xgb()
```

### Dashboard

```bash
cd web && npm ci && npm run build && cd ..
uvicorn app.api:app --port 7860          # open http://127.0.0.1:7860
# UI development with hot reload: (cd web && npm run dev) → http://127.0.0.1:5173
```

API: `POST /api/forecast`, `GET /api/track-record`, `GET /api/benchmark`; interactive docs at `/docs`.

Shareable links: `/?ticker=NVDA&interval=1d&target=2026-11-20T16:00#diagnostics`. `target` also accepts `next` or `hour`, and the hash selects a tab (`#diagnostics` or `#track`).

When an earnings date (from Yahoo Finance) falls inside the forecast window, the dashboard shows a warning and marks the date on the fan, because the model is trained on price history alone.

### Tests and benchmark

```bash
pytest -q tests                      # offline, synthetic data
python benchmarks/compare.py         # real held-out sessions, ~8 min on 6 processes
```

## Deployment

Every push to `main` syncs to the HuggingFace Space (`.github/workflows/sync_to_hf.yml`). The multi-stage `Dockerfile` builds the frontend with Node, then serves the API and the static frontend with uvicorn on port 7860.

Space secrets:
- `SUPABASE_URL` and `SUPABASE_KEY`: enable the **Track record** tab, which reads `forwardtest_logs`.
- `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY` (paper keys; `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` also work): show the live paper account on the **Portfolio** tab.

## Paper trading

`.github/workflows/paper_trade.yml` runs `scripts/paper_trade.py --execute` after the close on weekdays. On the last trading day of each week it rebalances the Alpaca **paper** account to the book's targets; orders fill at the next open. Add the two Alpaca secrets to the GitHub repository for it to trade; without them it prints the plan for a hypothetical $100,000 account.

```bash
python scripts/paper_trade.py --force          # dry run: plan against the real paper account (or $100k)
python scripts/paper_trade.py --force --execute  # submit to the paper account now
```

Safety: `leveledge.broker` refuses any non-paper Alpaca endpoint unless `LEVELEDGE_ALLOW_LIVE_TRADING=1`, never borrows (holdings at most 1.0x equity; a leftover margin balance is repaid first), trades whole shares only, and skips drift under the 2-point band. Before any order, `leveledge.safety` checks that the account is an active paper account, the prices come from the last completed session, no ETF moved more than 15% in a day, yfinance and Alpaca agree on prices within 3%, no order exceeds 90% of equity, holdings stay under the ceiling, no order buys on margin, and this decision's orders were not already sent. Any failure halts the run and posts an alert to the "Paper book weekly report" issue.

Copying the book in a real account: the paper account trades exactly what a Fidelity cash or margin account can (whole shares, no margin, market orders that fill at the open). Its orders fit only an account of the same size and holdings, so plan your own from your Fidelity positions:

```bash
python scripts/mirror_ticket.py --positions Portfolio_Positions_Nov-13-2026.csv   # Positions -> Download in Fidelity
python scripts/mirror_ticket.py --cash 25000                                       # a new account holding only cash
```

It prints sells then buys in whole shares, never spending more than the cash plus sale proceeds. The weekly report also estimates the dividends a real account receives and Alpaca's paper account does not credit.

Forward log: `.github/workflows/forward_log.yml` runs `scripts/log_forward.py` after every weekday close and commits that day's market outlook and the book's targets to the `forward-log` branch (`outlook.jsonl`). Each forecast is public before its outcome exists; the dashboard scores the log against S&P 500 closes. The weekly report also tracks untraded shadow books from public prices: DAA in place of the trend sleeve, the book with the 12-month trend filter on its core, and three greedy books (SSO, QLD or TQQQ while the S&P 500 or Nasdaq-100 is above its 200-day average, T-bills otherwise).

## Forward test

`.github/workflows/run_forwardtest.yml` runs `scripts/forwardtest.py` at 10:00, 12:00 and 14:00 ET on weekdays. It skips NYSE holidays and targets the early close on half-days. It logs QQQ, SPY, NVDA and TSLA at ±1–3% strikes to Supabase, with `model_type = leveledge_v2_stacked` (v1 rows are `xgboost_classifier`). `update_outcomes.yml` fills in the realised closes every hour. The dashboard's Track record tab scores both versions side by side.

## Limits

- yfinance keeps about 60 days of 2m–90m bars (7 days of 1m) and 730 days of 1h. Longer intraday history needs another source (see the roadmap).
- The model doesn't price scheduled events. The dashboard flags earnings inside the window, but macro releases (FOMC, CPI) are not flagged yet.
- Crypto (`BTC-USD`) is modelled 24/7; futures and FX tickers are not specifically supported.
- Intermediate fan bands are interpolated from the target forecast with the fitted variance term structure. They are not separately fitted forecasts.

## Project layout

```
src/leveledge/   forecaster: sessions.py  data.py  features.py  model.py  predictor.py  polygon_data.py
                 portfolio:  portfolio.py (engine)  live.py (the traded book)  broker.py (Alpaca paper)
                 market data: kalshi.py  edge.py (edge and event-level Kelly sizing)
app/api.py       FastAPI: forecast, track record, benchmark, portfolio, research, static UI
web/             React + Vite dashboard (hand-built SVG charts)
benchmarks/      compare.py (v1 vs v2), kalshi_edge.py, kalshi_maker.py, kalshi_horizon.py,
                 etf_trend.py, etf_research.py, paper_portfolio.py, legacy_v1.py, results/
scripts/         paper_trade.py, forwardtest.py, update_outcomes.py, backtests
tests/           offline pytest suite
```
