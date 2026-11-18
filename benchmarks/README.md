# benchmarks

The research log. Each script is one study that tests a trading or forecasting idea against a simple alternative, on data the idea hasn't seen, after costs. Studies that fail stay here with their numbers: knowing what doesn't work is half of LevelEdge's value.

## Following the protocol

Every study follows the same rules, and its docstring states them before any result:

1. **Pre-declare.** The rules, parameters, comparison and pass criteria are fixed before the study runs.
2. **Split in time.** Development data ends on 2019-12-31. The 2020+ holdout is read once, after the choice is made.
3. **Trade realistically.** Decisions use data available at the decision time, fills happen at the next open, and costs are charged (5 bp per side for ETFs, 10 bp for stocks).
4. **Measure uncertainty.** Sharpe differences come with block-bootstrap confidence intervals, and multi-variant searches report a deflated Sharpe ratio.
5. **Publish the verdict.** A failed idea isn't tuned until it passes.

## Studies

### Forecaster and prediction markets

| Study | Question | Verdict |
|---|---|---|
| [`compare.py`](compare.py) | Does the v2 forecaster beat v1 on held-out sessions? | **Adopted.** Brier 0.126 vs 0.185 on 4,800 forecasts (CI −0.070 to −0.049) |
| [`kalshi_edge.py`](kalshi_edge.py) | Do LevelEdge's probabilities beat Kalshi's hourly prices after fees? | **No edge.** Market Brier 0.1040 vs 0.1063 on liquid strikes; trading loses about 2¢ per contract |
| [`kalshi_maker.py`](kalshi_maker.py) | Does quoting around the model's fair value make money? | **Loses** 6–10¢ per fill in all 18 variants (adverse selection) |
| [`kalshi_horizon.py`](kalshi_horizon.py) | Does a morning forecast beat the market for the 4 PM settle (2024–2026)? | **No edge.** Brier 0.1155 vs market 0.1130 |

### Portfolio construction

| Study | Question | Verdict |
|---|---|---|
| [`etf_trend.py`](etf_trend.py) | Does 12-month trend following across 11 ETFs earn its keep? | **Real signal, kept as a sleeve.** Beats 99% of random pickers, matches 60/40 on Sharpe, with much smaller drawdowns |
| [`etf_research.py`](etf_research.py) | Does the best of 80 trend variants, chosen on 2007–2019, hold up? | **Fails the holdout.** Development and holdout rankings correlate at −0.73 |
| [`paper_portfolio.py`](paper_portfolio.py) | How does the traded book perform? (Writes the dashboard's backtest.) | 7.3%/yr, Sharpe 0.63, worst fall −24% vs 60/40's 8.0%, 0.62, −33% |
| [`literature_research.py`](literature_research.py) | Do published allocation strategies beat the trend sleeve? | **DAA-G12 wins the pre-declared selection but isn't proven** (holdout ΔSharpe +0.06, CI −0.11 to +0.21). Tracked as a shadow book. A 2% rebalancing band cuts orders from 419 to 64 a year. |
| [`construction_research.py`](construction_research.py) | Do risk-balanced, volatility-targeted designs beat the book? | **No better design.** The leverage frontier was later superseded by retail margin costs |
| [`reward_research.py`](reward_research.py) | What sleeve share and leverage maximise return within a drawdown limit? | Superseded: assumed borrowing at T-bill + 2% |
| [`alpha_research.py`](alpha_research.py) | Do ridge, spline, mixed-effects or boosted return models beat the trend rule? | **No gain** over the trend rule |
| [`bond_leg_research.py`](bond_leg_research.py) | Should the bond leg switch to short-term Treasuries on a falling trend? | **Rejected** |
| [`cadence_research.py`](cadence_research.py) | Should triggers be checked daily instead of weekly? | **Keep weekly.** Daily checks add trades without improving results |
| [`stock_factors.py`](stock_factors.py) | Do momentum, low-volatility or reversal factors beat equal-weight S&P 500 members (point-in-time)? | **None beats equal weight** |

### Risk, hedging and timing

| Study | Question | Verdict |
|---|---|---|
| [`hedge_research.py`](hedge_research.py) | Can warning signals (12-month trend, 10-month average, VIX > 30, curve inversion, drawdown stop, volatility scaling) sell the core before falls? | **Only the 12-month trend filter passes**: worst fall −35% → −21%, about 0.9 points a year less return. Tracked as a shadow book. |
| [`put_hedge_research.py`](put_hedge_research.py) | Do monthly SPY puts or collars protect the core? | **Fail.** The premium eats most of the protection |
| [`regime_research.py`](regime_research.py) | Can you forecast the size of moves, and tell safe zones from danger zones? | **Size: yes.** 80% ranges held 81–84% of outcomes; shown as the outlook card. **Timing: no.** HMM, jump models, VIX, turbulence and absorption ratio reach AUC of at most 0.65 |
| [`dip_research.py`](dip_research.py) | Does buying S&P 500 stocks or ETFs after big falls pay over 3 months? | **Fails.** A dipped stock beat the same-day average stock 49% of the time; no rule passed both periods |
| [`aggressive_research.py`](aggressive_research.py) | How much more return do leveraged ETFs or margin give, at what risk (2007–2026)? | **Margin loses** at retail rates. Leveraged ETFs with a 200-day rule add return with far bigger falls |
| [`aggressive_longrun.py`](aggressive_longrun.py) | Do those results hold back to 1960 (S&P) and 1985 (Nasdaq-100)? | Without a sell rule, 2x and 3x lost 91–100%. With the 200-day rule, worst falls of −59% to −90% |
| [`intraday_research.py`](intraday_research.py) | Do published day-trading strategies (opening-range breakout, intraday momentum) work after costs? | **Fail** after costs |

### Support files

| File | Purpose |
|---|---|
| [`legacy_v1.py`](legacy_v1.py) | A frozen copy of the v1 forecaster, used only by `compare.py` |
| [`results/`](results/README.md) | Each study's summary (JSON) and small result tables |
| `cache/` | Downloaded price data (not committed; rebuilt on first run) |

## Running a study

Most studies download prices from Yahoo Finance on first run and cache them in `cache/`.

```bash
python benchmarks/paper_portfolio.py     # minutes; writes results/paper_portfolio/summary.json
python benchmarks/compare.py             # about 8 minutes on 6 processes
```

Studies that read downloaded files from other sources (`stock_factors.py`, `dip_research.py`) run with `python -I` so files in the working directory can't shadow modules.

## Adding a study

1. Write the protocol in the module docstring first: data, rules, comparison and pass criteria.
2. Save outputs to `benchmarks/results/<study>/summary.json`.
3. Add the study to the tables above, with its verdict, and to the dashboard's Research section if it changes a decision.
4. If it changes the traded book, re-run `paper_portfolio.py` and the holdout protocol, and give the book a new `STRATEGY_ID`.
