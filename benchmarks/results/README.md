# benchmarks/results

Outputs of the research studies in [`benchmarks/`](../README.md). Each folder is named after its study, and `summary.json` is the study's verdict and headline numbers. The dashboard's Research section and `/api/research` read these files, so the dashboard updates when a study is re-run.

## Layout

| Path | Written by | Contents |
|---|---|---|
| `summary.json`, `predictions.csv` | `compare.py` | The v1-vs-v2 forecaster benchmark and its per-forecast predictions |
| `run1_ungated/` | `compare.py` (archived) | The same benchmark before the classifier gate was added |
| `history60/` | `compare.py --history-days 60` | The history-length test |
| `kalshi/`, `kalshi_horizon/` | `kalshi_edge.py`, `kalshi_maker.py`, `kalshi_horizon.py` | Market-edge tests; `predictions.csv.gz` holds each model-vs-market pair |
| `etf_trend/` | `etf_trend.py` | Trend-sleeve metrics, daily equity curves and weights |
| `etf_research/` | `etf_research.py` | The 80-variant search; `variants.csv` ranks every variant in both periods |
| `paper_portfolio/` | `paper_portfolio.py` | The traded book's backtest (shown on the Portfolio tab) |
| `literature/`, `construction/`, `reward/`, `alpha_research_main/`, `alpha_research_fresh/`, `bond_leg/`, `cadence/`, `stock_factors/` | The study of the same name | Portfolio-construction studies |
| `hedge/`, `put_hedge/`, `regime/`, `dip/`, `aggressive/`, `intraday/` | The study of the same name | Risk, hedging, timing and leverage studies; `dip/` also holds every event the dip study scored |

## Reading a summary

Most `summary.json` files share this shape:

```json
{
  "protocol": {"...": "the pre-declared settings"},
  "metrics":  {"strategy": {"cagr": 0.073, "sharpe": 0.63, "max_drawdown": -0.237}},
  "dev":      {"...": "the same metrics, 2007-2019"},
  "holdout":  {"...": "the same metrics, 2020 onward"}
}
```

Rates are decimals: `0.073` means 7.3% a year, and `-0.237` a 23.7% fall.

## What isn't committed

Bulky intermediates (daily equity curves of some studies, partial runs, raw forecasts) are listed in `.gitignore`. Re-run the study to regenerate them. Downloaded price data lives in `benchmarks/cache/` and isn't committed.
