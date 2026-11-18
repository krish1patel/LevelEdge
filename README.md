# LevelEdge

**Calibrated market forecasts, a paper-traded ETF portfolio, and a public, tamper-evident record of what did and didn't beat the market.**

[Live dashboard](https://leveledge.vercel.app) · [Documentation](docs/README.md) · [Research log](benchmarks/README.md) · [Forecast ledger](docs/forecast-ledger.md)

> Research and paper trading only. Nothing here is investment advice.

---

## Overview

LevelEdge answers one question well: *what is the probability that a price closes above a given level at a given time?* Around that forecaster it runs a weekly, rule-based ETF portfolio in a paper account, and it tests every trading idea against a simple alternative on data the idea hasn't seen, after costs. Ideas that fail are published with their numbers.

| Component | What it does | Where |
|---|---|---|
| **Forecaster** | Returns P(close above *K*) for any number of levels, plus the full forecast distribution and its out-of-sample record. | [`src/leveledge`](src/leveledge/README.md), [guide](docs/forecaster.md) |
| **Portfolio** | Trades 70% 60/40 (SPY/IEF) plus a 30% trend-following sleeve over 11 ETFs, weekly, in an Alpaca paper account, with pre-trade safety checks. No margin and whole shares only, so a real account can copy every trade. | [guide](docs/portfolio.md) |
| **Forecast ledger** | Commits every live forecast to a public, hash-chained log before its outcome exists, timestamps it with an outside authority, and scores it by appending. Anyone can verify the record. | [guide](docs/forecast-ledger.md) |
| **Research** | Logs about 20 pre-declared studies with development and 2020+ holdout periods: Kalshi edge, trend following, hedging, regime models, leverage, dip buying, and more. | [`benchmarks`](benchmarks/README.md) |
| **Dashboard** | Shows all of the above: a React frontend on Vercel and a FastAPI backend on AWS Lambda. | [`web`](web/README.md), [`app`](app/README.md) |

## Key results

Each claim links to the script that produced it.

| Question | Answer |
|---|---|
| Does the v2 forecaster beat v1? | **Yes.** Brier 0.126 vs 0.185 on 4,800 held-out forecasts (32% lower, 95% CI −0.070 to −0.049). [`compare.py`](benchmarks/compare.py) |
| Does it beat Kalshi's prices? | **No.** The market is better calibrated on liquid strikes, and trading the difference loses about 2¢ per contract after fees. [`kalshi_edge.py`](benchmarks/kalshi_edge.py) |
| Does the portfolio beat 60/40? | **Not on return; yes on crashes.** 7.3%/yr with a −24% worst fall against 8.0% and −33% (2007–2026). [`paper_portfolio.py`](benchmarks/paper_portfolio.py) |
| Can you time market falls? | **No.** Hidden Markov, jump models, the VIX and turbulence measures all failed out of sample. How *big* moves get is forecastable. [`regime_research.py`](benchmarks/regime_research.py) |
| Does leverage help in a retail account? | **Not on margin.** At Fidelity's ~11.6% rate, 1.5x earned less and fell further than 1x. [`aggressive_research.py`](benchmarks/aggressive_research.py) |
| Does buying the dip work? | **No.** A stock that fell 20% in a month beat the average stock bought the same day only 49% of the time. [`dip_research.py`](benchmarks/dip_research.py) |

The full log, including smaller studies, is in [benchmarks/README.md](benchmarks/README.md).

## Architecture

```text
                      ┌───────────────── GitHub Actions (scheduled) ─────────────────┐
  Yahoo Finance ────▶ │  forward test (3x daily) ─▶ Supabase + public `ledger` branch │
  Alpaca (paper) ◀──▶ │  paper trade (weekly) ─▶ safety checks ─▶ orders, report      │
  freetsa.org   ◀──── │  ledger reconcile, forward log, HuggingFace sync              │
                      └───────────────────────────────────────────────────────────────┘

  Browser ─▶ Vercel (React + Vite) ─▶ /api/* ─▶ AWS Lambda (FastAPI + leveledge) ─▶ data sources
```

For data flow and design decisions, see [docs/architecture.md](docs/architecture.md).

## Getting started

### Prerequisites

- Python 3.13
- Node.js 22 (only for the dashboard)

### Installing

```bash
git clone https://github.com/krish1patel/LevelEdge.git
cd LevelEdge
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[web,dev]"
```

### Making a forecast

```python
from datetime import datetime
from leveledge import forecast, US_EASTERN

target = datetime(2026, 11, 20, 16, 0, tzinfo=US_EASTERN)   # a future NYSE close
m = forecast("SPY", target, "15m")
print(m.prob_above([725, 735, 745]))   # for example [0.69 0.43 0.19]; never rises with the level
print(m.quantiles())                   # the forecast price range
```

### Running the dashboard locally

```bash
(cd web && npm ci && npm run build)
uvicorn app.api:app --port 7860        # then open http://127.0.0.1:7860
```

For hot reload while you edit the frontend, run `npm run dev` in `web/` and open <http://127.0.0.1:5173>. It forwards `/api` to port 7860.

### Running the tests

```bash
pytest -q tests                        # offline, about 7 seconds
```

The suite uses synthetic data and makes no network calls. See [tests/README.md](tests/README.md).

### Configuring credentials (optional)

Create `.env` in the repository root. Each key unlocks one feature; everything else runs without keys.

| Variable | Enables |
|---|---|
| `SUPABASE_URL`, `SUPABASE_KEY` | The live forward-test track record |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | The live paper account on the Portfolio tab, and paper trading |
| `POLYGON_API_KEY` | Polygon as an extra data source |

## Repository layout

| Path | Contents |
|---|---|
| [`src/leveledge/`](src/leveledge/README.md) | The Python library: forecaster, portfolio engine, broker, safety checks, ledger, regime models |
| [`app/`](app/README.md) | FastAPI backend for the dashboard |
| [`web/`](web/README.md) | React + Vite dashboard (hand-built SVG charts) |
| [`scripts/`](scripts/README.md) | Operational entry points run by the scheduled workflows |
| [`benchmarks/`](benchmarks/README.md) | Research studies; summaries in [`benchmarks/results/`](benchmarks/results/README.md) |
| [`tests/`](tests/README.md) | Offline pytest suite |
| [`deploy/`](deploy/README.md) | AWS Lambda and HuggingFace deployment |
| [`.github/workflows/`](.github/workflows/README.md) | Scheduled jobs: forward test, paper trading, ledger, sync |
| [`docs/`](docs/README.md) | Guides: architecture, forecaster, portfolio, ledger, deployment, operations |
| [`legacy/`](legacy/README.md) | The original v1 apps and scripts, kept for reference |
| `PRODUCT.md`, `DESIGN.md` | Product brief and design system for the dashboard |
| `CLAUDE.md` | Contributor guide: evidence, invariants and conventions |

## Safety

- **Paper trading only.** The broker client refuses any non-paper Alpaca endpoint unless `LEVELEDGE_ALLOW_LIVE_TRADING=1` is set, and order planning never spends more than cash plus sale proceeds.
- **Halts on bad inputs.** Eight pre-trade checks (stale data, implausible moves, price disagreement, oversized orders, margin, duplicate runs and more) halt a run and post an alert. No order is sent.
- **Pre-declared rules.** Changing a strategy requires a re-run of the holdout protocol and a new strategy ID. See [CLAUDE.md](CLAUDE.md) for the invariants.

## Maintainers

Krish Patel and Shlok Patel.
