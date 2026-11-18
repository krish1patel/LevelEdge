# Portfolio

LevelEdge trades one rule-based ETF portfolio, "the book", in an Alpaca paper account. It follows real retail-account rules (whole shares, no margin), so you can copy every trade into a brokerage account. This guide describes the book, the weekly trading loop, the safety checks, and how to mirror the trades.

## Understanding the book

| Part | Weight | Rule |
|---|---|---|
| Core | 70% | 60% SPY, 40% IEF, always held |
| Trend sleeve | 30% | 11 ETFs (SPY, QQQ, IWM, EFA, EEM, IEF, TLT, LQD, GLD, DBC, VNQ). Each ETF is held only while its 12-month return beats T-bills, sized to a 10% volatility target with EWMA volatility, at most 25% per ETF. |

- **Strategy ID:** `core6040_trend30_cash_v3`, defined in `src/leveledge/live.py`.
- **Schedule:** targets are computed after the close on the last NYSE trading day of each week, and orders fill at the next open.
- **No-trade band:** an ETF trades only when it drifts more than 2 percentage points from its target. This cut orders from 419 to 64 a year in testing.
- **Goal:** smaller crashes than 60/40 at a similar return per unit of risk, not beating the market.

### Backtest, 2007–2026

| | Return/yr | Sharpe | Worst fall | 2008 | 2020 | 2022 |
|---|---|---|---|---|---|---|
| **The book** | 7.3% | 0.63 | −24% | −23% | −17% | −18% |
| 60/40 | 8.0% | 0.62 | −33% | −32% | −19% | −21% |
| SPY | 10.7% | 0.54 | −55% | −55% | −33% | −24% |

The trend sleeve beat all 200 random-signal books built the same way. Re-run the backtest with `python benchmarks/paper_portfolio.py`, which writes `benchmarks/results/paper_portfolio/summary.json` for the dashboard.

### Why the book holds no margin

An earlier version held the book at 1.5x, with borrowing modelled at T-bill + 2%. A retail broker charges far more: Fidelity's rate was about 11.6% (T-bill + 7.5%) in November 2026. At that rate, 1.5x returned 6.6%/yr with a −37% worst fall, against 7.3% and −24% unlevered. The book returned to its original unlevered rules on 2026-11-16.

## Running the paper-trading loop

`.github/workflows/paper_trade.yml` runs `scripts/paper_trade.py --execute` after every weekday close. The script only trades on the weekly rebalance day.

To see the plan without sending orders, run a dry run:

```bash
python scripts/paper_trade.py --force             # plans against the real paper account, sends nothing
```

The script prints a JSON record with the targets, the planned orders and each safety check's result. Without Alpaca keys, it plans for a hypothetical $100,000 account.

To submit orders now (paper account only):

```bash
python scripts/paper_trade.py --force --execute
```

### Pre-trade safety checks

Before any order, `leveledge.safety.pretrade_checks` runs these checks. Any failure halts the run, sends nothing, and posts an alert.

| Check | Stops |
|---|---|
| `account_ok` | Trading on a blocked account or a non-paper account |
| `data_fresh` | Targets from anything but the last completed NYSE session |
| `moves_plausible` | Trading after any ETF moved more than 15% in a day (bad data, or a day for a human) |
| `prices_agree` | A Yahoo close more than 3% from Alpaca's price for a held ETF |
| `order_sizes` | Any order above 90% of equity, or total trading above 2x equity |
| `gross_ok` | Holdings above the book's ceiling (1.0x equity) |
| `no_borrowing` | Any buy on margin; a leftover margin balance must shrink |
| `not_duplicate` | Sending a decision's orders twice when a job re-runs |

A halted run is the system working. Never loosen a check to get a run through.

## Mirroring trades in a real account

The paper account's orders only fit an account with the same size and holdings. To trade your own account, plan its orders from its actual positions.

1. In Fidelity, open **Portfolio → Positions → Download** and save the CSV.
2. Run the ticket script:

   ```bash
   python scripts/mirror_ticket.py --positions Portfolio_Positions_Nov-16-2026.csv
   ```

   For an account that holds only cash, run `python scripts/mirror_ticket.py --cash 25000`.
3. Enter the sells first, then the buys, as day market orders before the next open.

The script prints whole-share orders that never spend more than your cash plus sale proceeds, and a table of target weights against weights after the orders. It leaves holdings outside the book alone. If the file holds several accounts, choose one with `--account`.

## Reading the weekly report

On each rebalance, `scripts/weekly_report.py` posts a comment on the "Paper book weekly report" GitHub issue. It covers:

- Equity, weekly and total return, distance below peak, and cash.
- Fill slippage against the open.
- The safety checks and the trades sent.
- Dividends a real account receives but Alpaca's paper account doesn't credit.
- The account against the book's own rules, the shadow books, 60/40 and SPY since the start.
- The nearest trend flips: the prices at which an ETF would enter or leave the sleeve.

### Shadow books

Shadow books are candidate strategies that aren't traded. They're rebuilt from public prices every week, so their record can't be adjusted after the fact.

| Shadow | Rule | Why it isn't traded |
|---|---|---|
| DAA sleeve | Keller & Keuning DAA-G12 in place of the trend sleeve | Won the pre-declared selection, but its holdout edge isn't significant |
| Trend filter on the core | Core SPY/IEF move to T-bills while their 12-month return trails T-bills | Passed its test (worst fall −35% → −21%); awaiting a decision |
| Greedy: SSO, QLD, TQQQ | A 2x or 3x leveraged ETF while its index is above the 200-day average, T-bills otherwise | Historical worst falls of −59% to −90% |

## Reading the market outlook

The Portfolio tab's outlook card forecasts how far the S&P 500 may move over 1 week, 1 month and 3 months. It averages three volatility models (VIX-implied, HAR on daily ranges, and GJR-GARCH with filtered historical simulation). Tested every Friday since 2000, its 80% ranges held 81–84% of outcomes. It forecasts the size of moves, not their direction, and it carries no buy or sell signal.

Each day's outlook is committed to the public `forward-log` branch before its outcome exists, and the card scores that live record as outcomes arrive.
