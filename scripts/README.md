# scripts

Operational entry points. The scheduled workflows in [`.github/workflows`](../.github/workflows/README.md) run these scripts, and you can run any of them locally. Each script's docstring lists its full options.

## Paper trading

| Script | Run by | What it does |
|---|---|---|
| `paper_trade.py` | `paper_trade.yml`, weekdays after the close | Computes the book's targets and, on the weekly rebalance day, plans whole-share orders, runs the safety checks and submits to the Alpaca paper account. Prints a JSON record. |
| `weekly_report.py` | `paper_trade.yml` | Writes the weekly Markdown report: returns, fills, trades, dividends, shadow books, trend flips |
| `post_report.sh` | All alerting workflows | Posts a Markdown file as a comment on the "Paper book weekly report" issue |
| `mirror_ticket.py` | You | Plans whole-share, no-margin orders for a real account from a Fidelity positions CSV |

```bash
python scripts/paper_trade.py --force                      # dry run against the paper account
python scripts/mirror_ticket.py --positions positions.csv  # orders for your own account
```

## Forecast record

| Script | Run by | What it does |
|---|---|---|
| `market_scan.py` | You | Runs the daily market scan locally and writes it as JSON (`SCAN_FILE=scan.json` shows it on the dashboard) |
| `forwardtest.py` | `run_forwardtest.yml`, every 20 minutes (each slot at most once) | Forecasts QQQ, SPY, NVDA and TSLA at 8 strikes and logs to Supabase. `--ledger-out FILE` also writes ledger records. |
| `update_outcomes.py` | `update_outcomes.yml`, hourly | Fills realised prices into Supabase forward-test rows |
| `ledger.py` | `run_forwardtest.yml`, `ledger_reconcile.yml` | Operates the verifiable ledger: `init`, `append`, `reconcile`, `anchor`, `verify` |
| `ledger_push.sh` | The same workflows | Runs a `ledger.py` command on the public `ledger` branch and pushes, retrying on the new head if another job pushed first |
| `log_forward.py` | `forward_log.yml`, weekdays after the close | Appends the day's S&P 500 outlook and the book's targets to the forward log |

```bash
python scripts/ledger.py verify                            # verify the public ledger
```

## Deployment

| Script | Run by | What it does |
|---|---|---|
| `sync_space.sh` | `sync_to_hf.yml`, on push to `main` | Pushes a snapshot to the HuggingFace Space, adding the Space's YAML header to the README |

The AWS and Vercel deployment scripts live in [`deploy/aws`](../deploy/aws/README.md).

## Environment

Scripts read credentials from the environment or from `.env` in the repository root. See the [root README](../README.md#configuring-credentials-optional).
