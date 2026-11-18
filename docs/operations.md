# Operations

LevelEdge runs on a schedule in GitHub Actions. This guide lists every job, where its output and alerts go, and what to do when one fails.

Scheduled workflows run only from the `main` branch. GitHub often starts them late, sometimes by hours; every job tolerates that.

## Scheduled jobs

| Workflow | Schedule (UTC) | Eastern time | What it does |
|---|---|---|---|
| `run_forwardtest.yml` | 14:00, 16:00, 18:00 Mon–Fri | 10 AM, 12 PM, 2 PM (EDT) | Forecasts QQQ, SPY, NVDA and TSLA at 8 strikes, logs to Supabase, and appends to the ledger with a timestamp |
| `update_outcomes.yml` | Every hour | — | Fills realised prices into Supabase forward-test rows |
| `paper_trade.yml` | 21:30 Mon–Fri | 4:30 PM EST / 5:30 PM EDT | Rebalances the paper account on the weekly rebalance day, then posts the weekly report |
| `forward_log.yml` | 22:00 Mon–Fri | 5 PM EST / 6 PM EDT | Commits the day's S&P 500 outlook to the `forward-log` branch |
| `ledger_reconcile.yml` | 22:30 Mon–Fri, plus 12:00 daily | 5:30 PM EST, plus a morning backstop | Scores matured forecasts and verifies the whole ledger |
| `sync_to_hf.yml` | On every push to `main` | — | Syncs the repository to the HuggingFace Space |

## Configuring secrets

Set these in the GitHub repository under **Settings → Secrets and variables → Actions**.

| Secret | Used by |
|---|---|
| `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (or `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`) | `paper_trade.yml` (paper keys only) |
| `SUPABASE_URL`, `SUPABASE_KEY` | `run_forwardtest.yml`, `update_outcomes.yml` |
| `HF_TOKEN` | `sync_to_hf.yml` |

`GITHUB_TOKEN` is provided automatically. Jobs that commit to the `ledger` or `forward-log` branch, or post alerts, request `contents: write` and `issues: write`.

## Receiving reports and alerts

Weekly reports and every failure alert are posted as comments on the **"Paper book weekly report"** issue. Subscribe to that issue to get them by email. Each alert links to the failed run and states the reason.

## Handling failures

### A paper-trade run halted on a safety check

The alert lists the failed checks. Nothing was sent.

1. Read the failed check's detail in the alert or in the run's `paper_trade.json` artifact.
2. Fix the cause. For example, wait for fresh data, or confirm an extreme move is real.
3. Re-run with **Actions → Paper Trade → Run workflow**, with **force** checked. The `not_duplicate` check prevents sending the same decision twice.

Never loosen a check to get a run through.

### The ledger failed verification

The alert includes the verifier's report and the first broken record.

1. Run `python src/leveledge/ledger.py verify <ledger URL>` locally to reproduce it.
2. Find the commit on the `ledger` branch that introduced the broken record.
3. Don't force-push or rewrite the branch. The forward-test job refuses to append to a broken chain until the problem is understood.

### The timestamp authority was unavailable

The forecasts are still appended, unanchored, and the next run anchors them. Forecasts whose first covering anchor comes after their target don't count as on time, so they're excluded from the headline score.

### The forward log skipped a day

The alert says which close was missed. That day's outlook stays missing; the log is append-only and isn't backfilled.

### AWS sent a budget email

Run `bash deploy/aws/pause.sh` to stop the API, then check the AWS Billing console for the cause. Resume with `bash deploy/aws/pause.sh resume`. See [Deployment](deployment.md#controlling-cost).

### Scheduled jobs stopped running

GitHub disables scheduled workflows after 60 days with no repository activity, and emails the repository owner. Re-enable them under the repository's **Actions** tab.

## Running jobs manually

Every workflow supports **Run workflow** under the Actions tab. You can also run the underlying scripts locally; see [scripts/README.md](../scripts/README.md).
