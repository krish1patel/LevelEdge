# Market scan

Every weekday after the close, LevelEdge runs its forecasting model on 100 major US stocks and funds and turns the results into plain-English notes on the dashboard's **Market scan** page. Nobody has to open 100 charts. This guide explains what the scan computes, how the AWS pipeline runs it, and how to run it yourself.

## What the scan computes

For each ticker, `leveledge.scan.analyze()`:

1. Downloads the ticker's full daily price history from Yahoo Finance.
2. Fits the v2 forecaster for the close 5 trading days ahead. The fit includes the model's walk-forward self-test on that ticker's own history.
3. Records the one-week 80% range, the expected move (half the range, as a percentage), the chance the price is higher in a week, and how often the model's 80% ranges held in testing.
4. Adds context from the price history: distance from the 200-day average (the long-term trend), how unusual the past month's swings are against the past five years, distance from the 52-week high, and recent returns.
5. Writes notes with `insights()`: unusually volatile or calm, near a trend flip, deep below its high, a big month, a model lean, or a reliability caution. `notable()` ranks the most noteworthy across all 100, with at most one note per ticker and three of each kind.

The model forecasts the size of moves well and rarely has a strong view on direction. Notes say so when it does lean.

## How the pipeline runs

```text
EventBridge Scheduler (5:45 PM ET, Mon-Fri)
  └─▶ leveledge-scan Lambda: {"action": "dispatch"} ─▶ 20 batches of 5 tickers
        └─▶ SQS queue leveledge-scan (dead-letter queue after 3 failed attempts)
              └─▶ leveledge-scan Lambda workers, at most 5 at once
                    └─▶ DynamoDB table leveledge-scan (one item per ticker per day; expires after 120 days)
                          └─▶ GET /api/scan ─▶ Market scan page
```

| Property | How it's achieved |
|---|---|
| Exact timing | EventBridge Scheduler in the `America/New_York` timezone. GitHub's scheduler, by contrast, often starts jobs hours late. |
| Parallelism with backpressure | The SQS event source mapping runs at most 5 workers. Extra batches wait in the queue. |
| Idempotency | A worker skips a ticker that already has a result, and writes with `attribute_not_exists`, so a redelivered message never duplicates or overwrites work. |
| Partial failure | A worker reports only the failed message back to SQS (`ReportBatchItemFailures`). After 3 attempts the message moves to the dead-letter queue. |
| Least privilege | The worker's role can use only its queue and its table; the public API can only read the table. |
| Cost | About 12 s of Lambda time per ticker at 1.5 GB: roughly 1,800 GB-seconds a day, about 10% of the free tier over a month. SQS, DynamoDB and the scheduler stay within their free tiers. |

The first cloud scan, on the 2026-11-17 close, analyzed all 100 tickers in about 5 minutes, with no failed batches.

## Running the scan

### On AWS

```bash
aws login
bash deploy/aws/scan.sh --run-now
```

`scan.sh` creates or updates every resource and is safe to re-run. `--run-now` starts a scan immediately. Check progress:

```bash
aws dynamodb query --table-name leveledge-scan --select COUNT \
  --key-condition-expression 'scan_date = :d' --expression-attribute-values '{":d":{"S":"2026-11-17"}}'
```

The count reaches 101 when the scan is done: 100 tickers plus one record for the scan itself.

### Locally

```bash
python scripts/market_scan.py --workers 6 --out scan.json   # about 2 minutes on a laptop
SCAN_FILE=scan.json uvicorn app.api:app --port 7860           # show it on the dashboard
```

## Changing the scan

- **Universe:** edit `UNIVERSE` in `src/leveledge/scan.py`.
- **Notes:** edit `insights()`. The API regenerates notes from the stored numbers on every read, so wording changes apply to past scans too.
- **Schedule:** edit the cron expression in `deploy/aws/scan.sh` and re-run it.
