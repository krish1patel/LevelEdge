"""
Market-scan worker (AWS Lambda, same container image as the API).

The AWS Lambda Web Adapter delivers non-HTTP events as POST /events. Two kinds arrive:

* EventBridge Scheduler, after each weekday close: {"action": "dispatch"}. The worker splits the
  universe into batches of BATCH tickers and sends one SQS message per batch, and records the scan's
  start in DynamoDB.
* SQS, one message (batch) at a time: {"Records": [...]} with {"scan_date", "tickers"}. The worker
  analyzes each ticker and writes one DynamoDB item per ticker with a conditional put, so a redelivered
  message never overwrites or duplicates a result. Tickers that fail are reported back as a partial
  batch failure, so SQS retries only that message; after 3 attempts it moves to the dead-letter queue.

This app has no public URL: the worker function has no Function URL, and the public API image runs
app.api instead. Run locally for a test with: SCAN_TABLE=... uvicorn app.worker:app
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request

os.environ.setdefault("LEVELEDGE_DISABLE_LOGGING", "1")

from leveledge import scan  # noqa: E402
from leveledge.constants import US_EASTERN  # noqa: E402

BATCH = int(os.environ.get("SCAN_BATCH", "5"))
TABLE = os.environ.get("SCAN_TABLE", "leveledge-scan")
QUEUE_URL = os.environ.get("SCAN_QUEUE_URL", "")
KEEP_DAYS = 120

app = FastAPI(title="LevelEdge scan worker", docs_url=None, redoc_url=None)


def _aws():
    import boto3
    return boto3.resource("dynamodb").Table(TABLE), boto3.client("sqs")


def scan_date(now: datetime) -> str:
    from leveledge.safety import last_completed_session
    return last_completed_session(now).strftime("%Y-%m-%d")


def _decimalize(x):
    """DynamoDB stores numbers as Decimal: convert floats (and nested floats) on the way in."""
    from decimal import Decimal
    if isinstance(x, float):
        return Decimal(str(x))
    if isinstance(x, dict):
        return {k: _decimalize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_decimalize(v) for v in x]
    return x


def dispatch(now: datetime | None = None) -> dict:
    now = now or datetime.now(tz=US_EASTERN)
    day = scan_date(now)
    table, sqs = _aws()
    tickers = [s for s, _, _ in scan.UNIVERSE]
    batches = [tickers[i:i + BATCH] for i in range(0, len(tickers), BATCH)]
    expires = int((now + timedelta(days=KEEP_DAYS)).timestamp())
    table.put_item(Item={"scan_date": day, "ticker": "#meta", "total": len(tickers), "batches": len(batches),
                         "started_at": datetime.now(timezone.utc).isoformat(), "expires": expires})
    table.put_item(Item={"scan_date": "latest", "ticker": "#pointer", "value": day})
    for i in range(0, len(batches), 10):                         # SQS sends at most 10 messages per call
        entries = [{"Id": str(j), "MessageBody": json.dumps({"scan_date": day, "tickers": b})}
                   for j, b in enumerate(batches[i:i + 10])]
        sqs.send_message_batch(QueueUrl=QUEUE_URL, Entries=entries)
    return {"scan_date": day, "tickers": len(tickers), "batches": len(batches)}


def process(records: list[dict]) -> dict:
    from botocore.exceptions import ClientError

    table, _ = _aws()
    failures = []
    for rec in records:
        body = json.loads(rec["body"])
        day, failed = body["scan_date"], []
        for t in body["tickers"]:
            if "Item" in table.get_item(Key={"scan_date": day, "ticker": t}):
                continue                                         # done by an earlier delivery: skip (idempotent)
            try:
                started = time.time()
                r = scan.analyze(t)
                r["seconds"] = round(time.time() - started, 2)
            except Exception as e:  # noqa: BLE001 - record it; SQS retries the message
                print(f"{t}: {e.__class__.__name__}: {e}")
                failed.append(t)
                continue
            item = _decimalize({"scan_date": day, "ticker": t, "data": r,
                                "expires": int((datetime.now(timezone.utc) + timedelta(days=KEEP_DAYS)).timestamp())})
            try:
                table.put_item(Item=item, ConditionExpression="attribute_not_exists(ticker)")
            except ClientError as e:
                if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                    raise
        if failed:
            failures.append({"itemIdentifier": rec["messageId"]})
    return {"batchItemFailures": failures}


@app.get("/healthz")
def health():
    return {"ok": True}


@app.post("/events")
async def events(request: Request):
    event = await request.json()
    if isinstance(event, dict) and event.get("action") == "dispatch":
        return dispatch()
    if isinstance(event, dict) and event.get("Records"):
        return process(event["Records"])
    return {"ignored": True}
