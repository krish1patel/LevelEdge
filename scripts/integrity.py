#!/usr/bin/env python3
"""
Run the research-integrity engine (leveledge.integrity) and write the trust scores.

  python scripts/integrity.py run                      # every strategy, locally, in parallel processes
  python scripts/integrity.py run --strategies book news_sentiment_spy
  python scripts/integrity.py submit                   # the same tasks, fanned out to AWS Lambda workers via SQS
  python scripts/integrity.py collect --run-id RUN     # gather the AWS results and score them

Both paths write benchmarks/results/integrity/summary.json, which the Research page shows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from leveledge import integrity as I  # noqa: E402

OUT = os.path.join(ROOT, "benchmarks", "results", "integrity")
_CTX: dict = {}


def _run(task: dict):
    uni = I.STRATEGIES[task["strategy"]]["universe"]
    if uni not in _CTX:
        _CTX[uni] = I.load_ctx(uni)
    t0 = time.time()
    return task["id"], I.run_task(task, _CTX[uni]), time.time() - t0


def write(results: dict, strategies: list[str], meta: dict) -> dict:
    import gzip
    os.makedirs(OUT, exist_ok=True)
    raw = os.path.join(OUT, "tasks.json.gz")                  # raw task results, so scoring can change without a rerun
    old = json.load(gzip.open(raw, "rt")) if os.path.exists(raw) and meta.get("partial") else {}
    old.update(results)
    with gzip.open(raw, "wt") as f:
        json.dump(old, f, default=float)
    reports = [I.aggregate(s, results) for s in strategies]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "weights": I.WEIGHTS, **meta,
           "strategies": sorted(reports, key=lambda r: -r["score"])}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "summary.json")
    prev = {}
    if os.path.exists(path) and meta.get("partial"):
        prev = {r["id"]: r for r in json.load(open(path))["strategies"]}
    if prev:
        prev.update({r["id"]: r for r in reports})
        out["strategies"] = sorted(prev.values(), key=lambda r: -r["score"])
    with open(path, "w") as f:
        json.dump(out, f, indent=1, default=float)
    for r in out["strategies"]:
        print(f"{r['grade']} {r['score']:5.1f}  {r['name']}")
        for line in r["findings"][:3]:
            print(f"         - {line}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "submit", "collect", "rescore"])
    ap.add_argument("--strategies", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--table", default=os.environ.get("SCAN_TABLE", "leveledge-scan"))
    ap.add_argument("--queue", default=os.environ.get("SCAN_QUEUE_NAME", "leveledge-scan"))
    a = ap.parse_args()
    strategies = a.strategies or list(I.STRATEGIES)
    tasks = [t for s in strategies for t in I.plan(s)]

    if a.cmd == "rescore":                                    # recompute scores from saved task results
        import gzip
        results = json.load(gzip.open(os.path.join(OUT, "tasks.json.gz"), "rt"))
        prev = json.load(open(os.path.join(OUT, "summary.json")))
        write(results, [s for s in strategies if f"{s}:base" in results],
              {k: prev.get(k) for k in ("executor", "run_id", "tasks", "compute_seconds")} | {"partial": False})
        return 0

    if a.cmd == "run":
        t0, results, secs = time.time(), {}, []
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for fut in as_completed([pool.submit(_run, t) for t in tasks]):
                tid, res, s = fut.result()
                results[tid] = res
                secs.append(s)
        print(f"{len(tasks)} tasks in {time.time() - t0:.0f} s on {a.workers} processes ({sum(secs):.0f} s of compute)\n")
        write(results, strategies, {"executor": "local", "tasks": len(tasks), "compute_seconds": round(sum(secs)),
                                    "partial": bool(a.strategies)})
        return 0

    import boto3
    if a.cmd == "submit":
        run_id = a.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        sqs = boto3.client("sqs")
        url = sqs.get_queue_url(QueueName=a.queue)["QueueUrl"]
        for i in range(0, len(tasks), 10):
            sqs.send_message_batch(QueueUrl=url, Entries=[{"Id": str(j), "MessageBody": json.dumps({"job": "integrity", "run_id": run_id, "task": t})}
                                                          for j, t in enumerate(tasks[i:i + 10])])
        boto3.resource("dynamodb").Table(a.table).put_item(Item={"scan_date": f"trust#{run_id}", "ticker": "#meta", "tasks": len(tasks),
                                                                  "strategies": strategies})
        print(f"Submitted {len(tasks)} tasks as run {run_id}. Collect with: python scripts/integrity.py collect --run-id {run_id}")
        return 0

    # collect
    from boto3.dynamodb.conditions import Key
    table = boto3.resource("dynamodb").Table(a.table)
    items, kw = [], {"KeyConditionExpression": Key("scan_date").eq(f"trust#{a.run_id}")}
    while True:
        page = table.query(**kw)
        items += page["Items"]
        if "LastEvaluatedKey" not in page:
            break
        kw["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    meta = next((i for i in items if i["ticker"] == "#meta"), {})
    done = {i["ticker"]: json.loads(i["data"]) for i in items if i["ticker"] != "#meta"}
    want = int(meta.get("tasks", len(tasks)))
    print(f"{len(done)} of {want} tasks finished")
    if len(done) < want:
        return 1
    strategies = list(meta.get("strategies") or strategies)
    write(done, strategies, {"executor": "aws", "run_id": a.run_id, "tasks": want,
                             "compute_seconds": round(sum(float(i.get("seconds", 0)) for i in items if i["ticker"] != "#meta")),
                             "partial": False})
    return 0


if __name__ == "__main__":
    sys.exit(main())
