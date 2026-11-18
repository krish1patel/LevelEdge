#!/usr/bin/env python3
"""
Run the daily market scan locally (the same analysis the AWS workers run) and write it as JSON.

  python scripts/market_scan.py                         # all ~100 tickers, 4 processes
  python scripts/market_scan.py --tickers NVDA SPY KO   # a few
  SCAN_FILE=scan.json uvicorn app.api:app               # show it on the dashboard

The production scan runs on AWS after each weekday close (deploy/aws/scan.sh).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
os.environ.setdefault("LEVELEDGE_DISABLE_LOGGING", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from leveledge import scan  # noqa: E402
from leveledge.constants import US_EASTERN  # noqa: E402
from leveledge.safety import last_completed_session  # noqa: E402


def one(t: str):
    started = time.time()
    try:
        r = scan.analyze(t)
        r["seconds"] = round(time.time() - started, 2)
        return t, r, None
    except Exception as e:  # noqa: BLE001
        return t, None, f"{e.__class__.__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="scan.json")
    a = ap.parse_args()
    tickers = a.tickers or [s for s, _, _ in scan.UNIVERSE]
    started = datetime.now(timezone.utc)
    rows, errors = [], {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(one, t) for t in tickers]):
            t, r, err = fut.result()
            if r:
                rows.append(r)
                print(f"{t:6s} ±{r['swing'] * 100:4.1f}%  {r['insights'][0]['text'][:90]}")
            else:
                errors[t] = err
                print(f"{t:6s} failed: {err}")
    out = {"scan_date": last_completed_session(datetime.now(tz=US_EASTERN)).strftime("%Y-%m-%d"),
           "started_at": started.isoformat(), "total": len(tickers), "rows": rows, "errors": errors}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"\n{len(rows)} of {len(tickers)} analyzed in {(datetime.now(timezone.utc) - started).seconds} s -> {a.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
