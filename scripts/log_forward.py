#!/usr/bin/env python3
"""
Append today's market outlook and the book's targets to the public forward log.

    python scripts/log_forward.py --log forward-log/outlook.jsonl

One JSON line per completed close, written before any of its outcomes are known.
The GitHub workflow commits the file to the `forward-log` branch, so every forecast
carries a public timestamp and cannot be edited after the fact without it showing.
Running twice for the same close does nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from leveledge import live, outlook  # noqa: E402
from leveledge.constants import US_EASTERN  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    args = ap.parse_args()
    seen = set()
    if os.path.exists(args.log):
        with open(args.log) as f:
            seen = {json.loads(line)["as_of_close"] for line in f if line.strip()}
    now = datetime.now(tz=US_EASTERN)
    o = outlook.market_outlook(now=now)
    if o["as_of_close"] in seen:
        print(f"Already logged the {o['as_of_close']} close; nothing to do.")
        return 0
    try:
        targets = live.current_targets(now=now)
    except Exception as e:  # noqa: BLE001 - the outlook still gets logged
        print(f"[warn] targets unavailable: {e}")
        targets = None
    entry = outlook.forward_entry(o, targets, now)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    with open(args.log, "a") as f:
        f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    m = {h["h"]: h for h in entry["horizons"]}
    print(f"Logged the {o['as_of_close']} close: S&P {o['spot']:.2f}; 1-month 80% range "
          f"{m[21]['ret']['q100']:+.1%} to {m[21]['ret']['q900']:+.1%}, resolves {m[21]['target_date']}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
