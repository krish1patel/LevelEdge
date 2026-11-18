#!/usr/bin/env python3
"""
Operate the verifiable forecast ledger (leveledge/ledger.py) on a local copy of the
public `ledger` branch. The GitHub workflows call these; scripts/ledger_push.sh commits.

  python scripts/ledger.py init      --path L        # genesis record if the file is new
  python scripts/ledger.py append    --path L --bodies forecasts.json
  python scripts/ledger.py reconcile --path L        # score matured forecasts (idempotent)
  python scripts/ledger.py anchor    --path L        # RFC 3161 timestamp on the head
  python scripts/ledger.py verify    --path L|URL [--check-tsa] [--check-closes]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from leveledge import ledger as L  # noqa: E402

PUBLIC_URL = "https://raw.githubusercontent.com/krish1patel/LevelEdge/ledger/ledger.jsonl"
CLOSE_SOURCE = "Yahoo Finance daily close (unadjusted) for the target session"
GENESIS = {
    "name": "LevelEdge forecast ledger",
    "rules": [
        "Every live forecast is appended here before its target time; records are never edited or deleted.",
        "Each record's hash covers the previous record's hash, so any edit breaks every later hash.",
        "Anchors are RFC 3161 timestamps from freetsa.org on the head hash; a forecast counts only if an "
        "anchor covering it was signed before its target time.",
        "Scores are appended after the target: above = close strictly above the level; Brier in parts per million.",
    ],
    "verify": "python src/leveledge/ledger.py verify " + PUBLIC_URL,
}


def close_for(ticker: str, target_iso: str) -> float | None:
    """The official close of the target session, once it is final (15 minutes after the target)."""
    import pandas as pd
    import yfinance as yf

    t = L.parse_time(target_iso)
    if datetime.now(timezone.utc) < t + timedelta(minutes=15):
        return None
    day = pd.Timestamp(t.date())
    h = yf.Ticker(ticker).history(start=day, end=day + pd.Timedelta(days=1), interval="1d", auto_adjust=False)
    if h.empty:
        return None
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    return float(h["Close"].loc[day]) if day in h.index else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "append", "reconcile", "anchor", "verify"])
    ap.add_argument("--path", default=None)
    ap.add_argument("--bodies", default=None)
    ap.add_argument("--check-tsa", action="store_true")
    ap.add_argument("--check-closes", action="store_true")
    ap.add_argument("--anchor", action="store_true", help="after appending, timestamp the new head (best effort)")
    a = ap.parse_args()

    if a.cmd == "verify":
        recs = L.load(a.path or PUBLIC_URL)
        rep = L.verify(recs, check_tsa=a.check_tsa,
                       certs_dir=L.fetch_tsa_certs(tempfile.mkdtemp()) if a.check_tsa else None)
        if a.check_closes:
            bad = []
            for r in recs:
                if r["kind"] == "score":
                    c = close_for(r["body"]["ticker"], r["body"]["target"])
                    if c is not None and abs(c - r["body"]["close"]) > 0.011:
                        bad.append(f"record {r['seq']}: {r['body']['ticker']} close {r['body']['close']} vs {c:.4f} now")
            rep["closes_checked"], rep["close_mismatches"] = True, bad
            rep["ok"] = rep["ok"] and not bad
        print(json.dumps(rep, indent=1))
        return 0 if rep["ok"] else 1

    led = L.Ledger(a.path)
    if a.cmd == "init":
        if not led.records:
            led.append("genesis", GENESIS)
        print(f"ledger has {len(led.records)} records")
        return 0
    if not led.records:
        raise SystemExit("ledger is empty: run init first")
    rep = L.verify(led.records)
    if not rep["ok"]:                                   # never build on a broken chain
        raise SystemExit("refusing to append: " + "; ".join(rep["problems"]))

    if a.cmd == "append":
        with open(a.bodies) as f:
            bodies = json.load(f)
        have = led.forecast_keys()
        new = [b for b in bodies if L.forecast_key(b) not in have]
        bad = [(b, L.check_forecast(b)) for b in new]
        bad = [(b, why) for b, why in bad if why]
        late = [b for b in new if L.parse_time(b["target"]) <= datetime.now(timezone.utc)]
        ok = [b for b in new if b not in late and all(b is not x for x, _ in bad)]
        for b in ok:
            led.append("forecast", b)
        for b, why in bad:
            print(f"  refused {b.get('ticker')} {b.get('interval')}: {why}")
        print(f"appended {len(ok)} forecasts ({len(bodies) - len(new)} already present, "
              f"{len(late)} refused: target already passed, {len(bad)} malformed)")
    elif a.cmd == "reconcile":
        added = L.reconcile(led, close_for, source=CLOSE_SOURCE)
        print(f"appended {len(added)} scores")
    if a.cmd == "anchor" or (a.anchor and led.head["kind"] != "anchor"):
        if led.head["kind"] == "anchor":
            print("head is already anchored")
        else:
            try:
                r = L.anchor(led)
                print(f"anchored record {r['body']['head_seq']} at {r['body']['gen_time']}")
            except Exception as e:  # noqa: BLE001 - the records still go in; the next run anchors them
                print(f"timestamp authority unavailable ({e}); records stay unanchored until the next run")
                if a.cmd == "anchor":
                    return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
