"""
Verifiable forecast ledger: an append-only, hash-chained record of every live forecast,
committed before its outcome exists, timestamped by an outside authority, and scored by
appending (never editing).

This file is standard-library only, so anyone can check the public ledger with nothing
but Python:

    python ledger.py verify https://raw.githubusercontent.com/krish1patel/LevelEdge/ledger/ledger.jsonl
    python ledger.py verify ledger.jsonl --check-tsa      # also check the timestamp tokens (needs openssl)

Record (one JSON object per line):
    {"seq": n, "kind": "genesis"|"forecast"|"score"|"anchor", "created_at": ISO-8601 UTC,
     "body": {...}, "prev": <hash of record n-1>, "hash": <this record's hash>}
    hash = SHA-256 of the canonical JSON of every field except "hash"

What it proves, and what it does not:
* Hash chain: changing, deleting, inserting or reordering any record changes every later
  hash, so a partial edit is detectable by anyone holding a later hash.
* Anchors: an RFC 3161 timestamp authority (freetsa.org) signs the chain's head hash. The
  token proves every record up to that head existed at the authority's time, which the
  ledger's operator cannot fake. Without anchors a full rewrite of the chain would go
  unnoticed; with them a forecast only counts if an anchor covering it predates its target.
* Scores are recomputed by the verifier from the committed probabilities and the recorded
  close. The close itself is a public fact the verifier can re-check (scripts/ledger.py).
* It proves the record was not altered. It does not prove the forecasts are any good.

Numbers that feed the hash are integers or plain decimals: probabilities are stored in
parts per million, so the canonical form never depends on how a language prints a float
like 1e-05.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from datetime import datetime, timezone

GENESIS_PREV = "0" * 64
PPM = 1_000_000
KINDS = ("genesis", "forecast", "score", "anchor")
TSA_URL = "https://freetsa.org/tsr"
TSA_CERTS = ("https://freetsa.org/files/cacert.pem", "https://freetsa.org/files/tsa.crt")


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
def canonical(obj) -> bytes:
    """One byte string per value: sorted keys, no spaces, UTF-8, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def record_hash(rec: dict) -> str:
    return hashlib.sha256(canonical({k: rec[k] for k in ("seq", "kind", "created_at", "body", "prev")})).hexdigest()


def make_record(prev: dict | None, kind: str, body: dict, created_at: str | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    rec = {"seq": 0 if prev is None else prev["seq"] + 1, "kind": kind,
           "created_at": created_at or utc_now(), "body": body,
           "prev": GENESIS_PREV if prev is None else prev["hash"]}
    rec["hash"] = record_hash(rec)
    return rec


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def to_ppm(p: float) -> int:
    return int(round(min(1.0, max(0.0, float(p))) * PPM))


# ---------------------------------------------------------------------------
# storage: an append-only JSON-lines file (git and the public branch do the rest)
# ---------------------------------------------------------------------------
class Ledger:
    def __init__(self, path: str):
        self.path = path
        self.records: list[dict] = load(path) if os.path.exists(path) else []

    @property
    def head(self) -> dict | None:
        return self.records[-1] if self.records else None

    def append(self, kind: str, body: dict, created_at: str | None = None) -> dict:
        """Append one record. The file is only ever opened for appending."""
        if not self.records and kind != "genesis":
            raise ValueError("the first record must be the genesis record")
        rec = make_record(self.head, kind, body, created_at)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(canonical(rec).decode() + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.append(rec)
        return rec

    def forecast_keys(self) -> set:
        return {forecast_key(r["body"]) for r in self.records if r["kind"] == "forecast"}

    def scored(self) -> set:
        return {r["body"]["forecast"] for r in self.records if r["kind"] == "score"}


def load(source: str) -> list[dict]:
    if re.match(r"^https?://", source):
        with urllib.request.urlopen(urllib.request.Request(source, headers={"User-Agent": "ledger-verify"}), timeout=60) as r:
            text = r.read().decode("utf-8")
    else:
        with open(source, encoding="utf-8") as f:
            text = f.read()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# forecasts and scores
# ---------------------------------------------------------------------------
def forecast_key(body: dict) -> tuple:
    """One forecast per model run, ticker and interval (makes re-appending idempotent)."""
    return (body["made_at"], body["ticker"], body["interval"], body["target"])


def check_forecast(body: dict) -> str | None:
    """Why a forecast body is malformed, or None. Checked before appending, because a bad
    record can never be removed later."""
    try:
        for k in ("type", "ticker", "interval", "made_at", "target", "levels", "p_above_ppm"):
            if k not in body:
                return f"missing {k}"
        if parse_time(body["target"]) <= parse_time(body["made_at"]):
            return "target is not after made_at"
        lv, p = body["levels"], body["p_above_ppm"]
        if not lv or len(lv) != len(p):
            return "levels and probabilities differ in length"
        if not all(isinstance(x, int) and 0 <= x <= PPM for x in p):
            return "probabilities must be integers in parts per million"
        pairs = sorted(zip(map(float, lv), p))
        if any(hi[1] > lo[1] for lo, hi in zip(pairs, pairs[1:])):
            return "P(above) rises with the level"
    except (TypeError, ValueError) as e:
        return f"unreadable: {e}"
    return None


def level_outcomes(levels: list, close: float) -> list[int]:
    return [1 if float(close) > float(k) else 0 for k in levels]       # "above" means strictly above


def brier_ppm(p_ppm: list[int], outcomes: list[int]) -> int:
    """Mean squared error in parts per million, exact integer arithmetic (rounded half up)."""
    num = sum((p - o * PPM) ** 2 for p, o in zip(p_ppm, outcomes))
    den = len(p_ppm) * PPM
    return (num + den // 2) // den


def covering_anchor(records: list[dict], seq: int) -> dict | None:
    """The earliest anchor whose head covers record ``seq``."""
    for r in records:
        if r["kind"] == "anchor" and r["body"]["head_seq"] >= seq:
            return r
    return None


def score_body(forecast: dict, close: float, records: list[dict], source: str) -> dict:
    b = forecast["body"]
    out = level_outcomes(b["levels"], close)
    anc = covering_anchor(records, forecast["seq"])
    on_time = bool(anc and parse_time(anc["body"]["gen_time"]) < parse_time(b["target"]))
    return {"forecast": forecast["hash"], "forecast_seq": forecast["seq"], "ticker": b["ticker"],
            "target": b["target"], "close": close, "close_source": source, "above": out,
            "brier_ppm": brier_ppm(b["p_above_ppm"], out),
            "anchored_before_target": on_time, "anchor_seq": anc["seq"] if anc else None}


def reconcile(ledger: Ledger, close_for, now: datetime | None = None, source: str = "") -> list[dict]:
    """Append one score per matured, unscored forecast. Safe to run any number of times:
    a forecast that already has a score is skipped, so a retried job never double-scores."""
    now = now or datetime.now(timezone.utc)
    done, added = ledger.scored(), []
    for f in [r for r in ledger.records if r["kind"] == "forecast"]:
        if f["hash"] in done or parse_time(f["body"]["target"]) > now:
            continue
        close = close_for(f["body"]["ticker"], f["body"]["target"])
        if close is None:
            continue                                                    # outcome not published yet: try next run
        added.append(ledger.append("score", score_body(f, round(float(close), 4), ledger.records, source),
                                   created_at=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")))
        done.add(f["hash"])
    return added


# ---------------------------------------------------------------------------
# anchors: RFC 3161 timestamps on the chain head
# ---------------------------------------------------------------------------
def _openssl(*args, data: bytes | None = None) -> bytes:
    return subprocess.run(["openssl", *args], input=data, capture_output=True, check=True).stdout


def request_timestamp(head_hash: str, tsa_url: str = TSA_URL) -> tuple[bytes, str]:
    """Ask the authority to sign the head hash; returns (token, its signing time in UTC)."""
    with tempfile.TemporaryDirectory() as d:
        data, q = os.path.join(d, "head"), os.path.join(d, "q.tsq")
        with open(data, "w") as f:
            f.write(head_hash)
        _openssl("ts", "-query", "-data", data, "-sha256", "-cert", "-out", q)
        with open(q, "rb") as f:
            req = urllib.request.Request(tsa_url, data=f.read(), headers={"Content-Type": "application/timestamp-query"})
        with urllib.request.urlopen(req, timeout=60) as r:
            token = r.read()
    return token, token_time(token)


def token_time(token: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tsr") as t:
        t.write(token)
        t.flush()
        text = _openssl("ts", "-reply", "-in", t.name, "-text").decode()
    if "Status: Granted" not in text:
        raise ValueError("timestamp not granted")
    m = re.search(r"Time stamp: (.+)", text)
    dt = datetime.strptime(m.group(1).strip().replace(" GMT", ""), "%b %d %H:%M:%S %Y")
    return dt.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def anchor(ledger: Ledger, tsa_url: str = TSA_URL) -> dict:
    head = ledger.head
    token, t = request_timestamp(head["hash"], tsa_url)
    return ledger.append("anchor", {"head": head["hash"], "head_seq": head["seq"], "tsa": tsa_url, "gen_time": t,
                                    "token": base64.b64encode(token).decode()})


def check_token(body: dict, certs_dir: str) -> bool:
    """Cryptographically verify an anchor's token against the authority's certificates."""
    with tempfile.TemporaryDirectory() as d:
        data, tsr = os.path.join(d, "head"), os.path.join(d, "t.tsr")
        with open(data, "w") as f:
            f.write(body["head"])
        with open(tsr, "wb") as f:
            f.write(base64.b64decode(body["token"]))
        try:
            _openssl("ts", "-verify", "-data", data, "-in", tsr, "-CAfile", os.path.join(certs_dir, "cacert.pem"),
                     "-untrusted", os.path.join(certs_dir, "tsa.crt"))
        except subprocess.CalledProcessError:
            return False
    return token_time(base64.b64decode(body["token"])) == body["gen_time"]


def fetch_tsa_certs(dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    for url in TSA_CERTS:
        with urllib.request.urlopen(url, timeout=60) as r, open(os.path.join(dest, url.rsplit("/", 1)[1]), "wb") as f:
            f.write(r.read())
    return dest


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def verify(records: list[dict], check_tsa: bool = False, certs_dir: str | None = None) -> dict:
    """Check everything that can be checked from the ledger alone. Returns a report;
    ``ok`` is False at the first broken record, with the reason."""
    problems: list[str] = []

    def fail(i, why):
        problems.append(f"record {i}: {why}")

    by_hash, scored = {}, set()
    prev = None
    for i, r in enumerate(records):
        if set(r) != {"seq", "kind", "created_at", "body", "prev", "hash"}:
            fail(i, "unexpected fields"); break
        if r["seq"] != i:
            fail(i, f"sequence number {r['seq']} (expected {i}): a record was removed, added or moved"); break
        if r["prev"] != (prev["hash"] if prev else GENESIS_PREV):
            fail(i, "does not point to the previous record's hash: the chain was edited"); break
        if record_hash(r) != r["hash"]:
            fail(i, "contents do not match its hash: the record was edited"); break
        if (i == 0) != (r["kind"] == "genesis"):
            fail(i, "only the first record may be the genesis record"); break
        if prev and parse_time(r["created_at"]) < parse_time(prev["created_at"]):
            fail(i, "created before the previous record"); break
        b = r["body"]
        if r["kind"] == "forecast":
            why = check_forecast(b)
            if why:
                fail(i, f"malformed forecast: {why}"); break
        elif r["kind"] == "score":
            f = by_hash.get(b["forecast"])
            if f is None or f["kind"] != "forecast":
                fail(i, "scores a forecast that is not earlier in the ledger"); break
            if b["forecast"] in scored:
                fail(i, "second score for the same forecast"); break
            scored.add(b["forecast"])
            if parse_time(r["created_at"]) < parse_time(f["body"]["target"]):
                fail(i, "scored before the forecast's target time"); break
            want = score_body(f, b["close"], records[:i], b.get("close_source", ""))
            for k in ("above", "brier_ppm", "anchored_before_target", "anchor_seq"):
                if want[k] != b[k]:
                    fail(i, f"{k} does not follow from the committed forecast and the recorded close"); break
            if problems:
                break
        elif r["kind"] == "anchor":
            if not (0 <= b["head_seq"] < i) or records[b["head_seq"]]["hash"] != b["head"]:
                fail(i, "anchor head does not match the chain"); break
            if check_tsa and not check_token(b, certs_dir):
                fail(i, "timestamp token does not verify"); break
        by_hash[r["hash"]] = r
        prev = r

    scores = [r["body"] for r in records if r["kind"] == "score"]
    on_time = [s for s in scores if s["anchored_before_target"]]
    forecasts = [r for r in records if r["kind"] == "forecast"]
    return {"ok": not problems, "problems": problems, "records": len(records),
            "forecasts": len(forecasts), "scored": len(scores), "scored_on_time": len(on_time),
            "pending": len(forecasts) - len(scores), "anchors": sum(r["kind"] == "anchor" for r in records),
            "head": records[-1]["hash"] if records else None, "head_seq": records[-1]["seq"] if records else None,
            "tsa_checked": bool(check_tsa),
            "brier_on_time": (sum(s["brier_ppm"] for s in on_time) / len(on_time) / PPM) if on_time else None}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] != "verify":
        print(__doc__)
        return 2
    check_tsa = "--check-tsa" in argv
    certs = fetch_tsa_certs(tempfile.mkdtemp()) if check_tsa else None
    rep = verify(load(argv[1]), check_tsa=check_tsa, certs_dir=certs)
    print(json.dumps(rep, indent=1))
    print("\nLEDGER VERIFIED" if rep["ok"] else "\nLEDGER FAILED VERIFICATION")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
