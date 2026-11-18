# Forecast ledger

The forecast ledger is a public, append-only record of every live forecast. Each forecast is committed before its outcome exists, timestamped by an independent authority, and later scored by appending a new record. Anyone can check that the record was never edited. This guide shows you how to verify it, explains the guarantees, and describes the record format.

## Verifying the ledger

You need Python 3 and, for the timestamp check, OpenSSL. The verifier, `src/leveledge/ledger.py`, uses only the standard library, so you can download that single file and run it anywhere.

```bash
python src/leveledge/ledger.py verify \
  https://raw.githubusercontent.com/krish1patel/LevelEdge/ledger/ledger.jsonl --check-tsa
```

A successful run ends with:

```text
LEDGER VERIFIED
```

The report above it lists the number of records, forecasts, scores and anchors, the head hash, and the Brier score of forecasts that were timestamped before their target.

To also re-check every recorded close against Yahoo Finance, run the full CLI from a checkout:

```bash
python scripts/ledger.py verify --check-closes
```

If verification fails, the report names the first broken record and the reason, for example `contents do not match its hash: the record was edited`.

## Understanding the guarantees

| Mechanism | What it proves |
|---|---|
| **Hash chain** | Each record's SHA-256 hash covers its contents and the previous record's hash. Editing, deleting, inserting or reordering any record changes every later hash. |
| **Outside timestamps** | After each batch of forecasts, freetsa.org signs the chain's newest hash (RFC 3161). The signed token proves every record up to that hash existed at that time, and the ledger's operator can't backdate it. A hash chain alone can't prove *when*: its owner could rebuild the whole chain with backdated forecasts. |
| **On-time rule** | A forecast counts only if a timestamp covering it was signed before its target time. |
| **Recomputed scores** | The verifier recomputes every score from the committed probabilities and the recorded close. A score that doesn't follow from them fails verification. |
| **Idempotent scoring** | Each forecast gets exactly one score, so a retried job can't double-score. |

### Limits

- The ledger proves the record wasn't edited after the fact. It doesn't prove the forecasts are good; the scores measure that.
- Closes come from Yahoo Finance. The verifier can re-check them but doesn't control the source.
- The file grows by roughly 100 forecasts per trading day. Splitting it into monthly files is the next step once it gets large.

## How records get written

1. **Forecasts.** After each forward-test run (10:00, 12:00 and 14:00 ET), `run_forwardtest.yml` appends one record per ticker and bar interval, holding eight levels and their probabilities. In the same push it requests a timestamp on the new head. A forecast whose target has passed is refused, and a malformed forecast is rejected before it's written, because a written record can never be removed.
2. **Scores.** After the close, `ledger_reconcile.yml` appends one score per matured forecast. A score holds the official close, which levels the price closed above, and the Brier score. The job then verifies the whole public ledger, token signatures included, and posts an alert if anything fails.
3. **Concurrency.** `scripts/ledger_push.sh` appends with optimistic concurrency. If another job pushed first, the push is rejected, and the attempt is rebuilt on the new head and retried. The branch history is only ever extended, never force-pushed.

## Record format

`ledger.jsonl` holds one JSON object per line:

```json
{"seq": 12, "kind": "forecast", "created_at": "2026-11-18T15:00:41.123456Z",
 "body": {...}, "prev": "<hash of record 11>", "hash": "<hash of this record>"}
```

`hash` is the SHA-256 of the canonical JSON (sorted keys, no spaces, UTF-8) of every field except `hash`.

| Kind | Body fields |
|---|---|
| `genesis` | The ledger's name, its rules, and how to verify it |
| `forecast` | `type`, `ticker`, `interval`, `made_at`, `target`, `last_bar`, `spot`, `levels`, `p_above_ppm`, `model`, `code` (the commit that produced it) |
| `anchor` | `head`, `head_seq`, `tsa`, `gen_time`, and `token` (the base64 RFC 3161 token) |
| `score` | `forecast` (its hash), `forecast_seq`, `ticker`, `target`, `close`, `above` (1 if the close was strictly above each level), `brier_ppm`, `anchored_before_target`, `anchor_seq` |

Probabilities and Brier scores are integers in parts per million, so the hashed text never depends on how a language prints a float such as `1e-05`.

## Working with the ledger in code

```python
from leveledge import ledger as L

records = L.load("https://raw.githubusercontent.com/krish1patel/LevelEdge/ledger/ledger.jsonl")
report = L.verify(records)
print(report["ok"], report["forecasts"], report["scored_on_time"], report["brier_on_time"])
```

The format is fixed: changing `canonical()`, `record_hash()` or the record fields would break every existing hash.
