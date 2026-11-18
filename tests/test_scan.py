import json
from datetime import datetime

import numpy as np
import pandas as pd

from leveledge import scan
from leveledge.constants import US_EASTERN


def _row(**kw):
    r = {"ticker": "XYZ", "name": "XYZ Corp", "swing": 0.03, "p_up": 0.53, "vol_pct": 0.5, "vs_sma200": 0.1,
         "from_high": -0.05, "ret_1m": 0.02, "coverage80": 0.8}
    r.update(kw)
    return r


def test_context_reads_trend_volatility_and_drawdown():
    rng = np.random.default_rng(0)
    calm = rng.normal(0.0005, 0.008, 1400)
    wild = rng.normal(0.0, 0.04, 21)                       # the last month swings far more than usual
    close = pd.Series(100 * np.exp(np.cumsum(np.r_[calm, wild])))
    c = scan.context(close)
    assert c["vol_pct"] > 0.95
    assert c["from_high"] <= 0
    assert c["ret_1d"] is not None and c["ret_12m"] is not None


def test_insights_flag_what_is_unusual_and_rank_it():
    notes = scan.insights(_row(vol_pct=0.97, from_high=-0.4, p_up=0.66))
    kinds = [n["kind"] for n in notes]
    assert kinds[0] in ("volatile", "drawdown") and "lean" in kinds
    assert all(notes[i]["weight"] >= notes[i + 1]["weight"] for i in range(len(notes) - 1))
    assert scan.insights(_row())[0]["kind"] == "steady"
    assert scan.insights(_row(vs_sma200=0.004))[0]["kind"] == "trend_edge"
    assert scan.insights(_row(coverage80=0.6))[0]["kind"] == "caution"


def test_notable_keeps_one_insight_per_ticker_and_skips_steady():
    rows = [dict(_row(ticker=t, vol_pct=v), insights=scan.insights(_row(vol_pct=v))) for t, v in (("A", 0.99), ("B", 0.9), ("C", 0.5))]
    top = scan.notable(rows)
    assert [n["ticker"] for n in top] == ["A", "B"]
    many = [dict(_row(ticker=f"T{i}", vol_pct=0.99), insights=scan.insights(_row(vol_pct=0.99))) for i in range(6)]
    trend = [dict(_row(ticker="E"), insights=scan.insights(_row(vs_sma200=0.004)))]
    mix = scan.notable(many + trend)
    assert sum(n["kind"] == "volatile" for n in mix) == 3 and any(n["kind"] == "trend_edge" for n in mix)


def test_week_target_is_five_sessions_ahead_and_skips_thanksgiving():
    t = scan.week_target(datetime(2026, 11, 20, 18, 0, tzinfo=US_EASTERN))      # Friday evening
    assert t.date().isoformat() == "2026-11-30"                                 # Mon, Tue, Wed, (Thu closed), Fri, Mon


class FakeTable:
    def __init__(self):
        self.items = {}

    def get_item(self, Key):
        k = (Key["scan_date"], Key["ticker"])
        return {"Item": self.items[k]} if k in self.items else {}

    def put_item(self, Item, ConditionExpression=None):
        from botocore.exceptions import ClientError
        k = (Item["scan_date"], Item["ticker"])
        if ConditionExpression and k in self.items:
            raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")
        self.items[k] = Item


class FakeSQS:
    def __init__(self):
        self.sent = []

    def send_message_batch(self, QueueUrl, Entries):
        assert len(Entries) <= 10
        self.sent += [json.loads(e["MessageBody"]) for e in Entries]


def test_worker_dispatch_batches_the_universe(monkeypatch):
    from app import worker
    table, sqs = FakeTable(), FakeSQS()
    monkeypatch.setattr(worker, "_aws", lambda: (table, sqs))
    out = worker.dispatch(datetime(2026, 11, 18, 17, 30, tzinfo=US_EASTERN))
    assert out["scan_date"] == "2026-11-18"
    assert sum(len(m["tickers"]) for m in sqs.sent) == len(scan.UNIVERSE)
    assert all(len(m["tickers"]) <= worker.BATCH for m in sqs.sent)
    assert table.items[("latest", "#pointer")]["value"] == "2026-11-18"


def test_worker_is_idempotent_and_reports_only_failed_messages(monkeypatch):
    from app import worker
    table = FakeTable()
    monkeypatch.setattr(worker, "_aws", lambda: (table, None))
    calls = []

    def fake_analyze(t):
        calls.append(t)
        if t == "BAD":
            raise RuntimeError("no data")
        return {"ticker": t, "swing": 0.03, "insights": []}

    monkeypatch.setattr(worker.scan, "analyze", fake_analyze)
    recs = [{"messageId": "m1", "body": json.dumps({"scan_date": "2026-11-18", "tickers": ["AAA", "BBB"]})},
            {"messageId": "m2", "body": json.dumps({"scan_date": "2026-11-18", "tickers": ["CCC", "BAD"]})}]
    out = worker.process(recs)
    assert out == {"batchItemFailures": [{"itemIdentifier": "m2"}]}
    assert {k[1] for k in table.items} == {"AAA", "BBB", "CCC"}
    calls.clear()
    worker.process(recs)                                   # SQS redelivers both messages
    assert calls == ["BAD"]                                # finished tickers are skipped, not recomputed
