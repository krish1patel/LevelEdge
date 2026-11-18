import copy
import json
from datetime import datetime, timezone

import pytest

from leveledge import ledger as L


def _fc(i=0, target="2026-11-18T21:00:00Z", made="2026-11-18T15:00:00Z"):
    return {"type": "level", "ticker": "SPY", "interval": "15m", "made_at": made, "target": target,
            "levels": [600.0 + i, 606.0 + i, 612.0 + i], "p_above_ppm": [800_000, 500_000, 200_000],
            "model": "test"}


def _anchor_body(led, t):
    h = led.head
    return {"head": h["hash"], "head_seq": h["seq"], "tsa": "test", "gen_time": t, "token": ""}


def _ledger(tmp_path, anchor_time="2026-11-18T15:05:00Z"):
    led = L.Ledger(str(tmp_path / "l.jsonl"))
    led.append("genesis", {"name": "t"}, "2026-11-18T14:00:00Z")
    led.append("forecast", _fc(0), "2026-11-18T15:00:01Z")
    led.append("forecast", _fc(1), "2026-11-18T15:00:02Z")
    led.append("anchor", _anchor_body(led, anchor_time), "2026-11-18T15:05:01Z")
    return led


def test_chain_verifies_and_reloads_identically(tmp_path):
    led = _ledger(tmp_path)
    rep = L.verify(led.records)
    assert rep["ok"], rep["problems"]
    assert L.load(led.path) == led.records
    assert rep["head"] == led.records[-1]["hash"]


def test_canonical_form_ignores_key_order():
    assert L.canonical({"b": 1, "a": [1, 2]}) == L.canonical({"a": [1, 2], "b": 1}) == b'{"a":[1,2],"b":1}'


@pytest.mark.parametrize("attack", ["edit_probability", "delete", "reorder", "insert", "edit_and_rehash"])
def test_every_kind_of_tampering_is_detected(tmp_path, attack):
    recs = copy.deepcopy(_ledger(tmp_path).records)
    if attack == "edit_probability":                      # change what was forecast
        recs[1]["body"]["p_above_ppm"][0] = 900_000
    elif attack == "delete":                              # hide an embarrassing forecast
        del recs[2]
    elif attack == "reorder":
        recs[1], recs[2] = recs[2], recs[1]
    elif attack == "insert":                              # sneak in a forecast after the fact
        recs.insert(2, L.make_record(recs[1], "forecast", _fc(5)))
    elif attack == "edit_and_rehash":                     # edit, then recompute that record's own hash
        recs[1]["body"]["p_above_ppm"][0] = 900_000
        recs[1]["hash"] = L.record_hash(recs[1])
    rep = L.verify(recs)
    assert not rep["ok"]
    assert rep["problems"]


def test_rebuilding_the_whole_chain_breaks_the_anchor(tmp_path):
    """A full rewrite yields a valid-looking chain, but the anchor signed the old head."""
    old = _ledger(tmp_path).records
    forged = [old[0]]
    for r in old[1:3]:
        body = copy.deepcopy(r["body"])
        body["p_above_ppm"] = [950_000, 500_000, 50_000]
        forged.append(L.make_record(forged[-1], r["kind"], body, r["created_at"]))
    forged.append(L.make_record(forged[-1], "anchor", old[3]["body"], old[3]["created_at"]))   # reuse the real token
    rep = L.verify(forged)
    assert not rep["ok"] and "anchor head" in rep["problems"][0]


def test_scoring_is_idempotent_and_recomputable(tmp_path):
    led = _ledger(tmp_path)
    after = datetime(2026, 11, 18, 22, tzinfo=timezone.utc)
    added = L.reconcile(led, lambda t, d: 607.5, now=after, source="test")
    assert len(added) == 2
    s = added[0]["body"]
    assert s["above"] == [1, 1, 0]                        # 607.5 > 600, > 606, not > 612
    # Brier = ((0.8-1)^2 + (0.5-1)^2 + (0.2-0)^2) / 3 = 0.11
    assert s["brier_ppm"] == 110_000
    assert s["anchored_before_target"] is True
    assert L.reconcile(led, lambda t, d: 607.5, now=after) == []      # a retried job adds nothing
    assert L.verify(led.records)["ok"]
    assert L.verify(led.records)["brier_on_time"] == pytest.approx(0.11)   # both forecasts: same outcomes at 607.5


def test_tampered_score_is_caught(tmp_path):
    led = _ledger(tmp_path)
    L.reconcile(led, lambda t, d: 607.5, now=datetime(2026, 11, 18, 22, tzinfo=timezone.utc))
    recs = copy.deepcopy(led.records)
    # forge a better score and re-chain everything after it
    recs[4]["body"]["brier_ppm"] = 1
    out = recs[:4]
    for r in recs[4:]:
        out.append(L.make_record(out[-1], r["kind"], r["body"], r["created_at"]))
    rep = L.verify(out)
    assert not rep["ok"] and "brier_ppm" in rep["problems"][0]


def test_unscored_until_the_target_and_the_close_exist(tmp_path):
    led = _ledger(tmp_path)
    assert L.reconcile(led, lambda t, d: 607.5, now=datetime(2026, 11, 18, 20, tzinfo=timezone.utc)) == []
    assert L.reconcile(led, lambda t, d: None, now=datetime(2026, 11, 18, 22, tzinfo=timezone.utc)) == []


def test_forecast_anchored_after_its_target_does_not_count(tmp_path):
    led = _ledger(tmp_path, anchor_time="2026-11-18T21:30:00Z")          # signed after the 21:00 target
    L.reconcile(led, lambda t, d: 607.5, now=datetime(2026, 11, 18, 22, tzinfo=timezone.utc))
    rep = L.verify(led.records)
    assert rep["ok"] and rep["scored"] == 2 and rep["scored_on_time"] == 0 and rep["brier_on_time"] is None


def test_malformed_forecasts_are_rejected_before_they_can_poison_the_chain():
    bad = _fc()
    bad["p_above_ppm"] = [200_000, 500_000, 800_000]                   # rises with the level
    assert "rises" in L.check_forecast(bad)
    assert L.check_forecast({**_fc(), "p_above_ppm": [0.8, 0.5, 0.2]}) is not None
    assert L.check_forecast({**_fc(), "target": "2026-11-18T14:00:00Z"}) is not None
    assert L.check_forecast(_fc()) is None


def test_first_record_must_be_genesis(tmp_path):
    led = L.Ledger(str(tmp_path / "x.jsonl"))
    with pytest.raises(ValueError):
        led.append("forecast", _fc())


def test_file_is_plain_json_lines(tmp_path):
    led = _ledger(tmp_path)
    lines = open(led.path).read().splitlines()
    assert len(lines) == 4 and all(json.loads(x)["hash"] for x in lines)


def test_forward_test_runs_each_slot_once(monkeypatch):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import forwardtest as F
    from leveledge.constants import US_EASTERN
    at = lambda h, m: datetime(2026, 11, 18, h, m, tzinfo=US_EASTERN)   # noqa: E731
    assert F.due_slot(at(9, 30)) is None and F.due_slot(at(15, 59)).hour == 14
    made = at(12, 40).isoformat()
    monkeypatch.setattr(L, "load", lambda url: [{"kind": "forecast", "body": {"made_at": made}}])
    assert F.slot_done(at(13, 0), F.due_slot(at(13, 0)))            # the 12:00 slot ran at 12:40
    assert not F.slot_done(at(14, 5), F.due_slot(at(14, 5)))        # the 14:00 slot hasn't
