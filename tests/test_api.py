import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(rth_bars):
    import app.api as api
    import leveledge.predictor as pred
    mp = pytest.MonkeyPatch()
    from leveledge.data import clean_bars
    mp.setattr(pred, "fetch_history",
               lambda ticker, interval, as_of=None: clean_bars(rth_bars, interval, ticker, as_of))
    mp.setattr(api, "_earnings_dates", lambda ticker: [])
    yield TestClient(api.app)
    mp.undo()


BODY = {"ticker": "SPY", "target": "2026-10-30T16:00", "interval": "15m",
        "levels": ["-1%", "+0.5%", "505"], "as_of": "2026-10-30T12:00"}


def test_forecast_shape(client):
    r = client.post("/api/forecast", json=BODY)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["meta"]["horizon_bars"] == 16
    assert [l["input"] for l in j["levels"]] == BODY["levels"]
    ps = sorted(j["levels"], key=lambda l: l["level"])
    assert all(a["p_above"] >= b["p_above"] for a, b in zip(ps, ps[1:]))
    assert len(j["fan"]) == 16 and j["fan"][-1]["q10"] < j["fan"][-1]["q90"]
    assert j["trust"]["verdict"] in {"model_adds_skill", "baseline", "unvalidated"}
    assert j["events"] == []


@pytest.mark.parametrize("patch,msg", [
    ({"levels": ["abc"]}, "Can't read level"),
    ({"interval": "7m"}, "Interval must be one of"),
    ({"ticker": "SPY; DROP"}, "Ticker should look like"),
    ({"target": "not a date"}, "dates like"),
])
def test_forecast_rejects_bad_input(client, patch, msg):
    r = client.post("/api/forecast", json={**BODY, **patch})
    assert r.status_code == 422 and msg in r.json()["detail"]


def test_track_record_without_credentials(client, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    assert client.get("/api/track-record").json()["status"] == "not_configured"
