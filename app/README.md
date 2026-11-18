# app

`app/api.py` is the FastAPI backend for the dashboard. It's a stateless JSON layer over the `leveledge` library, with short in-memory caches. In production it runs on AWS Lambda behind Vercel's `/api/*` rewrite. When `web/dist` exists, it also serves the built frontend, which is how the local server and the HuggingFace Space work.

## Running locally

```bash
(cd web && npm ci && npm run build)          # optional: serve the dashboard too
uvicorn app.api:app --reload --port 7860
```

Interactive API docs are at <http://127.0.0.1:7860/docs>.

## Endpoints

| Method and path | Returns | Cache |
|---|---|---|
| `POST /api/forecast` | P(above) for each level, quantiles, the horizon, and each level's out-of-sample record | Model fits are cached by the library |
| `GET /api/track-record?days=60` | Live forward-test scores by model from Supabase: Brier, log loss, calibration, daily Brier | Per window |
| `GET /api/benchmark` | The offline v1-vs-v2 benchmark (`benchmarks/results/summary.json`) | — |
| `GET /api/portfolio` | The book's targets and reasoning, its backtest, and the live paper account | 30 min (targets), 5 min (account) |
| `GET /api/outlook` | The S&P 500 outlook, its historical coverage, and the scored forward log | 30 min |
| `GET /api/ledger` | The public forecast ledger, verified from scratch, with scores by interval and calibration | 10 min |
| `GET /api/research` | Every study's results, from `benchmarks/results/*/summary.json` | — |
| `GET /api/health` | `ok`, supported intervals, suggested tickers and the server time | — |

### Making a forecast request

```bash
curl -s -X POST http://127.0.0.1:7860/api/forecast \
  -H 'Content-Type: application/json' \
  -d '{"ticker": "SPY", "interval": "15m", "target": "2026-11-20T16:00", "levels": ["-1%", "+1%", "745"]}'
```

`target` is an ISO datetime, read as US/Eastern when it has no offset. `levels` accepts prices or percent moves. `as_of` (optional) simulates a forecast made at a past time.

## Configuring

The API reads these environment variables, or `.env` locally:

| Variable | Effect |
|---|---|
| `SUPABASE_URL`, `SUPABASE_KEY` | Enable `/api/track-record` |
| `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` | Add the live paper account to `/api/portfolio` |
| `LEDGER_URL` | The ledger to verify (default: the public `ledger` branch) |
| `FORWARD_LOG_URL` | The forward log to score (default: the public `forward-log` branch) |
| `LEVELEDGE_DISABLE_LOGGING=1` | Stop forecasts being written to Supabase |

Each feature degrades to a clear empty state when its credentials are missing.

## Testing

`tests/test_api.py` exercises the endpoints with FastAPI's test client:

```bash
pytest -q tests/test_api.py
```
