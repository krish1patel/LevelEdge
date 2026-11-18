# tests

An offline pytest suite. It uses synthetic data and makes no network calls, so it runs anywhere in about 7 seconds.

## Running the tests

```bash
pip install -e ".[web,dev]"
pytest -q tests
```

A passing run ends with a line like `94 passed`. To run one file or one test:

```bash
pytest -q tests/test_ledger.py
pytest -q tests/test_ledger.py -k tampering
```

`conftest.py` sets `LEVELEDGE_DISABLE_LOGGING=1`, so no test writes to Supabase.

## Coverage

| File | Covers |
|---|---|
| `test_sessions.py` | Bar end times, NYSE sessions and half-days, dropping the in-progress bar, horizon arithmetic |
| `test_features.py` | No look-ahead, and scale invariance: the same features at $450 and $650 |
| `test_model.py` | Calibration on synthetic paths, monotone P(above), the classifier gate, no stacker intercept |
| `test_predictor.py` | `forecast()` caching and the backward-compatible `Predictor` API |
| `test_edge.py` | After-fee edge and Kelly sizing against market prices |
| `test_portfolio.py` | Momentum signals, volatility targeting, next-open fills and costs in the backtest |
| `test_strategies.py` | Published allocation rules (DAA and others) as their papers specify |
| `test_live.py` | The traded book's weights, the rebalance calendar, order planning (never borrows, repays margin with whole shares), the Fidelity ticket, the shadow books |
| `test_safety.py` | Each pre-trade check trips on the input it guards against; forward-log scoring and target dates |
| `test_regime.py` | HMM state recovery and causality, the GJR leverage effect, jump-model causality, CRPS |
| `test_ledger.py` | The hash chain, detection of every kind of tampering, rebuilt-chain detection by anchors, idempotent scoring, on-time rules, malformed-forecast rejection |
| `test_api.py` | The FastAPI endpoints through the test client |

## Writing tests

- Keep tests offline: build synthetic bars or pass `data=` frames instead of fetching.
- When you add an invariant to [CLAUDE.md](../CLAUDE.md), add a test that fails if it breaks.
- Name a test after the behaviour it protects, for example `test_plan_never_borrows`.
