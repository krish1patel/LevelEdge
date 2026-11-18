# legacy

Code from LevelEdge's v1 era, before the v2 forecaster, the dashboard and the portfolio. Nothing in production imports it. It's kept so earlier work stays reproducible and its history stays easy to find.

| Folder | Contents |
|---|---|
| [`examples/`](examples/README.md) | The original Streamlit apps, CLI predictors, plotting scripts and model-testing scripts |
| [`backtests/`](backtests/README.md) | v1-era walk-forward backtests on QQQ, ETH and SOL |
| [`maintenance/`](maintenance/README.md) | A one-off Supabase data fix |
| `regression_demo.py` | A Streamlit demo of the v1 regression price-ratio predictor |

Most of these scripts still run, because v2 keeps the `Predictor` API. Some need packages from the root `requirements.txt` (Streamlit, matplotlib).

```bash
streamlit run legacy/examples/streamlit_batch_predictor.py
```

New work belongs in `src/leveledge/`, `scripts/` or `benchmarks/`, not here.
