# legacy/examples

The original LevelEdge user interfaces and analysis scripts. They run on the backward-compatible `Predictor` API.

| Path | Contents |
|---|---|
| `streamlit_batch_predictor.py` | The original Streamlit UI: batch strike predictions with current price and metrics |
| `intraday_price_path.py` | A Streamlit view of a forecast price path through the session |
| `archive/` | Earlier CLI predictors, Streamlit prototypes, option-strike batch predictions, and prediction-log plotting and post-processing |
| `testing/` | v1 model checks: AUC per interval, a two-week backtest, calibration curves, a data-leakage test, a bull/bear threshold sweep |

Run a Streamlit app from the repository root:

```bash
streamlit run legacy/examples/streamlit_batch_predictor.py
```

The current dashboard replaces these apps; see [`web/`](../../web/README.md).
