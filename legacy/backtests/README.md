# legacy/backtests

v1-era walk-forward backtests of the `Predictor` on QQQ, ETH and SOL, at several bar intervals and strikes.

| Script | Asset |
|---|---|
| `backtest_qqq.py` | QQQ |
| `backtest_eth.py` | ETH-USD |
| `backtest_sol.py` | SOL-USD |

Run from the repository root; each script's docstring lists its options:

```bash
python legacy/backtests/backtest_qqq.py --days 10 --intervals 15m 1h --strikes -3 -2 -1 1 2 3
```

For the current, purged out-of-sample benchmark, use [`benchmarks/compare.py`](../../benchmarks/compare.py).
