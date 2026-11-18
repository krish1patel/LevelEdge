"""
Research-integrity engine: grade how far a strategy's backtest can be trusted, and say how it might
be fooling you.

A strategy (an entry in STRATEGIES) produces weekly target weights from prices. The engine breaks
the evaluation into independent tasks (``plan``), runs each (``run_task``, locally or on AWS Lambda
workers fed by SQS), and combines the results (``aggregate``) into eight checks and a 0-100 score:

  check          question                                                       weight
  holdout        Did it keep working on 2020+ data, which played no part in the design?   30
                 (scored against its own pre-2020 record and against the benchmark after 2020)
  benchmark      Does it beat a simple alternative by more than luck (block bootstrap)?   15
  luck           Does its timing beat the same holdings shuffled in time?                 15
  multiple       After counting every variant tried, is the best one more than noise?    10
                 (deflated Sharpe, Bailey & Lopez de Prado 2014)
  stability      Did variants that looked good before 2020 stay good after (rank corr)?  15
  lookahead      Are its decisions identical when it can't see future data (truncation)?  5
  costs          Does it survive four times the trading costs?                             5
  concentration  Does its edge survive without its single best year?                       5

A look-ahead failure caps the score at 20: no other evidence matters if the backtest saw the future.
"""
from __future__ import annotations

import itertools
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import live
from . import portfolio as P

DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
SHUFFLES, SHUFFLE_CHUNK, BLOCK_WEEKS = 200, 50, 4
WEIGHTS = {"holdout": 30, "benchmark": 15, "luck": 15, "multiple": 10, "stability": 15, "lookahead": 5, "costs": 5, "concentration": 5}
CACHE = os.environ.get("LEVELEDGE_CACHE") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "benchmarks", "cache")


# ---------------------------------------------------------------------------
# data context
# ---------------------------------------------------------------------------
@dataclass
class Ctx:
    opens: pd.DataFrame
    closes: pd.DataFrame
    cash: pd.Series
    sent: pd.Series | None = None
    _memo: dict = field(default_factory=dict)

    def memo(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    @property
    def vol(self):
        return self.memo("vol", lambda: np.log(self.closes).diff().apply(P.ewma_vol))

    def corr(self, how: str):
        return self.memo(("corr", how), lambda: P.rolling_corr(self.closes, P.rebalance_dates(self.closes.index, how), 126))

    def truncated(self, d: pd.Timestamp) -> "Ctx":
        sent = None if self.sent is None else self.sent.loc[:d]
        return Ctx(self.opens.loc[:d], self.closes.loc[:d], self.cash.loc[:d], sent)


def load_ctx(universe: str = "etf") -> Ctx:
    from . import strategies as S
    from .news import load_sentiment

    syms = sorted(set(P.UNIVERSE) | (set(S.DAA_SYMBOLS) if universe == "daa" else set()))
    try:                                         # the price cache is Parquet; the slim Lambda image has no pyarrow,
        import pyarrow  # noqa: F401              # and there the context is kept in memory instead (app/worker.py)
        cache = os.path.join(CACHE, "etf")
    except ImportError:
        cache = None
    prices = P.load_prices(syms, start="2003-01-01", cache_dir=cache)
    opens, closes, cash = P.panel(prices, syms)
    sent = load_sentiment(os.path.join(CACHE, "news")) if universe == "etf" else None
    return Ctx(opens, closes, cash, sent)


# ---------------------------------------------------------------------------
# strategies: decide(ctx, **params) -> weekly target weights (index dates, columns closes.columns)
# ---------------------------------------------------------------------------
def _cfg(lookback: int = 252) -> P.StrategyConfig:
    return P.StrategyConfig(lookback=lookback)


def _book(ctx: Ctx, lookback: int = 252):
    o, c = ctx.opens[list(P.UNIVERSE)], ctx.closes[list(P.UNIVERSE)]
    b, _ = live.book_decisions(o, c, ctx.cash, _cfg(lookback))
    return b


def _sleeve(ctx: Ctx, lookback: int = 252):
    o, c = ctx.opens[list(P.UNIVERSE)], ctx.closes[list(P.UNIVERSE)]
    return P.strategy_weights(o, c, ctx.cash, _cfg(lookback))


def _trend_filter(ctx: Ctx, lookback: int = 252):
    o, c = ctx.opens[list(P.UNIVERSE)], ctx.closes[list(P.UNIVERSE)]
    return live.trend_filter_decisions(o, c, ctx.cash, _cfg(lookback))


def _spy_rule(ctx: Ctx, on: pd.Series) -> pd.DataFrame:
    dates = P.rebalance_dates(ctx.closes.index, "weekly")
    on = on.reindex(dates)
    dates = on.dropna().index
    d = pd.DataFrame(0.0, index=dates, columns=ctx.closes.columns)
    d["SPY"] = on.loc[dates].to_numpy()
    return d


def _spy_sma(ctx: Ctx, window: int = 200):
    c = ctx.closes["SPY"]
    ma = c.rolling(window).mean()
    return _spy_rule(ctx, (c > ma).astype(float).where(ma.notna()))


def _news(ctx: Ctx, window: int = 252):
    from .news import available
    s = available(ctx.sent, ctx.closes.index)
    avg = s.rolling(window, min_periods=int(window * 0.8)).mean()
    return _spy_rule(ctx, (s > avg).astype(float).where(avg.notna()))


def _variant(ctx: Ctx, spec: str = "mom6|invvol|monthly|IEF"):
    sig, sz, reb, dfn = spec.split("|")
    o, c = ctx.opens[list(P.UNIVERSE)], ctx.closes[list(P.UNIVERSE)]
    return P.variant_weights(c, ctx.cash, ctx.vol[list(P.UNIVERSE)], ctx.corr(reb),
                             {"signal": sig, "sizing": sz, "rebalance": reb, "defensive": None if dfn == "cash" else dfn})


def _daa(ctx: Ctx):
    return live.shadow_decisions(ctx.closes)


def variant_names() -> list[str]:
    out = [f"{s}|{z}|{r}|{d or 'cash'}" for s, z, r, d in itertools.product(P.SIGNALS, ["equal", "invvol"], ["weekly", "monthly"], [None, "IEF"])]
    out += [f"{s}|voltarget|{r}|cash" for s, r in itertools.product(P.SIGNALS, ["weekly", "monthly"])]
    return out


LOOKBACKS = [126, 189, 252, 315]
STRATEGIES: dict[str, dict] = {
    "book": {"name": "The traded portfolio (70% 60/40 + 30% trend)", "decide": _book, "params": {"lookback": 252},
             "grid": [{"lookback": x} for x in LOOKBACKS], "benchmark": "sixty_forty", "universe": "etf"},
    "trend_sleeve": {"name": "Trend rule across 11 markets, alone", "decide": _sleeve, "params": {"lookback": 252},
                     "grid": [{"lookback": x} for x in LOOKBACKS], "benchmark": "sixty_forty", "universe": "etf"},
    "book_trend_filter": {"name": "Portfolio with the 12-month trend filter on its core", "decide": _trend_filter, "params": {"lookback": 252},
                          "grid": [{"lookback": x} for x in LOOKBACKS], "benchmark": "sixty_forty", "universe": "etf"},
    "spy_sma200": {"name": "S&P 500 with a 200-day-average sell rule", "decide": _spy_sma, "params": {"window": 200},
                   "grid": [{"window": x} for x in (100, 150, 200, 250)], "benchmark": "spy", "universe": "etf"},
    "news_sentiment_spy": {"name": "S&P 500 timed by news sentiment", "decide": _news, "params": {"window": 252},
                           "grid": [{"window": x} for x in LOOKBACKS], "benchmark": "spy", "universe": "etf"},
    "best_of_80": {"name": "Best of 80 trend variants, picked on 2007-2019", "decide": _variant, "params": {"spec": "mom6|invvol|monthly|IEF"},
                   "grid": [{"spec": v} for v in variant_names()], "benchmark": "sixty_forty", "universe": "etf"},
    "daa_book": {"name": "Portfolio with a published sleeve (Keller & Keuning DAA)", "decide": _daa, "params": {},
                 "grid": [], "benchmark": "sixty_forty", "universe": "daa"},
}


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
def plan(sid: str) -> list[dict]:
    st = STRATEGIES[sid]
    tasks = [{"id": f"{sid}:base", "strategy": sid, "kind": "base"}]
    tasks += [{"id": f"{sid}:grid:{i}", "strategy": sid, "kind": "grid", "i": i} for i in range(len(st["grid"]))]
    tasks += [{"id": f"{sid}:shuffle:{k}", "strategy": sid, "kind": "shuffle", "k": k} for k in range(SHUFFLES // SHUFFLE_CHUNK)]
    return tasks


def _bench(ctx: Ctx, which: str) -> pd.Series:
    w = {"SPY": 0.6, "IEF": 0.4} if which == "sixty_forty" else {"SPY": 1.0}
    return P.buy_and_hold(ctx.closes, w, ctx.cash, "monthly" if which == "sixty_forty" else None)["ret"]


def _bt(ctx: Ctx, dec: pd.DataFrame, cost: float = 5.0) -> pd.Series:
    return P.backtest(ctx.opens, ctx.closes, ctx.cash, dec.reindex(columns=ctx.closes.columns).fillna(0.0), cost)["ret"]


def _sharpe(r: pd.Series, cash: pd.Series) -> float:
    ex = r - cash.reindex(r.index).fillna(0.0)
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(P.TRADING_DAYS)) if ex.std() > 0 else 0.0


def _split(r: pd.Series, cash: pd.Series) -> dict:
    return {"full": _sharpe(r, cash), "dev": _sharpe(r.loc[:DEV_END], cash), "holdout": _sharpe(r.loc[HOLD_START:], cash)}


def run_task(task: dict, ctx: Ctx | None = None) -> dict:
    st = STRATEGIES[task["strategy"]]
    ctx = ctx or load_ctx(st["universe"])
    if task["kind"] == "grid":
        dec = st["decide"](ctx, **st["grid"][task["i"]])
        r = _bt(ctx, dec)
        return {"params": st["grid"][task["i"]], "sharpe": _split(r, ctx.cash),
                "daily_sharpe_dev": _sharpe(r.loc[:DEV_END], ctx.cash) / np.sqrt(P.TRADING_DAYS)}
    dec = st["decide"](ctx, **st["params"])
    if task["kind"] == "shuffle":
        rng = np.random.default_rng(1000 + task["k"])
        n = len(dec)
        blocks = [np.arange(i, min(i + BLOCK_WEEKS, n)) for i in range(0, n, BLOCK_WEEKS)]
        out = []
        for _ in range(SHUFFLE_CHUNK):
            order = np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])
            sh = pd.DataFrame(dec.to_numpy()[order], index=dec.index, columns=dec.columns)
            out.append(_split(_bt(ctx, sh), ctx.cash))
        return {"shuffles": out}

    # base: the strategy itself, its benchmark, costs, concentration, look-ahead
    r = _bt(ctx, dec)
    start = r.index[0]
    b = _bench(ctx, st["benchmark"]).loc[start:]
    ex = (r - b).dropna()
    years = ex.groupby(ex.index.year).sum()
    best = int(years.idxmax())
    r4 = _bt(ctx, dec, 20.0)
    look = []
    for d in pd.DatetimeIndex(dec.index[[int(len(dec) * f) for f in (0.3, 0.5, 0.7, 0.9)]]):
        part = st["decide"](ctx.truncated(d), **st["params"])
        if d in part.index:
            diff = float(np.abs(part.loc[d].reindex(dec.columns).fillna(0) - dec.loc[d].fillna(0)).max())
        else:
            diff = float("inf")
        look.append({"date": str(d.date()), "max_weight_diff": diff})
    return {
        "period": [str(start.date()), str(r.index[-1].date())],
        "metrics": {p: P.metrics(x, ctx.cash) for p, x in (("full", r), ("dev", r.loc[:DEV_END]), ("holdout", r.loc[HOLD_START:]))},
        "bench_metrics": {p: P.metrics(x, ctx.cash) for p, x in (("full", b), ("dev", b.loc[:DEV_END]), ("holdout", b.loc[HOLD_START:]))},
        "vs_bench": {p: P.sharpe_diff_ci(r.loc[a:z], b.loc[a:z], ctx.cash) for p, a, z in (("full", None, None), ("holdout", HOLD_START, None))},
        "sharpe": _split(r, ctx.cash), "sharpe_4x_costs": _split(r4, ctx.cash),
        "excess_by_year": {str(k): float(v) for k, v in years.items()}, "best_year": best,
        "sharpe_diff_without_best_year": _sharpe(r[r.index.year != best], ctx.cash) - _sharpe(b[b.index.year != best], ctx.cash),
        "lookahead": look,
    }


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _clip(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def aggregate(sid: str, results: dict[str, dict]) -> dict:
    """Combine one strategy's task results into checks, a 0-100 score, a grade and findings."""
    from scipy.stats import norm, spearmanr

    st = STRATEGIES[sid]
    base = results[f"{sid}:base"]
    grid = [results[f"{sid}:grid:{i}"] for i in range(len(st["grid"])) if f"{sid}:grid:{i}" in results]
    sh = [s for k, v in results.items() if k.startswith(f"{sid}:shuffle:") for s in v["shuffles"]]
    s = base["sharpe"]
    checks, findings = {}, []

    def check(name, score, status, detail):
        checks[name] = {"score": round(_clip(score), 3), "status": status, "detail": detail}

    # 1. holdout
    dev, hold = s["dev"], s["holdout"]
    bhold = base["bench_metrics"]["holdout"]["sharpe"]
    ratio = hold / dev if dev > 0 else (1.0 if hold > 0 else 0.0)
    vs_b = hold / bhold if bhold > 0 else (1.0 if hold > 0 else 0.0)
    sc = _clip(min(ratio, vs_b)) if hold > 0 else 0.0
    check("holdout", sc, "pass" if min(ratio, vs_b) >= 0.6 and hold > 0 else "warn" if hold > 0 and min(ratio, vs_b) >= 0.35 else "fail",
          f"Sharpe {dev:.2f} before 2020, {hold:.2f} after (benchmark after 2020: {bhold:.2f}). It kept {max(0, ratio) * 100:.0f}% of its "
          f"pre-2020 quality and reached {max(0, vs_b) * 100:.0f}% of the benchmark's.")
    if ratio < 0.6:
        findings.append(f"It looked {'much ' if ratio < 0.4 else ''}better before 2020 (Sharpe {dev:.2f}) than after ({hold:.2f}): the "
                        "early record may describe a market that no longer exists, or rules fitted to it.")
    if vs_b < 0.6 and ratio >= 0.6:
        findings.append(f"Since 2020 it reached only {max(0, vs_b) * 100:.0f}% of its benchmark's Sharpe ({hold:.2f} against {bhold:.2f}).")

    # 2. benchmark
    full = base["vs_bench"]["full"]
    lo, hi = full["ci95"]
    bname = "a 60/40 mix" if st["benchmark"] == "sixty_forty" else "simply holding the S&P 500"
    sc = 1.0 if lo > 0 else 0.6 if full["diff"] >= 0 else 0.3 if hi > 0 else 0.0
    check("benchmark", sc, "pass" if lo > 0 else "warn" if hi > 0 else "fail",
          f"Sharpe difference against {bname}: {full['diff']:+.2f} (95% interval {lo:+.2f} to {hi:+.2f}).")
    if lo <= 0:
        findings.append(f"Its edge over {bname} is {'within luck' if hi > 0 else 'negative'}: the 95% interval runs from {lo:+.2f} to {hi:+.2f}.")

    # 3. luck: timing against block-shuffled copies of its own holdings
    if sh:
        p = float(np.mean([x["full"] >= s["full"] for x in sh]))
        sc = 1.0 if p <= 0.05 else 0.7 if p <= 0.10 else 0.4 if p <= 0.20 else 0.1
        check("luck", sc, "pass" if p <= 0.05 else "warn" if p <= 0.2 else "fail",
              f"Its timing beat {100 * (1 - p):.0f}% of {len(sh)} copies with the same holdings shuffled in time.")
        if p > 0.05:
            findings.append(f"Its timing beat only {100 * (1 - p):.0f}% of shuffled copies of itself, so much of the result could come from what it owned, not when.")

    # 4. multiple testing: deflated Sharpe on the development period
    n = max(1, len(grid))
    sr_dev = dev / np.sqrt(P.TRADING_DAYS)
    T = 252 * 12.9
    if n >= 2:
        trials = np.array([g["daily_sharpe_dev"] for g in grid])
        v = np.var(trials, ddof=1)
        e = 0.5772156649
        sr0 = np.sqrt(v) * ((1 - e) * norm.ppf(1 - 1 / n) + e * norm.ppf(1 - 1 / (n * np.e)))
    else:
        sr0 = 0.0
    prob = float(norm.cdf((sr_dev - sr0) * np.sqrt(T - 1)))
    check("multiple", prob, "pass" if prob >= 0.95 else "warn" if prob >= 0.7 else "fail",
          f"Counting {n} variant{'s' if n > 1 else ''} tried, the chance its pre-2020 Sharpe is more than luck: {prob * 100:.0f}%.")
    if prob < 0.95:
        findings.append(f"After counting the {n} variants tried, there's only a {prob * 100:.0f}% chance the best one beat chance (95% is the usual bar).")

    # 5. stability across variants
    if len(grid) >= 3:
        rho = float(spearmanr([g["sharpe"]["dev"] for g in grid], [g["sharpe"]["holdout"] for g in grid]).correlation)
        check("stability", (rho + 1) / 2, "pass" if rho >= 0.3 else "warn" if rho >= 0 else "fail",
              f"Rank correlation of {len(grid)} variants before and after 2020: {rho:+.2f} (1 = same order, −1 = reversed).")
        if rho < 0:
            findings.append(f"Variants that looked best before 2020 tended to do worst after (rank correlation {rho:+.2f}): a sign of fitting noise.")
    else:
        check("stability", 0.5, "n/a", "Published parameters with no variants to compare; scored neutral.")

    # 6. look-ahead
    worst = max(x["max_weight_diff"] for x in base["lookahead"])
    ok = worst < 1e-6
    check("lookahead", 1.0 if ok else 0.0, "pass" if ok else "fail",
          f"Decisions recomputed on data cut at {len(base['lookahead'])} past dates {'matched exactly' if ok else f'differed by up to {worst:.3f}'}.")
    if not ok:
        findings.append("Its decisions change when future data is removed: the backtest saw the future. Nothing else in this report can be trusted.")

    # 7. costs
    keep = s["full"] and base["sharpe_4x_costs"]["full"] / s["full"] if s["full"] > 0 else 0.0
    check("costs", keep, "pass" if keep >= 0.8 else "warn" if keep >= 0.5 else "fail",
          f"At four times the trading costs it keeps {max(0, keep) * 100:.0f}% of its Sharpe ({base['sharpe_4x_costs']['full']:.2f}).")
    if keep < 0.8:
        findings.append(f"Higher trading costs erode it: at 4× costs only {max(0, keep) * 100:.0f}% of its Sharpe is left.")

    # 8. concentration
    d0, d1 = full["diff"], base["sharpe_diff_without_best_year"]
    if d0 > 0:
        sc = 1.0 if d1 > 0 else 0.3
        check("concentration", sc, "pass" if d1 > 0 else "warn",
              f"Without its best year ({base['best_year']}) its Sharpe advantage {'stays positive' if d1 > 0 else 'disappears'} ({d1:+.2f}).")
        if d1 <= 0:
            findings.append(f"Its whole advantage comes from one year ({base['best_year']}).")
    else:
        check("concentration", 0.5, "n/a", "It doesn't beat its benchmark, so there's no edge to concentrate; scored neutral.")

    score = sum(WEIGHTS[k] * checks[k]["score"] for k in WEIGHTS)
    if not ok:
        score = min(score, 20.0)
    grade = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "F"
    verdict = {"A": "Strong evidence", "B": "Credible, with caveats", "C": "Mixed evidence", "D": "Weak evidence", "F": "Don't trust it"}[grade]
    if not findings:
        findings.append("No check found a way it's fooling itself. That's evidence of a real effect, not a guarantee of future returns.")
    m, bm = base["metrics"], base["bench_metrics"]
    return {"id": sid, "name": st["name"], "benchmark": st["benchmark"], "score": round(score, 1), "grade": grade, "verdict": verdict,
            "checks": checks, "findings": findings, "period": base["period"], "variants": n, "shuffles": len(sh),
            "metrics": {p: {k: m[p][k] for k in ("cagr", "sharpe", "max_drawdown")} for p in m},
            "bench_metrics": {p: {k: bm[p][k] for k in ("cagr", "sharpe", "max_drawdown")} for p in bm}}


__all__ = ["STRATEGIES", "plan", "run_task", "aggregate", "load_ctx", "WEIGHTS"]
