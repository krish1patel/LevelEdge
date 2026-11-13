#!/usr/bin/env python3
"""
Can we forecast the market's next steps, and tell safe zones from danger zones?

Part A - distribution forecasts ("how likely is each next path"). Every Friday since
2000, forecast the S&P 500's log return over the next 5, 21 and 63 trading days as a
full distribution (199 quantiles), using only data available that day:
    hist      climatology: all past h-day returns (the no-skill benchmark)
    ewma      RiskMetrics EWMA volatility x empirical standardized quantiles
    vix       VIX-implied volatility x empirical standardized quantiles (absorbs the
              volatility risk premium)
    har       log-HAR on range-based daily variance (Corsi 2009)
    gjr_fhs   GJR-GARCH(1,1)-t, paths by filtered historical simulation
    combo     average of the gjr_fhs, har and vix quantiles (pre-declared combination)
Scores: CRPS (lower is better), 80% / 95% interval coverage, 5% VaR hit rate.

Part B - danger-zone probability. Event: the S&P 500 falls 5% or more over the next
21 trading days. Predictors: the Part A distributions, a two-state HMM, the
statistical jump model, 12-month trend, 10-month SMA, the VIX, the turbulence index
and the absorption ratio, each mapped to a probability out of sample (expanding,
refitted yearly, labels purged), plus a combined logistic model. Scores: Brier skill
vs the expanding base rate, AUC.

Part C - does a regime signal trade better than the 12-month trend filter? SPY core
to T-bills while the jump model (or HMM) says "bear", inside the live book, same
pre-declared adoption rule as hedge_research.py.

Protocol: development = origins 2000-2019 (Part B/C: 2008-2019, when the ETF panel
exists), holdout = 2020+. The jump penalty is the only chosen parameter, picked
on the development period by the paper's criterion (Sharpe of the 0/1 strategy).

Usage: python benchmarks/regime_research.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from leveledge import live  # noqa: E402
from leveledge import portfolio as P  # noqa: E402
from leveledge import regime as R  # noqa: E402
from hedge_research import CACHE as ETF_CACHE, core_overlay, simulate  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "regime")
OUT = os.path.join(ROOT, "benchmarks", "results", "regime")
H = (5, 21, 63)
LV = np.round(np.arange(0.005, 1.0, 0.005), 3)
DEV_END, HOLD_START = "2019-12-31", "2020-01-01"
EVENT = np.log(0.95) * 100            # 21-day loss of 5% or more (log %, as the returns)
LAMBDAS = (10.0, 25.0, 50.0, 100.0, 200.0)
METHODS = ("hist", "ewma", "vix", "har", "gjr_fhs", "combo")


def block_ci(d: np.ndarray, block: int = 13, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    T = len(d)
    k = int(np.ceil(T / block))
    means = []
    for _ in range(n):
        st = rng.integers(0, T - block, k)
        means.append(np.concatenate([d[s:s + block] for s in st])[:T].mean())
    return [round(float(np.percentile(means, 2.5)), 5), round(float(np.percentile(means, 97.5)), 5)]


def prob_below(q: np.ndarray, x: float) -> float:
    """P(Y <= x) from quantiles at LV (linear interpolation, clamped)."""
    return float(np.interp(x, q, LV, left=0.0, right=1.0))


def main():
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    px = P.load_prices(["^GSPC", "^VIX"], start="1990-01-01", cache_dir=CACHE)
    g = px["^GSPC"]
    idx = g.index
    C = g["Close"].to_numpy(float)
    r = np.r_[np.nan, np.diff(np.log(C))] * 100                       # daily log return, %
    cash = (px[P.TBILL]["Close"].reindex(idx).ffill() / 100 / P.TRADING_DAYS).fillna(0.0)
    rv = R.daily_variance(g) * 1e4                                     # %^2
    vix = px["^VIX"]["Close"].reindex(idx).ffill().to_numpy()
    ew = pd.Series(r).fillna(0).pow(2).ewm(alpha=0.06, adjust=False).mean().to_numpy()   # RiskMetrics lambda .94
    har_x = R.har_design(rv)
    fridays = pd.DatetimeIndex(pd.Series(idx, index=idx).groupby(idx.to_period("W-FRI")).last().to_numpy())
    origins = [idx.get_loc(d) for d in fridays if d >= pd.Timestamp("2000-01-01")]
    last = len(idx) - 1

    # ------------------------------------------------------------------ Part A
    print("Part A: distribution forecasts ...", flush=True)
    rows = []
    fit = None
    tabs: dict = {}
    refit_every = 13
    for j, i in enumerate(origins):
        if j % refit_every == 0:                   # quarterly refits on data up to i
            fit = R.gjr_fit(r[1:i + 1], x0=fit)
            tabs = {}
            for h in H:
                past_end = i - h                     # origins whose h-day outcome is known at i
                yh = (np.log(C[h:i + 1]) - np.log(C[:i + 1 - h])) * 100
                tabs[("hist", h)] = np.quantile(yh, LV)
                o = np.arange(30, past_end + 1)
                fut = yh[o]
                tabs[("ewma", h)] = np.quantile(fut / np.sqrt(h * ew[o]), LV)
                okv = np.isfinite(vix[o])
                tabs[("vix", h)] = np.quantile(fut[okv] / np.sqrt(h * vix[o][okv] ** 2 / P.TRADING_DAYS), LV)
                target = np.log(pd.Series(rv).rolling(h).mean().shift(-h).to_numpy())
                X = np.c_[np.ones(len(idx)), har_x.to_numpy()]
                ok = np.isfinite(X).all(1) & np.isfinite(target) & (np.arange(len(idx)) <= past_end)
                beta, *_ = np.linalg.lstsq(X[ok], target[ok], rcond=None)
                s2 = np.var(target[ok] - X[ok] @ beta)
                var_h = h * np.exp(X @ beta + 0.5 * s2)
                okh = ok & (np.arange(len(idx)) >= h)
                oo = np.where(okh)[0]
                oo = oo[oo <= past_end]
                tabs[("har", h)] = np.quantile(((np.log(C[oo + h]) - np.log(C[oo])) * 100) / np.sqrt(var_h[oo]), LV)
                tabs[("har_beta", h)] = (beta, s2)
        s2_next = float(R.gjr_filter(fit, r[1:i + 1])[-1])
        paths = R.gjr_simulate(fit, s2_next, max(H), n=2000, seed=j)
        Xi = np.r_[1.0, har_x.iloc[i].to_numpy()]
        for h in H:
            if i + h > last:
                continue
            y = (np.log(C[i + h]) - np.log(C[i])) * 100
            beta, s2 = tabs[("har_beta", h)]
            q = {"hist": tabs[("hist", h)],
                 "ewma": tabs[("ewma", h)] * np.sqrt(h * ew[i]),
                 "vix": tabs[("vix", h)] * np.sqrt(h * vix[i] ** 2 / P.TRADING_DAYS),
                 "har": tabs[("har", h)] * np.sqrt(h * np.exp(Xi @ beta + 0.5 * s2)),
                 "gjr_fhs": np.quantile(paths[:, h - 1], LV)}
            q["combo"] = (q["gjr_fhs"] + q["har"] + q["vix"]) / 3
            row = {"date": idx[i], "h": h, "y": y}
            for m in METHODS:
                qq = np.maximum.accumulate(q[m])
                row[f"{m}_crps"] = float(R.crps_from_quantiles(np.array([y]), qq[None], LV)[0])
                row[f"{m}_in80"] = bool(qq[np.searchsorted(LV, 0.1)] < y < qq[np.searchsorted(LV, 0.9)])
                row[f"{m}_in95"] = bool(qq[np.searchsorted(LV, 0.025)] < y < qq[np.searchsorted(LV, 0.975)])
                row[f"{m}_var5"] = bool(y < qq[np.searchsorted(LV, 0.05)])
                if h == 21:
                    row[f"{m}_p5"] = prob_below(qq, EVENT)
                row[f"{m}_width80"] = float(qq[np.searchsorted(LV, 0.9)] - qq[np.searchsorted(LV, 0.1)])
            rows.append(row)
        if j % 200 == 0:
            print(f"  {idx[i].date()} ({time.time() - t_start:.0f}s)", flush=True)
    A = pd.DataFrame(rows).set_index("date")

    resA = {}
    for h in H:
        resA[h] = {}
        for per, sl in (("dev", slice(None, DEV_END)), ("holdout", slice(HOLD_START, None))):
            a = A[A.h == h].loc[sl]
            resA[h][per] = {"n": len(a)}
            for m in METHODS:
                d = (a[f"{m}_crps"] - a["hist_crps"]).to_numpy()
                resA[h][per][m] = {"crps": round(float(a[f"{m}_crps"].mean()), 4),
                                   "crps_skill_vs_hist": round(float(1 - a[f"{m}_crps"].mean() / a["hist_crps"].mean()), 4),
                                   "crps_diff_ci95": block_ci(d) if m != "hist" else None,
                                   "cover80": round(float(a[f"{m}_in80"].mean()), 3),
                                   "cover95": round(float(a[f"{m}_in95"].mean()), 3),
                                   "var5_hits": round(float(a[f"{m}_var5"].mean()), 3),
                                   "width80": round(float(a[f"{m}_width80"].mean()), 2)}

    # ------------------------------------------------------------------ regimes: HMM and jump model
    print("Regime models (HMM, jump model) ...", flush=True)
    years = sorted({d.year for d in idx if d.year >= 2000})
    fit_days = [idx.searchsorted(pd.Timestamp(f"{y}-01-01")) for y in years]
    p_hmm = np.full(len(idx), np.nan)
    hm = None
    for k, fd in enumerate(fit_days):
        hm = R.hmm2_fit(r[1:fd], init=hm)
        p = R.hmm2_filter(hm, r[1:])
        end = fit_days[k + 1] if k + 1 < len(fit_days) else len(idx)
        p_hmm[fd:end] = p[fd - 1:end - 1]

    ex = pd.Series(r / 100, index=idx) - cash.reindex(idx).to_numpy()
    feats = R.jm_features(ex.fillna(0))
    F = feats.to_numpy()
    jm_state = {}
    for lam in LAMBDAS:
        st = np.full(len(idx), np.nan)
        for k, fd in enumerate(fit_days):
            tr = F[22:fd]
            mu, sd = tr.mean(0), tr.std(0)
            Z = (F - mu) / sd
            c, s_tr = R.jm_fit(Z[22:fd], lam)
            rets_tr = r[23:fd + 1]
            bear = int(np.argmin([np.nanmean(rets_tr[s_tr == s]) if (s_tr == s).any() else np.inf for s in (0, 1)]))
            on = R.jm_online(Z[22:], c, lam)
            end = fit_days[k + 1] if k + 1 < len(fit_days) else len(idx)
            st[fd:end] = (on[fd - 22:end - 22] == bear).astype(float)
        jm_state[lam] = st

    def zero_one(bear: np.ndarray, sl) -> dict:
        """Paper-style 0/1 strategy on the S&P 500: in the index unless bear (signal acts the next day)."""
        b = pd.Series(bear, index=idx).shift(1)
        ret = pd.Series(np.expm1(r / 100), index=idx)
        strat = np.where(b == 1, cash.to_numpy(), ret.to_numpy())
        sw = (b.diff().abs() > 0).astype(float).to_numpy() * 5 / 1e4
        s = pd.Series(strat - sw, index=idx).loc[sl].dropna()
        return P.metrics(s, cash)

    dev_sl, hold_sl = slice("2000-01-01", DEV_END), slice(HOLD_START, None)
    lam_scores = {lam: zero_one(jm_state[lam], dev_sl)["sharpe"] for lam in LAMBDAS}
    lam_best = max(lam_scores, key=lam_scores.get)
    jm = jm_state[lam_best]
    bh = pd.Series(np.expm1(r / 100), index=idx)
    resRegime = {"lambda_dev_sharpe": lam_scores, "lambda_selected": lam_best,
                 "zero_one": {per: {"buy_hold": P.metrics(bh.loc[sl].dropna(), cash), "jump_model": zero_one(jm, sl),
                                    "hmm": zero_one((p_hmm > 0.5).astype(float), sl)}
                              for per, sl in (("dev", dev_sl), ("holdout", hold_sl))},
                 "share_bear_jm": round(float(np.nanmean(jm[idx >= pd.Timestamp("2000-01-01")])), 3),
                 "switches_per_year_jm": round(float(np.nansum(np.abs(np.diff(jm[idx >= pd.Timestamp("2000-01-01")]))) / (len(years))), 2)}

    # ------------------------------------------------------------------ Part B: danger-zone probability
    print("Part B: danger-zone classifiers ...", flush=True)
    cfg = live.config()
    eprices = P.load_prices(cfg.universe, cache_dir=ETF_CACHE)
    eopens, ecloses, ecash = P.panel(eprices, cfg.universe)
    erets = np.log(ecloses).diff()
    turb = R.turbulence(erets).rolling(21).mean()
    ar = R.absorption_ratio(erets)
    ar_shift = (ar.rolling(15).mean() - ar.rolling(250).mean()) / ar.rolling(250).std()
    sma = pd.Series(C, index=idx).rolling(210).mean()
    mom12 = pd.Series(C, index=idx) / pd.Series(C, index=idx).shift(252) - 1 - cash.rolling(252).sum()

    A21 = A[A.h == 21].copy()
    raw = pd.DataFrame({"hmm": pd.Series(p_hmm, index=idx), "jump_model": pd.Series(jm, index=idx),
                        "trend12_off": (mom12 < 0).astype(float).where(mom12.notna()),
                        "below_sma10": (pd.Series(C, index=idx) < sma).astype(float).where(sma.notna()),
                        "log_vix": np.log(pd.Series(vix, index=idx)),
                        "turbulence": np.log(turb.reindex(idx).ffill()), "absorption_shift": ar_shift.reindex(idx).ffill()})
    FE = pd.DataFrame({"y": (A21["y"] <= EVENT).astype(int)}, index=A21.index).join(raw)
    for m in METHODS:
        FE[f"p_{m}"] = A21[f"{m}_p5"]
    signals = list(raw.columns)
    test_years = sorted({d.year for d in FE.index if d.year >= 2008})
    for name, cols in [(s, [s]) for s in signals] + [("all_logistic", signals + ["p_gjr_fhs"]), ("climatology", [])]:
        out = pd.Series(np.nan, index=FE.index)
        for y in test_years:
            te = (FE.index.year == y) & FE[cols].notna().all(1).to_numpy() if cols else (FE.index.year == y)
            cutoff = pd.Timestamp(f"{y}-01-01") - pd.Timedelta(days=32)     # purge the 21-day label window
            tr = FE.loc[:cutoff, ["y"] + cols].dropna()
            if len(tr) < 50 or tr["y"].nunique() < 2:
                continue
            if not cols:
                out[te] = tr["y"].mean()
                continue
            Xm = tr[cols].to_numpy()
            mu, sd = Xm.mean(0), Xm.std(0) + 1e-9
            model = LogisticRegression(C=1.0).fit((Xm - mu) / sd, tr["y"])
            if te.any():
                out[te] = model.predict_proba((FE.loc[te, cols].to_numpy() - mu) / sd)[:, 1]
        FE[f"p_{name}"] = out
    B = FE.loc["2008-01-01":].dropna()
    resB = {}
    for per, sl in (("dev", slice(None, DEV_END)), ("holdout", slice(HOLD_START, None))):
        b = B.loc[sl].dropna()
        clim = ((b["y"] - b["p_climatology"]) ** 2).mean()
        resB[per] = {"n": len(b), "base_rate": round(float(b["y"].mean()), 3)}
        for col in [c for c in b.columns if c.startswith("p_")]:
            p = b[col].clip(0, 1)
            br = float(((b["y"] - p) ** 2).mean())
            resB[per][col[2:]] = {"brier": round(br, 4), "brier_skill": round(1 - br / clim, 3),
                                  "auc": round(float(roc_auc_score(b["y"], p)), 3) if b["y"].nunique() > 1 else None}

    # ------------------------------------------------------------------ Part C: inside the live book
    print("Part C: regime overlays in the book ...", flush=True)
    book, trend = live.book_decisions(eopens, ecloses, ecash, cfg)
    dates = book.index
    tsm = P.momentum_signal(ecloses[list(live.CORE)], ecash, 252)

    def keep_spy(sig: pd.Series) -> pd.DataFrame:
        return pd.DataFrame({"SPY": sig.reindex(dates).ffill().fillna(1.0), "IEF": 1.0}, index=dates)

    runs = {"current": simulate(ecloses, ecash, book, live.BAND)["ret"],
            "core_tsmom12": simulate(ecloses, ecash, core_overlay(trend, tsm.loc[dates]), live.BAND)["ret"],
            "spy_core_jump_model": simulate(ecloses, ecash, core_overlay(trend, keep_spy(1 - pd.Series(jm, index=idx))), live.BAND)["ret"],
            "spy_core_hmm": simulate(ecloses, ecash, core_overlay(trend, keep_spy((pd.Series(p_hmm, index=idx) <= 0.5).astype(float))), live.BAND)["ret"]}
    start = pd.Timestamp("2008-01-02")
    resC = {}
    for k, s in runs.items():
        s = s.loc[start:]
        resC[k] = {"full": P.metrics(s, ecash), "dev": P.metrics(s.loc[:DEV_END], ecash), "holdout": P.metrics(s.loc[HOLD_START:], ecash)}
    cur = resC["current"]
    for k in ("spy_core_jump_model", "spy_core_hmm"):
        v = resC[k]
        v["adopt"] = bool(all(v[p]["max_drawdown"] >= cur[p]["max_drawdown"] + 0.05 and v[p]["sharpe"] >= cur[p]["sharpe"]
                              for p in ("dev", "holdout")) and v["full"]["cagr"] >= cur["full"]["cagr"] - 0.01)
        v["beats_trend_filter_holdout_sharpe"] = bool(v["holdout"]["sharpe"] > resC["core_tsmom12"]["holdout"]["sharpe"])

    res = {"A_distribution": resA, "regimes": resRegime, "B_danger_zone": resB, "C_book": resC,
           "runtime_s": round(time.time() - t_start)}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    A.to_csv(os.path.join(OUT, "forecasts.csv.gz"))

    # ------------------------------------------------------------------ print
    for h in H:
        print(f"\nPart A, {h}-day horizon")
        for per in ("dev", "holdout"):
            print(f"  {per} (n={resA[h][per]['n']})  {'method':8s} {'CRPS':>7s} {'skill':>6s} {'cover80':>7s} {'cover95':>7s} {'VaR5':>5s} {'width80':>7s}  diff CI")
            for m in METHODS:
                v = resA[h][per][m]
                print(f"  {'':22s}{m:8s} {v['crps']:7.3f} {v['crps_skill_vs_hist']:6.1%} {v['cover80']:7.1%} {v['cover95']:7.1%} {v['var5_hits']:5.1%} {v['width80']:7.2f}  {v['crps_diff_ci95']}")
    print("\nRegimes:", json.dumps({k: v for k, v in resRegime.items() if k != "zero_one"}))
    for per in ("dev", "holdout"):
        z = resRegime["zero_one"][per]
        print(f"  0/1 {per}: " + "; ".join(f"{k} sharpe {v['sharpe']:.2f} maxDD {v['max_drawdown']:.1%} cagr {v['cagr']:.1%}" for k, v in z.items()))
    for per in ("dev", "holdout"):
        print(f"\nPart B {per}: n={resB[per]['n']} base rate {resB[per]['base_rate']:.1%}")
        for k, v in resB[per].items():
            if isinstance(v, dict):
                print(f"  {k:18s} brier skill {v['brier_skill']:+.3f}  AUC {v['auc']}")
    print("\nPart C")
    for k, v in resC.items():
        print(f"  {k:22s} cagr {v['full']['cagr']:.1%} sharpe {v['full']['sharpe']:.2f} dd {v['full']['max_drawdown']:.1%} | dev sh {v['dev']['sharpe']:.2f} dd {v['dev']['max_drawdown']:.1%} | "
              f"20+ sh {v['holdout']['sharpe']:.2f} dd {v['holdout']['max_drawdown']:.1%} | adopt {v.get('adopt', '-')}")
    print(f"\nruntime {res['runtime_s']}s")


if __name__ == "__main__":
    main()
