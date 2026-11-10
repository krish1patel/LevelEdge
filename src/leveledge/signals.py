"""
Return forecasting for the ETF universe: features, pooled / spline / mixed-effects
models, and walk-forward evaluation with model choice inside the loop.

Everything is in volatility units so assets can share one model:
* features are returns divided by the asset's own volatility over that window;
* the target is next week's excess return divided by the ex-ante weekly vol.

Models
------
rule      today's trend rule (12-month return beats cash), as a forecast
ridge     pooled linear ridge regression
spline    pooled additive model: a cubic spline per feature, ridge-penalised
mixed     spline fixed effects + per-asset random intercepts and slopes on the
          linear features, shrunk toward the pooled effect. The ridge penalty on
          the asset deviations is the MAP estimate of a Gaussian random-effects
          model, so this is a penalised mixed-effects model.
xgb       shallow gradient-boosted trees (nonlinear benchmark)

All decisions use data known at the decision close only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from leveledge import portfolio as P

FEATURES = ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "rev_1w", "sma_gap", "vol_regime", "xs_rank_12m", "mkt_trend"]
LINEAR_RE = ["mom_1m", "mom_12m", "sma_gap"]       # random slopes per asset (mixed model)
WEEK = 5


# ---------------------------------------------------------------------------
# panel
# ---------------------------------------------------------------------------
def weekly_panel(closes: pd.DataFrame, cash: pd.Series, decision_dates: pd.DatetimeIndex,
                 market: str = "SPY") -> pd.DataFrame:
    """Long panel (date, asset) of features known at each decision close and the next-week target."""
    lr = np.log(closes).diff()
    vol = lr.apply(P.ewma_vol)                       # daily vol known at each close
    logp = np.log(closes)
    cash_cum = cash.cumsum()
    idx = closes.index
    pos = {d: idx.get_loc(d) for d in decision_dates if d in idx}
    rows = []
    sma = closes.rolling(200).mean()
    vol_avg = vol.rolling(252).mean()
    for d, i in pos.items():
        if i < 260:
            continue
        nxt = [j for j in pos.values() if j > i]
        j = min(nxt) if nxt else None
        v = vol.iloc[i]
        feats = {}
        for name, k in (("mom_1m", 21), ("mom_3m", 63), ("mom_6m", 126), ("mom_12m", 252)):
            ex = logp.iloc[i] - logp.iloc[i - k] - (cash_cum.iloc[i] - cash_cum.iloc[i - k])
            feats[name] = ex / (v * np.sqrt(k))
        feats["rev_1w"] = (logp.iloc[i] - logp.iloc[i - WEEK]) / (v * np.sqrt(WEEK))
        feats["sma_gap"] = np.log(closes.iloc[i] / sma.iloc[i]) / (v * np.sqrt(20))
        feats["vol_regime"] = np.log(v / vol_avg.iloc[i])
        feats["xs_rank_12m"] = feats["mom_12m"].rank(pct=True) - 0.5
        mk = feats["mom_12m"].get(market, np.nan)
        feats["mkt_trend"] = pd.Series(mk, index=closes.columns)
        if j is not None:
            h = j - i
            fwd = logp.iloc[j] - logp.iloc[i] - (cash_cum.iloc[j] - cash_cum.iloc[i])
            y = fwd / (v * np.sqrt(h))
        else:
            y = pd.Series(np.nan, index=closes.columns)
        for a in closes.columns:
            rows.append({"date": d, "asset": a, "sigma_d": float(v[a]), "y": float(y[a]),
                         **{f: float(feats[f][a]) for f in FEATURES}})
    df = pd.DataFrame(rows)
    df[FEATURES] = df[FEATURES].clip(-6, 6)          # robust to rare extreme ratios
    return df.dropna(subset=FEATURES).reset_index(drop=True)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class RuleModel:
    name = "rule"

    def fit(self, df, alpha=None):
        return self

    def predict(self, df):
        return np.where(df["mom_12m"] > 0, 0.1, -0.1)


class RidgeModel:
    name = "ridge"
    alphas = (10.0, 100.0, 1000.0, 10000.0)

    def __init__(self, alpha: float = 1000.0):
        self.alpha = alpha

    def _X(self, df, fit=False):
        X = df[FEATURES].to_numpy()
        if fit:
            self.sc = StandardScaler().fit(X)
        return self.sc.transform(X)

    def fit(self, df, alpha=None):
        self.alpha = alpha or self.alpha
        self.m = Ridge(alpha=self.alpha).fit(self._X(df, True), df["y"].to_numpy())
        return self

    def predict(self, df):
        return self.m.predict(self._X(df))


class SplineModel(RidgeModel):
    """Additive model: natural-ish cubic spline per feature (linear extrapolation)."""
    name = "spline"
    alphas = (100.0, 1000.0, 10000.0, 100000.0)

    def _X(self, df, fit=False):
        X = df[FEATURES].to_numpy()
        if fit:
            self.sp = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="linear").fit(X)
            self.sc = StandardScaler().fit(self.sp.transform(X))
        return self.sc.transform(self.sp.transform(X))


class MixedModel(SplineModel):
    """Spline fixed effects + per-asset random intercept and slopes (penalised mixed effects).

    Asset-deviation columns are scaled by ``re_scale`` < 1 so a single ridge
    penalty shrinks them ``1 / re_scale**2`` times harder than the fixed effects:
    the random-effect variance is a fraction of the fixed-effect prior variance.
    """
    name = "mixed"
    re_scale = 0.3

    def _X(self, df, fit=False):
        base = super()._X(df, fit)
        if fit:
            self.assets = sorted(df["asset"].unique())
            self.re_sc = StandardScaler().fit(df[LINEAR_RE].to_numpy())
        lin = self.re_sc.transform(df[LINEAR_RE].to_numpy())
        a = df["asset"].to_numpy()
        blocks = []
        for asset in self.assets:
            m = (a == asset).astype(float)[:, None]
            blocks.append(m * np.column_stack([np.ones(len(df)), lin]))
        return np.column_stack([base] + [b * self.re_scale for b in blocks])


class XGBModel:
    name = "xgb"
    alphas = (None,)

    def fit(self, df, alpha=None):
        from xgboost import XGBRegressor
        self.m = XGBRegressor(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=50, reg_lambda=10.0, n_jobs=1,
                              verbosity=0).fit(df[FEATURES].to_numpy(), df["y"].to_numpy())
        return self

    def predict(self, df):
        return self.m.predict(df[FEATURES].to_numpy())


MODELS = {"rule": RuleModel, "ridge": RidgeModel, "spline": SplineModel, "mixed": MixedModel, "xgb": XGBModel}


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------
def _score(y, f) -> float:
    """Out-of-sample R^2 against a zero forecast (the right null for excess returns)."""
    y, f = np.asarray(y), np.asarray(f)
    return float(1 - np.sum((y - f) ** 2) / np.sum(y ** 2))


def _purge(train: pd.DataFrame, before: pd.Timestamp) -> pd.DataFrame:
    """Drop training rows whose target overlaps ``before`` (one-week labels, one-week embargo)."""
    dates = sorted(train["date"].unique())
    keep = [d for d in dates if d < before]
    cut = set(keep[-2:])                                # last label ends inside the test window
    return train[train["date"].isin(keep) & ~train["date"].isin(cut)]


def fit_with_inner_cv(name: str, train: pd.DataFrame, val_years: int = 3):
    """Pick the model's penalty on the last ``val_years`` of the training window, then refit on all of it."""
    cls = MODELS[name]
    alphas = getattr(cls, "alphas", (None,))
    if len(alphas) > 1:
        last = train["date"].max()
        split = last - pd.DateOffset(years=val_years)
        inner_tr, inner_va = _purge(train, split), train[train["date"] >= split]
        best = max(alphas, key=lambda a: _score(inner_va["y"], cls().fit(inner_tr, a).predict(inner_va)))
    else:
        best = alphas[0]
    return cls().fit(train, best), best


def walk_forward(panel: pd.DataFrame, names: list[str], first_year: int, val_years: int = 3,
                 select_years: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yearly expanding-window refits. Returns OOS forecasts per model, plus the 'auto' column:
    the model chosen each year by its OOS-style R^2 over the previous ``select_years``
    (using only forecasts already made), so the choice itself is out of sample."""
    years = sorted({d.year for d in panel["date"] if d.year >= first_year})
    out = panel[panel["date"].dt.year >= first_year][["date", "asset", "y", "sigma_d"]].copy()
    log = []
    for name in names:
        out[name] = np.nan
    for yr in years:
        start = pd.Timestamp(f"{yr}-01-01")
        train = _purge(panel[panel["date"] < start], start)
        test_m = (out["date"].dt.year == yr).to_numpy()
        test = panel.loc[panel["date"].dt.year == yr]
        for name in names:
            m, a = fit_with_inner_cv(name, train, val_years)
            out.loc[test_m, name] = m.predict(test)
            log.append({"year": yr, "model": name, "alpha": a})
    # model selection inside the loop: for year Y use the model with the best R^2 on years Y-3..Y-1
    out["auto"] = np.nan
    chosen = []
    for yr in years:
        prev = out[(out["date"].dt.year < yr) & (out["date"].dt.year >= yr - select_years)]
        cand = [n for n in names if n != "rule"]
        pick = max(cand, key=lambda n: _score(prev["y"], prev[n])) if len(prev) else "spline"
        out.loc[out["date"].dt.year == yr, "auto"] = out.loc[out["date"].dt.year == yr, pick]
        chosen.append({"year": yr, "model": pick})
    return out, pd.DataFrame(log).merge(pd.DataFrame(chosen).rename(columns={"model": "auto_pick"}), on="year")


def forecast_stats(out: pd.DataFrame, col: str) -> dict:
    d = out.dropna(subset=[col, "y"])
    ic = d.groupby("date").apply(lambda g: g[col].rank().corr(g["y"].rank()) if g[col].nunique() > 1 else np.nan).dropna()
    return {"oos_r2": round(_score(d["y"], d[col]), 5),
            "ic_mean": round(float(ic.mean()), 4),
            "ic_t": round(float(ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))), 2) if len(ic) > 2 else None,
            "hit_rate": round(float((np.sign(d[col]) == np.sign(d["y"])).mean()), 4), "n": int(len(d))}


# ---------------------------------------------------------------------------
# forecasts -> weights
# ---------------------------------------------------------------------------
@dataclass
class Construction:
    target_vol: float = 0.10
    max_gross: float = 1.0
    max_weight: float = 0.25
    scale: str = "strength"        # "strength": weight by forecast size; "sign": equal risk when positive


def weights_from_forecasts(fc: pd.Series, daily_vol: pd.Series, corr: pd.DataFrame, c: Construction) -> pd.Series:
    """Long-only: risk budget proportional to the positive forecast, then scale to the vol target."""
    ann = daily_vol * np.sqrt(P.TRADING_DAYS)
    s = fc.clip(lower=0)
    if c.scale == "sign":
        s = (fc > 0).astype(float)
    ok = s.notna() & ann.notna() & (ann > 0)
    w = pd.Series(0.0, index=fc.index)
    if not (s[ok] > 0).any():
        return w
    w[ok] = s[ok] / ann[ok]
    cov = corr.loc[w.index, w.index].fillna(0).to_numpy() * np.outer(ann.fillna(0), ann.fillna(0))
    pv = float(np.sqrt(max(w.to_numpy() @ cov @ w.to_numpy(), 1e-12)))
    w *= c.target_vol / pv
    w = w.clip(upper=c.max_weight)
    if w.sum() > c.max_gross:
        w *= c.max_gross / w.sum()
    return w


def decisions_from(out: pd.DataFrame, col: str, closes: pd.DataFrame, corr: dict, c: Construction) -> pd.DataFrame:
    vol = np.log(closes).diff().apply(P.ewma_vol)
    rows = {}
    for d, g in out.groupby("date"):
        if d not in corr:
            continue
        fc = g.set_index("asset")[col].reindex(closes.columns)
        rows[d] = weights_from_forecasts(fc, vol.loc[d], corr[d], c)
    return pd.DataFrame(rows).T.reindex(columns=closes.columns).fillna(0.0)


__all__ = ["FEATURES", "weekly_panel", "MODELS", "walk_forward", "forecast_stats", "Construction",
           "weights_from_forecasts", "decisions_from", "fit_with_inner_cv"]
