"""
LevelEdge v2 forecaster: P(close at target > K) for any level K.

How it works
------------
1. **Volatility units.** Each historical bar *i* gets a horizon scale ``s_i``:
   an EWMA of de-seasonalised returns, projected over bars i+1..i+h with a
   time-of-day variance profile and GARCH-style mean reversion toward the
   long-run variance (persistence and an inflation factor are fitted by
   Gaussian likelihood on training rows only). The realised move
   ``z_i = log(C[i+h] / C[i]) / s_i`` is then roughly unit-scale in every
   volatility regime and at every time of day.
2. **Baseline** ``p_base(k)``: kernel-smoothed empirical survival function of
   ``z`` (conditioned on how many session closes the window crosses). It is
   fat-tailed (with a small Student-t(5) tail blend so extreme levels never
   collapse to 0/1), regime-adaptive and well calibrated.
3. **Classifier** ``p_xgb(k | features)``: XGBoost *boosted from the baseline*
   (``base_margin = logit(p_base)``), so trees only learn corrections to a
   calibrated prior. One model is trained on every row
   repeated at several thresholds ``k``, labelled ``z_i > k``, with a
   monotone-decreasing constraint on ``k``. One model serves every price level,
   and P(above) can never increase with the level.
4. **Stacker.** ``p = sigmoid(a*logit(p_xgb) + b*logit(p_base) + c)`` with
   ``a, b >= 0``, fitted on purged walk-forward out-of-fold predictions. If
   the classifier adds nothing, ``a`` goes to ~0 and the forecast falls back to
   the calibrated baseline instead of overfitting.

Validation is expanding-window walk-forward with a purge of ``h`` bars plus an
embargo of ``h`` bars, so no training label overlaps or abuts the test window.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, ndtr
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from leveledge.features import build_features, feature_columns
from leveledge.sessions import DAILY_MIN, Horizon, _future_stock_bar_ends

EPS = 1e-4
K_GRID = np.linspace(-8, 8, 801)


@dataclass
class ForecastConfig:
    thresholds_per_row: int = 8
    n_folds: int = 6
    test_fraction: float = 0.6
    max_rows: int = 6000
    warmup: int = 60
    min_rows: int = 250
    seed: int = 7
    gate_margin: float = 0.002  # log-loss gain the classifier must show on the held-out fold
    n_jobs: int = -1
    xgb: dict = field(default_factory=lambda: dict(
        n_estimators=200, max_depth=3, learning_rate=0.03, min_child_weight=40,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=10.0, max_bin=64,
        tree_method="hist", objective="binary:logistic", eval_metric="logloss",
    ))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def log_loss(y, p) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def safe_auc(y, p) -> float | None:
    y = np.asarray(y)
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, p))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table: mean forecast vs observed frequency."""
    df = pd.DataFrame({"p": p, "y": y})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(bins, len(df)), labels=False)
    out = df.groupby("bin").agg(forecast=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)


def summarize(y, p) -> dict:
    return {"log_loss": log_loss(y, p), "brier": brier(y, p), "auc": safe_auc(y, p), "n": int(len(y))}


# ---------------------------------------------------------------------------
# volatility scaling
# ---------------------------------------------------------------------------

class DiurnalVol:
    """Time-of-day variance profile, shrunk toward 1 for thin slots."""

    def __init__(self, shrink: float = 20.0):
        self.shrink = shrink
        self.profile: dict = {}

    def fit(self, lr: np.ndarray, slots: np.ndarray, mask: np.ndarray) -> "DiurnalVol":
        r2 = lr[mask] ** 2
        s = slots[mask]
        ok = np.isfinite(r2)
        r2, s = r2[ok], s[ok]
        overall = r2.mean() if len(r2) else 1.0
        df = pd.DataFrame({"s": s, "r2": r2}).groupby("s")["r2"].agg(["mean", "size"])
        w = df["size"] / (df["size"] + self.shrink)
        ratio = (w * df["mean"] / overall + (1 - w)).clip(0.05, 25)
        self.profile = ratio.to_dict()  # variance multipliers
        return self

    def var_factor(self, slots) -> np.ndarray:
        return np.array([self.profile.get(s, 1.0) for s in slots], dtype=float)


def ewma_sigma(x: np.ndarray, span: int = 40) -> np.ndarray:
    s = pd.Series(x ** 2).ewm(span=span, adjust=False, min_periods=span // 2).mean()
    return np.sqrt(s.to_numpy())


class VolScaler:
    """Horizon volatility with intraday seasonality and mean reversion.

    Horizon variance from bar i over the next h bars::

        S2_i = c2 * sum_j f_{i+j} * (VL + phi**j * (sig_i**2 - VL))

    ``f`` = time-of-day variance multipliers, ``sig`` = EWMA of de-seasonalised
    returns, ``VL`` = long-run variance. ``phi`` (via half-life) and the
    inflation ``c2`` are chosen by Gaussian likelihood of realised moves.
    """

    HALF_LIVES = (10, 40, 160, 640, np.inf)  # bars

    def fit(self, lr, slots, mask, fwd, h) -> "VolScaler":
        self.h = h
        self.diurnal = DiurnalVol().fit(lr, slots, mask)
        vf = self.diurnal.var_factor(slots)
        self.sig2 = ewma_sigma(np.nan_to_num(lr) / np.sqrt(vf)) ** 2
        self.VL = float(np.nanmean(self.sig2[mask]))
        F = self._window_sum(vf, 1.0)
        best = None
        for hl in self.HALF_LIVES:
            phi = 0.0 if hl == 0 else (1.0 if np.isinf(hl) else 0.5 ** (1 / hl))
            G = self._window_sum(vf, phi)
            S2 = self.VL * F + (self.sig2 - self.VL) * G
            S2 = np.where(S2 > 0, S2, np.nan)
            z2 = fwd ** 2 / S2
            ok = mask & np.isfinite(z2)
            if ok.sum() < 30:
                continue
            c2 = float(np.mean(z2[ok]))
            score = 0.5 * np.log(c2) + 0.5 * np.mean(np.log(S2[ok]))
            if best is None or score < best[0]:
                best = (score, phi, hl, c2, S2)
        _, self.phi, self.half_life, self.c2, S2 = best
        self.scale = np.sqrt(self.c2 * S2)
        return self

    def _window_sum(self, vf, phi):
        n, h = len(vf), self.h
        out = np.full(n, np.nan)
        acc = np.zeros(max(n - h, 0))
        for j in range(1, h + 1):
            acc += (phi ** j) * vf[j: n - h + j]
        out[: n - h] = acc
        return out

    def path(self, future_slots) -> np.ndarray:
        """Horizon scale after each of the next bars (cumulative), last = live()."""
        vf = self.diurnal.var_factor(future_slots)
        w = self.phi ** np.arange(1, len(vf) + 1)
        S2 = self.VL * np.cumsum(vf) + (self.sig2[-1] - self.VL) * np.cumsum(w * vf)
        S2 = np.where(S2 > 0, S2, self.VL * np.cumsum(vf))
        return np.sqrt(self.c2 * S2)

    def live(self, future_slots) -> float:
        return float(self.path(future_slots)[-1])


# ---------------------------------------------------------------------------
# baseline: kernel-smoothed empirical survival function of z
# ---------------------------------------------------------------------------

class EmpiricalBaseline:
    T_DF = 5

    def __init__(self, min_group: int = 200):
        self.min_group = min_group
        self.curves: dict = {}

    @classmethod
    def _curve(cls, z: np.ndarray) -> np.ndarray:
        from scipy.stats import t as student_t
        z = z[np.isfinite(z)]
        n = len(z)
        sd = z.std()
        iqr = np.subtract(*np.percentile(z, [75, 25])) / 1.349
        bw = 0.9 * max(min(sd, iqr if iqr > 0 else sd), 1e-3) * n ** -0.2
        zs = np.sort(z)
        chunks = [ndtr((zs[None, :] - K_GRID[j:j + 200, None]) / bw).mean(axis=1)
                  for j in range(0, len(K_GRID), 200)]
        emp = np.concatenate(chunks)
        # Student-t tail blend: the kernel has no mass beyond the sample range
        nu = cls.T_DF
        tail = student_t.sf((K_GRID - np.median(z)) / (sd * np.sqrt((nu - 2) / nu)), nu)
        w = max(0.02, 50.0 / (n + 50.0))
        return (1 - w) * emp + w * tail

    def fit(self, z: np.ndarray, groups: np.ndarray) -> "EmpiricalBaseline":
        self.curves = {"_all": self._curve(z)}
        for g in np.unique(groups):
            m = groups == g
            if m.sum() >= self.min_group:
                self.curves[int(g)] = self._curve(z[m])
        return self

    def prob(self, k, group) -> np.ndarray:
        k = np.atleast_1d(np.asarray(k, dtype=float))
        group = np.broadcast_to(np.atleast_1d(group), k.shape)
        out = np.empty_like(k)
        for g in np.unique(group):
            m = group == g
            curve = self.curves.get(int(g), self.curves["_all"])
            out[m] = np.interp(k[m], K_GRID, curve)
        return np.clip(out, EPS, 1 - EPS)


# ---------------------------------------------------------------------------
# stacker
# ---------------------------------------------------------------------------

def fit_stacker(lx, lb, y) -> tuple[float, float, float]:
    """Logistic regression on (logit p_xgb, logit p_base) with non-negative slopes.

    No intercept: an intercept fitted on the CV window learns that window's
    drift (e.g. "the market fell for six weeks") and bets on it continuing.
    The stacker may sharpen, flatten or mix the two forecasts but not tilt them.
    """
    A = np.column_stack([np.asarray(lx, float), np.asarray(lb, float), np.zeros(len(y))])
    y = np.asarray(y, float)
    lam = 1e-4

    def fg(w):
        p = np.clip(expit(A @ w), EPS, 1 - EPS)
        f = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + lam * (w[0] ** 2 + w[1] ** 2)
        g = A.T @ (p - y) / len(y) + lam * np.array([2 * w[0], 2 * w[1], 0.0])
        return f, g

    res = minimize(fg, x0=np.array([0.5, 0.5, 0.0]), jac=True, method="L-BFGS-B",
                   bounds=[(0, 3), (0, 3), (0, 0)])
    return tuple(float(v) for v in res.x)


def _lg(p):
    return logit(np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS))


# ---------------------------------------------------------------------------
# the forecaster
# ---------------------------------------------------------------------------

class LevelForecaster:
    """Fitted forecast of the price distribution at one horizon.

    Parameters
    ----------
    bars : cleaned, completed OHLCV bars (see :func:`leveledge.data.clean_bars`)
    horizon : :class:`leveledge.sessions.Horizon` from the last bar to the target
    """

    def __init__(self, bars: pd.DataFrame, interval_min: int, is_crypto: bool,
                 horizon: Horizon, config: ForecastConfig | None = None):
        self.cfg = config or ForecastConfig()
        self.interval_min = interval_min
        self.is_crypto = is_crypto
        self.horizon = horizon
        self.bars = bars
        self.current_price = float(bars["Close"].iloc[-1])
        self.fitted = False
        self._prepare()

    # ---- data preparation ----------------------------------------------------
    def _prepare(self) -> None:
        F = build_features(self.bars, self.interval_min, self.is_crypto)
        self.frame = F
        self.feature_names = feature_columns(F) + ["crossed", "log_scale"]
        h = self.horizon.bars
        c = self.bars["Close"].to_numpy(float)
        n = len(c)
        self.lr = F["_logret"].to_numpy(float)
        intraday = self.interval_min < DAILY_MIN
        self.slots = (F["minute_of_day"].to_numpy() if intraday and "minute_of_day" in F
                      else np.zeros(n, dtype=int))
        sess = F["_session"].to_numpy()

        fwd = np.full(n, np.nan)
        crossed = np.zeros(n)
        if n > h:
            fwd[: n - h] = np.log(c[h:] / c[: n - h])
            crossed[: n - h] = sess[h:] - sess[: n - h]
        self.fwd = fwd
        self.crossed = crossed

        start = min(self.cfg.warmup, max(0, n - h - self.cfg.min_rows))
        rows = np.arange(start, max(start, n - h))
        if len(rows) > self.cfg.max_rows:
            rows = rows[-self.cfg.max_rows:]
        self.rows = rows
        if len(rows) < 50:
            raise ValueError(
                f"Only {len(rows)} usable training rows for a {h}-bar horizon. "
                "Use a shorter interval, a nearer target, or more history."
            )
        X = F[feature_columns(F)].to_numpy(float)
        self.X_base = X

        # future time-of-day slots for the live forecast
        step = pd.Timedelta(minutes=self.interval_min)
        if intraday:
            anchor = self.horizon.anchor_time
            if self.is_crypto:
                fut = [anchor + step * (j + 1) for j in range(h)]
            else:
                fut = [e for e, _ in _future_stock_bar_ends(anchor, self.horizon.effective_target,
                                                            self.interval_min)[0]]
            self.future_slots = np.array([t.hour * 60 + t.minute for t in fut])
        else:
            from leveledge.sessions import nyse_schedule
            if self.is_crypto:
                fut = [self.horizon.anchor_time + step * (j + 1) for j in range(h)]
            else:
                closes = nyse_schedule(self.horizon.anchor_time, self.horizon.effective_target)["market_close"]
                fut = [c for c in closes if self.horizon.anchor_time < c <= self.horizon.effective_target][:h]
            self.future_slots = np.zeros(h, dtype=int)
        self.future_ends = pd.DatetimeIndex(fut)

    def _design(self, idx: np.ndarray, scale: np.ndarray, k: np.ndarray) -> np.ndarray:
        reps = len(k) // len(idx)
        base = np.column_stack([self.X_base[idx], self.crossed[idx], np.log(scale[idx])])
        return np.column_stack([np.repeat(base, reps, axis=0), k])

    def _sample_k(self, m: int, rng) -> np.ndarray:
        K = self.cfg.thresholds_per_row
        return np.where(rng.random((m, K)) < 0.7, rng.normal(0, 1.0, (m, K)),
                        rng.uniform(-4, 4, (m, K))).ravel()

    @staticmethod
    def _xgb_prob(model, X, pb) -> np.ndarray:
        """Classifier probability boosted from the baseline margin."""
        if model is None:
            return pb
        return model.predict_proba(X, base_margin=_lg(pb))[:, 1]

    def _predict_rows(self, comp: dict, idx: np.ndarray, k: np.ndarray):
        """(p_xgb, p_base) for historical rows ``idx`` at thresholds ``k`` (len = len(idx)*reps)."""
        reps = len(k) // len(idx)
        X = self._design(idx, comp["scale"], k)
        pb = comp["base"].prob(k, np.repeat(self.crossed[idx], reps))
        return self._xgb_prob(comp["model"], X, pb), pb

    def _fit_components(self, train_idx, rng, fit_model: bool = True) -> dict:
        mask = np.zeros(len(self.lr), dtype=bool)
        mask[train_idx] = True
        vol = VolScaler().fit(self.lr, self.slots, mask, self.fwd, self.horizon.bars)
        scale = vol.scale
        ok = train_idx[np.isfinite(scale[train_idx]) & (scale[train_idx] > 0)
                       & np.isfinite(self.fwd[train_idx])]
        z = self.fwd[ok] / scale[ok]
        base = EmpiricalBaseline().fit(z, self.crossed[ok])
        k = self._sample_k(len(ok), rng)
        reps = self.cfg.thresholds_per_row
        y = (np.repeat(z, reps) > k).astype(int)
        X = self._design(ok, scale, k)
        pb = base.prob(k, np.repeat(self.crossed[ok], reps))
        model = None
        if fit_model and 0 < y.mean() < 1:
            mono = tuple([0] * len(self.feature_names) + [-1])
            model = XGBClassifier(**self.cfg.xgb, monotone_constraints=mono,
                                  n_jobs=self.cfg.n_jobs, random_state=self.cfg.seed)
            model.fit(X, y, base_margin=_lg(pb))
        return dict(vol=vol, scale=scale, base=base, model=model, train=ok,
                    scale_median=float(np.median(scale[ok])))

    # ---- training --------------------------------------------------------------
    def fit(self, evaluate: bool = True) -> "LevelForecaster":
        rng = np.random.default_rng(self.cfg.seed)
        rows, h = self.rows, self.horizon.bars
        self.folds = []
        oof = []
        if evaluate and len(rows) >= self.cfg.min_rows:
            n_test = int(len(rows) * self.cfg.test_fraction)
            bounds = np.linspace(len(rows) - n_test, len(rows), self.cfg.n_folds + 1).astype(int)
            for f in range(self.cfg.n_folds):
                test = rows[bounds[f]: bounds[f + 1]]
                if len(test) < 10:
                    continue
                # purge: training labels must end before the test window, plus an embargo of h bars
                train = rows[rows + h < test[0] - h]
                if len(train) < 100:
                    continue
                comp = self._fit_components(train, rng)
                sc = comp["scale"]
                t = test[np.isfinite(sc[test]) & (sc[test] > 0)]
                if len(t) == 0:
                    continue
                k = self._sample_k(len(t), rng)
                reps = self.cfg.thresholds_per_row
                y = (np.repeat(self.fwd[t] / sc[t], reps) > k).astype(int)
                px, pb = self._predict_rows(comp, t, k)
                comp["test"] = t
                comp["clim"] = self.fwd[comp["train"]]
                self.folds.append(comp)
                hi_vol = np.repeat(sc[t] > comp["scale_median"], reps)
                oof.append(pd.DataFrame({"y": y, "px": px, "pb": pb, "fold": f, "high_vol": hi_vol}))

        self.gate = None
        if oof:
            O = pd.concat(oof, ignore_index=True)
            self.weights = self._gated_weights(O)
            O["p"] = self._stack(O["px"], O["pb"])
            self.oof = O
            self.metrics = {name: summarize(O["y"], O[col])
                            for name, col in [("blend", "p"), ("xgb", "px"), ("baseline", "pb")]}
            self.regime_metrics = {
                label: {name: summarize(g["y"], g[col]) for name, col in
                        [("blend", "p"), ("baseline", "pb")]}
                for label, g in [("low_vol", O[~O.high_vol]), ("high_vol", O[O.high_vol])] if len(g)
            }
            self.reliability = reliability(O["y"].to_numpy(), O["p"].to_numpy())
        else:
            self.weights = (0.0, 1.0, 0.0)  # no evidence the classifier helps: baseline only
            self.oof, self.metrics, self.regime_metrics, self.reliability = None, {}, {}, None

        # final model on everything (the classifier only if it earned a weight)
        self.final = self._fit_components(rows, rng, fit_model=self.weights[0] > 0)
        self.vol, self.scale, self.base, self.model = (self.final[k] for k in ("vol", "scale", "base", "model"))
        self.live_scale = self.vol.live(self.future_slots)
        self.fitted = True
        return self

    def _gated_weights(self, O: pd.DataFrame) -> tuple[float, float, float]:
        """Fit stack weights, but only let the classifier in if it also wins on
        the last fold, held out from the weight fit itself.

        On the 4,800-forecast benchmark the classifier layer was statistically
        indistinguishable from the baseline, so it must re-earn its weight on
        every fit or the forecast is the calibrated baseline alone.
        """
        folds = sorted(O["fold"].unique())
        if len(folds) >= 3:
            last = folds[-1]
            early, hold = O[O["fold"] != last], O[O["fold"] == last]
            w = fit_stacker(_lg(early["px"]), _lg(early["pb"]), early["y"].to_numpy())
            a, b, c = w
            p_hold = np.clip(expit(a * _lg(hold["px"]) + b * _lg(hold["pb"]) + c), EPS, 1 - EPS)
            ll_blend, ll_base = log_loss(hold["y"], p_hold), log_loss(hold["y"], hold["pb"])
            passed = bool(ll_blend < ll_base - self.cfg.gate_margin)
            self.gate = {"passed": passed, "holdout_log_loss_blend": ll_blend,
                         "holdout_log_loss_baseline": ll_base, "holdout_n": int(len(hold))}
            if not passed:
                return (0.0, 1.0, 0.0)
        return fit_stacker(_lg(O["px"]), _lg(O["pb"]), O["y"].to_numpy())

    def _stack(self, px, pb):
        a, b, c = self.weights
        return np.clip(expit(a * _lg(px) + b * _lg(pb) + c), EPS, 1 - EPS)

    # ---- inference -------------------------------------------------------------
    def k_of(self, levels) -> np.ndarray:
        levels = np.atleast_1d(np.asarray(levels, dtype=float))
        if (levels <= 0).any():
            raise ValueError("Price levels must be positive.")
        return np.log(levels / self.current_price) / self.live_scale

    def _components(self, k: np.ndarray):
        last = len(self.lr) - 1
        g = self.horizon.sessions_crossed if self.interval_min < DAILY_MIN else self.horizon.bars
        base_row = np.concatenate([self.X_base[last], [g, math.log(self.live_scale)]])
        X = np.column_stack([np.tile(base_row, (len(k), 1)), k])
        pb = self.base.prob(k, g)
        return self._xgb_prob(self.model, X, pb), pb

    def prob_above(self, levels) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        px, pb = self._components(self.k_of(levels))
        return self._stack(px, pb)

    def components(self, levels) -> pd.DataFrame:
        k = self.k_of(levels)
        px, pb = self._components(k)
        return pd.DataFrame({"level": np.atleast_1d(levels), "k_sigma": k, "p_xgb": px,
                             "p_baseline": pb, "p": self._stack(px, pb)})

    def curve(self, n: int = 241, span_sigma: float = 4.0) -> pd.DataFrame:
        """P(above) on a level grid spanning +/- ``span_sigma`` horizon sigmas."""
        k = np.linspace(-span_sigma, span_sigma, n)
        levels = self.current_price * np.exp(k * self.live_scale)
        px, pb = self._components(k)
        p = np.minimum.accumulate(self._stack(px, pb))  # guard tiny float non-monotonicity
        return pd.DataFrame({"level": levels, "k_sigma": k, "p_above": p, "p_baseline": pb})

    def fan(self, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
        """Forecast range after every future bar up to the target.

        The target distribution's shape (in vol units) is scaled by the fitted
        variance term structure at each intermediate bar. Intermediate bands are
        an interpolation, not separately fitted forecasts.
        """
        cur = self.curve(n=801, span_sigma=6)
        cdf = 1 - cur["p_above"].to_numpy()
        zq = {q: float(np.interp(q, cdf, cur["k_sigma"])) for q in qs}
        path = self.vol.path(self.future_slots)
        ratio = path / path[-1]
        out = pd.DataFrame({"time": self.future_ends[: len(ratio)], "step": np.arange(1, len(ratio) + 1)})
        for q, z in zq.items():
            out[f"q{int(round(q * 100)):02d}"] = self.current_price * np.exp(z * self.live_scale * ratio)
        return out

    def quantiles(self, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> dict:
        cur = self.curve(n=801, span_sigma=6)
        cdf = 1 - cur["p_above"].to_numpy()
        return {q: float(np.interp(q, cdf, cur["level"])) for q in qs}

    # ---- diagnostics ------------------------------------------------------------
    def level_backtest(self, level: float) -> dict | None:
        """Out-of-fold performance for *this* level's move (same relative move
        applied at every historical test bar), vs climatology."""
        if not self.folds:
            return None
        thr = math.log(level / self.current_price)
        ys, ps, pbs, pcs, pxs = [], [], [], [], []
        for fd in self.folds:
            t = fd["test"]
            k = thr / fd["scale"][t]
            y = (self.fwd[t] > thr).astype(int)
            px, pb = self._predict_rows(fd, t, k)
            clim = np.clip(np.mean(fd["clim"][np.isfinite(fd["clim"])] > thr), EPS, 1 - EPS)
            ys.append(y); pxs.append(px); pbs.append(pb); pcs.append(np.full(len(t), clim))
            ps.append(self._stack(px, pb))
        y, p, pb, pc, px = map(np.concatenate, (ys, ps, pbs, pcs, pxs))
        out = {"blend": summarize(y, p), "baseline": summarize(y, pb), "xgb": summarize(y, px),
               "climatology": summarize(y, pc), "base_rate": float(y.mean())}
        # A level almost never (or almost always) crossed historically has a
        # near-zero climatology Brier, which makes the skill ratio meaningless.
        informative = 0.02 <= out["base_rate"] <= 0.98 and out["climatology"]["brier"] > 0.005
        out["informative"] = bool(informative)
        out["brier_skill"] = (1 - out["blend"]["brier"] / out["climatology"]["brier"]) if informative else None
        pb_ = (p >= 0.6).astype(int)
        out["precision_at_60"] = float((y[pb_ == 1]).mean()) if pb_.sum() else None
        out["avg_precision"] = _avg_precision(y, p)
        return out

    def interval_coverage(self, lo: float = 0.1, hi: float = 0.9) -> float | None:
        """Share of out-of-fold outcomes that landed inside the forecast's [lo, hi] range."""
        if not self.folds:
            return None
        grid = np.linspace(-5, 5, 81)
        hits = []
        for fd in self.folds:
            t = fd["test"]
            k = np.tile(grid, len(t))
            px, pb = self._predict_rows(fd, t, k)
            S = self._stack(px, pb).reshape(len(t), len(grid))
            S = np.minimum.accumulate(S, axis=1)
            cdf = 1 - S
            z = self.fwd[t] / fd["scale"][t]
            qlo = np.array([np.interp(lo, r, grid) for r in cdf])
            qhi = np.array([np.interp(hi, r, grid) for r in cdf])
            hits.append((z >= qlo) & (z <= qhi))
        return float(np.concatenate(hits).mean())

    def feature_importance(self, top: int = 12) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        names = self.feature_names + ["level (k)"]
        gain = self.model.get_booster().get_score(importance_type="gain")
        s = pd.Series({names[int(key[1:])]: v for key, v in gain.items()})
        return (s / s.sum()).sort_values(ascending=False).head(top)

    def summary(self) -> dict:
        idx = self.bars.index
        return {
            "rows": int(len(self.rows)), "history_start": idx[0], "history_end": idx[-1],
            "horizon_bars": self.horizon.bars, "effective_target": self.horizon.effective_target,
            "sessions_crossed": self.horizon.sessions_crossed, "horizon_sigma": self.live_scale,
            "weights": dict(zip(["xgb", "baseline", "intercept"], self.weights)),
            "vol_half_life_bars": self.vol.half_life, "vol_inflation": self.vol.c2,
            "metrics": self.metrics, "regime_metrics": self.regime_metrics, "gate": self.gate,
        }


def _avg_precision(y, p) -> float | None:
    from sklearn.metrics import average_precision_score
    if np.asarray(y).min() == np.asarray(y).max():
        return None
    return float(average_precision_score(y, p))
