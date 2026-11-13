"""
Market-state models: volatility, regimes and return distributions (numpy/scipy only).

Volatility / distribution
    gjr_fit, gjr_simulate   GJR-GARCH(1,1) with Student-t errors (Glosten, Jagannathan &
                            Runkle 1993), multi-day paths by filtered historical simulation
                            (Barone-Adesi, Giannopoulos & Vosper 1999).
    daily_variance          range-based daily variance (Parkinson 1980) plus the overnight gap.
    har_design              log-HAR regressors (Corsi 2009): daily, weekly, monthly variance.
Regimes
    hmm2_fit, hmm2_filter   two-state Gaussian hidden Markov model (Hamilton 1989), fitted by
                            EM; filtered (past-only) probability of the turbulent state.
    jm_features, jm_fit,    statistical jump model (Nystrup et al. 2020; Shu, Yu & Mulvey
    jm_online               2024): k-means with a penalty on switching; the online state at t
                            is the last state of the optimal path through data up to t.
    turbulence              Mahalanobis distance of cross-asset returns (Kritzman & Li 2010).
    absorption_ratio        variance share of the top eigenvectors (Kritzman et al. 2011).
Scoring
    pinball, crps_from_quantiles
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, special
from scipy.signal import lfilter


# ---------------------------------------------------------------------------
# GJR-GARCH(1,1)-t
# ---------------------------------------------------------------------------
def _gjr_s2(e: np.ndarray, omega, alpha, gamma, beta) -> np.ndarray:
    """Conditional variance path; s2[t] uses shocks up to t-1. Starts at the sample variance."""
    x = omega + (alpha + gamma * (e < 0)) * e * e
    u = np.empty_like(e)
    u[0] = np.var(e)
    u[1:] = x[:-1]
    return lfilter([1.0], [1.0, -beta], u)


def _t_ll(z, s, nu):
    return (special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
            - np.log(s) - (nu + 1) / 2 * np.log1p(z * z / (nu - 2)))


def gjr_fit(r: np.ndarray, x0: dict | None = None) -> dict:
    """Fit on daily returns in percent. Returns params, last variance and standardized residuals."""
    r = np.asarray(r, float)

    def nll(p):
        mu, omega, alpha, gamma, beta, nu = p
        if alpha + gamma / 2 + beta >= 0.999 or alpha + gamma < 0:
            return 1e10
        e = r - mu
        s2 = _gjr_s2(e, omega, alpha, gamma, beta)
        s = np.sqrt(np.maximum(s2, 1e-12))
        return -float(np.sum(_t_ll(e / s, s, nu)))

    start = [x0[k] for k in ("mu", "omega", "alpha", "gamma", "beta", "nu")] if x0 else [0.05, 0.02, 0.01, 0.12, 0.88, 7.0]
    bounds = [(-1, 1), (1e-6, 1.0), (0.0, 0.5), (-0.5, 0.5), (0.0, 0.999), (2.1, 60)]
    res = optimize.minimize(nll, start, method="L-BFGS-B", bounds=bounds)
    mu, omega, alpha, gamma, beta, nu = res.x
    e = r - mu
    s2 = _gjr_s2(e, omega, alpha, gamma, beta)
    next_s2 = omega + (alpha + gamma * (e[-1] < 0)) * e[-1] ** 2 + beta * s2[-1]
    return {"mu": mu, "omega": omega, "alpha": alpha, "gamma": gamma, "beta": beta, "nu": nu,
            "z": e / np.sqrt(s2), "s2": s2, "next_s2": float(next_s2)}


def gjr_filter(fit: dict, r: np.ndarray) -> np.ndarray:
    """Next-day variance after each observation of ``r`` (fixed parameters)."""
    e = np.asarray(r, float) - fit["mu"]
    s2 = _gjr_s2(e, fit["omega"], fit["alpha"], fit["gamma"], fit["beta"])
    return fit["omega"] + (fit["alpha"] + fit["gamma"] * (e < 0)) * e * e + fit["beta"] * s2


def gjr_simulate(fit: dict, s2_next: float, horizon: int, n: int = 2000, seed: int = 0) -> np.ndarray:
    """Cumulative return paths (n x horizon, percent) by bootstrapping standardized residuals."""
    rng = np.random.default_rng(seed)
    z = fit["z"][np.isfinite(fit["z"])]
    s2 = np.full(n, s2_next)
    out = np.empty((n, horizon))
    acc = np.zeros(n)
    for h in range(horizon):
        e = np.sqrt(s2) * rng.choice(z, n)
        acc += fit["mu"] + e
        out[:, h] = acc
        s2 = fit["omega"] + (fit["alpha"] + fit["gamma"] * (e < 0)) * e * e + fit["beta"] * s2
    return out


# ---------------------------------------------------------------------------
# realized-variance proxy and HAR
# ---------------------------------------------------------------------------
def daily_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Parkinson range variance plus the squared overnight gap (log units)."""
    park = np.log(ohlc["High"] / ohlc["Low"]) ** 2 / (4 * np.log(2))
    gap = np.log(ohlc["Open"] / ohlc["Close"].shift(1)) ** 2
    return (park + gap.fillna(0)).clip(lower=1e-10)


def har_design(rv: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"d": np.log(rv), "w": np.log(rv.rolling(5).mean()), "m": np.log(rv.rolling(22).mean())})


# ---------------------------------------------------------------------------
# two-state Gaussian HMM
# ---------------------------------------------------------------------------
def _emis(x, mu, sd):
    return np.exp(-0.5 * ((x[:, None] - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) + 1e-300


def _forward(b0, b1, A, pi):
    T = len(b0)
    a = np.empty((T, 2))
    c = np.empty(T)
    A00, A01, A10, A11 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    p0, p1 = pi[0] * b0[0], pi[1] * b1[0]
    for t in range(T):
        if t:
            q0, q1 = a0, a1  # noqa: F821 - set in the previous iteration
            p0 = (q0 * A00 + q1 * A10) * b0[t]
            p1 = (q0 * A01 + q1 * A11) * b1[t]
        s = p0 + p1
        a0, a1 = p0 / s, p1 / s
        a[t, 0], a[t, 1], c[t] = a0, a1, s
    return a, c


def hmm2_fit(x: np.ndarray, init: dict | None = None, iters: int = 200, tol: float = 1e-7) -> dict:
    """EM for a 2-state Gaussian HMM on daily returns (percent). State 1 = higher variance."""
    x = np.asarray(x, float)
    if init:
        A, mu, sd, pi = init["A"].copy(), init["mu"].copy(), init["sd"].copy(), init["pi"].copy()
    else:
        s = x.std()
        A, mu, sd, pi = np.array([[0.98, 0.02], [0.05, 0.95]]), np.array([0.05, -0.05]), np.array([0.7 * s, 1.8 * s]), np.array([0.5, 0.5])
    prev = -np.inf
    T = len(x)
    for _ in range(iters):
        b = _emis(x, mu, sd)
        a, c = _forward(b[:, 0].tolist(), b[:, 1].tolist(), A, pi)
        # backward (scaled)
        beta = np.empty((T, 2))
        beta[-1] = 1.0
        b0, b1 = b[:, 0].tolist(), b[:, 1].tolist()
        A00, A01, A10, A11 = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
        n0, n1 = 1.0, 1.0
        for t in range(T - 2, -1, -1):
            u0, u1 = b0[t + 1] * n0, b1[t + 1] * n1
            n0, n1 = (A00 * u0 + A01 * u1) / c[t + 1], (A10 * u0 + A11 * u1) / c[t + 1]
            beta[t, 0], beta[t, 1] = n0, n1
        g = a * beta
        g /= g.sum(1, keepdims=True)
        xi = (a[:-1, :, None] * A[None] * (b[1:] * beta[1:])[:, None, :]) / c[1:, None, None]
        A = xi.sum(0) / xi.sum((0, 2))[:, None]
        w = g.sum(0)
        mu = (g * x[:, None]).sum(0) / w
        sd = np.sqrt((g * (x[:, None] - mu) ** 2).sum(0) / w).clip(min=1e-3)
        pi = g[0]
        ll = float(np.log(c).sum())
        if ll - prev < tol * abs(ll):
            break
        prev = ll
    if sd[0] > sd[1]:                       # keep state 1 = turbulent
        A, mu, sd, pi = A[::-1, ::-1], mu[::-1], sd[::-1], pi[::-1]
    return {"A": A, "mu": mu, "sd": sd, "pi": pi, "ll": prev}


def hmm2_filter(fit: dict, x: np.ndarray) -> np.ndarray:
    """Filtered probability of the turbulent state at each t (data up to t only)."""
    b = _emis(np.asarray(x, float), fit["mu"], fit["sd"])
    a, _ = _forward(b[:, 0].tolist(), b[:, 1].tolist(), fit["A"], fit["pi"])
    return a[:, 1]


# ---------------------------------------------------------------------------
# statistical jump model
# ---------------------------------------------------------------------------
def jm_features(r: pd.Series) -> pd.DataFrame:
    """Shu, Yu & Mulvey (2024) style features from daily excess log returns."""
    f = {}
    for hl in (5, 10, 21):
        m = r.ewm(halflife=hl).mean()
        dd = np.sqrt((r.clip(upper=0) ** 2).ewm(halflife=hl).mean()) + 1e-6
        f[f"ret_{hl}"], f[f"logdd_{hl}"], f[f"sortino_{hl}"] = m, np.log(dd), m / dd
    return pd.DataFrame(f)


def _viterbi2(c0: list, c1: list, lam: float):
    T = len(c0)
    v0, v1 = c0[0], c1[0]
    bp0, bp1 = [0] * T, [1] * T
    for t in range(1, T):
        if v0 <= v1 + lam:
            n0, bp0[t] = v0, 0
        else:
            n0, bp0[t] = v1 + lam, 1
        if v1 <= v0 + lam:
            n1, bp1[t] = v1, 1
        else:
            n1, bp1[t] = v0 + lam, 0
        m = min(n0, n1)
        v0, v1 = n0 - m + c0[t], n1 - m + c1[t]
    k = 0 if v0 <= v1 else 1
    s = [0] * T
    for t in range(T - 1, -1, -1):
        s[t] = k
        k = bp0[t] if k == 0 else bp1[t]
    return np.array(s)


def jm_fit(X: np.ndarray, lam: float, n_init: int = 4, max_iter: int = 30, seed: int = 0):
    """Two-state jump model: minimise sum ||x_t - c_s(t)||^2 + lam * #switches. Returns centroids, states."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        c = X[rng.choice(len(X), 2, replace=False)].copy()
        s_prev = None
        for _ in range(max_iter):
            cost = ((X[:, None, :] - c[None]) ** 2).sum(-1)
            s = _viterbi2(cost[:, 0].tolist(), cost[:, 1].tolist(), lam)
            if s_prev is not None and np.array_equal(s, s_prev):
                break
            s_prev = s
            for k in (0, 1):
                if (s == k).any():
                    c[k] = X[s == k].mean(0)
        cost = ((X[:, None, :] - c[None]) ** 2).sum(-1)
        obj = float(cost[np.arange(len(s)), s].sum() + lam * np.count_nonzero(np.diff(s)))
        if best is None or obj < best[0]:
            best = (obj, c.copy(), s.copy())
    return best[1], best[2]


def jm_online(X: np.ndarray, c: np.ndarray, lam: float) -> np.ndarray:
    """Online state at each t: the end state of the optimal path through X[:t+1]."""
    cost = ((X[:, None, :] - c[None]) ** 2).sum(-1)
    c0, c1 = cost[:, 0].tolist(), cost[:, 1].tolist()
    v0, v1 = c0[0], c1[0]
    out = [0 if v0 <= v1 else 1]
    for t in range(1, len(c0)):
        n0, n1 = min(v0, v1 + lam) + c0[t], min(v1, v0 + lam) + c1[t]
        m = min(n0, n1)
        v0, v1 = n0 - m, n1 - m
        out.append(0 if v0 <= v1 else 1)
    return np.array(out)


# ---------------------------------------------------------------------------
# cross-asset fragility
# ---------------------------------------------------------------------------
def turbulence(rets: pd.DataFrame, window: int = 750) -> pd.Series:
    """Mahalanobis distance of each day's returns from the trailing window (past only)."""
    R = rets.to_numpy(float)
    out = np.full(len(R), np.nan)
    for t in range(window, len(R)):
        hist = R[t - window:t]
        ok = np.isfinite(hist).all(0) & np.isfinite(R[t])
        if ok.sum() < 3:
            continue
        h = hist[:, ok]
        d = R[t, ok] - h.mean(0)
        cov = np.cov(h, rowvar=False)
        out[t] = float(d @ np.linalg.pinv(cov) @ d) / ok.sum()
    return pd.Series(out, index=rets.index)


def absorption_ratio(rets: pd.DataFrame, window: int = 500, share: float = 0.2) -> pd.Series:
    """Fraction of variance explained by the top ``share`` of eigenvectors (EWMA cov, halflife 250)."""
    R = rets.to_numpy(float)
    n = R.shape[1]
    k = max(1, int(round(n * share)))
    w = 0.5 ** (np.arange(window)[::-1] / 250)
    out = np.full(len(R), np.nan)
    for t in range(window, len(R), 1):
        h = R[t - window + 1:t + 1]
        if not np.isfinite(h).all():
            continue
        hc = h - np.average(h, axis=0, weights=w)
        cov = (hc * w[:, None]).T @ hc / w.sum()
        ev = np.linalg.eigvalsh(cov)[::-1]
        out[t] = ev[:k].sum() / ev.sum()
    return pd.Series(out, index=rets.index)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def pinball(y: np.ndarray, q: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Mean pinball loss per row: y (n,), q (n, L) quantiles at ``levels`` (L,)."""
    d = y[:, None] - q
    return np.mean(np.maximum(levels * d, (levels - 1) * d), axis=1)


def crps_from_quantiles(y, q, levels):
    """CRPS approximated as twice the average pinball loss over a dense quantile grid."""
    return 2 * pinball(np.asarray(y, float), np.asarray(q, float), np.asarray(levels, float))
