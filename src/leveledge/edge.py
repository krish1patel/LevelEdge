"""
Edge and position sizing against a market price.

Conventions
-----------
* A YES contract costs ``ask`` and pays $1 if the price settles above the strike.
* A NO contract costs ``1 - bid`` and pays $1 otherwise.
* Fills are taker fills at the quoted price, plus Kalshi's fee
  (:func:`leveledge.kalshi.taker_fee`). Edge is expected profit per contract
  after costs: ``p - ask - fee`` for YES, ``bid - p - fee`` for NO.

Sizing
------
Strikes in one event are nested bets on a single outcome, so they are sized
jointly: :func:`size_event` maximises expected log wealth over the event's
outcome intervals (implied by the model's own monotone probabilities), then
scales by a Kelly fraction and caps the total stake per event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from leveledge.kalshi import taker_fee


def evaluate_quotes(p, bid, ask, fee_rate: float = 0.07) -> pd.DataFrame:
    """Best side per contract and its after-cost edge (vectorised)."""
    p, bid, ask = (np.asarray(x, dtype=float) for x in (p, bid, ask))
    cost_yes = ask + taker_fee(ask, fee_rate)
    cost_no = (1 - bid) + taker_fee(1 - bid, fee_rate)
    edge_yes = p - cost_yes
    edge_no = (1 - p) - cost_no
    yes = edge_yes >= edge_no
    return pd.DataFrame({
        "edge_yes": edge_yes,
        "edge_no": edge_no,
        "side": np.where(yes, "yes", "no"),
        "edge": np.where(yes, edge_yes, edge_no),
        "cost": np.where(yes, cost_yes, cost_no),          # all-in price per contract
        "p_win": np.where(yes, p, 1 - p),
    })


def kelly_single(p_win, cost):
    """Full-Kelly bankroll fraction for one binary bet paying $1 at all-in ``cost``."""
    p_win, cost = np.asarray(p_win, float), np.asarray(cost, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (p_win - cost) / (1 - cost)
    return np.clip(np.nan_to_num(f, nan=0.0), 0.0, 1.0)


def size_event(strikes, p_above, sides, costs, edges, *, kelly_mult: float = 0.25,
               event_cap: float = 0.05, min_edge: float = 0.02) -> np.ndarray:
    """Bankroll fraction to stake on each contract of one event.

    Parameters
    ----------
    strikes, p_above : the event's strikes and the model's P(settle > strike).
    sides            : "yes"/"no" per contract (from :func:`evaluate_quotes`).
    costs, edges     : all-in cost per contract and after-cost edge.
    kelly_mult       : fraction of the joint Kelly solution to take (0.25 = quarter Kelly).
    event_cap        : maximum total stake on the event, as a bankroll fraction.
    min_edge         : contracts with less edge than this are not traded.

    Returns stake fractions (dollars staked / bankroll); contracts = stake * bankroll / cost.
    """
    strikes = np.asarray(strikes, float)
    p_above = np.asarray(p_above, float)
    costs = np.asarray(costs, float)
    edges = np.asarray(edges, float)
    sides = np.asarray(sides)
    n = len(strikes)
    stake = np.zeros(n)
    live = np.where(edges >= min_edge)[0]
    if len(live) == 0:
        return stake

    # outcome intervals between sorted strikes; probabilities from the model's own curve
    order = np.argsort(strikes)
    ks, pa = strikes[order], np.minimum.accumulate(np.clip(p_above[order], 0, 1))
    probs = np.concatenate([[1 - pa[0]], pa[:-1] - pa[1:], [pa[-1]]])  # state j: price in (k_{j-1}, k_j]
    probs = np.clip(probs, 0, None)
    probs = probs / probs.sum()
    # payoff matrix: contract i wins in state j?
    rank = np.empty(n, int)
    rank[order] = np.arange(n)
    states = np.arange(n + 1)
    above = states[None, :] > rank[:, None]          # settle above strike i in state j
    wins = np.where((sides == "yes")[:, None], above, ~above)

    W = wins[live].astype(float) / costs[live][:, None]   # gross return per $ staked, by state

    def neg_growth(x):
        wealth = 1 - x.sum() + x @ W
        if np.any(wealth <= 1e-9):
            return 1e6
        return -float(probs @ np.log(wealth))

    def grad(x):
        wealth = 1 - x.sum() + x @ W
        wealth = np.maximum(wealth, 1e-9)
        return -((W - 1) @ (probs / wealth))

    x0 = np.minimum(kelly_single(np.where(sides[live] == "yes", p_above[live], 1 - p_above[live]),
                                 costs[live]), event_cap) / max(len(live), 1)
    cap_full = min(1.0, event_cap / max(kelly_mult, 1e-9))   # cap applies after the Kelly scaling
    res = minimize(neg_growth, x0, jac=grad, method="SLSQP",
                   bounds=[(0.0, cap_full)] * len(live),
                   constraints=[{"type": "ineq", "fun": lambda x: cap_full - x.sum()}],
                   options={"maxiter": 200, "ftol": 1e-10})
    x = np.clip(res.x if res.success else x0, 0, None)
    stake[live] = kelly_mult * x
    total = stake.sum()
    if total > event_cap:
        stake *= event_cap / total
    stake[stake < 1e-5] = 0.0
    return stake


__all__ = ["evaluate_quotes", "kelly_single", "size_event"]
