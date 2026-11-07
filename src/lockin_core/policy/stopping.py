"""One slot, one player: exact optimal stopping by backward induction (design section 11).

With games k = 0..n-1 left, score X_k and miss probability q_k, the last game forced:

    V_{n-1} = (1 - q_{n-1}) E[X_{n-1}]
    V_k     = (1 - q_k) E[max(X_k, V_{k+1})] + q_k V_{k+1}

and the rule is lock game k iff X_k >= c_k = V_{k+1}: the value of continuing, not his average.
Distributions are discrete (values, weights); ``normal_grid`` gives a fine normal discretisation
for the closed-form checks.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def normal_grid(mu: float, sd: float, q: int = 4001) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint-quantile discretisation (accurate to ~1e-4 SD for q = 4001)."""
    u = (np.arange(q) + 0.5) / q
    return mu + sd * stats.norm.ppf(u), np.full(q, 1.0 / q)


def e_max(values: np.ndarray, weights: np.ndarray, c: float) -> float:
    return float(np.sum(weights * np.maximum(values, c)))


def stopping_values(games: list[tuple[np.ndarray, np.ndarray, float]]) -> list[float]:
    """``games[k] = (values, weights, miss probability)``; returns [V_0, ..., V_{n-1}, 0]."""
    n = len(games)
    V = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        v, w, q = games[k]
        cont = V[k + 1]
        ex = float(np.sum(w * v)) if k == n - 1 else e_max(v, w, cont)
        V[k] = (1 - q) * ex + q * cont
    return V


def thresholds(games: list[tuple[np.ndarray, np.ndarray, float]]) -> list[float]:
    """Lock threshold after each game k (= V_{k+1}); the last game's score is forced."""
    V = stopping_values(games)
    return V[1:]


def threshold_with_rho(mu: float, sd: float, q: float, rho: float, iters: int = 200) -> float:
    """One forced game left, with a within-week shock: x = (1-q)(mu + rho (x - mu))."""
    x = mu
    for _ in range(iters):
        x = (1 - q) * (mu + rho * (x - mu))
    return x


def premium_sd(n_games: int) -> float:
    """Optimal-stopping value over the mean, in SD units, for n i.i.d. standard normal games."""
    v, w = normal_grid(0.0, 1.0)
    return stopping_values([(v, w, 0.0)] * n_games)[0]
