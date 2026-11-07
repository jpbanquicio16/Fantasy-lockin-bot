"""Continuation tables for the base policy (design section 13), frozen per recommendation.

For player p and his remaining game k (q_out: ruled out before tip; q_scr: late scratch):

    W_in[p,k] = (1 - q_scr) E[max(X, W[p,k+1])] + q_scr W[p,k+1]
    W[p,k]    = (1 - q_out) W_in[p,k] + q_out W[p,k+1]

with W[p,n] = 0 and E[X] in place of the maximum for his team's final game (that score counts
whatever it is). W is his value before game k's status is known; W_in once he is known to be
available for it. Tables depend only on projections known at the root, never on a scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuationTables:
    W: np.ndarray  # [P, K+1]
    W_in: np.ndarray  # [P, K+1]
    mean: np.ndarray  # [P, K] E[X | plays]
    tonight: np.ndarray  # [P, K] (1 - q_scr) E[X]: seating order
    flex: np.ndarray  # [S] roster players eligible for each slot


def build_tables(values: np.ndarray, weights: np.ndarray, q_out: np.ndarray, q_scr: np.ndarray,
                 n_games: np.ndarray, final: np.ndarray, flex: np.ndarray) -> ContinuationTables:
    """``values``/``weights`` [P, K, Q]: the score distribution given he plays; ``final`` [P, K]
    marks his team's final game of the week (normally game n-1)."""
    P, K, _ = values.shape
    W = np.zeros((P, K + 1))
    W_in = np.zeros((P, K + 1))
    mean = (values * weights).sum(-1)
    for p in range(P):
        for k in range(int(n_games[p]) - 1, -1, -1):
            cont = W[p, k + 1]
            if final[p, k]:
                ex = mean[p, k]
            else:
                ex = float(np.sum(weights[p, k] * np.maximum(values[p, k], cont)))
            W_in[p, k] = (1 - q_scr[p, k]) * ex + q_scr[p, k] * cont
            W[p, k] = (1 - q_out[p, k]) * W_in[p, k] + q_out[p, k] * cont
    tonight = (1 - q_scr) * mean
    return ContinuationTables(W, W_in, mean, tonight, flex.astype(float))


def quantile_grid(samples: np.ndarray, q: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint quantiles of the played-game samples (values, equal weights)."""
    if samples.size == 0:
        return np.zeros(q), np.full(q, 1.0 / q)
    u = (np.arange(q) + 0.5) / q
    return np.quantile(samples.astype(float), u), np.full(q, 1.0 / q)
