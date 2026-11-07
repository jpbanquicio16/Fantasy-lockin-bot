"""Maximum-weight assignment of players to open slots, for many scenarios at once.

Two variants (design section 13):

* **fast** - greedy, vectorised across scenarios: players in decreasing value, each into the
  eligible free slot that the fewest roster players can fill (ties: lowest slot index). Used in
  the production simulator.
* **exact** - SciPy ``linear_sum_assignment`` per scenario row, memoised on the information
  state. Used in tests and toy validation.

Players with value <= 0 are never seated (the bench is worth 0).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

NEG = -1e18


def greedy_total(V: np.ndarray, elig: np.ndarray, free: np.ndarray, flex: np.ndarray) -> np.ndarray:
    """``V`` [n, P] (NEG = not available), ``elig`` [P, S], ``free`` [n, S] -> total [n]."""
    n, P = V.shape
    if n == 0:
        return np.zeros(0)
    free = free.copy()
    order = np.argsort(-V, axis=1, kind="stable")
    total = np.zeros(n)
    rows = np.arange(n)
    big = flex.max() + 1.0
    cost_base = flex[None, :]
    for r in range(P):
        i = order[:, r]
        v = V[rows, i]
        pos = v > 0
        if not pos.any():
            break
        cand = elig[i] & free
        cost = np.where(cand, cost_base, big)
        j = cost.argmin(1)
        ok = pos & cand[rows, j]
        total += np.where(ok, v, 0.0)
        free[rows[ok], j[ok]] = False
    return total


class ExactAssigner:
    def __init__(self, elig: np.ndarray):
        self.elig = elig
        self.memo: dict[tuple, float] = {}

    def total(self, V: np.ndarray, free: np.ndarray) -> np.ndarray:
        out = np.zeros(V.shape[0])
        for r in range(V.shape[0]):
            v = V[r]
            key = (np.round(v, 6).tobytes(), free[r].tobytes())
            hit = self.memo.get(key)
            if hit is None:
                hit = self._solve(v, free[r])
                self.memo[key] = hit
            out[r] = hit
        return out

    def _solve(self, v: np.ndarray, free: np.ndarray) -> float:
        players = np.flatnonzero(v > 0)
        slots = np.flatnonzero(free)
        if len(players) == 0 or len(slots) == 0:
            return 0.0
        Wm = np.where(self.elig[np.ix_(players, slots)], v[players][:, None], 0.0)
        r, c = linear_sum_assignment(Wm, maximize=True)
        return float(Wm[r, c].sum())


def assignment_total(V: np.ndarray, elig: np.ndarray, free: np.ndarray, flex: np.ndarray,
                     exact: ExactAssigner | None = None) -> np.ndarray:
    return exact.total(V, free) if exact is not None else greedy_total(V, elig, free, flex)


def best_lineups(weights: np.ndarray, k: int = 5) -> list[tuple[float, list[tuple[int, int]]]]:
    """Up to ``k`` good assignments of a [P, S] weight matrix (NEG = forbidden): the optimum,
    then the best with each of its pairs forbidden in turn (a cheap Murty-style list)."""
    def solve(Wm):
        Wpos = np.where(Wm > 0, Wm, 0.0)
        r, c = linear_sum_assignment(Wpos, maximize=True)
        pairs = [(int(a), int(b)) for a, b in zip(r, c) if Wm[a, b] > 0]
        return float(sum(Wm[a, b] for a, b in pairs)), pairs

    if weights.size == 0:
        return []
    best = solve(weights)
    out = [best]
    seen = {tuple(sorted(best[1]))}
    for a, b in best[1]:
        Wm = weights.copy()
        Wm[a, b] = NEG
        alt = solve(Wm)
        key = tuple(sorted(alt[1]))
        if key not in seen:
            seen.add(key)
            out.append(alt)
    out.sort(key=lambda x: -x[0])
    return out[:k]
