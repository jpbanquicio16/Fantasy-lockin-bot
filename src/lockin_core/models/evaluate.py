"""Pure short-horizon forecast diagnostics, scored only after outcomes are observed."""
from __future__ import annotations

import numpy as np


def availability(probabilities, outcomes) -> dict:
    p, y = np.asarray(probabilities, float), np.asarray(outcomes, float)
    if p.ndim != 1 or p.shape != y.shape or not len(p):
        raise ValueError("Supply equally sized, nonempty probability and outcome vectors")
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Probabilities must be in [0, 1] and outcomes binary")
    q = np.clip(p, 1e-12, 1-1e-12)
    bins = np.minimum((p * 10).astype(int), 9)
    return {"n": len(p), "brier": float(np.mean((p-y)**2)),
            "log_loss": float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q))),
            "reliability": [{"decile": b, "n": int((bins == b).sum()),
                             "predicted": float(p[bins == b].mean()), "observed": float(y[bins == b].mean())}
                            for b in range(10) if (bins == b).any()]}


def distributions(draws, outcomes) -> dict:
    x, y = np.asarray(draws, float), np.asarray(outcomes, float)
    if x.ndim != 2 or len(x) != len(y) or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Supply finite [observations, scenarios] draws and matching outcomes")
    ordered = np.sort(x, axis=1)
    n = x.shape[1]
    # E|X-y| - 1/2 E|X-X'| without an O(n^2) pairwise array.
    spread = np.sum((2*np.arange(1, n+1)-n-1)*ordered, axis=1) / n**2
    quantiles = np.quantile(x, [0.1, 0.5, 0.9], axis=1)
    pinball = {}
    for q, pred in zip((0.1, 0.5, 0.9), quantiles):
        err = y-pred
        pinball[str(q)] = float(np.maximum(q*err, (q-1)*err).mean())
    return {"n": len(y), "crps": float((np.abs(x-y[:, None]).mean(1)-spread).mean()),
            "mae_median": float(np.abs(quantiles[1]-y).mean()), "pinball": pinball,
            "coverage_80": float(((y >= quantiles[0]) & (y <= quantiles[2])).mean())}
