"""Deterministic Sleeper fantasy scoring: one game's box score -> fantasy points.

Points are integers in the settings' smallest unit (half-points for this league), so
sums never drift and equality tests are exact. Three entry points share one rule set:

* ``score_units`` - NumPy, for the simulator's millions of sampled games;
* ``score_frame`` - Polars, for historical game logs;
* ``score_line`` - one game, with status and an interval when fouls are unknown.

A game that is not final is ``pending`` and is never scored. A game whose foul counts
are missing is scored as an interval [FP - 2 k_max, FP] and flagged, never as if the
player had committed no fouls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import polars as pl

from lockin_core.scoring.settings import (
    BONUS_KEYS,
    BOX_COLUMNS,
    DD_CATEGORIES,
    FOUL_COLUMNS,
    MISS_KEYS,
    STAT_KEYS,
    ScoringSettings,
)

ArrayMap = Mapping[str, np.ndarray]
MAX_UNKNOWN_FOULS = 2  # k_max: technical + flagrant fouls assumed possible when unrecorded


def _get(cols: ArrayMap, name: str, n: int) -> np.ndarray:
    v = cols.get(name)
    if v is None:
        return np.zeros(n, dtype=np.int64)
    return np.asarray(v).astype(np.int64, copy=False)


def _length(cols: ArrayMap) -> int:
    for v in cols.values():
        return int(np.size(v))
    return 0


def bonus_indicators(cols: ArrayMap, stacking: bool) -> dict[str, np.ndarray]:
    n = _length(cols)
    pts = _get(cols, "pts", n)
    cats = sum((_get(cols, c, n) >= 10).astype(np.int64) for c in DD_CATEGORIES)
    b40 = pts >= 40 if stacking else (pts >= 40) & (pts < 50)
    return {
        "dd": (cats >= 2).astype(np.int64),
        "td": (cats >= 3).astype(np.int64),
        "bonus_pt_40p": b40.astype(np.int64),
        "bonus_pt_50p": (pts >= 50).astype(np.int64),
        "bonus_ast_15p": (_get(cols, "ast", n) >= 15).astype(np.int64),
        "bonus_reb_20p": (_get(cols, "reb", n) >= 20).astype(np.int64),
    }


def score_units(cols: ArrayMap, settings: ScoringSettings, stacking: bool | None = None) -> np.ndarray:
    """Fantasy points in integer units of 1/settings.units_per_point."""
    stacking = settings.stacking() if stacking is None else stacking
    n = _length(cols)
    total = np.zeros(n, dtype=np.int64)
    bonuses = None
    for key, w in settings.unit_weights().items():
        if key in STAT_KEYS:
            total += w * _get(cols, STAT_KEYS[key], n)
        elif key in MISS_KEYS:
            att, made = MISS_KEYS[key]
            total += w * (_get(cols, att, n) - _get(cols, made, n))
        elif key in BONUS_KEYS:
            if bonuses is None:
                bonuses = bonus_indicators(cols, stacking)
            total += w * bonuses[key]
        else:  # combination stat such as pts_reb_ast
            total += w * sum(_get(cols, part, n) for part in key.split("_"))
    return total


@dataclass(frozen=True)
class ScoreResult:
    status: str  # final | pending
    units: int | None = None  # exact score, None if pending or fouls unknown
    low: int | None = None
    high: int | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def pending(cls) -> "ScoreResult":
        return cls("pending", flags=("pending",))


def score_line(box: Mapping[str, float | None], settings: ScoringSettings,
               final: bool = True) -> ScoreResult:
    if not final:
        return ScoreResult.pending()
    settings.require_exact()
    cols = {k: np.array([0 if v is None else v]) for k, v in box.items() if k in BOX_COLUMNS}
    base = int(score_units(cols, settings)[0])
    unknown = settings.scores_fouls and any(box.get(c) is None for c in FOUL_COLUMNS)
    if not unknown:
        return ScoreResult("final", base, base, base)
    worst = min(settings.unit_weights().get(c, 0) for c in FOUL_COLUMNS)
    return ScoreResult("final", None, base + worst * MAX_UNKNOWN_FOULS, base, ("fouls_unknown",))


def score_points(box: Mapping[str, float | None], settings: ScoringSettings) -> float:
    r = score_line(box, settings)
    if r.units is None:
        raise ValueError(f"no exact score: {r.flags}")
    return r.units / settings.units_per_point


def score_frame(df: pl.DataFrame, settings: ScoringSettings, out: str = "fp") -> pl.DataFrame:
    """Add ``{out}_units`` (null when fouls are unknown), ``{out}_low``/``{out}_high``
    and ``{out}`` (float points). Rows must already be final, played games."""
    settings.require_exact()
    cols = {c: df[c].fill_null(0).to_numpy() for c in BOX_COLUMNS if c in df.columns}
    units = score_units(cols, settings)
    unknown = np.zeros(df.height, dtype=bool)
    if settings.scores_fouls:
        for c in FOUL_COLUMNS:
            unknown |= df[c].is_null().to_numpy() if c in df.columns else True
    worst = min(settings.unit_weights().get(c, 0) for c in FOUL_COLUMNS)
    low = units + np.where(unknown, worst * MAX_UNKNOWN_FOULS, 0)
    res = settings.units_per_point
    exact = pl.Series(f"{out}_units", units, dtype=pl.Int64)
    return df.with_columns(
        pl.when(pl.Series(unknown)).then(None).otherwise(exact).alias(f"{out}_units"),
        pl.Series(f"{out}_low", low, dtype=pl.Int64),
        pl.Series(f"{out}_high", units, dtype=pl.Int64),
        pl.Series(out, units / res, dtype=pl.Float64),
    )
