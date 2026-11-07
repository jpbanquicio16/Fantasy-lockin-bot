"""Shared preparation for fitting and projecting: game histories, cohorts, EW weights.

Point-in-time contract: every function that looks at history takes the rows already
filtered to what was known at the cutoff (``tables.latest_revision(box, as_of)``) and
never reads anything later.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

from lockin_core.config import ET, Paths
from lockin_core.data import tables as T

RATE_STATS = ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m")
MINUTE_BANDS = (0.0, 12.0, 20.0, 28.0)  # band b: minutes per game >= MINUTE_BANDS[b]
AVAIL_BANDS = (0.0, 10.0, 20.0, 30.0)
GROUPS = ("G", "F", "C")


def position_group(positions: list[str] | None) -> str:
    pos = set(positions or [])
    if "C" in pos:
        return "C"
    if pos & {"PG", "SG"} and "PF" not in pos:
        return "G"
    return "F"


def band(minutes: float, edges: tuple[float, ...] = MINUTE_BANDS) -> int:
    return int(np.searchsorted(np.asarray(edges), minutes, side="right") - 1)


def history(paths: Paths, seasons: list[int], as_of: datetime | None = None) -> pl.DataFrame:
    """Latest known revision of every listed player-game (played or dnp), oldest first."""
    frames = []
    for s in seasons:
        b = T.read(paths, "box", s)
        if b is not None:
            frames.append(T.latest_revision(b, as_of))
    if not frames:
        raise FileNotFoundError("no box tables; run `lockin ingest`")
    df = pl.concat(frames, how="vertical_relaxed")
    if as_of is not None:  # a game counts only once it had finished by the cutoff date
        df = df.filter(pl.col("date") < as_of.astimezone(ET).date())
    return df.sort("player_id", "date", "game_id")


def with_groups(df: pl.DataFrame, players: pl.DataFrame | None) -> pl.DataFrame:
    if players is None:
        return df.with_columns(pl.lit("F").alias("group"))
    g = pl.DataFrame({"player_id": players["player_id"],
                      "group": [position_group(p) for p in players["positions"].to_list()]})
    return df.join(g, on="player_id", how="left").with_columns(pl.col("group").fill_null("F"))


def ew_minutes_asof(df: pl.DataFrame, half_life: float) -> pl.DataFrame:
    """``ew_min``: EW mean minutes over a player's earlier *played* games (excluding the
    current row), i.e. what was known before the game. Null before his first game."""
    alpha = 1 - 0.5 ** (1 / half_life)
    out = []
    for (_pid,), g in df.group_by("player_id", maintain_order=True):
        mins = g["min"].to_numpy()
        played = (g["played"] == "played").to_numpy()
        ew = np.full(len(g), np.nan)
        cur, have = 0.0, False
        for i in range(len(g)):
            ew[i] = cur if have else np.nan
            if played[i]:
                cur = mins[i] if not have else (1 - alpha) * cur + alpha * mins[i]
                have = True
        out.append(g.with_columns(pl.Series("ew_min", ew)))
    return pl.concat(out) if out else df.with_columns(pl.lit(None, dtype=pl.Float64).alias("ew_min"))


def recency_weights(n: int, half_life: float, prev_season: np.ndarray, prev_weight: float) -> np.ndarray:
    """Weight of each of ``n`` games ordered oldest -> newest."""
    age = np.arange(n - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / half_life)
    return w * np.where(prev_season, prev_weight, 1.0)


def et_today(ts: datetime) -> date:
    return ts.astimezone(ET).date()
