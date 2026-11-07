"""Same-day availability: P(play) for one player-game, split into what resolves before tip
and what does not (design section 9).

* **Listed on the latest official report for that game**: the status prior (Out 0, Doubtful
  0.10, Questionable 0.55, Probable 0.93, Available 0.99; all Assumption) until the archive
  calibrates the cell (status x hours to tip, 200 player-games per cell).
* **Team reported and he is not listed**: the healthy model (below), capped at 0.99.
* **Game beyond the report's horizon**: Sleeper's tag. Out/IR/suspended players return after
  a sampled number of team games (the role scenario, drawn in the sampler); day-to-day tags
  decay toward healthy with lead time; healthy players use their own calibrated play rate,
  the fitted decay with games ahead and the back-to-back factor, with floors for a rotation
  player who played his last game.

The simulator needs two probabilities: ``q_out``, ruled out before tip (revealed at the tip,
so a manager can bench him), and ``q_scr``, a late scratch after the lineup is set (a started
player who does not play: no entry, or a forced zero in his team's final game).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from lockin_core.config import ModelConfig
from lockin_core.models.fit import K_AHEAD, Params

REPORT_KIND = {"out": "out", "doubtful": "doubtful", "questionable": "questionable",
               "probable": "probable", "available": "available"}
LEAD_BUCKETS = ((0, 2, "0-2h"), (2, 8, "2-8h"), (8, 24, "8-24h"), (24, 1e9, ">24h"))
DTD_DECAY = (0.55, 0.70, 0.85)  # Sleeper GTD/DTD tag: next game, 1-3 days, 4+ days (Assumption)
MIN_CELL = 200


def lead_bucket(hours: float) -> str:
    for lo, hi, name in LEAD_BUCKETS:
        if lo <= hours < hi:
            return name
    return ">24h"


@dataclass(frozen=True)
class GameAvailability:
    p_play: float
    q_out: float
    q_scr: float
    source: str


def split(p: float, q_scr: float, source: str) -> GameAvailability:
    p = float(np.clip(p, 0.0, 1.0))
    q_scr = float(np.clip(min(q_scr, 1.0 - p), 0.0, 1.0))
    q_out = 0.0 if q_scr >= 1.0 else float(np.clip(1.0 - p / (1.0 - q_scr), 0.0, 1.0))
    return GameAvailability(p, q_out, q_scr, source)


def healthy_play_prob(proj, k: int, lead_days: int, b2b: bool, params: Params, cfg: ModelConfig) -> float:
    """P(plays his team's k-th upcoming game) for a player with no injury tag."""
    if proj.p_override is not None and k == 1:
        return proj.p_override
    table = params.avail_played[proj.avail_band]
    p = params.calibrate_play(proj.play_rate) * table[min(k, K_AHEAD) - 1] / table[0]
    a = cfg.availability
    if proj.last_played and proj.avail_band >= 2:
        if lead_days <= 0:
            p = max(p, a.healthy_floor_same_day)
        elif lead_days == 1:
            p = max(p, a.healthy_floor_next_day)
    if b2b and proj.avail_band >= 2:
        p *= params.b2b_factor
    return float(np.clip(p, 0.0, 1.0))


def game_availability(proj, k: int, lead_hours: float, b2b: bool, report: str | None,
                      team_reported: bool, params: Params, cfg: ModelConfig,
                      calibration: dict[tuple[str, str], float] | None = None) -> GameAvailability:
    a = cfg.availability
    lead_days = int(max(lead_hours, 0) // 24)
    if report is not None:
        kind = REPORT_KIND.get(report.lower(), "available")
        cal = (calibration or {}).get((kind, lead_bucket(lead_hours)))
        p = cal if cal is not None else a.status_prior.get(kind, 0.99)
        if kind in ("available", "probable"):  # healthy, but the play-rate model knows a deep-bench role
            p = min(p, healthy_play_prob(proj, k, lead_days, b2b, params, cfg))
        late = a.late_scratch + (a.questionable_late_share * (1 - p) if kind == "questionable" else 0.0)
        return split(p, late if p > 0 else 0.0, f"report:{kind}" + (" (calibrated)" if cal is not None else ""))
    kind = proj.status_kind
    healthy = healthy_play_prob(proj, k, lead_days, b2b, params, cfg)
    if team_reported:
        return split(min(healthy, a.status_prior.get("available", 0.99)), a.late_scratch, "not listed")
    if kind == "dtd":
        prior = DTD_DECAY[0] if lead_days <= 0 else DTD_DECAY[1] if lead_days <= 3 else DTD_DECAY[2]
        p = min(healthy, prior)
        return split(p, a.late_scratch + a.questionable_late_share * (1 - p), "sleeper:dtd")
    if kind == "doubtful":
        return split(min(healthy, a.status_prior["doubtful"]), a.late_scratch, "sleeper:doubtful")
    if kind == "probable":
        return split(min(healthy, a.status_prior["probable"]), a.late_scratch, "sleeper:probable")
    # healthy, or out/ir/sus (whose return is sampled; this is P(play) once back)
    return split(healthy, a.late_scratch, "healthy" if kind == "healthy" else f"sleeper:{kind} (if back)")


# --- calibration from the report archive (design section 9, P5) ----------------------
def fit_status_calibration(reports: pl.DataFrame, box: pl.DataFrame, games: pl.DataFrame,
                           min_cell: int = MIN_CELL) -> tuple[dict[tuple[str, str], float], pl.DataFrame]:
    """Observed P(played) by status x hours-to-tip cell from archived reports joined to box
    scores. Cells with at least ``min_cell`` player-games replace the prior; the table of all
    cells is returned for the report."""
    if reports is None or reports.is_empty():
        return {}, pl.DataFrame()
    g = games.select("game_id", "date", "home", "away", "tip_at")
    teams = pl.concat([g.select("game_id", "date", "tip_at", pl.col("home").alias("team")),
                       g.select("game_id", "date", "tip_at", pl.col("away").alias("team"))])
    r = (reports.filter(~pl.col("not_submitted") & pl.col("player_id").is_not_null())
         .join(teams, left_on=["team", "game_date"], right_on=["team", "date"], how="inner")
         .with_columns(((pl.col("tip_at") - pl.col("published_at")).dt.total_minutes() / 60).alias("lead_h"))
         .filter(pl.col("lead_h") >= 0))
    b = box.select("game_id", "player_id", (pl.col("played") == "played").alias("played"))
    j = r.join(b, on=["game_id", "player_id"], how="inner")
    if j.is_empty():
        return {}, pl.DataFrame()
    j = j.with_columns(pl.col("status").str.to_lowercase().alias("kind"),
                       pl.col("lead_h").map_elements(lead_bucket, return_dtype=pl.Utf8).alias("bucket"))
    # one observation per player-game and bucket: his latest report in that bucket
    j = j.sort("published_at").unique(["game_id", "player_id", "bucket"], keep="last")
    cells = j.group_by("kind", "bucket").agg(pl.len().alias("n"), pl.col("played").mean().alias("p_played"))
    cal = {(k, bk): float(p) for k, bk, n, p in cells.iter_rows() if n >= min_cell}
    return cal, cells.sort("kind", "bucket")
