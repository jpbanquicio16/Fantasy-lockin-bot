"""Reconcile the scoring engine against Sleeper's own numbers (the engine's release gate).

Two independent checks:

1. **Every game, against ``pts_std``.** Sleeper's stats feed carries its own score of
   each game under its standard settings, which match this league's weights except that
   ``pts_std`` ignores technical and flagrant fouls. The engine without fouls must agree
   to the half-point on every game; fifty-point games settle ``threshold_bonus_mode``.
2. **Foul games, against the league.** Matchup ``players_points`` is the league's score
   of each player's current game. A foul game is identified when that number equals
   exactly one of his games this week; it confirms the deduction when it equals the
   score with fouls and refutes it when it equals the score without.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import polars as pl

from lockin_core.scoring.engine import score_frame
from lockin_core.scoring.settings import ScoringSettings


@dataclass
class Reconciliation:
    games: int = 0
    mismatches: list[dict] = field(default_factory=list)
    fifty_point_games: int = 0
    stack_matches: int = 0
    exclusive_matches: int = 0
    foul_confirm: list[dict] = field(default_factory=list)
    foul_refute: list[dict] = field(default_factory=list)
    league_checked: int = 0
    league_unmatched: list[dict] = field(default_factory=list)
    current_season: int | None = None

    @property
    def bonus_mode(self) -> str | None:
        if self.stack_matches and not self.exclusive_matches:
            return "stack"
        if self.exclusive_matches and not self.stack_matches:
            return "exclusive"
        return None

    @property
    def tech_definition(self) -> str | None:
        return "sleeper_tf" if self.foul_confirm and not self.foul_refute else None

    @property
    def current_mismatches(self) -> list[dict]:
        return [m for m in self.mismatches if m["season"] == self.current_season]

    @property
    def passed(self) -> bool:
        """Exact on every current-season game; historical feed rows whose own ``pts_std``
        disagrees with their stat line (1 in 54,597 when first run) are logged, not fatal."""
        return (self.games >= 30 and not self.current_mismatches
                and len(self.mismatches) <= self.games * 1e-4)

    def markdown(self) -> str:
        lines = [
            "# Scoring reconciliation", "",
            f"* Games compared with Sleeper's `pts_std` (fouls excluded on both sides): **{self.games}**; "
            f"mismatches: **{len(self.mismatches)}** ({len(self.current_mismatches)} this season) -> "
            f"{'PASS' if self.passed else 'FAIL'}",
            f"* Fifty-point games: {self.fifty_point_games}; matched by stacking bonuses: "
            f"{self.stack_matches}; by exclusive bonuses: {self.exclusive_matches} -> "
            f"`threshold_bonus_mode: {self.bonus_mode}`",
            f"* Foul games identified in league matchup points: {len(self.foul_confirm) + len(self.foul_refute)}; "
            f"deduction confirmed {len(self.foul_confirm)}, refuted {len(self.foul_refute)} -> "
            f"`tech_definition: {self.tech_definition}`",
            f"* League matchup points checked against the engine: {self.league_checked}; "
            f"not equal to any of the player's games that week: {len(self.league_unmatched)}",
        ]
        if self.mismatches:
            lines += ["", "## pts_std mismatches (first 20)", ""]
            lines += [f"* {m}" for m in self.mismatches[:20]]
        if self.league_unmatched:
            lines += ["", "## League points with no matching game (first 20)",
                      "Usually a stat correction published after the matchup was scored.", ""]
            lines += [f"* {m}" for m in self.league_unmatched[:20]]
        return "\n".join(lines) + "\n"


def against_pts_std(box: pl.DataFrame, settings: ScoringSettings, rec: Reconciliation,
                    season: int | None = None) -> None:
    played = box.filter((pl.col("played") == "played") & pl.col("pts_std").is_not_null())
    base = settings.without_fouls()
    res = base.units_per_point
    stack = score_frame(played, base.model_copy(update={"threshold_bonus_mode": "stack"}), "s")
    excl = score_frame(played, base.model_copy(update={"threshold_bonus_mode": "exclusive"}), "e")
    df = played.with_columns(stack["s_units"], excl["e_units"],
                             (pl.col("pts_std") * res).round().cast(pl.Int64).alias("std_units"))
    rec.games += df.height
    fifty = df.filter(pl.col("pts") >= 50)
    rec.fifty_point_games += fifty.height
    rec.stack_matches += fifty.filter(pl.col("s_units") == pl.col("std_units")).height
    rec.exclusive_matches += fifty.filter(pl.col("e_units") == pl.col("std_units")).height
    mode = settings.threshold_bonus_mode or "stack"
    col = "s_units" if mode == "stack" else "e_units"
    bad = df.filter(pl.col(col) != pl.col("std_units"))
    for r in bad.head(50).iter_rows(named=True):
        rec.mismatches.append({"season": season} | {k: r[k] for k in (
            "date", "player_id", "pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "pts_std")}
            | {"engine": r[col] / res})


def against_league(box: pl.DataFrame, settings: ScoringSettings, week_points: dict[int, dict[str, float]],
                   rec: Reconciliation) -> None:
    """``week_points``: week -> {player_id: league players_points} from matchups."""
    played = box.filter(pl.col("played") == "played")
    with_f = score_frame(played, settings, "f")
    no_f = score_frame(played, settings.without_fouls(), "n")
    df = played.select("week", "date", "player_id", "tf", "ff").with_columns(
        with_f["f"], no_f["n"])
    by = {(w, p): g for (w, p), g in df.group_by("week", "player_id")}
    for week, points in week_points.items():
        for pid, shown in points.items():
            g = by.get((week, pid))
            if g is None:
                continue
            if shown == 0 and not (g["f"] == 0).any():
                continue  # no game locked yet, or a forced DNP final game
            rec.league_checked += 1
            scores = Counter(g["f"].to_list())
            if shown not in scores:
                rec.league_unmatched.append({"week": week, "player_id": pid, "league": shown,
                                             "engine_games": sorted(scores)})
            for r in g.filter((pl.col("tf") + pl.col("ff")) > 0).iter_rows(named=True):
                others = Counter(scores)
                others[r["f"]] -= 1
                if shown == r["f"] and others[r["f"]] <= 0 and shown != r["n"]:
                    rec.foul_confirm.append({"week": week, "player_id": pid, "date": r["date"],
                                             "tf": r["tf"], "ff": r["ff"], "league": shown})
                elif shown == r["n"] and shown not in scores:
                    rec.foul_refute.append({"week": week, "player_id": pid, "date": r["date"],
                                            "league": shown, "with_fouls": r["f"]})
