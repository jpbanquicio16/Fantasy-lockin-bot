import json

import numpy as np
import polars as pl
import pytest

from lockin_core.scoring.engine import score_frame, score_line, score_points, score_units
from lockin_core.scoring.settings import SLEEPER_DEFAULT, OpenSettingError, ScoringSettings

STACK = ScoringSettings(threshold_bonus_mode="stack", tech_definition="sleeper_tf")
EXCL = ScoringSettings(threshold_bonus_mode="exclusive", tech_definition="sleeper_tf")


def line(**kw):
    base = {c: 0 for c in ("pts", "reb", "ast", "stl", "blk", "tov", "fg3m", "tf", "ff")}
    return base | kw


def test_golden_pts_std_reproduced_exactly(fixtures_dir):
    """Release gate: every golden game matches Sleeper's own (foul-free) score."""
    data = json.loads((fixtures_dir / "sleeper_std_2025.json").read_text())
    s = ScoringSettings.from_sleeper(data["settings"], "stack", "sleeper_tf").without_fouls()
    bad = [(g["name"], score_points(g["box"], s), g["pts_std"]) for g in data["games"]
           if score_points(g["box"], s) != g["pts_std"]]
    assert not bad, bad[:5]
    assert len(data["games"]) >= 30


def test_fifty_point_games_settle_stacking(fixtures_dir):
    data = json.loads((fixtures_dir / "sleeper_std_2025.json").read_text())
    fifties = [g for g in data["games"] if g["box"]["pts"] >= 50]
    assert fifties
    for g in fifties:
        assert score_points(g["box"], STACK.without_fouls()) == g["pts_std"]
        assert score_points(g["box"], STACK) - score_points(g["box"], EXCL) == 2.0


def test_league_deducts_sleeper_tf_and_ff(fixtures_dir):
    data = json.loads((fixtures_dir / "league_foul_games.json").read_text())
    s = ScoringSettings.from_sleeper(data["settings"], "stack", "sleeper_tf")
    assert len(data["games"]) >= 10
    for g in data["games"]:
        assert score_points(g["box"], s) == g["league_points"]
        assert score_points(g["box"], s.without_fouls()) == g["pts_std"]
        assert g["league_points"] < g["pts_std"]


@pytest.mark.parametrize("cat", ["pts", "reb", "ast", "stl", "blk"])
def test_double_double_boundary(cat):
    other = "reb" if cat != "reb" else "ast"
    nine = score_points(line(**{cat: 9, other: 10}), STACK)
    ten = score_points(line(**{cat: 10, other: 10}), STACK)
    w = SLEEPER_DEFAULT[cat]
    assert ten - nine == w + 1  # one more of the stat, plus the double-double


@pytest.mark.parametrize("pts,stack,excl", [(39, 19.5, 19.5), (40, 22, 22), (49, 26.5, 26.5),
                                            (50, 29, 27), (52, 30, 28)])
def test_point_thresholds_both_modes(pts, stack, excl):
    assert score_points(line(pts=pts), STACK) == stack
    assert score_points(line(pts=pts), EXCL) == excl


def test_triple_double_on_steals_and_blocks():
    assert score_points(line(pts=10, stl=10, blk=10), STACK) == 5 + 20 + 20 + 3


def test_quadruple_double_still_three():
    assert score_points(line(pts=10, reb=10, ast=10, blk=10), STACK) == 5 + 10 + 10 + 20 + 3


def test_negative_total_allowed():
    assert score_points(line(tov=5, tf=1), STACK) == -7


def test_unknown_fouls_give_interval_not_zero():
    r = score_line(line(pts=20) | {"tf": None, "ff": None}, STACK)
    assert r.units is None and "fouls_unknown" in r.flags
    assert r.high == 20 and r.low == 20 - 2 * 2 * 2  # two unknown fouls at -2 FP, in half-points
    df = pl.DataFrame({"pts": [20, 20], "tf": [None, 0], "ff": [None, 0]},
                      schema={"pts": pl.Int32, "tf": pl.Int32, "ff": pl.Int32})
    out = score_frame(df, STACK)
    assert out["fp_units"].to_list() == [None, 20]


def test_pending_game_is_not_scored():
    assert score_line(line(pts=30), STACK, final=False).status == "pending"


def test_open_settings_refuse_exact_scoring():
    with pytest.raises(OpenSettingError):
        score_line(line(pts=55), ScoringSettings())


def test_vectorised_and_scalar_paths_agree():
    rng = np.random.default_rng(0)
    cols = {c: rng.integers(0, 15, 500) for c in ("pts", "reb", "ast", "stl", "blk", "tov", "tf", "ff")}
    cols["pts"] = rng.integers(0, 60, 500)
    cols["fg3m"] = np.minimum(rng.integers(0, 8, 500), cols["pts"] // 3)
    units = score_units(cols, STACK)
    for i in range(0, 500, 37):
        box = {k: int(v[i]) for k, v in cols.items()}
        assert score_line(box, STACK).units == units[i]
