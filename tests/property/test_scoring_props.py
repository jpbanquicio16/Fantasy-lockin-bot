from hypothesis import given
from hypothesis import strategies as st

from lockin_core.scoring.engine import score_line
from lockin_core.scoring.settings import ScoringSettings

S = ScoringSettings(threshold_bonus_mode="stack", tech_definition="sleeper_tf")
stat = st.integers(0, 30)


@given(stat, stat, stat, stat, stat, stat)
def test_one_more_rebound_never_lowers(pts, reb, ast, stl, blk, tov):
    box = dict(pts=pts, reb=reb, ast=ast, stl=stl, blk=blk, tov=tov, fg3m=0, tf=0, ff=0)
    assert score_line(box | {"reb": reb + 1}, S).units >= score_line(box, S).units + 2


@given(stat, stat, stat, stat, stat, stat)
def test_one_more_turnover_costs_exactly_one(pts, reb, ast, stl, blk, tov):
    box = dict(pts=pts, reb=reb, ast=ast, stl=stl, blk=blk, tov=tov, fg3m=0, tf=0, ff=0)
    assert score_line(box | {"tov": tov + 1}, S).units == score_line(box, S).units - 2


@given(st.integers(0, 70), st.integers(0, 20))
def test_a_three_is_worth_two(two_pt_pts, threes):
    box = dict(pts=two_pt_pts, reb=0, ast=0, stl=0, blk=0, tov=0, fg3m=0, tf=0, ff=0)
    more = box | {"pts": two_pt_pts + 3, "fg3m": 1}
    gain = score_line(more, S).units - score_line(box, S).units
    assert gain >= 4  # 1.5 via points + 0.5 via the three, plus any bonus crossed
