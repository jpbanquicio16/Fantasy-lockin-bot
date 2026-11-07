from datetime import datetime, timedelta, timezone
import numpy as np

from lockin_core.config import LeagueConfig, ModelConfig, SimConfig
from lockin_core.models.fit import Params
from lockin_core.models.project import PlayerProjection
from lockin_core.policy.exact_dp import ToyGame, ToyPlayer, ToyWeek, toy_problem
from lockin_core.schedule.calendar import Game, WeekCalendar
from lockin_core.scoring.settings import ScoringSettings
from lockin_core.sim.build import WeekModel
from lockin_core.sim.problem import RootEntry
from lockin_core.state.build import Context
from lockin_core.state.league import LeagueRules
from lockin_core.state.models import SideState, PlayerInfo, EligibleScore, ValidatedState


def small_model(now=None):
    now = now or datetime(2026, 11, 4, 12, tzinfo=timezone.utc)
    cfg = LeagueConfig(league_id="test", my_roster_id=1)
    scfg = SimConfig(planning_scenarios=50, scenarios={"start": 50, "max": 50},
                     candidates={"screen_scenarios": 50}, close_call={"model_draws": 2, "scenarios_per_draw": 20},
                     wait={"min_branch_scenarios": 2})
    scoring = ScoringSettings(threshold_bonus_mode="stack", tech_definition="sleeper_tf")
    rules = LeagueRules(league_id="test", season=2026, total_rosters=12, starting_slots=["G"],
                       bench_slots=1, reserve_slots=0, scoring=scoring)
    wk = ToyWeek(("G",), (ToyPlayer(("PG",), (ToyGame(0, .1, .01, (30, 70), (.5, .5)),)),), days=1)
    prob = toy_problem(wk, tip_minute=180)
    game = Game("g0_0", 2026, 3, now.date(), "AAA", "BBB", now+timedelta(hours=3), "espn", "scheduled", now+timedelta(hours=6))
    old = Game("old", 2026, 3, now.date()-timedelta(days=1), "AAA", "CCC", now-timedelta(days=1), "espn", "final", now-timedelta(hours=20))
    cal = WeekCalendar(2026, 3, [old, game], now-timedelta(days=2), now+timedelta(days=3))
    entry = EligibleScore("3:1:0:old", "0", "old", 60, 0, False, now+timedelta(hours=2, minutes=50), "observed")
    me = SideState(1, "owner", ("G",), {0: "0"}, {}, (entry,), ("0",), (), frozenset())
    info = PlayerInfo(player_id="0", name="P0", team="AAA", positions=["PG"])
    state = ValidatedState(now, 2026, 3, "test", me, None, {"0": info}, 2, "test_snapshot")
    ctx = Context(cfg, rules, cal, cfg.rules, report_published_at=now, player_team={"0": "AAA"})
    params = Params(version="test", fitted_at=now-timedelta(days=7), seasons=[2025], rate_prior={}, tau2={}, phi={},
                    avail_played=[[.99]*12]*4, avail_dnp=[[.7]*12]*4, b2b_factor=1, minutes_sd=[3]*4,
                    ramp=[.8,.9,1], fp_cv=[.3]*4, foul_mean=.01, foul_k=10, ot_rate=.05,
                    residual_keys=["test"], n_player_seasons=1)
    projection = PlayerProjection("0", "P0", "AAA", ["PG"], "G", 30, 3,
                                 np.array([.7,.2,.15,.04,.02,.08,.07]), np.full(7, 100.),
                                 .01, None, "healthy", 0, True, .99, 0, "test")
    prob.occupant[0] = 0
    prob.entries = [RootEntry(0, 60, False, entry.entry_id)]
    return WeekModel(state, ctx, {"0": projection}, params, {"test": np.ones((10, 7))}, ModelConfig(), scfg,
                     {"AAA": [game]}, {"AAA": {game.date}}, prob, None, ["0"])
