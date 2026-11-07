"""From a validated state to the simulator: projections, scenario streams, side problems.

``WeekModel`` holds everything frozen when a recommendation starts: projections for every
player on an involved team (teammates are needed to allocate minutes), the remaining
schedule, and one ``SideProblem`` per roster with continuation tables built from a separate
planning stream. Scenario streams (screen, final, band) are drawn on demand from seeds
derived from the snapshot id, so a recommendation is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from lockin_core.config import ModelConfig, Paths, SimConfig
from lockin_core.data import tables as T
from lockin_core.models.data import history
from lockin_core.models.fit import Params
from lockin_core.models.project import PlayerProjection, project_players
from lockin_core.policy.tables import build_tables, quantile_grid
from lockin_core.scoring.engine import score_frame
from lockin_core.sim.problem import RootEntry, SideProblem, SideScenarios, SimPlayer
from lockin_core.sim.scenarios import ScenarioSet, sample, seed_from, team_schedule
from lockin_core.state.build import Context
from lockin_core.state.models import PlayerInfo, SideState, ValidatedState


def _minutes(a: datetime, b: datetime) -> float:
    return (a - b).total_seconds() / 60.0


def baseline_fp(paths: Paths, seasons: list[int], pids: set[str], scoring, cutoff: datetime,
                last_n: int = 40) -> dict[str, tuple[float, float]]:
    """(mean, 75th percentile) of each player's last ``last_n`` played games, in units: the
    thresholds of the "lock above his average / 75th percentile" baselines."""
    try:
        h = history(paths, seasons, as_of=cutoff)
    except FileNotFoundError:
        return {}
    h = h.filter(pl.col("player_id").is_in(list(pids)) & (pl.col("played") == "played"))
    if h.is_empty():
        return {}
    s = score_frame(h, scoring)
    out = {}
    for (pid,), g in s.sort("date").group_by("player_id"):
        u = g["fp_units"].drop_nulls().to_numpy()[-last_n:]
        if len(u):
            out[pid] = (float(u.mean()), float(np.percentile(u, 75)))
    return out


@dataclass
class WeekModel:
    state: ValidatedState
    ctx: Context
    projections: dict[str, PlayerProjection]
    params: Params
    lib: dict[str, np.ndarray]
    mcfg: ModelConfig
    scfg: SimConfig
    team_games: dict
    team_dates: dict
    me: SideProblem
    opp: SideProblem | None
    relevant: list[str]
    calibration: dict | None = None

    # --- scenarios -----------------------------------------------------------------
    def scenarios(self, n: int, stream: str, batches: int = 1, rate_draws: bool = False) -> ScenarioSet:
        seed = self.scfg.seed if self.scfg.seed is not None else seed_from(self.state.snapshot_id, stream)
        if self.scfg.seed is not None:
            seed = seed_from(str(self.scfg.seed), stream)
        return sample(self.projections, self.params, self.lib, self.ctx.rules.scoring, self.mcfg, self.relevant,
                      self.team_games, self.team_dates, self.state.t, n, seed, self.ctx.reports, self.ctx.reported,
                      batches=batches, rate_draws=rate_draws, shock_sd=0.0, calibration=self.calibration,
                      stream=stream)

    def side_scenarios(self, sc: ScenarioSet, prob: SideProblem) -> SideScenarios:
        N, P, K = sc.n, prob.P, prob.K
        out = np.ones((N, P, K), bool)
        scr = np.zeros((N, P, K), bool)
        units = np.zeros((N, P, K))
        for p, pl_ in enumerate(prob.players):
            i = sc.index.get(pl_.pid)
            if i is None:
                continue
            k = min(pl_.n_games, sc.K)
            out[:, p, :k] = sc.out[:, i, :k]
            scr[:, p, :k] = sc.scr[:, i, :k]
            units[:, p, :k] = sc.units[:, i, :k]
        return SideScenarios(out, scr, units)

    def side(self, which: str) -> SideProblem | None:
        return self.me if which == "me" else self.opp


def _side_problem(side: SideState, state: ValidatedState, ctx: Context, team_games: dict,
                  base: dict[str, tuple[float, float]]) -> SideProblem:
    t0 = state.t
    cal = ctx.calendar
    players: list[SimPlayer] = []
    for pid in side.active:
        info: PlayerInfo | None = state.players.get(pid)
        if info is None:
            continue
        games = team_games.get(info.team, []) if info.team else []
        tips = np.array([_minutes(g.tip_at, t0) for g in games])
        ends = np.array([max(_minutes(g.end_at, t0), 1.0) for g in games])
        final = np.array([cal.is_team_final(info.team, g.game_id) for g in games], bool)
        live = np.array([g.status in ("live", "awaiting_stats") for g in games], bool)
        m, p75 = base.get(pid, (0.0, 0.0))
        players.append(SimPlayer(pid, info.name, list(info.positions), tips, ends, final, live,
                                 [g.game_id for g in games], m, p75))
    idx = {p.pid: i for i, p in enumerate(players)}
    S = len(side.slots)
    locked = np.zeros(S, bool)
    occ = np.full(S, -1)
    used = np.zeros(len(players), bool)
    for i, l in side.locked.items():
        locked[i] = True
        if l.player_id in idx:
            used[idx[l.player_id]] = True
            occ[i] = idx[l.player_id]
    for i, pid in side.assignment.items():
        if pid in idx and not locked[i]:
            occ[i] = idx[pid]
    entries = [RootEntry(idx[e.player_id], float(e.fp_half), e.is_team_final, e.entry_id,
                         must_decide=(e.deadline_at - t0).total_seconds() < 15 * 60)
               for e in side.entries if e.player_id in idx]
    started_live = np.zeros(len(players), bool)
    for pid in side.frozen:
        if pid in idx and idx[pid] in occ:
            started_live[idx[pid]] = True
    return SideProblem(list(side.slots), players, locked, float(side.locked_units), occ, used, entries, started_live)


def _attach_tables(prob: SideProblem, sc: ScenarioSet, q: int) -> None:
    P, K = prob.P, prob.K
    values = np.zeros((P, K, q))
    weights = np.full((P, K, q), 1.0 / q)
    q_out = np.ones((P, K))
    q_scr = np.zeros((P, K))
    final = np.zeros((P, K), bool)
    n_games = np.array([p.n_games for p in prob.players])
    for p, pl_ in enumerate(prob.players):
        i = sc.index.get(pl_.pid)
        for k in range(pl_.n_games):
            final[p, k] = pl_.final[k]
            if i is None or k >= sc.K:
                continue
            plays = sc.plays[:, i, k]
            values[p, k], weights[p, k] = quantile_grid(sc.units[plays, i, k], q)
            av = sc.avail[i][k] if k < len(sc.avail[i]) else None
            if pl_.live[k]:
                q_out[p, k] = 0.0
                q_scr[p, k] = 1.0 - plays.mean()
            else:
                # empirical rates from the planning draws (they include the sampled return
                # of long-term absentees, which the per-game probabilities do not)
                q_out[p, k] = sc.out[:, i, k].mean()
                q_scr[p, k] = (sc.scr[:, i, k].sum() / max((~sc.out[:, i, k]).sum(), 1))
            _ = av
    prob.tables = build_tables(values, weights, q_out, q_scr, n_games, final, prob.flex)


def build_week_model(state: ValidatedState, ctx: Context, paths: Paths, params: Params, lib: dict,
                     mcfg: ModelConfig, scfg: SimConfig, manual: dict | None = None,
                     calibration: dict | None = None) -> WeekModel:
    sides = [state.me] + ([state.opp] if state.opp is not None else [])
    relevant = sorted({p for s in sides for p in s.active if p in state.players})
    teams = {state.players[p].team for p in relevant if state.players[p].team}
    players_df = T.read(paths, "players")
    pinfo: dict[str, PlayerInfo] = dict(state.players)
    if players_df is not None:
        for r in players_df.filter(pl.col("team").is_in(list(teams))).iter_rows(named=True):
            pinfo.setdefault(r["player_id"], PlayerInfo(player_id=r["player_id"], name=r["name"], team=r["team"],
                                                        positions=r["positions"], injury_status=r["injury_status"]))
    only = {p for p, i in pinfo.items() if i.team in teams}
    seasons = list(ctx.cfg.history_seasons)
    projections = project_players(paths, pinfo, state.season, state.t, params, mcfg, seasons, manual=manual,
                                  only=only)
    tg, td = team_schedule(ctx.calendar, teams)
    base = baseline_fp(paths, [*seasons, state.season], set(relevant), ctx.rules.scoring, state.t)
    me = _side_problem(state.me, state, ctx, tg, base)
    opp = _side_problem(state.opp, state, ctx, tg, base) if state.opp is not None else None
    if opp is not None and state.opp_locked_manual is not None:
        opp.locked_units = state.opp_locked_manual * state.units_per_point
    wm = WeekModel(state, ctx, projections, params, lib, mcfg, scfg, tg, td, me, opp, relevant, calibration)
    planning = wm.scenarios(scfg.planning_scenarios, "planning")
    _attach_tables(me, planning, scfg.base_policy.quantile_grid)
    if opp is not None:
        _attach_tables(opp, planning, scfg.base_policy.quantile_grid)
    wm.planning = planning  # type: ignore[attr-defined]
    return wm
