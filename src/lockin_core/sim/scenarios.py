"""Shared simulated futures for the rest of the matchup week, one draw per scenario.

For every remaining player-game of every relevant player (yours and the opponent's):

1. **Role**: players tagged Out/IR/suspended return after a sampled number of team games.
2. **Availability**: ruled out before tip (``out``, revealed at the tip, so a manager can bench
   him) and late scratch after the lineup is set (``scr``), from ``models.availability``.
3. **Team minutes** allocated inside the scenario among the teammates who play, so a
   teammate's absence raises minutes only where he is absent.
4. **Box score** from real residual vectors (his own once he has enough games this season,
   else his cohort's) on his per-minute rates, scored exactly by the engine.

Every candidate action is evaluated on the same ``ScenarioSet`` (common random numbers). The
main sets use posterior-mean rates; the model band uses ``batches`` > 1, where each batch
shares one draw of every player's rates and of the availability priors.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

from lockin_core.config import ET, ModelConfig
from lockin_core.models.availability import GameAvailability, game_availability
from lockin_core.models.data import RATE_STATS
from lockin_core.models.fit import Params
from lockin_core.models.project import PlayerProjection
from lockin_core.schedule.calendar import Game
from lockin_core.scoring.engine import score_units
from lockin_core.scoring.settings import ScoringSettings

LIVE = ("live", "awaiting_stats")


def seed_from(snapshot_id: str, stream: str) -> int:
    return int(hashlib.sha256(f"{snapshot_id}|{stream}".encode()).hexdigest()[:8], 16)


@dataclass
class ScenarioSet:
    pids: list[str]
    index: dict[str, int]
    game_ids: list[list[str]]  # per player: remaining games (live first), chronological
    n: int
    batches: int
    batch: np.ndarray  # [N]
    out: np.ndarray  # [N, P, K] bool
    scr: np.ndarray  # [N, P, K] bool
    units: np.ndarray  # [N, P, K] int32, score if he plays (0 otherwise)
    avail: list[list[GameAvailability]] = field(default_factory=list)
    stream: str = ""

    @property
    def K(self) -> int:
        return self.out.shape[2]

    @property
    def plays(self) -> np.ndarray:
        return ~self.out & ~self.scr

    def subset(self, rows: np.ndarray) -> "ScenarioSet":
        return ScenarioSet(self.pids, self.index, self.game_ids, int(len(rows)), self.batches, self.batch[rows],
                           self.out[rows], self.scr[rows], self.units[rows], self.avail, self.stream)

    def k_of(self, pid: str, game_id: str) -> int | None:
        i = self.index.get(pid)
        if i is None:
            return None
        try:
            return self.game_ids[i].index(game_id)
        except ValueError:
            return None


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sample(projections: dict[str, PlayerProjection], params: Params, lib: dict[str, np.ndarray],
           scoring: ScoringSettings, mcfg: ModelConfig, relevant: list[str],
           team_games: dict[str, list[Game]], team_dates: dict[str, set[date]], now: datetime, n: int,
           seed: int, reports: dict[tuple[str, str], str] | None = None,
           reported: set[tuple[str, str]] | None = None, batches: int = 1, rate_draws: bool = False,
           shock_sd: float = 0.0, calibration: dict | None = None, minutes_cap: float = 42.0,
           allocation_strength: float = 1.0, stream: str = "") -> ScenarioSet:
    """``team_games``: each team's remaining games this week (live and scheduled), by tip.
    ``reports``: (player_id, game_id) -> official status; ``reported``: (team, game_id) blocks
    whose team has submitted (not listed there = available)."""
    rng = np.random.default_rng(seed)
    reports = reports or {}
    reported = reported or set()
    batches = max(1, min(batches, n))
    batch = np.repeat(np.arange(batches), int(np.ceil(n / batches)))[:n]
    rel = [p for p in relevant if p in projections and projections[p].team]
    index = {p: i for i, p in enumerate(rel)}
    P = len(rel)
    gids = [[g.game_id for g in team_games.get(projections[p].team, [])] for p in rel]
    K = max([len(g) for g in gids] + [1])
    out = np.ones((n, P, K), bool)
    scr = np.zeros((n, P, K), bool)
    units = np.zeros((n, P, K), np.int32)
    avail_tab: list[list[GameAvailability]] = [[] for _ in rel]
    sd_logit = mcfg.availability.prior_uncertainty_logit_sd
    delta = rng.normal(0, sd_logit, batches)[batch] if batches > 1 else None

    for team in sorted(team_games):
        games = team_games[team]
        members = sorted(p for p, pr in projections.items() if pr.team == team)
        if not games or not members or not any(p in index for p in members):
            continue
        Kt, nj = len(games), len(members)
        dates = team_dates.get(team, set())
        mu = np.array([projections[p].minutes_mu for p in members])
        sd = np.array([projections[p].minutes_sd for p in members])
        plays_t = np.zeros((n, nj, Kt), bool)
        ramp = np.ones((n, nj, Kt))
        lam_all = np.zeros((n if rate_draws else 1, nj, len(RATE_STATS)))
        for j, p in enumerate(members):
            pr = projections[p]
            if rate_draws:
                shape = np.maximum(pr.rate * pr.rate_conc, 1e-6)
                draws = rng.gamma(shape, 1.0 / pr.rate_conc, size=(batches, len(RATE_STATS)))
                lam_all[:, j, :] = draws[batch]
            else:
                lam_all[0, j, :] = pr.rate
            kind = pr.status_kind
            # role scenario: the team game (1-based among remaining) a tagged-out player returns on
            if kind in ("out", "ir", "sus"):
                if pr.out_until is not None:
                    first = next((kk + 1 for kk, g in enumerate(games) if g.date > pr.out_until), Kt + 1)
                    ret = np.full(n, first)
                elif pr.return_hazard > 0:
                    ret = 1 + rng.geometric(pr.return_hazard, n)
                else:
                    ret = np.full(n, Kt + 1)
                for kk, g in enumerate(games):
                    st = reports.get((p, g.game_id))
                    if st is not None and st.lower() != "out":
                        ret = np.minimum(ret, kk + 1)  # the report says he is back
            else:
                ret = np.ones(n, int)
            for kk, g in enumerate(games):
                lead_h = (g.tip_at - now).total_seconds() / 3600
                b2b = (g.date - timedelta(days=1)) in dates
                st = reports.get((p, g.game_id))
                av = game_availability(pr, kk + 1, lead_h, b2b, st, (team, g.game_id) in reported,
                                       params, mcfg, calibration)
                if p in index:
                    avail_tab[index[p]].append(av)
                back = (kk + 1) >= ret if st is None else np.ones(n, bool)
                if st is not None and st.lower() == "out":
                    back = np.zeros(n, bool)
                q_out = np.full(n, av.q_out)
                if delta is not None and 0 < av.q_out < 1 and av.source not in ("healthy", "not listed"):
                    q_out = _sigmoid(_logit(av.q_out) + delta)
                if g.status in LIVE:  # tipped: nothing left to reveal; he plays or he does not
                    o = np.zeros(n, bool)
                    s_ = ~back | (rng.random(n) < (1.0 - av.p_play))
                else:
                    o = ~back | (rng.random(n) < q_out)
                    s_ = ~o & (rng.random(n) < av.q_scr)
                plays_t[:, j, kk] = ~o & ~s_
                if p in index:
                    i = index[p]
                    out[:, i, kk] = o
                    scr[:, i, kk] = s_
            ramp_n = 3 if kind in ("out", "ir", "sus") else pr.ramp_left
            if ramp_n:
                back_cnt = np.cumsum(plays_t[:, j, :], axis=1)
                start = 3 - ramp_n
                for r in range(ramp_n):
                    ramp[:, j, :] = np.where(back_cnt == r + 1, params.ramp[start + r], ramp[:, j, :])
        ot = rng.random((n, Kt)) < params.ot_rate
        target = 240.0 + 25.0 * ot
        mean_m = np.minimum(mu[None, :, None] * ramp, minutes_cap) * plays_t
        tot = mean_m.sum(axis=1)
        scale = np.where(tot > 0, (target / np.maximum(tot, 1e-6)) ** allocation_strength, 1.0)
        m = np.minimum(mean_m * scale[:, None, :], minutes_cap)
        m = np.where(plays_t, np.clip(m + sd[None, :, None] * rng.standard_normal((n, nj, Kt)), 1.0, 53.0), 0.0)
        shock = (np.exp(shock_sd * rng.standard_normal((n, nj)) - shock_sd**2 / 2) if shock_sd > 0
                 else np.ones((n, nj)))
        for j, p in enumerate(members):
            i = index.get(p)
            if i is None:
                continue
            pr = projections[p]
            res = pr.own_resid if pr.resid_key == "own" and pr.own_resid is not None else lib[pr.resid_key]
            lam = lam_all[:, j, :]
            for kk in range(Kt):
                mm = m[:, j, kk]
                u = res[rng.integers(0, len(res), n)]
                mean = mm[:, None] * lam * shock[:, j][:, None] * u
                cnt = np.floor(mean + rng.random(mean.shape)).astype(np.int64)
                cols = {c: cnt[:, ci] for ci, c in enumerate(RATE_STATS)}
                cols["fg3m"] = np.minimum(cols["fg3m"], cols["pts"] // 3)
                cols["tf"] = rng.poisson(pr.foul_rate, n)
                sc = score_units(cols, scoring).astype(np.int32)
                units[:, i, kk] = np.where(plays_t[:, j, kk], sc, 0)
    return ScenarioSet(rel, index, gids, n, batches, batch, out, scr, units, avail_tab, stream)


def team_schedule(calendar, teams: set[str]) -> tuple[dict[str, list[Game]], dict[str, set[date]]]:
    """Remaining games (live and scheduled) and all game dates of each team this week."""
    tg, td = {}, {}
    for t in sorted(teams):
        games = calendar.team_games(t)
        td[t] = {g.date for g in games}
        tg[t] = [g for g in games if g.status in ("scheduled", *LIVE)]
    return tg, td


def et_dates(ts: list[datetime]) -> set[date]:
    return {t.astimezone(ET).date() for t in ts}
