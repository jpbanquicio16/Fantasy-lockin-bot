"""Point-in-time projections: each player's game-level model as known at a cutoff.

For every player on an NBA roster the projection holds the three parts of the
generative model of one game (design section 11):

* availability - P(plays) for each of his team's upcoming games, from his injury
  status at the cutoff and the fitted table; "Out"/"IR" players return after a
  sampled number of games (a role scenario drawn once per simulated week);
* minutes if he plays - exponentially weighted recent minutes (fast half-life,
  because roles change in minutes first) with a ramp after a long absence;
* production per minute - EW rates shrunk to his cohort with a gamma posterior whose
  spread is the model (epistemic) uncertainty, and real residual vectors for the shape
  of a single game.

Only games that had finished before the cutoff date are read (``models.data.history``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import polars as pl

from lockin_core.config import ModelConfig, Paths
from lockin_core.models.data import (
    AVAIL_BANDS,
    RATE_STATS,
    band,
    et_today,
    history,
    position_group,
    recency_weights,
)
from lockin_core.models.fit import Params
from lockin_core.scoring.settings import ScoringSettings
from lockin_core.state.models import PlayerInfo

STATUS_KIND = {
    "OUT": "out", "IR": "ir", "SUS": "sus", "SUSPENDED": "sus", "DTD": "dtd", "GTD": "dtd",
    "QUESTIONABLE": "dtd", "DOUBTFUL": "doubtful", "PROBABLE": "probable",
}
LINEAR_WEIGHT_KEYS = {"pts": "pts", "reb": "reb", "ast": "ast", "stl": "stl", "blk": "blk",
                      "tov": "to", "fg3m": "tpm"}
DEFAULT_MINUTES = 6.0
OUT_OF_ROTATION_MINUTES = 8.0
PLAY_RATE_HALF_LIFE = 5.0  # listed games
PLAY_RATE_PREV_SEASON = 0.3
PLAY_RATE_PRIOR_GAMES = 4.0


@dataclass
class PlayerProjection:
    player_id: str
    name: str
    team: str | None
    positions: list[str]
    group: str
    minutes_mu: float
    minutes_sd: float
    rate: np.ndarray  # posterior mean per minute, RATE_STATS order
    rate_conc: np.ndarray  # gamma concentration: shape = rate * conc, scale = 1 / conc
    foul_rate: float
    status: str | None
    status_kind: str  # healthy | dtd | doubtful | probable | out | ir | sus
    return_hazard: float
    last_played: bool | None
    play_rate: float  # P(plays his next game) when healthy, from his own listed games
    ramp_left: int  # games of post-absence ramp still to come
    resid_key: str
    own_resid: np.ndarray | None = None
    games_this_season: int = 0
    minutes_this_season: float = 0.0
    out_until: date | None = None
    p_override: float | None = None
    notes: list[str] = field(default_factory=list)
    fp_mean_units: float = 0.0  # realised FP per played game (recent seasons), for baselines
    fp_p75_units: float = 0.0

    @property
    def band(self) -> int:
        return band(self.minutes_mu)

    @property
    def avail_band(self) -> int:
        return band(self.minutes_mu, AVAIL_BANDS)

    def linear_fp(self, scoring: ScoringSettings) -> float:
        """Expected FP per played game from the linear stat weights (no bonuses)."""
        w = scoring.weights
        fp = sum(w.get(LINEAR_WEIGHT_KEYS[c], 0.0) * self.minutes_mu * self.rate[i]
                 for i, c in enumerate(RATE_STATS))
        return fp + (w.get("tf", 0.0)) * self.foul_rate


def _availability_kind(status: str | None) -> str:
    return STATUS_KIND.get((status or "").upper(), "healthy")


def ramp_state(seq: list[bool]) -> tuple[int, int]:
    """(games of post-absence ramp still to come, length of a trailing 3+ game absence).

    ``seq`` is played/did-not-play over his listed games, oldest first.
    """
    runs = [(k, len(list(g))) for k, g in itertools.groupby(seq)]
    if not runs:
        return 0, 0
    last, n = runs[-1]
    if not last:
        return (3, n) if n >= 3 else (0, 0)
    if len(runs) >= 2 and runs[-2][1] >= 3 and n < 3:
        return 3 - n, 0
    return 0, 0


def play_rate_seq(seq: list[bool], prev: list[bool], prior: float, injured_now: bool = False) -> float:
    """Shrunk EW share of listed games played (oldest first), ignoring completed injury
    absences (runs of 3+ did-not-play followed by a return), which say nothing about his
    role - and, when an injury status explains it, the absence he is in now."""
    keep = [True] * len(seq)
    i = 0
    while i < len(seq):
        if seq[i]:
            i += 1
            continue
        j = i
        while j < len(seq) and not seq[j]:
            j += 1
        if (j - i >= 3 and j < len(seq)) or (injured_now and j == len(seq)):
            for t in range(i, j):
                keep[t] = False
        i = j
    xs = [(x, pv) for x, pv, kp in zip(seq, prev, keep) if kp]
    if not xs:
        return prior
    n = len(xs)
    w = 0.5 ** (np.arange(n - 1, -1, -1, dtype=float) / PLAY_RATE_HALF_LIFE)
    w = w * np.array([PLAY_RATE_PREV_SEASON if pv else 1.0 for _, pv in xs])
    x = np.array([float(v) for v, _ in xs])
    return float((w @ x + PLAY_RATE_PRIOR_GAMES * prior) / (w.sum() + PLAY_RATE_PRIOR_GAMES))


def play_rate(rows: pl.DataFrame, season: int, prior: float, injured_now: bool = False) -> float:
    seq = (rows["played"] == "played").to_list()
    prev = (rows["season"] < season).to_list()
    return play_rate_seq(seq, prev, prior, injured_now)


def _own_residuals(g: pl.DataFrame, min_minutes: float) -> np.ndarray | None:
    g = g.filter(pl.col("min") >= min_minutes)
    if g.height == 0:
        return None
    m = g["min"].to_numpy()
    cols = []
    for c in RATE_STATS:
        x = g[c].to_numpy().astype(float)
        r = x.sum() / m.sum()
        cols.append(np.where(r > 0, (x / m) / max(r, 1e-12), 1.0))
    return np.column_stack(cols).astype(np.float32)


def project_players(paths: Paths, players: dict[str, PlayerInfo], season: int, cutoff: datetime,
                    params: Params, cfg: ModelConfig, history_seasons: list[int],
                    manual: dict[str, dict] | None = None, only: set[str] | None = None,
                    hist: pl.DataFrame | None = None,
                    injured_this_season: set[str] | None = None) -> dict[str, PlayerProjection]:
    """Projections for every player with a team (or in ``only``) as known at ``cutoff``.

    ``injured_this_season``: players seen with an injury status in a snapshot this season
    before the cutoff; their did-not-play games are not read as lost rotation spots.
    """
    manual = manual or {}
    injured_this_season = injured_this_season or set()
    if hist is None:
        hist = history(paths, [*history_seasons, season], as_of=cutoff)
    wanted = {p for p, info in players.items() if info.team} if only is None else set(only)
    hist = hist.filter(pl.col("player_id").is_in(list(wanted)))
    groups = {pid: g for (pid,), g in hist.group_by("player_id")}
    hl = cfg.half_life_games
    today = et_today(cutoff)
    out = {}
    for pid in sorted(wanted):
        info = players.get(pid)
        if info is None:
            continue
        g = groups.get(pid)
        group = position_group(info.positions)
        rows = g.sort("date") if g is not None else None
        played = rows.filter(pl.col("played") == "played") if rows is not None else None
        notes = []
        if played is not None and played.height:
            prev = (played["season"] < season).to_numpy()
            mins = played["min"].to_numpy()
            w_min = recency_weights(len(mins), hl["min"], prev, cfg.previous_season_weight)
            mu_m = float((w_min * mins).sum() / w_min.sum())
        else:
            mins, prev, mu_m = np.zeros(0), np.zeros(0, bool), DEFAULT_MINUTES
            notes.append("no NBA games: cohort prior and deep-bench minutes")
        kind = _availability_kind(info.injury_status)
        cur_listed = rows.filter(pl.col("season") == season) if rows is not None else None
        if (kind == "healthy" and cur_listed is not None and cur_listed.height >= 3
                and not (cur_listed["played"] == "played").any() and pid not in injured_this_season
                and mu_m > OUT_OF_ROTATION_MINUTES):
            notes.append(f"listed {cur_listed.height} times this season without playing: "
                         f"out of the rotation ({mu_m:.0f} -> {OUT_OF_ROTATION_MINUTES:.0f} min if he plays)")
            mu_m = OUT_OF_ROTATION_MINUTES
        b = band(mu_m)
        rate = np.zeros(len(RATE_STATS))
        conc = np.zeros(len(RATE_STATS))
        for i, c in enumerate(RATE_STATS):
            prior = params.rate_prior[group][b][c]
            k = params.shrink_k(group, b, c)
            if len(mins):
                w = recency_weights(len(mins), hl[c], prev, cfg.previous_season_weight)
                X, M = float((w * played[c].to_numpy()).sum()), float((w * mins).sum())
            else:
                X = M = 0.0
            rate[i] = (X + k * prior) / (M + k)
            conc[i] = (M + k) / params.phi[c]
        cur = played.filter(pl.col("season") == season) if played is not None else None
        if played is not None and played.height:
            fouls = float((played["tf"] + played["ff"]).sum())
            foul_rate = (fouls + params.foul_k * params.foul_mean) / (played.height + params.foul_k)
        else:
            foul_rate = params.foul_mean
        last_played, ramp_left = None, 0
        if rows is not None and rows.height:
            seq = (rows["played"] == "played").to_list()
            last_played = seq[-1]
            ramp_left, missed = ramp_state(seq)
            if missed:
                notes.append(f"missed his last {missed} listed games: minutes ramp on return")
        prior_p = params.avail_played[band(mu_m, AVAIL_BANDS)][0]
        p_rate = (play_rate(rows, season, prior_p, injured_now=kind != "healthy")
                  if rows is not None and rows.height else prior_p)
        hazard = {"out": cfg.availability.out_return_hazard, "ir": cfg.availability.ir_return_hazard,
                  "sus": 0.0}.get(kind, 0.0)
        own = None
        enough = cur is not None and cur.filter(
            pl.col("min") >= cfg.min_minutes_residual_game).height >= cfg.min_games_own_residuals
        if enough:
            own = _own_residuals(cur, cfg.min_minutes_residual_game)
        proj = PlayerProjection(
            player_id=pid, name=info.name, team=info.team, positions=info.positions, group=group,
            minutes_mu=mu_m, minutes_sd=params.minutes_sd[b], rate=rate, rate_conc=conc,
            foul_rate=foul_rate, status=info.injury_status, status_kind=kind, return_hazard=hazard,
            last_played=last_played, play_rate=p_rate, ramp_left=ramp_left,
            resid_key="own" if own is not None else f"{group}{b}", own_resid=own,
            games_this_season=cur.height if cur is not None else 0,
            minutes_this_season=float(cur["min"].sum()) if cur is not None and cur.height else 0.0,
            notes=notes)
        _apply_manual(proj, manual.get(pid), today)
        out[pid] = proj
    return out


def _apply_manual(proj: PlayerProjection, entry: dict | None, today: date) -> None:
    if not entry:
        return
    if "minutes" in entry:
        proj.minutes_mu = float(entry["minutes"])
        proj.notes.append(f"minutes overridden to {proj.minutes_mu:.0f}")
    if entry.get("out_until"):
        until = date.fromisoformat(entry["out_until"])
        if until >= today:
            proj.status_kind, proj.out_until, proj.return_hazard = "out", until, 0.0
            proj.notes.append(f"assumed out through {until:%a %d %b} (override)")
    if entry.get("p_play") is not None:
        proj.p_override = float(entry["p_play"])
        proj.status_kind = "healthy" if proj.status_kind in ("out", "ir", "sus") else proj.status_kind
        proj.notes.append(f"P(play next game) overridden to {proj.p_override:.2f}")
