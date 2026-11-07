"""Build and validate the state a recommendation runs on (design sections 3, 5 and 7).

Sources, in order of authority:

1. **Manual entries** - locks you recorded, passes, "not eligible", score corrections, the
   opponent's locked total, tip-off corrections.
2. **The lineup ledger** - tip-off snapshots: who was starting when each game tipped.
3. **Matchup points** - ``starters_points`` per slot is the score that slot carries now
   (verified on this league by the waiver assistant, 30 Oct 2026): equal to an *earlier* game
   of its occupant means that game is **locked**; equal to his *latest* game means **pending**
   (locked already, or still lockable until his team's next tip - the API cannot tell, so you
   are asked once); a pending score whose deadline has passed is locked.
4. **Box scores** (Sleeper's stats feed) scored by the engine; Sleeper's own ``players_points``
   confirms the number.

Blocking inputs stop the run (roster, slots, schedule, scoring); stale or uncertain ones are
flags that every recommendation carries.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

import polars as pl

from lockin_core.config import LeagueConfig, Paths, RuleSwitches, code_version, now_utc, tzdata_version
from lockin_core.data import tables as T
from lockin_core.ingest.collect import completion_times
from lockin_core.ingest.sleeper import SleeperClient
from lockin_core.schedule.calendar import WeekCalendar, build_calendar
from lockin_core.scoring.engine import score_frame
from lockin_core.state.ledger import resolve_tipoffs, starters_at
from lockin_core.state.league import LeagueRules
from lockin_core.state.models import EligibleScore, LockedSlot, PlayerInfo, SideState, ValidatedState
from lockin_core.state.store import Store

MAX_AGE = {"rosters": timedelta(minutes=10), "matchups": timedelta(minutes=10), "schedule": timedelta(hours=2),
           "stats": timedelta(hours=6), "players_map": timedelta(hours=26), "injury_report": timedelta(minutes=45)}


class BlockingInputError(RuntimeError):
    """A recommendation cannot be made without this input (design section 5)."""


@dataclass
class WeekGameScore:
    game_id: str
    team: str | None
    played: str  # played | dnp
    units: int | None
    revision: int
    all_units: frozenset[int] = frozenset()  # the score under every revision seen (stat corrections)

    def shows(self, units: int) -> bool:
        return units == self.units or units in self.all_units


@dataclass
class Context:
    """Inputs behind a ValidatedState that projections and explanations need."""

    cfg: LeagueConfig
    rules: LeagueRules
    calendar: WeekCalendar
    switches: RuleSwitches
    reports: dict[tuple[str, str], str] = field(default_factory=dict)  # (player, game) -> status
    reported: set[tuple[str, str]] = field(default_factory=set)  # (team, game) blocks submitted
    report_published_at: datetime | None = None
    scores: dict[str, list[WeekGameScore]] = field(default_factory=dict)
    sleeper_points: dict[str, float] = field(default_factory=dict)
    roster_names: dict[int, str] = field(default_factory=dict)
    player_team: dict[str, str | None] = field(default_factory=dict)


def resolve_my_roster(cfg: LeagueConfig, client: SleeperClient, rosters) -> int:
    if cfg.my_roster_id is not None:
        return int(cfg.my_roster_id)
    if not cfg.username:
        raise BlockingInputError("set username or my_roster_id in configs/league.yaml")
    user = client.user(cfg.username)
    if user is None:
        raise BlockingInputError(f"Sleeper user {cfg.username!r} not found")
    for r in rosters:
        co = r.model_extra.get("co_owners") or []
        if r.owner_id == user.user_id or user.user_id in co:
            return r.roster_id
    raise BlockingInputError(f"{cfg.username} owns no roster in league {cfg.league_id}")


def week_scores(box: pl.DataFrame, rules: LeagueRules, week: int) -> dict[str, list[WeekGameScore]]:
    every = box.filter((pl.col("week") == week) & (pl.col("played") == "played"))
    revs: dict[tuple[str, str], set[int]] = {}
    if every.height:
        sc_all = score_frame(every, rules.scoring)
        for pid, gid, u in zip(sc_all["player_id"], sc_all["game_id"], sc_all["fp_units"]):
            if u is not None:
                revs.setdefault((pid, gid), set()).add(int(u))
    wk = T.latest_revision(box).filter(pl.col("week") == week)
    played = wk.filter(pl.col("played") == "played")
    scored = score_frame(played, rules.scoring) if played.height else played.with_columns(
        pl.lit(None, dtype=pl.Int64).alias("fp_units"))
    wk = wk.join(scored.select("game_id", "player_id", "fp_units"), on=["game_id", "player_id"], how="left")
    out: dict[str, list[WeekGameScore]] = {}
    for r in wk.sort("date").iter_rows(named=True):
        out.setdefault(r["player_id"], []).append(WeekGameScore(
            r["game_id"], r["team"], r["played"], r["fp_units"] if r["played"] == "played" else None, r["revision"],
            frozenset(revs.get((r["player_id"], r["game_id"]), set()))))
    return out


def report_maps(paths: Paths, calendar: WeekCalendar, now: datetime):
    reps = T.read(paths, "injury_reports")
    if reps is None or reps.is_empty():
        return {}, set(), None
    cur = T.report_status_as_of(reps, now)
    if cur.is_empty():
        return {}, set(), None
    by_team_date = {}
    for g in calendar.games:
        for t in g.teams():
            by_team_date[(t, g.date)] = g.game_id
    statuses, reported = {}, set()
    for r in cur.iter_rows(named=True):
        gid = by_team_date.get((r["team"], r["game_date"]))
        if gid is None:
            continue
        reported.add((r["team"], gid))
        if r["player_id"]:
            statuses[(r["player_id"], gid)] = r["status"]
    return statuses, reported, cur["published_at"].max()


# --- one roster's week ---------------------------------------------------------------
def derive_side(roster_id: int, owner: str | None, slots: list[str], starters: list[str | None],
                players_all: list[str], reserve: list[str], starters_points: list[float] | None,
                ctx: Context, store: Store, league_id: str, week: int, now: datetime, res: int,
                manual: bool = True) -> SideState:
    cal = ctx.calendar
    notes: list[str] = []
    active = tuple(p for p in players_all if p not in set(reserve))
    locked = {}
    # A displayed weekly score is not a lock-state API. Only explicit lock records fix slots.
    for m in store.locks(league_id, week, roster_id, as_of=now) if manual else []:
        i = m["slot_index"]
        if not 0 <= i < len(slots):
            raise BlockingInputError("recorded lock refers to a missing slot; correct the lock history")
        locked[i] = LockedSlot(i, m["player_id"], m["game_id"], int(m["fp_half"]), "manual")
    used = {l.player_id for l in locked.values()}
    assignment = {i: pid for i, pid in enumerate(starters[:len(slots)])
                  if pid and i not in locked and pid not in used}
    passes = {e.key for kind in ("pass", "not_eligible") for e in store.manual(kind, as_of=now)} if manual else set()
    corrections = {e.key: e.value for e in store.manual("score", as_of=now)} if manual else {}
    not_locked = {e.key for e in store.manual("not_locked", as_of=now)} if manual else set()
    entries = []
    frozen = set()
    for i, pid in assignment.items():
        team = ctx_player_team(ctx, pid)
        team_games = cal.team_games(team)
        if any(g.status in ("live", "awaiting_stats") for g in team_games):
            frozen.add(pid)
        completed = [g for g in team_games if g.status == "final" and g.end_at <= now]
        if not completed:
            continue
        game = completed[-1]
        deadline, final = cal.deadline(team, game.game_id)
        key = f"{week}:{roster_id}:{pid}:{game.game_id}"
        if deadline <= now:
            if manual and key not in passes and key not in not_locked:
                notes.append(f"lock_state_uncertain:{pid}: record any past lock or confirm this game was passed")
            continue
        if key in passes:
            continue
        snap, status = _tip_status(store, roster_id, cal, game.game_id, now)
        if status not in ("observed", "confirmed_manual"):
            if manual:
                notes.append(f"eligibility_uncertain:{pid}:{game.tip_at.isoformat()}: confirm starters at tip")
            continue
        if not snap or pid not in snap:
            continue  # being a starter now never makes a bench game eligible
        # Leaving the starting lineup after tip relinquishes an open score.
        obs = store.observations(roster_id, game.tip_at, now)
        if any(o["poll_ok"] and pid not in o["starters"] for o in obs):
            continue
        score = next((g for g in reversed(ctx.scores.get(pid, [])) if g.game_id == game.game_id), None)
        if score is None and key not in corrections:
            notes.append(f"awaiting_box_score:{pid}")
            continue
        if score is not None and score.played != "played" and not final:
            continue
        units = corrections[key]["fp_half"] if key in corrections else (score.units if score.played == "played" else 0)
        if units is None:
            notes.append(f"awaiting_box_score:{pid}")
            continue
        # A final DNP can only supply a zero fallback; it cannot recover an earlier score.
        entries.append(EligibleScore(key, pid, game.game_id, int(units), snap.index(pid), final, deadline,
                                     "manual" if status == "confirmed_manual" else "observed",
                                     "manual" if key in corrections else "engine",
                                     locked_unknown=manual and key not in not_locked))
    return SideState(roster_id, owner, tuple(slots), assignment, locked, tuple(entries), active, tuple(reserve),
                     frozenset(frozen), notes)


def ctx_player_team(ctx: Context, pid: str) -> str | None:
    return ctx.player_team.get(pid)


def _tip_status(store: Store, roster_id: int, cal: WeekCalendar, game_id: str, as_of=None):
    g = cal.game(game_id)
    if g is None:
        return None, "unknown"
    return starters_at(store, roster_id, g.tip_at, as_of=as_of)


def _slot_in(snap: list | None, pid: str) -> int | None:
    if not snap or pid not in snap:
        return None
    return snap.index(pid)


# --- the whole state -------------------------------------------------------------------
def build_state(cfg: LeagueConfig, paths: Paths | None = None, client: SleeperClient | None = None,
                store: Store | None = None, now: datetime | None = None, objective: str = "points",
                save: bool = True, strict: bool = True) -> tuple[ValidatedState, Context]:
    paths = (paths or Paths()).ensure()
    client = client or SleeperClient(paths=paths)
    store = store or Store(paths)
    now = now or now_utc()
    if not cfg.league_id:
        raise BlockingInputError("set league_id in configs/league.yaml")
    flags: list[str] = []
    blocking: list[str] = []

    st = client.state(max_age_s=600)
    season, week = int(st.season), int(st.week)
    league = client.league(cfg.league_id)
    rules = LeagueRules.from_league(league, cfg)
    if not rules.starting_slots:
        blocking.append("no starting slots in roster_positions")
    names = {u.user_id: u.display_name for u in client.users(cfg.league_id)}
    raw = client.rosters(cfg.league_id, max_age_s=60)
    my_id = resolve_my_roster(cfg, client, raw)
    rosters = {r.roster_id: r for r in raw}
    if my_id not in rosters:
        raise BlockingInputError(f"roster {my_id} not in league")
    matchups = {m.roster_id: m for m in client.matchups(cfg.league_id, week, max_age_s=60)}
    opp_id = None
    mine = matchups.get(my_id)
    if mine and mine.matchup_id is not None:
        opp_id = next((rid for rid, m in matchups.items() if m.matchup_id == mine.matchup_id and rid != my_id), None)
    if mine is None:
        flags.append("no_matchup_data")

    games = T.read(paths, "games", season)
    box = T.read(paths, "box", season)
    players_df = T.read(paths, "players")
    if box is None and store.manual("completed_game", as_of=now):
        box = pl.DataFrame(schema=T.BOX_SCHEMA)
    if games is None or box is None or players_df is None:
        raise BlockingInputError("tables missing; run `lockin ingest` first")
    if rules.scoring.open_settings():
        raise BlockingInputError(f"scoring settings not settled: {rules.scoring.open_settings()} "
                                 "(run `lockin reconcile` and set them in configs/league.yaml)")
    cal = build_calendar(games, season, week, now, cfg.rules.deadline_margin_minutes,
                         completion=completion_times(client, season))
    for entry in store.manual("completed_game", as_of=now, key_prefix=f"{cfg.league_id}:"):
        gid = entry.value["game_id"]
        ended = datetime.fromisoformat(entry.value["completed_at"])
        cal.games = [replace(g, status="final", end_at=ended) if g.game_id == gid else g for g in cal.games]
    flags += [f"schedule:{i}" for i in cal.issues]
    ctx = Context(cfg, rules, cal, cfg.rules, roster_names={rid: names.get(r.owner_id or "", f"roster {rid}")
                                                           for rid, r in rosters.items()})
    ctx.scores = week_scores(box.filter(pl.col("ingested_at") <= now), rules, week)
    ctx.reports, ctx.reported, ctx.report_published_at = report_maps(paths, cal, now)
    if mine is not None and mine.players_points:
        ctx.sleeper_points = dict(mine.players_points)
    if opp_id is not None and matchups[opp_id].players_points:
        ctx.sleeper_points.update(matchups[opp_id].players_points)

    rostered = {p for r in raw for p in (r.players or [])}
    pinfo: dict[str, PlayerInfo] = {}
    for r in players_df.iter_rows(named=True):
        if r["player_id"] in rostered:
            pinfo[r["player_id"]] = PlayerInfo(player_id=r["player_id"], name=r["name"], team=r["team"],
                                               positions=r["positions"], injury_status=r["injury_status"],
                                               years_exp=r["years_exp"])
    for pid in rostered - set(pinfo):
        flags.append(f"unknown_player:{pid}")
    ctx.player_team = {p: i.team for p, i in pinfo.items()}
    for pid, info in pinfo.items():
        nxt = next((g for g in cal.remaining_team_games(info.team, now, include_live=False)), None)
        if nxt is not None and (pid, nxt.game_id) in ctx.reports:
            info.report_status = ctx.reports[(pid, nxt.game_id)]

    res = rules.scoring.units_per_point
    sides = {}
    for rid in [my_id] + ([opp_id] if opp_id is not None else []):
        r = rosters[rid]
        resolve_tipoffs(store, cfg.league_id, rid, cal, now)
        starters = [p if p and p != "0" else None for p in (r.starters or [])]
        sides[rid] = derive_side(rid, ctx.roster_names.get(rid), rules.starting_slots, starters, r.players or [],
                                 r.reserve or [], matchups[rid].starters_points if rid in matchups else None,
                                 ctx, store, cfg.league_id, week, now, res, manual=True)
    me = sides[my_id]
    opp = sides.get(opp_id) if opp_id is not None else None
    opp_manual = None
    if opp is not None:
        m = store.manual("opp_locked", as_of=now, key_prefix=f"{week}:{opp_id}")
        m = [e for e in m if e.key == f"{week}:{opp_id}"]
        if m:
            opp_manual = float(m[-1].value["total"])
            flags.append("opp_locked_manual")
    freshness = {
        "rosters": client.fetched.get(f"/league/{cfg.league_id}/rosters", now),
        "matchups": client.fetched.get(f"/league/{cfg.league_id}/matchups/{week}", now),
        "schedule": games["checked_at"].max(),
        "stats": client.store.checked_at("sleeper_app", f"/stats/nba/{season}/{week}", {"season_type": "regular"})
        or box["ingested_at"].max(),
        "players_map": client.store.checked_at("sleeper_v1", "/players/nba", None) or players_df["ingested_at"].max(),
    }
    if ctx.report_published_at is not None:
        freshness["injury_report"] = ctx.report_published_at
    for src, ts in freshness.items():
        if ts is not None and now - ts > MAX_AGE[src]:
            if src == "injury_report" and not cal.is_game_day(now.astimezone(cal.week_start.tzinfo).date()):
                continue
            flags.append(f"stale:{src}")
    unconf = [e for e in me.entries if e.locked_unknown]
    if unconf:
        flags.append(f"pending_lock_unconfirmed:{len(unconf)}")
        blocking.append("confirm lock status for: " + ", ".join(pinfo[e.player_id].name for e in unconf))
    unc = [e for e in me.entries if e.eligibility == "uncertain"]
    if unc:
        flags.append(f"eligibility_uncertain:{len(unc)}")
    blocking.extend(n for n in me.notes if n.startswith(("eligibility_uncertain:", "lock_state_uncertain:")))
    for side in (me, opp):
        if side is None:
            continue
        for pid in side.active:
            if pid not in pinfo or not pinfo[pid].positions:
                blocking.append(f"missing player positions: {pid}")
    relevant_teams = {pinfo[p].team for p in me.active if p in pinfo}
    for g in cal.games:
        if g.status == "scheduled" and set(g.teams()) & relevant_teams and g.tip_source in ("default", "espn_tbd"):
            blocking.append(f"missing confirmed tip time: {g.game_id}")
    # a finished game of a starter with no box score yet: that player's advice waits
    for i, pid in me.assignment.items():
        team = pinfo[pid].team if pid in pinfo else None
        for g in cal.team_games(team):
            if g.status == "awaiting_stats":
                flags.append(f"awaiting_box_score:{pid}")
    body = {"t": now.isoformat(), "season": season, "week": week, "me": _side_json(me),
            "opp": _side_json(opp) if opp else None, "flags": flags, "objective": objective,
            "league_id": cfg.league_id, "units_per_point": res, "blocking": blocking}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:6]
    sid = f"s_{now:%Y%m%dT%H%M%SZ}_{digest}"
    state = ValidatedState(now, season, week, cfg.league_id, me, opp, pinfo, res, sid, objective,  # type: ignore[arg-type]
                           versions={"code": code_version(), "tzdata": tzdata_version()},
                           freshness=freshness, flags=flags, blocking=blocking, opp_locked_manual=opp_manual)
    if save:
        store.save_snapshot(sid, now, body)
        for e in me.entries:
            store.upsert_entry({"entry_id": e.entry_id, "league_id": cfg.league_id, "roster_id": my_id,
                                "week": week, "player_id": e.player_id, "game_id": e.game_id,
                                "slot_at_tip": e.slot_at_tip, "fp_half": e.fp_half, "fp_source": e.fp_source,
                                "box_revision": None, "is_team_final": int(e.is_team_final),
                                "deadline_at": e.deadline_at.isoformat(), "status": "open"})
            if not any(d["entry_id"] == e.entry_id for d in store.deadlines()):
                store.upsert_deadline(e.entry_id, e.deadline_at,
                                      {"player": pinfo[e.player_id].name, "player_id": e.player_id,
                                       "game_id": e.game_id, "score": e.fp_half / res,
                                       "default": "confirm lock status and eligibility in Sleeper"}, None, [120, 30, 10])
    if blocking and strict:
        raise BlockingInputError("; ".join(blocking))
    return state, ctx


def _side_json(s: SideState | None) -> dict | None:
    if s is None:
        return None
    return {"roster_id": s.roster_id, "slots": list(s.slots), "assignment": s.assignment,
            "locked": {i: vars(l) for i, l in s.locked.items()},
            "entries": [vars(e) | {"deadline_at": e.deadline_at.isoformat()} for e in s.entries],
            "active": list(s.active), "reserve": list(s.reserve), "frozen": sorted(s.frozen), "notes": s.notes}
