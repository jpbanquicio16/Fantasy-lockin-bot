"""Fetch everything the system needs and rebuild the typed tables from the raw stores.

* Sleeper stats feed and schedule (past seasons once; this season's latest two weeks
  re-fetched, because stat corrections arrive 24-48 hours later).
* Sleeper player map (at most daily, as Sleeper asks); every version adds injury-status
  observations, which cannot be reconstructed later.
* ESPN tip times for the current and next week (and, with ``history_tips``, every date
  of the past seasons, for replay chronology).
* Every official injury-report PDF published since the last one archived; each is parsed,
  matched to Sleeper players by normalised name and team, and appended to a table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import polars as pl
import yaml

from lockin_core.config import LeagueConfig, Paths, et_date, now_utc
from lockin_core.data import tables as T
from lockin_core.ingest.espn import EspnClient
from lockin_core.ingest.nba_injury import InjuryReportClient, InjuryReportParseError, parse_pdf
from lockin_core.ingest.sleeper import MIN, ScheduleGame, SleeperClient


@dataclass
class IngestReport:
    season: int
    week: int
    lines: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def log(self, msg: str) -> None:
        self.lines.append(msg)


def completion_times(client: SleeperClient, season: int) -> dict[str, datetime]:
    """When each game was first seen complete, from the stored schedule versions."""
    out: dict[str, datetime] = {}
    paths = client.store.versions("sleeper_app", f"/schedule/nba/regular/{season}", None)
    if len(paths) <= 1:
        return out
    for p in paths:
        rec = client.store.read(p)
        for g in rec.body or []:
            if g.get("status") == "complete" and g["game_id"] not in out:
                out[g["game_id"]] = rec.fetched_at
    first = client.store.read(paths[0])
    for g in first.body or []:
        if g.get("status") == "complete":
            out.pop(g["game_id"], None)
    return out


def tip_dates(games: pl.DataFrame, season: int, current_season: int, current_week: int,
              history_tips: bool) -> list[date]:
    if season == current_season:
        sub = games.filter(pl.col("week").is_between(max(1, current_week - 1), current_week + 1))
    elif history_tips:
        sub = games
    else:
        return []
    return sorted(set(sub["date"].to_list()))


def fetch_tips(espn: EspnClient, dates: list[date], today: date, report: IngestReport) -> list:
    out = []
    failed = 0
    for d in dates:
        age: float | None
        if d < today - timedelta(days=2):
            age = None  # finished dates never change
        elif d <= today + timedelta(days=1):
            age = 10 * MIN
        else:
            age = 6 * 3600.0
        try:
            out.extend(espn.scoreboard(d, age))
        except Exception as e:  # tip times degrade to defaults, flagged per game
            failed += 1
            report.issues.append(f"ESPN scoreboard {d}: {e}")
    if failed:
        report.log(f"ESPN: {failed} of {len(dates)} dates failed")
    return out


def ingest_season(client: SleeperClient, espn: EspnClient, paths: Paths, season: int, current_season: int,
                  current_week: int, report: IngestReport, refetch_weeks: bool = False,
                  history_tips: bool = False) -> None:
    sched: list[ScheduleGame] = client.schedule(season, max_age_s=None if season < current_season else 5 * MIN)
    checked = client.fetched[f"/schedule/nba/regular/{season}"]
    games = T.games_frame(sched, season, checked)
    old = T.read(paths, "games", season)
    dates = tip_dates(games, season, current_season, current_week, history_tips)
    espn_games = fetch_tips(espn, dates, et_date(now_utc()), report) if dates else []
    games = T.with_tip_times(games, espn_games, now_utc() if dates else None)
    if old is not None and "tip_at" in old.columns:  # keep tips fetched earlier for other dates
        prev = old.filter(pl.col("tip_source") != "default").select("game_id", "tip_at", "tip_source", "espn_state")
        games = (games.join(prev, on="game_id", how="left", suffix="_old")
                 .with_columns(*[pl.when(pl.col("tip_source") == "default").then(pl.col(f"{c}_old"))
                                 .otherwise(pl.col(c)).fill_null(pl.col(c)).alias(c)
                                 for c in ("tip_at", "tip_source", "espn_state")])
                 .drop("tip_at_old", "tip_source_old", "espn_state_old"))
    T.write(games, paths, "games", season)
    last_week = current_week if season == current_season else int(games["week"].max())
    versions, totals = [], {}
    for w in range(1, last_week + 1):
        age: float | None = 0.0 if season == current_season and (refetch_weeks or w >= current_week - 1) else None
        rows = client.week_stats(season, w, age)
        totals.update(T.team_total_rows(rows))
        for rec in client.week_stats_versions(season, w):
            versions.append((T.stat_rows_from_record(rec), rec.fetched_at))
    box = T.build_box(versions, season, games, completion_times(client, season))
    report.issues.extend(f"season {season}: {i}" for i in T.validate_box(box, totals))
    T.write(box, paths, "box", season)
    latest = T.latest_revision(box)
    n_played = latest.filter(pl.col("played") == "played").height
    n_default = games.filter(pl.col("week").is_between(current_week, current_week + 1)
                             & (pl.col("tip_source") == "default")).height if season == current_season else 0
    report.log(f"season {season}: {games.height} games ({(games['status'] == 'complete').sum()} complete), "
               f"{n_played} player-games played, {latest.height - n_played} dnp"
               + (f"; tip times for {len(dates)} dates ({n_default} games this/next week without one)"
                  if dates else ""))


def ingest_players(client: SleeperClient, paths: Paths, report: IngestReport) -> None:
    client.players()
    versions = client.store.versions("sleeper_v1", "/players/nba", None)
    injuries = None
    for p in versions:
        rec = client.store.read(p)
        injuries = T.merge_injuries(injuries, T.injury_observations(rec.body, rec.fetched_at))
    latest = client.store.read(versions[-1])
    players = T.players_frame(latest.body, latest.fetched_at)
    T.write(players, paths, "players")
    T.write(injuries, paths, "injuries")
    report.log(f"players: {players.filter(pl.col('team').is_not_null()).height} on NBA teams; "
               f"{len(versions)} player-map snapshots, {injuries.height} status observations")


def load_overrides(paths: Paths) -> dict[str, str]:
    p = paths.configs / "id_overrides.yaml"
    if not p.exists():
        return {}
    return {str(k): str(v) for k, v in (yaml.safe_load(p.read_text()) or {}).items()}


def ingest_injury_reports(paths: Paths, report: IngestReport, client: InjuryReportClient | None = None,
                          fetch: bool = True, lookback_hours: float = 4.0, now: datetime | None = None) -> int:
    client = client or InjuryReportClient(paths)
    if fetch:
        try:
            new = client.discover(now=now, lookback_hours=lookback_hours)
            report.log(f"injury reports: {len(new)} new PDF(s)")
        except Exception as e:
            report.issues.append(f"injury report fetch failed: {e}")
    players = T.read(paths, "players")
    if players is None:
        report.issues.append("injury reports not matched: no player map yet")
        return 0
    table = T.read(paths, "injury_reports")
    done = set(table["report_name"].unique().to_list()) if table is not None else set()
    overrides = load_overrides(paths)
    frames, unmatched = [], set()
    for p in client.archive.all():
        if p.name in done:
            continue
        meta = client.archive.meta(p)
        try:
            rows = parse_pdf(p.read_bytes())
        except InjuryReportParseError as e:
            report.issues.append(f"{p.name}: parse failed ({e}); previous report and Sleeper status stand")
            continue
        matches = T.match_report_players(rows, players, overrides)
        unmatched |= {r.player_raw for r, (pid, _) in zip(rows, matches) if pid is None and r.player_raw}
        frames.append(T.injury_report_frame(
            p.name, datetime.fromisoformat(meta["published_at"]),
            datetime.fromisoformat(meta["fetched_at"]) if meta.get("fetched_at") else None, rows, matches))
    if frames:
        new = pl.concat(frames, how="vertical_relaxed")
        table = new if table is None else pl.concat([table, new], how="vertical_relaxed")
        T.write(table.sort("published_at", "team", "player_raw"), paths, "injury_reports")
    if unmatched:
        report.issues.append(f"injury reports: {len(unmatched)} names not matched to a Sleeper player "
                             f"(add to configs/id_overrides.yaml if rostered): {sorted(unmatched)[:8]}")
    report.log(f"injury reports: {len(frames)} parsed this run, "
               f"{0 if table is None else table['report_name'].n_unique()} on file")
    return len(frames)


def ingest(league: LeagueConfig, paths: Paths | None = None, client: SleeperClient | None = None,
           espn: EspnClient | None = None, history: bool = True, refetch_weeks: bool = False,
           history_tips: bool = False, injury: bool = True, players: bool = True) -> IngestReport:
    paths = (paths or Paths()).ensure()
    client = client or SleeperClient(paths=paths)
    espn = espn or EspnClient(paths=paths)
    state = client.state(max_age_s=10 * MIN)
    season, week = int(state.season), int(state.week)
    report = IngestReport(season, week)
    seasons = [*(s for s in league.history_seasons if history and s < season), season]
    for s in seasons:
        ingest_season(client, espn, paths, s, season, week, report, refetch_weeks, history_tips)
    if players:
        ingest_players(client, paths, report)
    if injury:
        ingest_injury_reports(paths, report)
    return report


def refresh_live(paths: Paths, client: SleeperClient, espn: EspnClient, season: int, week: int,
                 report: IngestReport | None = None) -> IngestReport:
    """The cheap refresh the event loop runs while games are live: this season's schedule,
    today's and tomorrow's tip times and status, and this week's stats."""
    report = report or IngestReport(season, week)
    ingest_season(client, espn, paths, season, season, week, report)
    return report
