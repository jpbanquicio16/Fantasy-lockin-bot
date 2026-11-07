"""Raw Sleeper responses -> typed, validated, bitemporal Parquet tables.

Tables (``data/tables``)
------------------------
games_{season}     one row per game: date (US Eastern), Sleeper week, teams, status, starters
box_{season}       one row per (game, player, revision); ``played`` is played | dnp
injuries           append-only injury observations from every player-map snapshot
players            the latest player map: team, eligibility, injury status

Every source row carries ``published_at`` (when Sleeper says it changed; nullable) and
``ingested_at`` (when this system first saw it). ``knowledge_at`` is the earlier-known
of the two rules in the design: publication time when present, else ingestion time.

A zero is the most dangerous wrong value here, so ``played`` is explicit: a player-game
is ``dnp`` only when the game is complete and Sleeper lists the player with an empty
stat line; games that are not complete are never written.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

import polars as pl

from lockin_core.config import ET, Paths, UTC, from_ms
from lockin_core.ingest.raw import RawRecord
from lockin_core.ingest.sleeper import ScheduleGame, StatRow

STAT_MAP = {
    "pts": "pts", "reb": "reb", "ast": "ast", "stl": "stl", "blk": "blk", "to": "tov",
    "tpm": "fg3m", "tpa": "fg3a", "fgm": "fgm", "fga": "fga", "ftm": "ftm", "fta": "fta",
    "oreb": "oreb", "dreb": "dreb", "pf": "pf", "tf": "tf", "ff": "ff",
}
COUNT_COLS = list(STAT_MAP.values())
BOX_SCHEMA = {
    "season": pl.Int32, "week": pl.Int32, "date": pl.Date, "game_id": pl.Utf8,
    "player_id": pl.Utf8, "team": pl.Utf8, "opp": pl.Utf8, "home": pl.Boolean,
    "played": pl.Utf8, "started": pl.Boolean, "min": pl.Float64,
    **{c: pl.Int32 for c in COUNT_COLS},
    "pts_std": pl.Float64, "revision": pl.Int64,
    "published_at": pl.Datetime("us", "UTC"), "ingested_at": pl.Datetime("us", "UTC"),
}


class ValidationError(RuntimeError):
    pass


# --- games -----------------------------------------------------------------------
def games_frame(games: list[ScheduleGame], season: int, checked_at: datetime) -> pl.DataFrame:
    rows = [
        {
            "season": season, "week": g.week, "date": g.date, "game_id": g.game_id,
            "home": g.home.team, "away": g.away.team, "status": g.status,
            "home_starters": g.home.starters or [], "away_starters": g.away.starters or [],
            "checked_at": checked_at,
        }
        for g in games
    ]
    df = pl.DataFrame(rows, schema={
        "season": pl.Int32, "week": pl.Int32, "date": pl.Date, "game_id": pl.Utf8,
        "home": pl.Utf8, "away": pl.Utf8, "status": pl.Utf8,
        "home_starters": pl.List(pl.Utf8), "away_starters": pl.List(pl.Utf8),
        "checked_at": pl.Datetime("us", "UTC"),
    })
    if df["game_id"].n_unique() != df.height:
        raise ValidationError(f"games {season}: duplicate game ids")
    return df.sort("date", "game_id")


def team_games(games: pl.DataFrame) -> pl.DataFrame:
    """One row per team per game (the long form the projections and simulator use)."""
    cols = ["season", "week", "date", "game_id", "status"]
    home = games.select(*cols, pl.col("home").alias("team"), pl.col("away").alias("opp"),
                        pl.lit(True).alias("is_home"))
    away = games.select(*cols, pl.col("away").alias("team"), pl.col("home").alias("opp"),
                        pl.lit(False).alias("is_home"))
    return pl.concat([home, away]).sort("date", "team")


# --- box scores ------------------------------------------------------------------
def box_rows(rows: Iterable[StatRow], season: int, games: pl.DataFrame,
             ingested_at: datetime, completed_at: dict[str, datetime] | None = None) -> list[dict]:
    """Rows from one fetch of a week. Skips team totals, games that are not complete, and
    rows fetched before the schedule first showed the game complete (partial lines)."""
    completed_at = completed_at or {}
    status = dict(zip(games["game_id"], games["status"]))
    home_of = dict(zip(games["game_id"], games["home"]))
    starters: dict[str, set[str]] = {}
    for gid, hs, as_ in zip(games["game_id"], games["home_starters"].to_list(),
                            games["away_starters"].to_list()):
        starters[gid] = set(hs or []) | set(as_ or [])
    out = []
    for r in rows:
        if r.player_id.startswith("TEAM_") or status.get(r.game_id) != "complete":
            continue
        done = completed_at.get(r.game_id)
        if done is not None and ingested_at < done:
            continue
        s = r.stats
        # Older seasons carry seconds played ("sp") and no "min".
        minutes = float(s["min"]) if s.get("min") is not None else float(s.get("sp") or 0.0) / 60
        minutes = round(minutes, 2)
        played = "played" if s and minutes > 0 else "dnp"
        ts = r.last_modified or r.updated_at
        rec = {
            "season": season, "week": r.week, "date": r.date, "game_id": r.game_id,
            "player_id": r.player_id, "team": r.team, "opp": r.opponent,
            "home": home_of.get(r.game_id) == r.team if r.team else None,
            "played": played,
            "started": (r.player_id in starters[r.game_id]) if starters.get(r.game_id) else None,
            "min": minutes,
            # Sleeper's feed omits zero counts, so an absent key is a recorded zero.
            **{dst: int(s.get(src, 0) or 0) for src, dst in STAT_MAP.items()},
            "pts_std": float(s["pts_std"]) if "pts_std" in s else (0.0 if played == "played" else None),
            "revision": int(ts or 0),
            "published_at": from_ms(ts),
            "ingested_at": ingested_at,
        }
        out.append(rec)
    return out


def team_total_rows(rows: Iterable[StatRow]) -> dict[tuple[str, str], int]:
    return {(r.game_id, r.player_id.removeprefix("TEAM_")): int(r.stats.get("pts", 0))
            for r in rows if r.player_id.startswith("TEAM_") and r.stats}


def build_box(versions: Iterable[tuple[list[StatRow], datetime]], season: int,
              games: pl.DataFrame, completed_at: dict[str, datetime] | None = None) -> pl.DataFrame:
    """All revisions of every week fetch; identical revisions keep their first sighting."""
    rows: list[dict] = []
    for stat_rows, ingested_at in versions:
        rows.extend(box_rows(stat_rows, season, games, ingested_at, completed_at))
    if not rows:
        return pl.DataFrame(schema=BOX_SCHEMA)
    df = pl.DataFrame(rows, schema=BOX_SCHEMA)
    return (
        df.sort("ingested_at")
        .unique(["game_id", "player_id", "revision", *COUNT_COLS, "played"], keep="first",
                maintain_order=True)
        .sort("date", "game_id", "player_id", "revision")
    )


def validate_box(df: pl.DataFrame, team_totals: dict[tuple[str, str], int] | None = None) -> list[str]:
    """Hard failures raise; soft issues (team totals off after a correction) are returned."""
    if df.is_empty():
        return []
    key = pl.struct("game_id", "player_id", "revision", "ingested_at")
    if df.select(key.is_duplicated().any()).item():
        raise ValidationError("box: conflicting rows for one (game, player, revision)")
    bad =df.filter((pl.col("min") < 0) | (pl.col("min") > 70))
    if bad.height:
        raise ValidationError(f"box: minutes out of range in {bad.height} rows")
    neg = df.filter(pl.any_horizontal([pl.col(c) < 0 for c in COUNT_COLS]))
    if neg.height:
        raise ValidationError(f"box: negative counts in {neg.height} rows")
    if df.filter(pl.col("fg3m") * 3 > pl.col("pts")).height:
        raise ValidationError("box: more threes than points allow")
    if df.filter((pl.col("played") == "dnp") & (pl.col("pts") > 0)).height:
        raise ValidationError("box: a dnp row has statistics")
    issues = []
    if team_totals:
        latest = latest_revision(df).filter(pl.col("played") == "played")
        sums = latest.group_by("game_id", "team").agg(pl.col("pts").sum())
        off = [(g, t, p, team_totals[(g, t)]) for g, t, p in sums.iter_rows()
               if (g, t) in team_totals and team_totals[(g, t)] != p]
        if off:
            issues.append(f"{len(off)} team-games where player points do not sum to the team total "
                          f"(e.g. {off[0]}); usually a pending stat correction")
    return issues


def knowledge_at() -> pl.Expr:
    return pl.coalesce("published_at", "ingested_at")


def latest_revision(box: pl.DataFrame, as_of: datetime | None = None) -> pl.DataFrame:
    """The newest revision of each player-game known at ``as_of`` (None = now)."""
    df = box if as_of is None else box.filter(knowledge_at() <= as_of)
    return df.sort("revision", "ingested_at").unique(["game_id", "player_id"], keep="last").sort(
        "date", "game_id", "player_id")


# --- players and injuries --------------------------------------------------------
INJURY_SCHEMA = {
    "player_id": pl.Utf8, "injury_status": pl.Utf8, "body_part": pl.Utf8, "notes": pl.Utf8,
    "published_at": pl.Datetime("us", "UTC"), "ingested_at": pl.Datetime("us", "UTC"),
}
PLAYER_SCHEMA = {
    "player_id": pl.Utf8, "name": pl.Utf8, "team": pl.Utf8, "positions": pl.List(pl.Utf8),
    "status": pl.Utf8, "injury_status": pl.Utf8, "years_exp": pl.Int32,
    "birth_date": pl.Utf8, "search_rank": pl.Int64, "news_updated": pl.Datetime("us", "UTC"),
    "ingested_at": pl.Datetime("us", "UTC"),
}
FANTASY_POSITIONS = {"PG", "SG", "SF", "PF", "C"}


def players_frame(players: dict[str, dict], ingested_at: datetime) -> pl.DataFrame:
    rows = []
    for pid, p in players.items():
        pos = [x for x in (p.get("fantasy_positions") or []) if x in FANTASY_POSITIONS]
        if not pos:
            continue  # team defences and staff
        rows.append({
            "player_id": pid, "name": p.get("full_name") or f"{p.get('first_name')} {p.get('last_name')}",
            "team": p.get("team"), "positions": pos, "status": p.get("status"),
            "injury_status": p.get("injury_status"), "years_exp": p.get("years_exp"),
            "birth_date": p.get("birth_date"), "search_rank": p.get("search_rank"),
            "news_updated": from_ms(p.get("news_updated")), "ingested_at": ingested_at,
        })
    return pl.DataFrame(rows, schema=PLAYER_SCHEMA)


def injury_observations(players: dict[str, dict], ingested_at: datetime) -> pl.DataFrame:
    """Every rostered NBA player's status at this snapshot, including "healthy" (null)."""
    rows = [
        {"player_id": pid, "injury_status": p.get("injury_status"),
         "body_part": p.get("injury_body_part"), "notes": p.get("injury_notes"),
         "published_at": from_ms(p.get("news_updated")), "ingested_at": ingested_at}
        for pid, p in players.items() if p.get("team")
    ]
    return pl.DataFrame(rows, schema=INJURY_SCHEMA)


def merge_injuries(existing: pl.DataFrame | None, new: pl.DataFrame) -> pl.DataFrame:
    """Append-only: a row is new when the player's status or its publication time changed."""
    df = new if existing is None else pl.concat([existing, new])
    return df.sort("ingested_at").unique(
        ["player_id", "injury_status", "body_part", "published_at"], keep="first", maintain_order=True
    )


def status_as_of(injuries: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Each player's latest known status at ``as_of`` (knowledge = ingestion time here:
    ``news_updated`` covers any news, not only the status, so it cannot date a status)."""
    return (
        injuries.filter(pl.col("ingested_at") <= as_of)
        .sort("ingested_at")
        .unique("player_id", keep="last")
        .select("player_id", "injury_status", "body_part", "published_at", "ingested_at")
    )


# --- storage ---------------------------------------------------------------------
def table_path(paths: Paths, name: str, season: int | None = None) -> Path:
    return paths.tables / (f"{name}_{season}.parquet" if season is not None else f"{name}.parquet")


def write(df: pl.DataFrame, paths: Paths, name: str, season: int | None = None) -> Path:
    p = table_path(paths, name, season)
    p.parent.mkdir(parents=True, exist_ok=True)
    import uuid
    tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
    df.write_parquet(tmp)
    if name in ("games", "players"):
        from lockin_core.config import now_utc
        archive = paths.tables / "history" / p.stem
        archive.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(tmp, archive / (now_utc().strftime("%Y%m%dT%H%M%S%fZ") + ".parquet"))
    tmp.replace(p)
    return p


def read(paths: Paths, name: str, season: int | None = None) -> pl.DataFrame | None:
    p = table_path(paths, name, season)
    return pl.read_parquet(p) if p.exists() else None


def read_seasons(paths: Paths, name: str, seasons: Iterable[int]) -> pl.DataFrame:
    frames = [f for s in seasons if (f := read(paths, name, s)) is not None]
    if not frames:
        raise FileNotFoundError(f"no {name} tables for seasons {list(seasons)}; run `lockin ingest`")
    return pl.concat(frames, how="vertical_relaxed")


def stat_rows_from_record(rec: RawRecord) -> list[StatRow]:
    return [StatRow(**r) for r in rec.body or [] if r.get("game_id") and r["game_id"] != "season"]


def ensure_utc(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


# --- tip times (ESPN) ------------------------------------------------------------
DEFAULT_TIP_ET = (19, 30)  # used only when ESPN has no matching game; flagged as tip_source=default


def with_tip_times(games: pl.DataFrame, espn_games: list, checked_at: datetime | None) -> pl.DataFrame:
    """Add ``tip_at`` (scheduled tip, UTC), ``tip_source`` and ``espn_state`` to Sleeper's games,
    matching ESPN games on (ET date of the tip, home, away)."""
    by_key = {}
    for e in espn_games:
        by_key[(e.tip_at.astimezone(ET).date(), e.home, e.away)] = e
    tips, src, state = [], [], []
    for d, h, a in zip(games["date"].to_list(), games["home"].to_list(), games["away"].to_list()):
        e = by_key.get((d, h, a))
        if e is not None:
            tips.append(e.tip_at)
            src.append("espn" if e.time_valid else "espn_tbd")
            state.append(e.state)
        else:
            tips.append(datetime(d.year, d.month, d.day, *DEFAULT_TIP_ET, tzinfo=ET).astimezone(UTC))
            src.append("default")
            state.append(None)
    return games.with_columns(
        pl.Series("tip_at", tips, dtype=pl.Datetime("us", "UTC")),
        pl.Series("tip_source", src, dtype=pl.Utf8),
        pl.Series("espn_state", state, dtype=pl.Utf8),
        pl.lit(checked_at, dtype=pl.Datetime("us", "UTC")).alias("tips_checked_at"),
    )


# --- official injury reports -----------------------------------------------------
REPORT_SCHEMA = {
    "report_name": pl.Utf8, "published_at": pl.Datetime("us", "UTC"), "fetched_at": pl.Datetime("us", "UTC"),
    "game_date": pl.Date, "game_time": pl.Utf8, "matchup": pl.Utf8, "team": pl.Utf8,
    "player_raw": pl.Utf8, "player_key": pl.Utf8, "player_id": pl.Utf8, "match": pl.Utf8,
    "status": pl.Utf8, "reason": pl.Utf8, "not_submitted": pl.Boolean,
}


def match_report_players(rows: list, players: pl.DataFrame, overrides: dict[str, str] | None = None
                         ) -> list[tuple[str | None, str]]:
    """(player_id, how) per report row: ``override`` | ``name_team`` | ``name_only`` (unique name,
    flagged) | ``none``. Names are normalised (accents, punctuation, suffixes dropped)."""
    from lockin_core.ingest.nba_injury import sleeper_name_key

    overrides = overrides or {}
    by_name_team: dict[tuple[str, str], list[str]] = {}
    by_name: dict[str, list[str]] = {}
    for pid, name, team in zip(players["player_id"].to_list(), players["name"].to_list(),
                               players["team"].to_list()):
        k = sleeper_name_key(name or "")
        by_name.setdefault(k, []).append(pid)
        if team:
            by_name_team.setdefault((k, team), []).append(pid)
    out = []
    for r in rows:
        if r.not_submitted or not r.player_key:
            out.append((None, "none"))
        elif r.player_raw in overrides:
            out.append((overrides[r.player_raw], "override"))
        elif len(by_name_team.get((r.player_key, r.team), [])) == 1:
            out.append((by_name_team[(r.player_key, r.team)][0], "name_team"))
        elif len(by_name.get(r.player_key, [])) == 1:
            out.append((by_name[r.player_key][0], "name_only"))
        else:
            out.append((None, "none"))
    return out


def injury_report_frame(name: str, published_at: datetime, fetched_at: datetime | None, rows: list,
                        matches: list[tuple[str | None, str]]) -> pl.DataFrame:
    recs = [{
        "report_name": name, "published_at": published_at, "fetched_at": fetched_at,
        "game_date": r.game_date, "game_time": r.game_time, "matchup": r.matchup, "team": r.team,
        "player_raw": r.player_raw, "player_key": r.player_key, "player_id": pid, "match": how,
        "status": r.status, "reason": r.reason, "not_submitted": r.not_submitted,
    } for r, (pid, how) in zip(rows, matches)]
    return pl.DataFrame(recs, schema=REPORT_SCHEMA)


def report_status_as_of(reports: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Latest published report at ``as_of`` (knowledge time = publication instant): one row per
    (team, game_date) block and per listed player; teams marked not submitted carry their
    previous report's rows (design section 9: "carry each player's previous status")."""
    known = reports.filter(pl.col("published_at") <= as_of)
    if known.is_empty():
        return known
    latest = known["published_at"].max()
    cur = known.filter(pl.col("published_at") == latest)
    pending = cur.filter(pl.col("not_submitted")).select("team", "game_date").unique()
    keep = cur.filter(~pl.col("not_submitted"))
    if pending.height:
        prev = (known.filter((pl.col("published_at") < latest) & ~pl.col("not_submitted"))
                .join(pending, on=["team", "game_date"], how="semi"))
        if prev.height:
            last = prev.group_by("team", "game_date").agg(pl.col("published_at").max().alias("_p"))
            prev = prev.join(last, on=["team", "game_date"]).filter(pl.col("published_at") == pl.col("_p")).drop("_p")
            keep = pl.concat([keep, prev], how="vertical_relaxed")
    return keep


def read_as_of(paths: Paths, name: str, as_of: datetime, season: int | None = None) -> pl.DataFrame | None:
    """An immutable schedule/player-map snapshot known at the requested instant."""
    prefix = table_path(paths, name, season).stem
    cutoff = as_of.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    versions = sorted(p for p in (paths.tables / "history" / prefix).glob("*.parquet") if p.stem <= cutoff)
    return pl.read_parquet(versions[-1]) if versions else None
