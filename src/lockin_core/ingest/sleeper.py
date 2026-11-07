"""Sleeper: league state (documented v1 API) plus game data (undocumented stats feed).

Read-only and unauthenticated. Two hosts:

* ``api.sleeper.app/v1`` - documented: state, league, users, rosters, matchups,
  transactions, players, trending adds.
* ``api.sleeper.com`` - undocumented, used by Sleeper's own app: per-game box scores
  for every player in a week (Sportradar, including technical and flagrant fouls and
  explicit did-not-play rows) and the season schedule with game status.

Responses are parsed into Pydantic models so a renamed field fails loudly, and every
response is kept in the raw store. Sleeper asks for under 1,000 calls a minute and the
5 MB players file at most daily.
"""

from __future__ import annotations

import datetime as dt
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from lockin_core.config import Paths
from lockin_core.ingest.raw import HttpClient, RawRecord, RawStore

MIN = 60.0
HOUR = 3600.0
DAY = 86400.0


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class SleeperState(_Loose):
    week: int
    season: str
    season_type: str
    leg: int | None = None
    season_start_date: str | None = None


class League(_Loose):
    league_id: str
    name: str = ""
    season: str
    sport: str = "nba"
    status: str | None = None
    total_rosters: int
    roster_positions: list[str]
    scoring_settings: dict[str, float] = Field(default_factory=dict)
    settings: dict[str, Any] = Field(default_factory=dict)
    previous_league_id: str | None = None


class User(_Loose):
    user_id: str
    display_name: str | None = None


class Roster(_Loose):
    roster_id: int
    owner_id: str | None = None
    players: list[str] | None = None
    starters: list[str] | None = None
    reserve: list[str] | None = None
    settings: dict[str, Any] = Field(default_factory=dict)


class Matchup(_Loose):
    roster_id: int
    matchup_id: int | None = None
    points: float | None = None
    starters: list[str] | None = None
    starters_points: list[float] | None = None
    players: list[str] | None = None
    players_points: dict[str, float] | None = None


class Transaction(_Loose):
    transaction_id: str
    type: str
    status: str
    created: int
    status_updated: int | None = None
    leg: int | None = None
    adds: dict[str, int] | None = None
    drops: dict[str, int] | None = None
    roster_ids: list[int] = Field(default_factory=list)
    settings: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class TeamSide(_Loose):
    team: str
    starters: list[str] | None = None
    points: int | None = None


class ScheduleGame(_Loose):
    game_id: str
    date: dt.date
    week: int
    status: str  # pre_game | in_game | complete | postponed | canceled
    home: TeamSide
    away: TeamSide


class StatRow(_Loose):
    player_id: str
    game_id: str
    date: dt.date | None = None
    week: int | None = None
    team: str | None = None
    opponent: str | None = None
    stats: dict[str, float] = Field(default_factory=dict)
    last_modified: int | None = None
    updated_at: int | None = None


class SleeperClient:
    def __init__(self, store: RawStore | None = None, paths: Paths | None = None,
                 v1: HttpClient | None = None, app: HttpClient | None = None):
        paths = (paths or Paths()).ensure()
        self.store = store or RawStore(paths.raw)
        self.v1 = v1 or HttpClient("https://api.sleeper.app/v1", min_interval=0.08)
        self.app = app or HttpClient("https://api.sleeper.com", min_interval=0.25)
        self.fetched: dict[str, datetime] = {}  # endpoint -> when its content was checked

    def _get(self, host: str, path: str, params: dict | None, max_age_s: float | None) -> Any:
        client = self.v1 if host == "v1" else self.app
        rec: RawRecord = self.store.fetch(
            f"sleeper_{host}", path, params, lambda: client.get_json(path, params), max_age_s
        )
        checked = self.store.checked_at(f"sleeper_{host}", path, params) or rec.fetched_at
        self.fetched[path] = checked
        return rec.body

    # --- documented ------------------------------------------------------------------
    def state(self, max_age_s: float = HOUR) -> SleeperState:
        return SleeperState(**self._get("v1", "/state/nba", None, max_age_s))

    def user(self, username: str) -> User | None:
        body = self._get("v1", f"/user/{username}", None, DAY)
        return User(**body) if body else None

    def league(self, league_id: str, max_age_s: float = HOUR) -> League:
        return League(**self._get("v1", f"/league/{league_id}", None, max_age_s))

    def users(self, league_id: str) -> list[User]:
        return [User(**u) for u in self._get("v1", f"/league/{league_id}/users", None, DAY) or []]

    def rosters(self, league_id: str, max_age_s: float = MIN) -> list[Roster]:
        return [Roster(**r) for r in self._get("v1", f"/league/{league_id}/rosters", None, max_age_s) or []]

    def matchups(self, league_id: str, week: int, max_age_s: float = MIN) -> list[Matchup]:
        body = self._get("v1", f"/league/{league_id}/matchups/{week}", None, max_age_s)
        return [Matchup(**m) for m in body or []]

    def transactions(self, league_id: str, week: int, max_age_s: float = 5 * MIN) -> list[Transaction]:
        """``round`` is the Sleeper week (verified: each transaction's ``leg`` equals it)."""
        body = self._get("v1", f"/league/{league_id}/transactions/{week}", None, max_age_s)
        return [Transaction(**t) for t in body or []]

    def players(self, max_age_s: float = DAY) -> dict[str, dict]:
        return self._get("v1", "/players/nba", None, max_age_s)

    def trending_adds(self, hours: int = 24, limit: int = 50) -> list[dict]:
        return self._get("v1", "/players/nba/trending/add",
                         {"lookback_hours": hours, "limit": limit}, HOUR) or []

    # --- undocumented (Sleeper's own app) ---------------------------------------------
    def schedule(self, season: int, max_age_s: float | None = 10 * MIN) -> list[ScheduleGame]:
        body = self._get("app", f"/schedule/nba/regular/{season}", None, max_age_s)
        return [ScheduleGame(**g) for g in body or []]

    def week_stats(self, season: int, week: int, max_age_s: float | None) -> list[StatRow]:
        """Every player-game of one Sleeper week, including empty did-not-play rows."""
        body = self._get("app", f"/stats/nba/{season}/{week}", {"season_type": "regular"}, max_age_s)
        return [StatRow(**r) for r in body or [] if r.get("game_id") and r["game_id"] != "season"]

    def week_stats_versions(self, season: int, week: int) -> list[RawRecord]:
        """Every stored revision of a week's stats (stat corrections)."""
        paths = self.store.versions("sleeper_app", f"/stats/nba/{season}/{week}",
                                    {"season_type": "regular"})
        return [self.store.read(p) for p in paths]
