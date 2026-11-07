"""The matchup week's schedule: tip times, game status, each team's final game, deadlines.

Rules encoded (design section 1):

* A game belongs to the week of its **scheduled US Eastern date** (``week_assignment``);
  Sleeper's own ``week`` field is used and checked against the Monday-Sunday rule.
* A completed score must be locked before **his team's next scheduled tip** (whether or not
  he plays it, ``deadline_on_inactive: team_tip``), minus a safety margin
  (``deadline_margin_minutes``). A score from his team's **final game of the week** stays
  until week end (``final_game_pending_mode: auto_until_week_end``).
* Postponed or cancelled games leave the week and every final-game map and deadline is
  recomputed from the new schedule (``postponement_rule``).
* All instants are UTC; Adelaide time is only for display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from lockin_core.config import ET, UTC

GAME_MINUTES = 150  # tip to final when no final time is observed (Assumption)
WEEK_END_AFTER_LAST_TIP = timedelta(hours=4)


@dataclass(frozen=True)
class Game:
    game_id: str
    season: int
    week: int
    date: date  # scheduled ET date
    home: str
    away: str
    tip_at: datetime  # scheduled tip, UTC
    tip_source: str  # espn | espn_tbd | default
    status: str  # scheduled | live | awaiting_stats | final | postponed
    end_at: datetime  # observed or expected final, UTC

    def teams(self) -> tuple[str, str]:
        return (self.home, self.away)

    @property
    def tipped(self) -> bool:
        return self.status in ("live", "awaiting_stats", "final")


def combined_status(sleeper: str, espn: str | None, tip_at: datetime, now: datetime) -> str:
    """Sleeper is the authority for final (its box scores are what it scores); ESPN or the clock
    says a game has started or ended before Sleeper's schedule catches up."""
    if sleeper in ("postponed", "canceled"):
        return "postponed"
    if sleeper == "complete":
        return "final"
    if espn == "postponed":
        return "postponed"
    if espn == "final":
        return "awaiting_stats"
    if sleeper == "in_game" or espn == "live" or tip_at <= now:
        return "live"
    return "scheduled"


@dataclass
class WeekCalendar:
    season: int
    week: int
    games: list[Game]  # this week's games, by tip (postponed excluded)
    week_start: datetime
    week_end: datetime
    margin: timedelta = timedelta(minutes=10)
    postponed: list[Game] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    # --- lookups -------------------------------------------------------------------
    def team_games(self, team: str | None) -> list[Game]:
        return [g for g in self.games if team in g.teams()] if team else []

    def team_final(self, team: str | None) -> Game | None:
        tg = self.team_games(team)
        return tg[-1] if tg else None

    def is_team_final(self, team: str, game_id: str) -> bool:
        f = self.team_final(team)
        return f is not None and f.game_id == game_id

    def game(self, game_id: str) -> Game | None:
        return next((g for g in self.games if g.game_id == game_id), None)

    def next_team_game(self, team: str, after_game_id: str) -> Game | None:
        tg = self.team_games(team)
        ids = [g.game_id for g in tg]
        if after_game_id not in ids:
            return None
        i = ids.index(after_game_id)
        return tg[i + 1] if i + 1 < len(tg) else None

    def deadline(self, team: str, game_id: str) -> tuple[datetime, bool]:
        """(lock deadline of the score from ``game_id``, whether it is his team's final game)."""
        nxt = self.next_team_game(team, game_id)
        if nxt is None:
            return self.week_end, True
        return nxt.tip_at - self.margin, False

    def remaining_team_games(self, team: str | None, now: datetime, include_live: bool = True) -> list[Game]:
        out = []
        for g in self.team_games(team):
            if g.status == "scheduled" or (include_live and g.status == "live"):
                out.append(g)
        return out

    def tip_batches(self, after: datetime, teams: set[str] | None = None) -> list[tuple[datetime, list[Game]]]:
        """Distinct scheduled tip instants after ``after`` (optionally only games involving
        ``teams``): all players tipping at one minute are decided jointly (one epoch)."""
        by: dict[datetime, list[Game]] = {}
        for g in self.games:
            if g.status != "scheduled" or g.tip_at <= after:
                continue
            if teams is not None and not (set(g.teams()) & teams):
                continue
            by.setdefault(g.tip_at, []).append(g)
        return sorted(by.items())

    def live_games(self) -> list[Game]:
        return [g for g in self.games if g.status in ("live", "awaiting_stats")]

    def is_game_day(self, d: date) -> bool:
        return any(g.date == d for g in self.games)


def week_bounds(dates: list[date]) -> tuple[datetime, datetime]:
    first = min(dates)
    monday = first - timedelta(days=first.weekday())
    start = datetime.combine(monday, time(0), tzinfo=ET).astimezone(UTC)
    end = datetime.combine(monday + timedelta(days=7), time(0), tzinfo=ET).astimezone(UTC)
    return start, end


def build_calendar(games: pl.DataFrame, season: int, week: int, now: datetime,
                   margin_minutes: int = 10, game_minutes: float = GAME_MINUTES,
                   completion: dict[str, datetime] | None = None) -> WeekCalendar:
    completion = completion or {}
    wk = games.filter(pl.col("week") == week)
    if wk.is_empty():
        raise ValueError(f"no games in week {week} of season {season}")
    issues = []
    start, end = week_bounds(wk["date"].to_list())
    out, postponed = [], []
    has_tips = "tip_at" in wk.columns
    for r in wk.iter_rows(named=True):
        if has_tips and r["tip_at"] is not None:
            tip = r["tip_at"]
            src = r["tip_source"]
        else:
            d = r["date"]
            tip = datetime(d.year, d.month, d.day, 19, 30, tzinfo=ET).astimezone(UTC)
            src = "default"
        status = combined_status(r["status"], r.get("espn_state"), tip, now)
        done = completion.get(r["game_id"])
        end_at = done if (status == "final" and done) else tip + timedelta(minutes=game_minutes)
        if status in ("live", "awaiting_stats") and end_at <= now:
            end_at = now + timedelta(minutes=5)  # still being played or scored
        g = Game(r["game_id"], season, int(r["week"]), r["date"], r["home"], r["away"], tip, src, status, end_at)
        if not (start.astimezone(ET).date() <= g.date < end.astimezone(ET).date()):
            issues.append(f"game {g.game_id} on {g.date} is outside the Monday-Sunday bounds of week {week}")
        (postponed if status == "postponed" else out).append(g)
    out.sort(key=lambda g: (g.tip_at, g.game_id))
    if out:
        end = max(end, max(g.tip_at for g in out) + WEEK_END_AFTER_LAST_TIP)
    defaults = [g for g in out if g.tip_source == "default" and g.status == "scheduled"]
    if defaults:
        issues.append(f"{len(defaults)} scheduled games have no tip time (7:30 p.m. ET assumed)")
    return WeekCalendar(season, week, out, start, end, timedelta(minutes=margin_minutes), postponed, issues)


def fmt_local(ts: datetime | None, tz: ZoneInfo | str, with_day: bool = True) -> str:
    if ts is None:
        return "-"
    z = ZoneInfo(tz) if isinstance(tz, str) else tz
    loc = ts.astimezone(z)
    return loc.strftime("%a %d %b %H:%M" if with_day else "%H:%M")


def fmt_delta(td: timedelta) -> str:
    s = int(td.total_seconds())
    if s < 0:
        return "passed"
    h, m = divmod(s // 60, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"
