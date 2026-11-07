"""ESPN's public scoreboard: scheduled tip times and live game status.

Sleeper's schedule has dates and status but no tip times, and every lock deadline is a
tip time, so tip times come from ESPN's unofficial scoreboard (one request per ET date,
no key). ESPN team codes differ from the NBA tricodes Sleeper uses for six teams; the map
below converts them and an unknown code fails loudly. ESPN is never the scoring source.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel

from lockin_core.config import Paths
from lockin_core.ingest.raw import HttpClient, RawStore

ESPN_TO_NBA = {"GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
NBA_TEAMS = {
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL",
    "MEM", "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR",
    "UTA", "WAS",
}
STATE = {"pre": "scheduled", "in": "live", "post": "final"}


class UnknownTeamCode(ValueError):
    pass


def nba_code(espn_abbr: str) -> str:
    code = ESPN_TO_NBA.get(espn_abbr, espn_abbr)
    if code not in NBA_TEAMS:
        raise UnknownTeamCode(f"ESPN team code {espn_abbr!r} has no NBA tricode mapping")
    return code


class EspnGame(BaseModel):
    espn_id: str
    tip_at: datetime  # scheduled tip, UTC
    time_valid: bool
    home: str  # NBA tricode
    away: str
    state: str  # scheduled | live | final | postponed
    detail: str = ""
    period: int | None = None


def parse_scoreboard(body: dict[str, Any]) -> list[EspnGame]:
    out = []
    for e in body.get("events", []) or []:
        comp = (e.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): c.get("team", {}).get("abbreviation") for c in comp.get("competitors", [])}
        st = (e.get("status") or comp.get("status") or {}).get("type", {})
        name = st.get("name", "")
        state = "postponed" if "POSTPONED" in name or "CANCELED" in name else STATE.get(st.get("state"), "scheduled")
        ts = e.get("date") or comp.get("date")
        out.append(EspnGame(
            espn_id=str(e["id"]), tip_at=datetime.fromisoformat(ts.replace("Z", "+00:00")),
            time_valid=bool(comp.get("timeValid", True)), home=nba_code(teams["home"]),
            away=nba_code(teams["away"]), state=state, detail=st.get("detail", ""),
            period=(e.get("status") or {}).get("period")))
    return out


class EspnClient:
    def __init__(self, store: RawStore | None = None, paths: Paths | None = None,
                 http: HttpClient | None = None):
        paths = (paths or Paths()).ensure()
        self.store = store or RawStore(paths.raw)
        self.http = http or HttpClient("https://site.api.espn.com/apis/site/v2/sports/basketball/nba",
                                       min_interval=0.5)

    def scoreboard(self, d: date, max_age_s: float | None) -> list[EspnGame]:
        params = {"dates": d.strftime("%Y%m%d")}
        rec = self.store.fetch("espn", "/scoreboard", params,
                               lambda: self.http.get_json("/scoreboard", params), max_age_s)
        return parse_scoreboard(rec.body)

    def checked_at(self, d: date) -> datetime | None:
        return self.store.checked_at("espn", "/scoreboard", {"dates": d.strftime("%Y%m%d")})
