"""League rules read from the Sleeper league object at run time (design section 5).

Verified for LeFantasy (waiver assistant probe, 30 Oct 2026): ``roster_positions`` lists the
starting slots then BN; ``settings.reserve_slots`` is the IL size; ``reserve_allow_*`` flags
say which statuses may use IL. Starting slots are read, never assumed: this league has one C
slot (PG, SG, G, SF, PF, F, C, UTIL, UTIL), not the two the design assumed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from lockin_core.config import LeagueConfig
from lockin_core.ingest.sleeper import League
from lockin_core.scoring.settings import ScoringSettings
from lockin_core.sim.problem import SLOT_ACCEPTS

NON_STARTING = {"BN", "IL", "TAXI"}
IL_FLAGS: dict[str, set[str]] = {
    "reserve_allow_out": {"OUT", "IR"}, "reserve_allow_sus": {"SUS", "SUSPENDED"},
    "reserve_allow_dnr": {"DNR"}, "reserve_allow_doubtful": {"DOUBTFUL"},
    "reserve_allow_dtd": {"DTD", "QUESTIONABLE", "GTD"}, "reserve_allow_cov": {"COV"},
    "reserve_allow_na": {"NA"},
}


class LeagueRules(BaseModel):
    league_id: str
    season: int
    total_rosters: int
    starting_slots: list[str]
    bench_slots: int
    reserve_slots: int
    il_statuses: list[str] = Field(default_factory=list)
    playoff_week_start: int = 22
    playoff_teams: int | None = None
    scoring: ScoringSettings

    @classmethod
    def from_league(cls, league: League, cfg: LeagueConfig) -> "LeagueRules":
        st = league.settings
        starting = [p for p in league.roster_positions if p not in NON_STARTING]
        unknown = [p for p in starting if p not in SLOT_ACCEPTS]
        if unknown:
            raise ValueError(f"unsupported roster slots {unknown}")
        il = sorted({s for flag, statuses in IL_FLAGS.items() if st.get(flag) for s in statuses})
        return cls(
            league_id=league.league_id, season=int(league.season), total_rosters=league.total_rosters,
            starting_slots=starting, bench_slots=sum(p == "BN" for p in league.roster_positions),
            reserve_slots=int(st.get("reserve_slots", 0) or 0), il_statuses=il,
            playoff_week_start=int(st.get("playoff_week_start", 22) or 22),
            playoff_teams=st.get("playoff_teams"),
            scoring=ScoringSettings.from_sleeper(league.scoring_settings or None, cfg.scoring.threshold_bonus_mode,
                                                 cfg.scoring.tech_definition),
        )

    def is_playoffs(self, week: int) -> bool:
        return week >= self.playoff_week_start
