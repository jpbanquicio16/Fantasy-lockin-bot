"""Typed state: players, the eligible-score ledger entries, one roster's week, the validated
state every recommendation runs on (design section 22 interfaces)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

SlotType = Literal["PG", "SG", "G", "SF", "PF", "F", "C", "UTIL"]


class PlayerInfo(BaseModel):
    player_id: str
    name: str
    team: str | None = None
    positions: list[str]
    injury_status: str | None = None  # Sleeper's tag
    report_status: str | None = None  # latest official report for his next game, if listed
    years_exp: int | None = None


@dataclass(frozen=True)
class EligibleScore:
    """A completed game the player started, not yet locked or passed (the ledger E_t)."""

    entry_id: str
    player_id: str
    game_id: str
    fp_half: int  # score in league units (half-points)
    slot_at_tip: int | None
    is_team_final: bool
    deadline_at: datetime  # UTC; week end for final-game entries
    eligibility: Literal["observed", "inferred", "manual", "uncertain"] = "inferred"
    fp_source: Literal["engine", "sleeper_confirmed", "manual"] = "engine"
    locked_unknown: bool = False  # the API cannot tell "locked already" from "still lockable"
    note: str = ""


@dataclass(frozen=True)
class LockedSlot:
    slot_index: int
    player_id: str
    game_id: str | None
    fp_half: int
    source: Literal["api_inferred", "manual"]


@dataclass
class SideState:
    """One roster's position in the current matchup week."""

    roster_id: int
    owner: str | None
    slots: tuple[str, ...]  # starting slot types in order
    assignment: dict[int, str]  # open slot index -> current occupant
    locked: dict[int, LockedSlot]
    entries: tuple[EligibleScore, ...]
    active: tuple[str, ...]  # not on IL
    reserve: tuple[str, ...]
    frozen: frozenset[str]  # in a game that has tipped and is not final
    notes: list[str] = field(default_factory=list)

    @property
    def used_players(self) -> frozenset[str]:
        return frozenset(l.player_id for l in self.locked.values())

    @property
    def locked_units(self) -> int:
        return sum(l.fp_half for l in self.locked.values())

    def slot_of(self, player_id: str) -> int | None:
        return next((i for i, p in self.assignment.items() if p == player_id), None)

    @property
    def bench(self) -> list[str]:
        started = set(self.assignment.values()) | {l.player_id for l in self.locked.values()}
        return [p for p in self.active if p not in started]


@dataclass
class ValidatedState:
    t: datetime
    season: int
    week: int
    league_id: str
    me: SideState
    opp: SideState | None
    players: dict[str, PlayerInfo]
    units_per_point: int
    snapshot_id: str
    objective: Literal["points", "win"] = "points"
    versions: dict[str, str] = field(default_factory=dict)
    freshness: dict[str, datetime] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    blocking: list[str] = field(default_factory=list)
    opp_locked_manual: float | None = None

    def name(self, pid: str) -> str:
        p = self.players.get(pid)
        return p.name if p else pid
