"""The decision problem for one roster's rest of week, in the simulator's terms.

Times are minutes from the decision instant t0 (a tip at or before 0 has happened). Players
are indexed 0..P-1 and their remaining games 0..n-1 (a game in progress first). Everything
here is deterministic; outcomes live in ``SideScenarios``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lockin_core.policy.tables import ContinuationTables

POSITIONS = ("PG", "SG", "SF", "PF", "C")
SLOT_ACCEPTS: dict[str, frozenset[str]] = {
    "PG": frozenset({"PG"}), "SG": frozenset({"SG"}), "G": frozenset({"PG", "SG"}),
    "SF": frozenset({"SF"}), "PF": frozenset({"PF"}), "F": frozenset({"SF", "PF"}),
    "C": frozenset({"C"}), "UTIL": frozenset(POSITIONS),
}


def eligibility(positions: list[list[str]], slots: list[str]) -> np.ndarray:
    return np.array([[bool(SLOT_ACCEPTS[s] & set(pos)) for s in slots] for pos in positions], dtype=bool).reshape(
        len(positions), len(slots))


def slot_classes(slots: list[str]) -> list[np.ndarray]:
    """Slots grouped by acceptance set (C and C, UTIL and UTIL count once)."""
    seen: dict[frozenset, list[int]] = {}
    for i, s in enumerate(slots):
        seen.setdefault(SLOT_ACCEPTS[s], []).append(i)
    return [np.array(v) for v in seen.values()]


@dataclass
class SimPlayer:
    pid: str
    name: str
    positions: list[str]
    tips: np.ndarray  # [n] minutes from t0
    ends: np.ndarray  # [n]
    final: np.ndarray  # [n] bool: his team's final game of the week
    live: np.ndarray  # [n] bool: tipped before t0, not final
    game_ids: list[str] = field(default_factory=list)
    fp_mean: float = 0.0  # realised FP per game (units), for the "lock above average" baselines
    fp_p75: float = 0.0

    @property
    def n_games(self) -> int:
        return len(self.tips)


@dataclass
class RootEntry:
    p: int
    units: float
    final: bool
    entry_id: str = ""
    must_decide: bool = False  # deadline before any later epoch (no wait)


@dataclass
class SideProblem:
    slots: list[str]
    players: list[SimPlayer]
    locked: np.ndarray  # [S] bool
    locked_units: float
    occupant: np.ndarray  # [S] int, -1 empty
    used: np.ndarray  # [P] bool (locked this week)
    entries: list[RootEntry]
    started_live: np.ndarray  # [P] bool: seated in a game in progress at t0
    tables: ContinuationTables | None = None

    def __post_init__(self):
        self.elig = eligibility([p.positions for p in self.players], self.slots)
        self.flex = self.elig.sum(0).astype(float)
        self.classes = slot_classes(self.slots)
        self.K = max([p.n_games for p in self.players] + [1])

    @property
    def P(self) -> int:
        return len(self.players)

    @property
    def S(self) -> int:
        return len(self.slots)

    def index(self, pid: str) -> int:
        return next(i for i, p in enumerate(self.players) if p.pid == pid)

    def classes_of(self, p: int) -> list[np.ndarray]:
        return [c for c in self.classes if self.elig[p, c[0]]]


@dataclass
class SideScenarios:
    out: np.ndarray  # [N, P, K] bool
    scr: np.ndarray  # [N, P, K] bool
    units: np.ndarray  # [N, P, K] float

    @property
    def n(self) -> int:
        return self.out.shape[0]

    def subset(self, rows) -> "SideScenarios":
        return SideScenarios(self.out[rows], self.scr[rows], self.units[rows])


@dataclass(frozen=True)
class Action:
    """A root action. Locks name their slot; ``lineup`` (player -> slot) fixes the seating of the
    next tip batch (other tippers benched); waits only label entries the action keeps open."""

    locks: tuple[tuple[int, int], ...] = ()
    passes: tuple[int, ...] = ()
    waits: tuple[int, ...] = ()
    lineup: tuple[tuple[int, int], ...] | None = None
    label: str = "base"

    def key(self) -> tuple:
        return (tuple(sorted(self.locks)), tuple(sorted(self.passes)),
                None if self.lineup is None else tuple(sorted(self.lineup)))


@dataclass(frozen=True)
class PolicyConfig:
    lock_rule: str = "earmark"  # earmark | never | above_mean | above_p75 | independent | independent_best
    defer: bool = True  # decide each entry at its deadline (v2) or right after its game (4 Nov greedy)
    seat_by: str = "tonight_mean"  # tonight_mean | one_player_gain | highest_projection
    assignment: str = "fast"  # fast | exact
    value_open_entries: bool = True


BASE = PolicyConfig()
GREEDY_NOV4 = PolicyConfig(defer=False)
BASELINES: dict[str, PolicyConfig] = {
    "base_v2": BASE,
    "greedy_nov4": GREEDY_NOV4,
    "independent": PolicyConfig(lock_rule="independent", defer=False, seat_by="highest_projection"),
    "above_average": PolicyConfig(lock_rule="above_mean", defer=False, seat_by="highest_projection"),
    "above_p75": PolicyConfig(lock_rule="above_p75", defer=False, seat_by="highest_projection"),
    "never_lock": PolicyConfig(lock_rule="never", defer=False, seat_by="highest_projection"),
}
