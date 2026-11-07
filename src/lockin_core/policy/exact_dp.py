"""Exact dynamic programming for small roster-weeks: the test oracle (design section 12).

Toy model (one tip batch a day, the games of a day all finish before the next day):

* day d: statuses of the day's games are revealed (ruled out with q_out); then the decision
  epoch - every open entry of a player who tips today is due (lock into a slot or pass),
  others may be locked, passed or kept; final-game entries (fallbacks) may be locked or kept;
  then a feasible set of available tippers is seated; then games are played (late scratch
  q_scr, else a discrete score). A non-final game's score becomes an open entry (its holder
  must stay seated); his team's final game's score (0 if scratched) becomes a fallback.
* week end: every kept fallback counts.

``wait=False`` is the policy class that must decide each entry right after its game. Values
are exact (no Monte Carlo); the hindsight bound uses Monte Carlo.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from lockin_core.policy.tables import build_tables
from lockin_core.sim.problem import SLOT_ACCEPTS, SideProblem, SideScenarios, SimPlayer, slot_classes


@dataclass(frozen=True)
class ToyGame:
    day: int
    q_out: float
    q_scr: float
    values: tuple[int, ...]  # units
    probs: tuple[float, ...]

    @property
    def mean(self) -> float:
        return float(np.dot(self.values, self.probs))


@dataclass(frozen=True)
class ToyPlayer:
    positions: tuple[str, ...]
    games: tuple[ToyGame, ...]  # by day; the last is his team's final game


@dataclass(frozen=True)
class ToyWeek:
    slots: tuple[str, ...]
    players: tuple[ToyPlayer, ...]
    days: int = 7

    def classes(self) -> list[tuple[int, ...]]:
        return [tuple(int(i) for i in c) for c in slot_classes(list(self.slots))]


@dataclass
class ExactDP:
    week: ToyWeek
    wait: bool = True
    nonneg: bool = field(init=False)

    def __post_init__(self):
        w = self.week
        self.cls = w.classes()
        self.cls_accepts = [SLOT_ACCEPTS[w.slots[c[0]]] for c in self.cls]
        self.P = len(w.players)
        self.elig_cls = [[bool(set(pl.positions) & acc) for acc in self.cls_accepts] for pl in w.players]
        self.game_on = [{g.day: (k, g) for k, g in enumerate(pl.games)} for pl in w.players]
        self.n_games = [len(pl.games) for pl in w.players]
        self.nonneg = all(v >= 0 for pl in w.players for g in pl.games for v in g.values)
        self._fit = lru_cache(maxsize=None)(self._feasible)
        self._pre = lru_cache(maxsize=None)(self._v_pre)

    # --- feasibility: can these players all sit in the remaining slot classes? ------
    def _feasible(self, players: tuple[int, ...], counts: tuple[int, ...]) -> bool:
        if not players:
            return True
        if len(players) > sum(counts):
            return False
        p, rest = players[0], players[1:]
        for c, n in enumerate(counts):
            if n and self.elig_cls[p][c]:
                if self._fit(rest, counts[:c] + (n - 1,) + counts[c + 1:]):
                    return True
        return False

    def value(self, day: int = 0, open_: tuple = (), fb: tuple = (), locked_classes: tuple = (),
              used: frozenset = frozenset()) -> float:
        """Optimal expected units from the start of ``day`` with these open entries
        ((player, units), ...), fallbacks, classes already locked and players already used."""
        counts = [len(c) for c in self.cls]
        for c in locked_classes:
            counts[c] -= 1
        return self._pre(day, tuple(counts), frozenset(used), tuple(sorted(open_)), tuple(sorted(fb)))

    def class_of(self, slot: str) -> int:
        return next(i for i, c in enumerate(self.cls) if self.week.slots[c[0]] == slot)

    # --- recursion ---------------------------------------------------------------
    def _v_pre(self, d, counts, used, open_, fb):
        if d >= self.week.days:
            return float(sum(s for _, s in fb))
        tippers = [p for p in range(self.P) if p not in used and d in self.game_on[p]]
        total = 0.0
        for mask in itertools.product((True, False), repeat=len(tippers)):
            pr = 1.0
            avail = []
            for p, ok in zip(tippers, mask):
                q = self.game_on[p][d][1].q_out
                pr *= (1 - q) if ok else q
                if ok:
                    avail.append(p)
            if pr == 0:
                continue
            total += pr * self._v_dec(d, counts, used, open_, fb, tuple(avail))
        return total

    def _entry_options(self, p, s, counts, due, final):
        """Each option: (lock class or None, keep?)."""
        opts = []
        for c, n in enumerate(counts):
            if n and self.elig_cls[p][c]:
                opts.append(("lock", c))
        if final:  # a fallback may also be given up, freeing its slot for a later starter
            opts.append(("keep", None))
            opts.append(("drop", None))
        else:
            opts.append(("pass", None))
            if not due:
                opts.append(("keep", None))
        return opts

    def _v_dec(self, d, counts, used, open_, fb, avail):
        items = [(p, s, False) for p, s in open_] + [(p, s, True) for p, s in fb]
        option_lists = [self._entry_options(p, s, counts, d in self.game_on[p], fin) for p, s, fin in items]
        best = -np.inf
        for combo in itertools.product(*option_lists):
            cnt = list(counts)
            reward = 0.0
            used2 = set(used)
            keep_open, keep_fb, ok = [], [], True
            for (p, s, fin), (kind, c) in zip(items, combo):
                if kind == "lock":
                    if cnt[c] == 0:
                        ok = False
                        break
                    cnt[c] -= 1
                    reward += s
                    used2.add(p)
                elif kind == "keep":
                    (keep_fb if fin else keep_open).append((p, s))
            if not ok:
                continue
            cnt_t = tuple(cnt)
            holders = tuple(sorted(p for p, _ in keep_open + keep_fb))
            if not self._fit(holders, cnt_t):
                continue
            cands = [p for p in avail if p not in used2 and p not in holders]
            fb_t = tuple(sorted(keep_fb))
            open_t = tuple(sorted(keep_open))
            used_t = frozenset(used2)
            seatings = self._seatings(cands, holders, cnt_t)
            for seated in seatings:
                v = reward + self._v_play(d, cnt_t, used_t, open_t, fb_t, seated)
                if v > best:
                    best = v
        return best

    def _seatings(self, cands, holders, counts):
        feas = []
        for r in range(len(cands), -1, -1):
            for sub in itertools.combinations(cands, r):
                if self._fit(tuple(sorted(holders + sub)), counts):
                    feas.append(sub)
        if self.nonneg:  # seating more is weakly better: an entry can be passed at no cost
            maximal = [s for s in feas if not any(set(s) < set(t) for t in feas)]
            return maximal
        return feas

    def _v_play(self, d, counts, used, open_, fb, seated):
        if not seated:
            return self._next(d, counts, used, open_, fb, ())
        per = []
        for p in seated:
            k, g = self.game_on[p][d]
            final = k == self.n_games[p] - 1
            outs = [(g.q_scr, (p, 0 if final else None, final))]
            for v, w in zip(g.values, g.probs):
                outs.append(((1 - g.q_scr) * w, (p, v, final)))
            per.append(outs)
        total = 0.0
        for combo in itertools.product(*per):
            pr = 1.0
            new_open, new_fb = list(open_), list(fb)
            fresh = []
            for q, (p, v, final) in combo:
                pr *= q
                if v is None:
                    continue
                if final:
                    new_fb.append((p, v))
                else:
                    new_open.append((p, v))
                    fresh.append((p, v))
            if pr == 0:
                continue
            total += pr * self._next(d, counts, used, tuple(sorted(new_open)), tuple(sorted(new_fb)), tuple(fresh))
        return total

    def _next(self, d, counts, used, open_, fb, fresh):
        if self.wait or not fresh:
            return self._pre(d + 1, counts, used, open_, fb)
        # must decide each new entry right after its game: lock (any free class) or pass
        best = -np.inf
        fresh_set = set(fresh)
        others = tuple(e for e in open_ if e not in fresh_set)
        opts = []
        for p, s in fresh:
            o = [("pass", None)] + [("lock", c) for c, n in enumerate(counts) if n and self.elig_cls[p][c]]
            opts.append(o)
        for combo in itertools.product(*opts):
            cnt = list(counts)
            used2 = set(used)
            reward = 0.0
            ok = True
            for (p, s), (kind, c) in zip(fresh, combo):
                if kind == "lock":
                    if cnt[c] == 0:
                        ok = False
                        break
                    cnt[c] -= 1
                    reward += s
                    used2.add(p)
            if not ok:
                continue
            v = reward + self._pre(d + 1, tuple(cnt), frozenset(used2), others, fb)
            best = max(best, v)
        return best


# --- toy generation, hindsight, conversion to the simulator ----------------------------
TOY_POSITIONS = (("PG",), ("SG",), ("SF",), ("PF",), ("PG", "SF"), ("SG", "PF"), ("C",))


def random_week(rng: np.random.Generator, n_players: int = 5, slots: tuple[str, ...] = ("G", "F", "UTIL"),
                days: int = 7, max_games: int = 4, late_scratch: float = 0.02) -> ToyWeek:
    """Design section 12's toys: G-only, F-only, G/F or C players; 1-4 games on distinct days;
    0-50% ruled-out chance per game; a 2% late scratch; three-point score distributions."""
    kinds = [("PG",), ("SF",), ("PG", "SF"), ("C",)]
    players = []
    for _ in range(n_players):
        pos = kinds[int(rng.integers(len(kinds)))]
        n = int(rng.integers(1, max_games + 1))
        game_days = sorted(rng.choice(days, size=n, replace=False).tolist())
        mu = rng.uniform(20, 40)
        spread = rng.uniform(4, 12)
        games = []
        for dd in game_days:
            vals = tuple(int(round(2 * x)) for x in (mu - spread, mu, mu + spread))
            probs = tuple(float(x) for x in rng.dirichlet((3, 4, 3)))
            games.append(ToyGame(int(dd), float(rng.uniform(0, 0.5)), late_scratch, vals, probs))
        players.append(ToyPlayer(pos, tuple(games)))
    return ToyWeek(tuple(slots), tuple(players), days)


def toy_scenarios(week: ToyWeek, n: int, seed: int) -> SideScenarios:
    rng = np.random.default_rng(seed)
    P = len(week.players)
    K = max(len(p.games) for p in week.players)
    out = np.ones((n, P, K), bool)
    scr = np.zeros((n, P, K), bool)
    units = np.zeros((n, P, K))
    for i, pl in enumerate(week.players):
        for k, g in enumerate(pl.games):
            out[:, i, k] = rng.random(n) < g.q_out
            scr[:, i, k] = rng.random(n) < g.q_scr
            units[:, i, k] = np.asarray(g.values)[rng.choice(len(g.values), size=n, p=np.asarray(g.probs))]
    return SideScenarios(out, scr, units)


def hindsight(week: ToyWeek, sc: SideScenarios) -> np.ndarray:
    """Best eligible game per slot with every score known: max-weight matching of slots to
    players' best played games."""
    from lockin_core.sim.problem import eligibility

    elig = eligibility([list(p.positions) for p in week.players], list(week.slots))
    plays = ~sc.out & ~sc.scr
    best = np.where(plays, sc.units, -np.inf).max(axis=2)
    out = np.zeros(sc.n)
    for r in range(sc.n):
        Wm = np.where(elig & np.isfinite(best[r])[:, None], np.maximum(best[r], 0)[:, None], 0.0)
        a, b = linear_sum_assignment(Wm, maximize=True)
        out[r] = Wm[a, b].sum()
    return out


def toy_problem(week: ToyWeek, tip_minute: int = 19 * 60, game_minutes: int = 150) -> SideProblem:
    P = len(week.players)
    K = max(len(p.games) for p in week.players)
    players = []
    Q = max(len(g.values) for p in week.players for g in p.games)
    values = np.zeros((P, K, Q))
    weights = np.zeros((P, K, Q))
    q_out = np.zeros((P, K))
    q_scr = np.zeros((P, K))
    final = np.zeros((P, K), bool)
    for i, pl in enumerate(week.players):
        tips = np.array([g.day * 1440 + tip_minute for g in pl.games], float)
        fin = np.zeros(len(pl.games), bool)
        fin[-1] = True
        means = [g.mean for g in pl.games]
        players.append(SimPlayer(str(i), f"P{i}", list(pl.positions), tips, tips + game_minutes, fin,
                                 np.zeros(len(pl.games), bool), [f"g{i}_{k}" for k in range(len(pl.games))],
                                 fp_mean=float(np.mean(means)),
                                 fp_p75=float(np.mean([np.percentile(np.repeat(g.values, [int(100 * w) for w in g.probs]), 75)
                                                       for g in pl.games]))))
        for k, g in enumerate(pl.games):
            values[i, k, :len(g.values)] = g.values
            weights[i, k, :len(g.values)] = g.probs
            q_out[i, k], q_scr[i, k] = g.q_out, g.q_scr
            final[i, k] = k == len(pl.games) - 1
    S = len(week.slots)
    prob = SideProblem(list(week.slots), players, np.zeros(S, bool), 0.0, np.full(S, -1), np.zeros(P, bool), [],
                       np.zeros(P, bool))
    n_games = np.array([p.n_games for p in players])
    prob.tables = build_tables(values, weights, q_out, q_scr, n_games, final, prob.flex)
    return prob
