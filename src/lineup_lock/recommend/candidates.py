"""Candidate root actions, generated around the base policy's own answer (design section 16).

1. The base action a0: every open entry kept until its deadline (the base policy decides it
   there, after the information scheduled before it), the next tip batch seated by its rule.
2. One change per entry: lock into each free eligible slot type (identical slots once), or pass.
3. Lineup variants when the next event is a tip batch with your players: the top seatings from
   the assignment solver under three weightings, plus "bench X" for each tipper.
4. Deduplicate by resulting state; check legality; cap at 60, dropping lineup variants first and
   never the base action or a due entry's lock-now / pass versions.
"""

from __future__ import annotations

import numpy as np

from lockin_core.policy.assign import NEG, best_lineups
from lockin_core.sim.problem import Action, SideProblem


def lock_slot(prob: SideProblem, p: int, cls: np.ndarray, taken: set[int] = frozenset()) -> int | None:
    """The slot of class ``cls`` a lock of player p would use: his own slot if it is in the class,
    else the first unlocked one (slots held by players in live games are not available)."""
    frozen_slots = {j for j in range(prob.S) if prob.occupant[j] >= 0 and prob.started_live[prob.occupant[j]]}
    free = [int(j) for j in cls if not prob.locked[j] and j not in frozen_slots and j not in taken]
    if not free:
        return None
    own = [j for j in free if prob.occupant[j] == p]
    return own[0] if own else free[0]


from lockin_core.state.legality import is_legal


def next_batch(prob: SideProblem) -> tuple[float, list[tuple[int, int]]] | None:
    """The first future tip batch of this roster and its (player, game) pairs, if no final of a
    game in progress comes before it (otherwise the next epoch is that final)."""
    tips = sorted({float(t) for pl in prob.players for t in pl.tips if t > 0})
    if not tips:
        return None
    t = tips[0]
    first_final = min([float(e) for pl in prob.players for e, lv in zip(pl.ends, pl.live) if lv] + [np.inf])
    if first_final < t:
        return None
    pairs = [(p, k) for p, pl in enumerate(prob.players) for k in range(pl.n_games)
             if float(pl.tips[k]) == t and not prob.used[p]]
    return t, pairs


def lineup_variants(prob: SideProblem, top: int = 5) -> list[Action]:
    nb = next_batch(prob)
    if nb is None or prob.tables is None:
        return []
    _, pairs = nb
    T = prob.tables
    holders = {e.p for e in prob.entries if not e.final}
    tippers = [(p, k) for p, k in pairs if p not in holders]
    if not tippers:
        return []
    frozen_slots = {j for j in range(prob.S) if prob.occupant[j] >= 0 and prob.started_live[prob.occupant[j]]}
    held_slots = {j for j in range(prob.S) if prob.occupant[j] in holders}
    slots = [j for j in range(prob.S) if not prob.locked[j] and j not in frozen_slots and j not in held_slots]
    if not slots:
        return []
    flex = prob.flex
    weightings = {
        "tonight": lambda p, k, j: T.tonight[p, k],
        "gain": lambda p, k, j: T.W[p, k] - T.W[p, k + 1],
        "flex_penalty": lambda p, k, j: T.tonight[p, k] * (1 - 0.03 * (flex[j] - flex[slots].min())),
    }
    out, seen = [], set()
    for name, f in weightings.items():
        Wm = np.full((len(tippers), len(slots)), NEG)
        for a, (p, k) in enumerate(tippers):
            for b, j in enumerate(slots):
                if prob.elig[p, j]:
                    Wm[a, b] = max(f(p, k, j), 1e-6)
        for _, pairs_ab in best_lineups(Wm, top):
            lineup = tuple(sorted((tippers[a][0], slots[b]) for a, b in pairs_ab))
            if lineup in seen:
                continue
            seen.add(lineup)
            out.append(Action(lineup=lineup, label=f"lineup:{name}"))
    if out:  # "bench X" for each player the first lineup starts
        first = dict(out[0].lineup)
        for p in first:
            lu = tuple(sorted((q, j) for q, j in first.items() if q != p))
            if lu not in seen:
                seen.add(lu)
                out.append(Action(lineup=lu, label=f"bench:{p}"))
    return out


def generate(prob: SideProblem, cap: int = 60, top_lineups: int = 5) -> tuple[list[Action], list[Action]]:
    """(candidates, protected)."""
    a0 = Action(label="base")
    acts = [a0]
    protected = [a0]
    for e in prob.entries:
        for cls in prob.classes_of(e.p):
            j = lock_slot(prob, e.p, cls)
            if j is None:
                continue
            a = Action(locks=((e.p, j),), label=f"lock:{e.p}:{j}")
            acts.append(a)
            if e.must_decide:
                protected.append(a)
        if not e.final:
            a = Action(passes=(e.p,), label=f"pass:{e.p}")
            acts.append(a)
            if e.must_decide:
                protected.append(a)
    lineups = lineup_variants(prob, top_lineups)
    acts += lineups
    out, seen = [], set()
    for a in acts:
        if not is_legal(prob, a) or a.key() in seen:
            continue
        seen.add(a.key())
        out.append(a)
    if len(out) > cap:
        prot = {a.key() for a in protected}
        keep = [a for a in out if a.lineup is None or a.key() in prot]
        rest = [a for a in out if a not in keep]
        out = keep[:cap] + rest[:max(0, cap - len(keep))]
    return out, [a for a in protected if is_legal(prob, a)]


def combine(prob: SideProblem, a: Action, b: Action) -> Action | None:
    """a with b's entry decisions added (for coordinate ascent); None if they conflict."""
    a_ps = {p for p, _ in a.locks} | set(a.passes)
    b_ps = {p for p, _ in b.locks} | set(b.passes)
    if a_ps & b_ps:
        return None
    c = Action(locks=tuple(sorted(a.locks + b.locks)), passes=tuple(sorted(a.passes + b.passes)),
               lineup=a.lineup if a.lineup is not None else b.lineup, label=f"{a.label}+{b.label}")
    return c if is_legal(prob, c) else None
