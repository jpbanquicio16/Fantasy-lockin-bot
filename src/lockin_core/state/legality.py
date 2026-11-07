"""Pure legality checks used at candidate generation and at simulation entry."""
from lockin_core.sim.problem import Action, SideProblem


def is_legal(prob: SideProblem, action: Action) -> bool:
    open_entries = {e.p: e for e in prob.entries}
    locked_players = [p for p, _ in action.locks]
    locked_slots = [j for _, j in action.locks]
    if len(set(locked_players)) != len(locked_players) or len(set(locked_slots)) != len(locked_slots):
        return False
    if len(set(action.passes)) != len(action.passes) or len(set(action.waits)) != len(action.waits):
        return False
    if set(action.passes) & set(locked_players) or set(action.waits) & (set(action.passes) | set(locked_players)):
        return False
    if any(p not in open_entries for p in (*action.passes, *action.waits)):
        return False
    for p, j in action.locks:
        if not 0 <= p < prob.P or not 0 <= j < prob.S:
            return False
        if p not in open_entries or prob.locked[j] or prob.used[p] or not prob.elig[p, j]:
            return False
        if p not in prob.occupant or prob.started_live[p]:
            return False
        occupant = prob.occupant[j]
        if occupant >= 0 and occupant != p and prob.started_live[occupant]:
            return False
        if open_entries[p].final and occupant not in (-1, p):
            return False
    if action.lineup is not None:
        players, slots = set(), set(locked_slots)
        for p, j in action.lineup:
            if not 0 <= p < prob.P or not 0 <= j < prob.S:
                return False
            if p in players or j in slots or prob.locked[j] or prob.used[p] or p in locked_players:
                return False
            if not prob.elig[p, j]:
                return False
            occupant = prob.occupant[j]
            if occupant >= 0 and prob.started_live[occupant] and occupant != p:
                return False
            if prob.started_live[p] and occupant != p:
                return False
            players.add(p)
            slots.add(j)
    return True
