"""Vectorised chronological simulator of the rest of the week under base policy v2.

One run plays a root action and then the base policy through every event of the week, for
all scenarios at once (design section 13):

1. **A game ends that is not his team's last of the week**: each Sleeper starter who played gets
   an open entry (his score, deadline his team's next tip). He must stay in a starting slot.
2. **His team's last game ends**: the entry (his score, or 0 if he did not play) is decided at
   once by the lock rule with his own value 0; if not locked it is the slot's fallback, which
   counts at week end unless a later starter takes that slot. It may lock only into his own
   slot or an empty one.
3. **A tip batch**: statuses are revealed; each open entry whose player tips now is due (lock
   rule); free slots are seated - holders of open entries first, then available tippers by
   tonight's expected score into the eligible free slot the fewest roster players can fill,
   fallback slots last; a held entry that stops a tipper from being seated is decided now and
   seating redone.

**Lock rule (earmarking)**: tau = A(O + p at his value | S) - max_j A(O | S - j); lock into the
maximising j if the score >= tau. Policies never see a scenario's future: every decision
reads only outcomes revealed by its time (finals, statuses at tips) and the frozen tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lockin_core.policy.assign import NEG, ExactAssigner, greedy_total
from lockin_core.sim.problem import Action, PolicyConfig, SideProblem, SideScenarios


from lockin_core.state.information import FutureInformationError, InformationView
from lockin_core.state.legality import is_legal


@dataclass
class SimResult:
    total: np.ndarray  # [N] week points (units) for this roster
    contributed: np.ndarray | None = None  # [N, P] one of his games counted
    first_seated: np.ndarray | None = None  # [N, P] started at the first tip batch
    forced_zero: np.ndarray | None = None  # [N] slots ending on a 0 from a DNP final game
    first_occ: np.ndarray | None = None  # [N, S] occupants right after the first tip batch
    locks_at_root: np.ndarray | None = None  # [N] units locked by the root action


class WeekSim:
    def __init__(self, prob: SideProblem, policy: PolicyConfig):
        if prob.tables is None:
            raise ValueError("problem has no continuation tables")
        self.prob = prob
        self.pol = policy
        self.T = prob.tables
        self.elig = prob.elig
        self.flex = prob.flex
        self.exact = ExactAssigner(prob.elig) if policy.assignment == "exact" else None
        ev: dict[tuple[float, int], list[tuple[int, int]]] = {}
        for p, pl in enumerate(prob.players):
            for k in range(pl.n_games):
                if pl.tips[k] > 0:
                    ev.setdefault((float(pl.tips[k]), 1), []).append((p, k))
                if pl.ends[k] > 0:
                    ev.setdefault((float(pl.ends[k]), 0), []).append((p, k))
        self.events = sorted(ev.items())  # finals (0) before tips (1) at the same minute

    # --- helpers -----------------------------------------------------------------
    def _assign(self, V, free):
        if self.exact is not None:
            return self.exact.total(V, free)
        return greedy_total(V, self.elig, free, self.flex)

    def _values(self, rows):
        """Each player's value at the current time, for the scenario rows given."""
        T, s = self.T, self
        P = self.prob.P
        nk = s.nk
        ar = np.arange(P)
        base = T.W[ar, nk]
        avail = T.W_in[ar, np.minimum(nk, T.W.shape[1] - 1)]
        after = T.W[ar, np.minimum(nk + 1, T.W.shape[1] - 1)]
        kn = s.known[rows]
        V = np.where(kn == 1, avail[None, :], np.where(kn == -1, after[None, :], base[None, :]))
        live = s.liveg >= 0
        if live.any():
            lk = np.where(live, s.liveg, 0)
            win_live = T.W_in[ar, lk]
            V = np.where(live[None, :] & s.started[rows], win_live[None, :], V)
        eo = s.eopen[rows]
        reg = eo & ~s.efinal[rows]
        fb = eo & s.efinal[rows]
        if self.pol.value_open_entries:
            V = np.where(reg, np.maximum(V, s.eunits[rows]), V)
        V = np.where(fb, s.eunits[rows], V)
        V = np.where(s.used[rows] | s.frozen[rows], NEG, V)
        return V

    def _frozen_slots(self, rows):
        occ = self.occ[rows]
        fr = self.frozen[rows]
        out = np.zeros(occ.shape, bool)
        has = occ >= 0
        r, c = np.nonzero(has)
        out[r, c] = fr[r, occ[r, c]]
        return out

    def _lock_rule(self, p: int, rows: np.ndarray, own: np.ndarray, allowed: np.ndarray | None = None):
        """(lock?, slot) for player p's open entry in each row (score = eunits[rows, p])."""
        n = len(rows)
        s = self.eunits[rows, p]
        V = self._values(rows)
        V[:, p] = NEG
        free = ~self.locked[rows] & ~self._frozen_slots(rows)
        Vw = V.copy()
        Vw[:, p] = own
        A1 = self._assign(Vw, free)
        best = np.full(n, NEG)
        bslot = np.full(n, -1)
        for cls in self.prob.classes_of(p):
            cand = free[:, cls]
            if allowed is not None:
                cand = cand & allowed[:, cls]
            has = cand.any(1)
            if not has.any():
                continue
            j = cls[cand.argmax(1)]
            hr = np.flatnonzero(has)
            f2 = free[hr].copy()
            f2[np.arange(len(hr)), j[hr]] = False
            A2 = self._assign(V[hr], f2)
            upd = A2 > best[hr]
            best[hr[upd]] = A2[upd]
            bslot[hr[upd]] = j[hr][upd]
        tau = A1 - best
        lock = (bslot >= 0) & (s >= tau - 1e-9)
        return lock, bslot

    def _simple_rule(self, p: int, rows: np.ndarray, k_next: int, allowed: np.ndarray | None = None):
        """Baseline lock rules: a fixed threshold, first eligible free slot (or the
        least-flexible one for ``independent_best``)."""
        pl = self.prob.players[p]
        rule = self.pol.lock_rule
        s = self.eunits[rows, p]
        if rule == "never":
            return np.zeros(len(rows), bool), np.full(len(rows), -1)
        if rule == "above_mean":
            thr = pl.fp_mean
        elif rule == "above_p75":
            thr = pl.fp_p75
        else:  # independent: his own one-player continuation
            thr = self.T.W[p, min(k_next, self.T.W.shape[1] - 1)]
        free = ~self.locked[rows] & ~self._frozen_slots(rows) & self.elig[p][None, :]
        if allowed is not None:
            free &= allowed
        if rule == "independent_best":
            cost = np.where(free, self.flex[None, :], 1e9)
            j = cost.argmin(1)
        else:
            j = free.argmax(1)
        ok = free[np.arange(len(rows)), j]
        return ok & (s >= thr), np.where(ok, j, -1)

    def _decide(self, p: int, rows: np.ndarray, own: np.ndarray, k_next: int, allowed=None):
        if len(rows) == 0:
            return
        if self.pol.lock_rule == "earmark":
            lock, slot = self._lock_rule(p, rows, own, allowed)
        else:
            lock, slot = self._simple_rule(p, rows, k_next, allowed)
        lr = rows[lock]
        if len(lr):
            self._apply_lock(lr, p, slot[lock])
        # a passed regular entry is released; a final-game entry not locked stays as the fallback
        pr = rows[~lock]
        reg = pr[~self.efinal[pr, p]]
        self.eopen[reg, p] = False

    def _apply_lock(self, rows: np.ndarray, p: int, slots: np.ndarray):
        if rows.size and self.used[rows, p].any():
            raise RuntimeError("player locked more than once")
        if rows.size and (self.locked[rows, slots]).any():
            raise RuntimeError("lock into a slot that is already locked")
        prev = self.occ[rows] == p
        r, c = np.nonzero(prev)
        self.occ[rows[r], c] = -1
        self.locked[rows, slots] = True
        self.occ[rows, slots] = p
        self.total[rows] += self.eunits[rows, p]
        self.used[rows, p] = True
        self.eopen[rows, p] = False
        if self.contributed is not None:
            self.contributed[rows, p] = True

    # --- events ------------------------------------------------------------------
    def _final(self, games: list[tuple[int, int]]):
        decisions = []
        for p, k in sorted(games):
            pl = self.prob.players[p]
            self.liveg[p] = -1
            st = self.started[:, p] & ~self.used[:, p]
            self.frozen[:, p] = False
            played = st & ~self.view.scratched(p, k)
            rows = np.flatnonzero(st if pl.final[k] else played)
            self.eopen[rows, p] = True
            self.efinal[rows, p] = pl.final[k]
            self.eunits[rows, p] = np.where(played[rows], self.view.final_score(p, k)[rows], 0.0)
            if pl.final[k] and self.forced is not None:
                self.forced[rows] += (~played[rows]).astype(int)
            decisions.append((p, k, rows))
            self.started[:, p] = False
        for p, k, rows in decisions:
            if not len(rows):
                continue
            if self.prob.players[p].final[k]:
                occ = self.occ[rows]
                allowed = ((occ == p) | (occ == -1)) & ~self.locked[rows]
                self._decide(p, rows, np.zeros(len(rows)), k + 1, allowed)
            elif not self.pol.defer:
                self._decide(p, rows, np.full(len(rows), self.T.W[p, k + 1]), k + 1)

    def _tip(self, tippers: list[tuple[int, int]], lineup: dict[int, int] | None):
        N = self.N
        rows_all = np.arange(N)
        for p, k in tippers:
            self.known[:, p] = np.where(self.view.status(p, k), -1, 1)
        # every open entry whose player tips now is due (deferred entries, and root entries
        # under policies that otherwise decide right after the game)
        for p, k in sorted(tippers):
            rows = np.flatnonzero(self.eopen[:, p] & ~self.efinal[:, p] & ~self.used[:, p])
            if len(rows):
                own = np.where(self.known[rows, p] == 1, self.T.W_in[p, k], self.T.W[p, k + 1])
                self._decide(p, rows, own, k)
        self._seat(tippers, lineup)
        for p, k in tippers:
            self.liveg[p] = k
            self.nk[p] = k + 1
            seated = (self.occ == p).any(1) & ~self.used[:, p] & (self.known[:, p] == 1)
            self.started[:, p] = seated
            self.frozen[:, p] = seated
            self.known[:, p] = 0
            if self.first_seated is not None and not self._seen_tip:
                self.first_seated[:, p] = seated
        if self.first_seated is not None and not self._seen_tip:
            self.first_occ = self.occ.copy()
        self._seen_tip = True
        _ = rows_all

    def _seat(self, tippers, lineup):
        N, S = self.N, self.prob.S
        rows = np.arange(N)
        flex = self.flex
        big = 1e9
        free = ~self.locked & ~self._frozen_slots(rows)
        occ_old = self.occ.copy()
        new = np.where(~free, occ_old, -1)  # locked slots and frozen players stay
        # 1. holders of regular open entries keep a starting slot (their own if possible)
        holders = [p for p in range(self.prob.P) if (self.eopen[:, p] & ~self.efinal[:, p] & ~self.used[:, p]).any()]
        for p in holders:
            hr = np.flatnonzero(self.eopen[:, p] & ~self.efinal[:, p] & ~self.used[:, p] & ~self.frozen[:, p])
            if not len(hr):
                continue
            cand = free[hr] & self.elig[p][None, :]
            own = (occ_old[hr] == p) & cand
            cost = np.where(own, -1.0, np.where(cand, flex[None, :], big))
            j = cost.argmin(1)
            ok = cand[np.arange(len(hr)), j]
            r_ok, j_ok = hr[ok], j[ok]
            new[r_ok, j_ok] = p
            free[r_ok, j_ok] = False
            r_bad = hr[~ok]
            if len(r_bad):  # nowhere to sit: decide now (a lock needs a free slot, so this passes)
                self.eopen[r_bad, p] = False
        # fallback holders: keep their slot unless a tipper worth more needs it
        fb_unit = np.full((N, S), NEG)
        fb_who = np.full((N, S), -1)
        for p in range(self.prob.P):
            m = self.eopen[:, p] & self.efinal[:, p] & ~self.used[:, p]
            if not m.any():
                continue
            r = np.flatnonzero(m)
            slot_of = np.argmax(occ_old[r] == p, axis=1)
            has = (occ_old[r] == p).any(1) & free[r, slot_of]
            r, slot_of = r[has], slot_of[has]
            fb_unit[r, slot_of] = self.eunits[r, p]
            fb_who[r, slot_of] = p
        fb_mask = fb_who >= 0
        # 2. available tippers by tonight's expected score (or the lineup override)
        tonight = {p: (self.T.tonight[p, k] if self.pol.seat_by != "one_player_gain"
                       else self.T.W[p, k] - self.T.W[p, k + 1]) for p, k in tippers}
        if self.pol.seat_by == "highest_projection":
            tonight = {p: self.T.mean[p, k] for p, k in tippers}
        order = sorted(tippers, key=lambda pk: (-tonight[pk[0]], pk[0]))
        seated_t = {}
        for p, k in order:
            avail = (self.known[:, p] == 1) & ~self.used[:, p] & ~(new == p).any(1)
            if lineup is not None:
                if p not in lineup:
                    seated_t[p] = np.zeros(N, bool)
                    continue
                j = lineup[p]
                ok = avail & free[:, j] & self.elig[p, j]
                r = np.flatnonzero(ok)
                if fb_mask[r, j].any():
                    disp = r[fb_mask[r, j]]
                    self.eopen[disp, fb_who[disp, j]] = False
                    fb_mask[disp, j] = False
                new[r, j] = p
                free[r, j] = False
                seated_t[p] = ok
                continue
            hr = np.flatnonzero(avail)
            if not len(hr):
                seated_t[p] = np.zeros(N, bool)
                continue
            c1 = free[hr] & ~fb_mask[hr] & self.elig[p][None, :]
            if self.pol.seat_by == "highest_projection":
                cost1 = np.where(c1, np.arange(S)[None, :], big)
            else:
                cost1 = np.where(c1, flex[None, :], big)
            j1 = cost1.argmin(1)
            ok1 = c1[np.arange(len(hr)), j1]
            val = self.T.W_in[p, k]
            c2 = free[hr] & fb_mask[hr] & self.elig[p][None, :] & (fb_unit[hr] < val)
            cost2 = np.where(c2, flex[None, :], big)
            j2 = cost2.argmin(1)
            ok2 = ~ok1 & c2[np.arange(len(hr)), j2]
            j = np.where(ok1, j1, j2)
            ok = ok1 | ok2
            r, jj = hr[ok], j[ok]
            disp = fb_mask[r, jj]
            if disp.any():
                dr, dj = r[disp], jj[disp]
                self.eopen[dr, fb_who[dr, dj]] = False
                fb_mask[dr, dj] = False
            new[r, jj] = p
            free[r, jj] = False
            s = np.zeros(N, bool)
            s[r] = True
            seated_t[p] = s
        # 3. a held entry that stops an available tipper from being seated is decided now
        if lineup is None and self.pol.lock_rule == "earmark" and self.pol.defer:
            for p, k in order:
                miss = (self.known[:, p] == 1) & ~self.used[:, p] & ~seated_t.get(p, np.zeros(N, bool)) \
                    & ~(new == p).any(1)
                if not miss.any():
                    continue
                r = np.flatnonzero(miss)
                occ_r = new[r]
                hp = np.where(occ_r >= 0, occ_r, 0)
                is_holder = (occ_r >= 0) & self.eopen[r[:, None], hp] & ~self.efinal[r[:, None], hp] \
                    & ~self.locked[r] & self.elig[p][None, :]
                has = is_holder.any(1)
                if not has.any():
                    continue
                r = r[has]
                j = is_holder[has].argmax(1)
                h_of = new[r, j]
                for h in np.unique(h_of):
                    rr = r[h_of == h]
                    jj = j[h_of == h]
                    k_h = int(self.nk[h])
                    own = np.where(self.known[rr, h] == 1, self.T.W_in[h, k_h], self.T.W[h, k_h])
                    # his slot is needed now: lock him in it, or release it to the tipper
                    self.occ[rr] = new[rr]
                    allowed = np.zeros((len(rr), S), bool)
                    allowed[np.arange(len(rr)), jj] = True
                    self._decide(h, rr, own, k_h, allowed)
                    released = ~self.eopen[rr, h] & ~self.used[rr, h]
                    rel_r, rel_j = rr[released], jj[released]
                    new[rel_r, rel_j] = p
                    free[rel_r, rel_j] = False
        # fallback holders keep their slots where not displaced
        r, c = np.nonzero(fb_mask & (new == -1))
        new[r, c] = fb_who[r, c]
        free[r, c] = False
        # a fallback holder who lost his slot to a lock may move to another free slot he fits
        for p in range(self.prob.P):
            m = self.eopen[:, p] & self.efinal[:, p] & ~self.used[:, p] & ~(new == p).any(1)
            if not m.any():
                continue
            hr = np.flatnonzero(m)
            cand = free[hr] & self.elig[p][None, :]
            j = np.where(cand, flex[None, :], big).argmin(1)
            ok = cand[np.arange(len(hr)), j]
            new[hr[ok], j[ok]] = p
            free[hr[ok], j[ok]] = False
        # anything locked during this batch keeps its locked occupant
        lk = self.locked & (new == -1)
        new[lk] = self.occ[lk]
        self.occ = new

    def _close(self):
        """Week end: each unlocked slot scores its occupant's eligible final game (fallback)."""
        for p in range(self.prob.P):
            m = self.eopen[:, p] & self.efinal[:, p] & ~self.used[:, p]
            if not m.any():
                continue
            seated = (self.occ == p).any(1) & ~(self.locked & (self.occ == p)).any(1)
            cnt = m & seated
            self.total[cnt] += self.eunits[cnt, p]
            if self.contributed is not None:
                self.contributed[cnt, p] = True

    # --- run ---------------------------------------------------------------------
    def run(self, sc: SideScenarios, action: Action | None = None, record: bool = False) -> SimResult:
        prob = self.prob
        N, P, S = sc.n, prob.P, prob.S
        if action is not None and not is_legal(prob, action):
            raise ValueError("Illegal root action")
        self.N = N
        self.view = InformationView(prob, sc)
        self.locked = np.repeat(prob.locked[None, :], N, 0)
        self.total = np.full(N, float(prob.locked_units))
        self.occ = np.repeat(prob.occupant[None, :], N, 0).astype(int)
        self.used = np.repeat(prob.used[None, :], N, 0)
        self.eopen = np.zeros((N, P), bool)
        self.efinal = np.zeros((N, P), bool)
        self.eunits = np.zeros((N, P))
        for e in prob.entries:
            self.eopen[:, e.p] = True
            self.efinal[:, e.p] = e.final
            self.eunits[:, e.p] = e.units
        self.known = np.zeros((N, P), np.int8)
        self.nk = np.array([int((pl.tips <= 0).sum()) for pl in prob.players], dtype=int)
        self.liveg = np.array([int(np.flatnonzero(pl.live)[0]) if pl.live.any() else -1
                               for pl in prob.players], dtype=int)
        self.started = np.repeat(prob.started_live[None, :], N, 0)
        self.frozen = self.started.copy()
        self.contributed = np.zeros((N, P), bool) if record else None
        self.first_seated = np.zeros((N, P), bool) if record else None
        self.forced = np.zeros(N, int) if record else None
        self.first_occ = None
        self._seen_tip = False
        lineup = None
        if action is not None:
            for p, j in action.locks:
                if not self.eopen[0, p]:
                    raise ValueError(f"lock of player {p} who holds no open entry")
                self._apply_lock(np.arange(N), p, np.full(N, j))
            for p in action.passes:
                self.eopen[:, p] = False
            if action.lineup is not None:
                lineup = dict(action.lineup)
        for (t, kind), games in self.events:
            self.view.advance(t)
            if kind == 0:
                self._final(games)
            else:
                self._tip(games, lineup)
                lineup = None
        self._close()
        return SimResult(self.total.copy(), self.contributed, self.first_seated, self.forced, self.first_occ)


def simulate(prob: SideProblem, sc: SideScenarios, action: Action | None = None,
             policy: PolicyConfig | None = None, record: bool = False) -> SimResult:
    return WeekSim(prob, policy or PolicyConfig()).run(sc, action, record)
