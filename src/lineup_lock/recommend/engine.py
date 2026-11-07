"""Rollout recommendations: the action that is best when base policy v2 plays the rest of the
week, on shared scenarios (design sections 12-16).

1. Candidates around the base action (``candidates``), screened on 500 scenarios; entries whose
   slots overlap are improved together by coordinate ascent (two passes).
2. Finalists within 2 paired SE of the best (at most 12, plus the protected base action and each
   due entry's lock-now / pass) re-evaluated on a **fresh** stream from 2,000 scenarios, doubling
   until the top action's paired SE against each runner-up is under 0.2 FP and the top two keep
   their order, up to 16,000 scenarios or 20 seconds (then labelled Monte Carlo-limited).
3. For every open entry: lock now (each slot) / pass now / keep open, around the chosen action;
   the value of waiting and the conditional rule from the same scenarios split by his next-game
   status; the default if you are unreachable at the deadline; a close-call label from the model
   band (30 parameter draws); the score at which the decision would flip.
4. Expected points and P(win) are both computed and never added. The opponent is simulated with
   the same base policy, with "never locks" and "locks any above-average score" as sensitivity.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from lockin_core.config import SimConfig, code_version, config_hash, tzdata_version
from lockin_core.schedule.calendar import fmt_local
from lockin_core.sim.build import WeekModel
from lockin_core.sim.engine import WeekSim
from lockin_core.sim.problem import BASELINES, Action, PolicyConfig, SideProblem, SideScenarios
from lineup_lock.recommend.candidates import combine, generate, is_legal, lock_slot

OPP_POLICIES = {"base": BASELINES["base_v2"], "never_lock": BASELINES["never_lock"],
                "lock_above_average": BASELINES["above_average"]}


def paired_se(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0


def p_win(me: np.ndarray, opp: np.ndarray, tie: float = 0.5) -> float:
    return float((me > opp).mean() + tie * (me == opp).mean())


@dataclass
class Evaluation:
    action: Action
    u: np.ndarray  # [N] week totals (units)

    @property
    def mean(self) -> float:
        return float(self.u.mean())


class Recommender:
    def __init__(self, wm: WeekModel, scfg: SimConfig, policy: PolicyConfig | None = None):
        self.wm = wm
        self.scfg = scfg
        bp = scfg.base_policy
        self.policy = policy or PolicyConfig(assignment=bp.assignment, seat_by=bp.seat_by,
                                             value_open_entries=bp.value_open_entries, defer=bp.defer_to_deadline)
        self.me: SideProblem = wm.me
        self.sim = WeekSim(self.me, self.policy)
        self.res = wm.state.units_per_point
        self.timings: dict[str, float] = {}

    # --- evaluation --------------------------------------------------------------
    def evaluate(self, acts: list[Action], ss: SideScenarios, record: bool = False):
        return {a.key(): Evaluation(a, self.sim.run(ss, a).total) for a in acts}

    def screen(self, acts: list[Action], protected: list[Action]) -> list[Action]:
        cc = self.scfg.candidates
        sc = self.wm.scenarios(cc.screen_scenarios, "screen")
        ss = self.wm.side_scenarios(sc, self.me)
        ev = self.evaluate(acts, ss)
        opp = None
        if getattr(self, "objective", "points") == "win":
            so = self.wm.side_scenarios(sc, self.wm.opp)
            opp = WeekSim(self.wm.opp, OPP_POLICIES["base"]).run(so).total
            ev = {k: Evaluation(e.action, (e.u > opp).astype(float) + self.wm.ctx.switches.tie_value * (e.u == opp))
                  for k, e in ev.items()}
        best = max(ev.values(), key=lambda e: e.mean)
        singles = [e.action for e in ev.values() if (e.action.locks or e.action.passes) and e.action.lineup is None]
        for _ in range(2):  # coordinate ascent: add other entries' best decisions while they help
            improved = False
            for o in sorted(singles, key=lambda a: -ev[a.key()].mean):
                c = combine(self.me, best.action, o)
                if c is None or c.key() in ev:
                    continue
                u = self.sim.run(ss, c).total
                if opp is not None:
                    u = (u > opp).astype(float) + self.wm.ctx.switches.tie_value * (u == opp)
                e = Evaluation(c, u)
                ev[c.key()] = e
                if e.mean > best.mean + 1e-9:
                    best, improved = e, True
            if not improved:
                break
        ranked = sorted(ev.values(), key=lambda e: -e.mean)
        keep = [e.action for e in ranked if best.mean - e.mean <= cc.keep_within_paired_se * paired_se(best.u, e.u)
                or e is best][:cc.finalists]
        for a in protected:
            if a.key() not in {k.key() for k in keep}:
                keep.append(a)
        self.screened = len(ev)
        return keep

    def final(self, acts: list[Action], deadline_s: float, fixed_scenarios: int | None = None):
        sc_cfg = self.scfg.scenarios
        n = fixed_scenarios or min(sc_cfg.start, sc_cfg.max)
        target = sc_cfg.target_paired_se_fp * self.res
        prev_order = None
        t0 = time.time()
        while True:
            sc = self.wm.scenarios(n, "final")
            ss = self.wm.side_scenarios(sc, self.me)
            ev = self.evaluate(acts, ss)
            ranked = sorted(ev.values(), key=lambda e: -e.mean)
            best = ranked[0]
            ses = [paired_se(best.u, e.u) for e in ranked[1:]]
            order = [e.action.key() for e in ranked[:2]]
            done = all(s < target for s in ses) and (prev_order is None or order == prev_order or len(ranked) < 2)
            elapsed = time.time() - t0
            limited = False
            if fixed_scenarios or done or n >= sc_cfg.max or elapsed * 2.2 > min(sc_cfg.wall_clock_s, deadline_s):
                limited = not done
                return sc, ss, ev, ranked, n, limited
            prev_order = order
            n = min(n * 2, sc_cfg.max)

    # --- per-entry analysis -------------------------------------------------------
    def entry_options(self, best: Action, p: int) -> dict[str, Action]:
        """Lock (each slot type) / pass / keep for entry p, everything else as in ``best``."""
        locks = tuple((q, j) for q, j in best.locks if q != p)
        passes = tuple(q for q in best.passes if q != p)
        taken = {j for _, j in locks}
        out = {"keep": Action(locks=locks, passes=passes, lineup=best.lineup, label=f"keep:{p}")}
        e = next(e for e in self.me.entries if e.p == p)
        if not e.final:
            out["pass"] = Action(locks=locks, passes=tuple(sorted(passes + (p,))), lineup=best.lineup,
                                 label=f"pass:{p}")
        for cls in self.me.classes_of(p):
            j = lock_slot(self.me, p, cls, taken)
            if j is None:
                continue
            a = Action(locks=tuple(sorted(locks + ((p, j),))), passes=passes, lineup=best.lineup,
                       label=f"lock:{p}:{j}")
            if is_legal(self.me, a):
                out[f"lock:{self.me.slots[j]}:{j}"] = a
        return out

    # --- main --------------------------------------------------------------------
    def run(self, objective: str | None = None, epoch_type: str = "manual", trigger: str = "",
            fixed_scenarios: int | None = None) -> dict:
        t_start = time.time()
        wm, me, res, scfg = self.wm, self.me, self.res, self.scfg
        state, ctx = wm.state, wm.ctx
        objective = objective or state.objective
        self.objective = objective
        opponent_confirmed = (wm.opp is not None and state.opp is not None and state.opp_locked_manual is not None
                              and abs(state.opp.locked_units / res - state.opp_locked_manual) < 1e-9)
        if objective == "win" and not opponent_confirmed:
            raise ValueError("Record the opponent's locked total and matching individual slot locks before requesting P(win)")
        acts, protected = generate(me, scfg.candidates.max_actions, scfg.candidates.top_lineups)
        self.timings["candidates"] = time.time() - t_start
        finalists = self.screen(acts, protected)
        self.timings["screen"] = time.time() - t_start
        sc, ss, ev, ranked, n, mc_limited = self.final(finalists, scfg.scenarios.wall_clock_s, fixed_scenarios)
        self.timings["final"] = time.time() - t_start
        # opponent
        opp_u: dict[str, np.ndarray] = {}
        if opponent_confirmed:
            so = wm.side_scenarios(sc, wm.opp)
            for name in scfg.opponent_policies:
                pol = OPP_POLICIES.get(name)
                if pol is not None:
                    opp_u[name] = WeekSim(wm.opp, pol).run(so).total
        tie = ctx.switches.tie_value
        pw = {a: {n_: p_win(e.u, o, tie) for n_, o in opp_u.items()} for a, e in ev.items()}
        by_points = ranked[0]
        if objective == "win" and opp_u:
            chosen = max(ranked, key=lambda e: pw[e.action.key()]["base"])
        else:
            chosen = by_points
        best = chosen.action
        # Turn per-entry tie-breaks/defaults into one legal joint action before rendering.
        # In win mode retain the action selected by win utility.
        if objective == "points":
            for e in me.entries:
                opts = self.entry_options(best, e.p)
                values = {k: self.sim.run(ss, a).total for k, a in opts.items()}
                card = self._entry_card(e, opts, values, ss, sc)
                proposed = opts[card["_choice"]]
                if is_legal(me, proposed):
                    best = proposed
            chosen = Evaluation(best, self.sim.run(ss, best).total)
            if best.key() not in ev:
                ev[best.key()] = chosen
                pw[best.key()] = {name: p_win(chosen.u, o, tie) for name, o in opp_u.items()}
        rec_run = self.sim.run(ss, best, record=True)
        # per-entry analysis around the chosen action, on the same scenarios
        band_sc = None
        entries_out = []
        for e in me.entries:
            opts = self.entry_options(best, e.p)
            oev = {k: (ev[a.key()].u if a.key() in ev else self.sim.run(ss, a).total) for k, a in opts.items()}
            card = self._entry_card(e, opts, oev, ss, sc)
            locked_slot = dict(best.locks).get(e.p)
            if locked_slot is not None:
                choice = next(k for k, a in opts.items() if (e.p, locked_slot) in a.locks)
                card.update(advice="lock_now", lock_slot=me.slots[locked_slot], lock_slot_index=locked_slot,
                            lock_slot_if_locking=me.slots[locked_slot], _choice=choice)
            elif e.p in best.passes:
                card.update(advice="pass", lock_slot=None, _choice="pass")
            else:
                card.update(advice="keep_fallback" if e.final else "wait", lock_slot=None, _choice="keep")
                if card.get("conditional_rule") is None:
                    card["conditional_rule"] = {"ruled_out": card["default_if_unattended"],
                                                "available": card["default_if_unattended"],
                                                "reason": "Re-evaluate at the next event; use the stored default if no update arrives."}
            card["why"] = explain_entry(card)
            entries_out.append(card)
        # model band for each entry's top two options
        if entries_out and scfg.close_call.model_draws > 1:
            B, per = scfg.close_call.model_draws, scfg.close_call.scenarios_per_draw
            band_sc = wm.scenarios(B * per, "band", batches=B, rate_draws=True)
            bss = wm.side_scenarios(band_sc, me)
            for card in entries_out:
                self._band(card, bss, band_sc.batch, B)
        for card in entries_out:  # without a band run: Monte Carlo label only
            if "_opts" in card:
                for k in ("_opts", "_u", "_choice"):
                    card.pop(k, None)
                card.setdefault("label", "clear" if abs(card["delta_vs_runner_up_fp"]) > 2 * card["paired_se_fp"]
                                else "toss-up")
        lineup = self._lineup(best, rec_run, sc)
        players = self._players(rec_run, sc)
        locked_me = me.locked_units
        out = {
            "rec_id": f"r_{state.t:%Y-%m-%dT%H:%MZ}_{hashlib.sha256(state.snapshot_id.encode()).hexdigest()[:4]}",
            "created_at": state.t.isoformat(),
            "expires_at": self._expires().isoformat(),
            "epoch_type": epoch_type, "trigger": trigger, "objective": objective,
            "season": state.season, "week": state.week, "snapshot_id": state.snapshot_id,
            "summary": {
                "expected_points": round(chosen.mean / res, 2),
                "expected_points_best_points_action": round(by_points.mean / res, 2),
                "locked_points": locked_me / res,
                "p_win": round(pw[best.key()].get("base", np.nan), 3) if opp_u else None,
                "opp_expected_points": round(float(opp_u["base"].mean()) / res, 2) if "base" in opp_u else None,
                "points_and_win_disagree": bool(opp_u and max(ranked, key=lambda e: pw[e.action.key()]["base"]).action.key()
                                                   != by_points.action.key()),
            },
            "best_action": self._action_json(best),
            "alternatives": [self._action_json(e.action) | {"expected_points": round(e.mean / res, 2),
                                                             "vs_best": round((e.mean - chosen.mean) / res, 2),
                                                             "paired_se": round(paired_se(chosen.u, e.u) / res, 2),
                                                             "p_win": (round(pw[e.action.key()].get("base", 0), 3)
                                                                       if opp_u else None)}
                             for e in ranked[:6]],
            "entries": entries_out,
            "lineup_next_tip": lineup,
            "players": players,
            "matchup": self._matchup(pw, best, opp_u, ranked),
            "could_change": self._could_change(entries_out),
            "data_freshness": {k: v.isoformat() if isinstance(v, datetime) else str(v)
                               for k, v in state.freshness.items()},
            "flags": list(state.flags) + (["monte_carlo_limited"] if mc_limited else []),
            "versions": {**state.versions, "projection": wm.params.version,
                         "config": config_hash(scfg, wm.mcfg, ctx.cfg), "tzdata": tzdata_version(),
                         "code": code_version()},
            "n_scenarios": n, "n_candidates_screened": self.screened, "n_finalists": len(finalists),
            "runtime_s": round(time.time() - t_start, 2), "timings": {k: round(v, 2) for k, v in self.timings.items()},
        }
        out["fingerprint"] = fingerprint(out)
        return out

    # --- pieces --------------------------------------------------------------------
    def _name(self, p: int) -> str:
        return self.me.players[p].name

    def _action_json(self, a: Action) -> dict:
        me, res = self.me, self.res
        units = {e.p: e.units for e in me.entries}
        return {"label": a.label,
                "locks": [{"player": self._name(p), "player_id": me.players[p].pid, "slot": me.slots[j],
                           "slot_index": j, "score": units.get(p, 0) / res} for p, j in a.locks],
                "passes": [{"player": self._name(p), "player_id": me.players[p].pid} for p in a.passes],
                "lineup": None if a.lineup is None else [{"player": self._name(p), "player_id": me.players[p].pid, "slot": me.slots[j], "slot_index": j}
                                                         for p, j in a.lineup]}

    def _entry_card(self, e, opts: dict[str, Action], oev: dict[str, np.ndarray], ss: SideScenarios, sc) -> dict:
        me, res, scfg = self.me, self.res, self.scfg
        pl = me.players[e.p]
        state, cal = self.wm.state, self.wm.ctx.calendar
        se_entry = next(x for x in state.me.entries if x.entry_id == e.entry_id)
        means = {k: float(v.mean()) for k, v in oev.items()}
        lock_keys = [k for k in means if k.startswith("lock:")]
        best_lock = max(lock_keys, key=lambda k: means[k]) if lock_keys else None
        now_opts = [k for k in (best_lock, "pass") if k is not None and k in means]
        best_now = max(now_opts, key=lambda k: means[k]) if now_opts else "keep"
        can_wait = (not e.must_decide) and not e.final
        voi_keep = means["keep"] - means[best_now] if best_now in means else 0.0
        # value of waiting from the same scenarios, split by his next-game status
        rule, voi_rule, branch_n = None, None, None
        nxt = next((k for k in range(pl.n_games) if pl.tips[k] > 0), None)
        if nxt is not None and best_lock is not None and "pass" in means:
            z = ss.out[:, e.p, nxt]
            nz = int(z.sum())
            branch_n = {"ruled_out": nz, "available": int(len(z) - nz)}
            if 0 < nz < len(z):
                q = {}
                for br, m in (("ruled_out", z), ("available", ~z)):
                    q[br] = {"lock": float(oev[best_lock][m].mean()), "pass": float(oev["pass"][m].mean())}
                pz = nz / len(z)
                ideal = pz * max(q["ruled_out"].values()) + (1 - pz) * max(q["available"].values())
                voi_rule = ideal - max(means[best_lock], means["pass"])
                rule = {br: max(v, key=v.get) for br, v in q.items()}
                rule["values"] = {br: {k: round(x / res, 2) for k, x in v.items()} for br, v in q.items()}
                rule["enough_scenarios"] = min(nz, len(z) - nz) >= scfg.wait.min_branch_scenarios
        margin = scfg.wait.min_voi_fp * res
        report_fresh = (self.wm.ctx.report_published_at is not None
                        and (state.t - self.wm.ctx.report_published_at).total_seconds() <= 45 * 60)
        wait = (can_wait and report_fresh and rule is not None and rule["enough_scenarios"]
                and next_report(state.t) < se_entry.deadline_at and voi_keep >= margin)
        if e.final:
            advice = "lock_now" if best_lock and means[best_lock] >= means["keep"] - 1e-9 else "keep_fallback"
            choice = best_lock if advice == "lock_now" else "keep"
        elif wait:
            advice, choice = "wait", "keep"
        else:
            advice, choice = ("lock_now", best_lock) if best_now == best_lock else ("pass", "pass")
        others = sorted([k for k in means if k != choice], key=lambda k: -means[k])
        runner = others[0] if others else None
        delta = means[choice] - means[runner] if runner else 0.0
        se = paired_se(oev[choice], oev[runner]) if runner else 0.0
        flip = None
        if best_lock is not None:
            alt = max([k for k in ("pass", "keep") if k in means], key=lambda k: means[k])
            flip = (e.units - (means[best_lock] - means[alt])) / res  # lock pays for scores at or above this
        default = (best_lock if (best_lock and means[best_lock] >= means.get("pass", -np.inf)) else "pass")
        lock_slot_name = best_lock.split(":")[1] if best_lock else None
        cur = next((j for j in range(me.S) if me.occupant[j] == e.p), None)
        move_note = None
        if best_lock is not None:
            j_lock = int(best_lock.split(":")[2])
            if cur is not None and j_lock != cur:
                other = me.occupant[j_lock]
                move_note = (f"move {pl.name} from {me.slots[cur]} to {me.slots[j_lock]} first"
                             + (f" ({me.players[other].name} goes to {me.slots[cur]} or the bench)" if other >= 0 else ""))
        game = cal.game(pl.game_ids[nxt]) if nxt is not None else None
        values = {}
        for k, v in means.items():
            label = {"keep": "keep_open", "pass": "pass_now"}.get(k, f"lock_{k.split(':')[1]}_now" if k.startswith("lock:") else k)
            values[label] = round(v / res, 2)
        card = {
            "entry_id": e.entry_id, "player": pl.name, "player_id": pl.pid, "score": e.units / res,
            "game_id": se_entry.game_id, "is_team_final": e.final,
            "eligible_slots": sorted({me.slots[c[0]] for c in me.classes_of(e.p)}),
            "advice": advice, "lock_slot": lock_slot_name if advice == "lock_now" else None,
            "lock_slot_if_locking": lock_slot_name,
            "lock_slot_index": int(best_lock.split(":")[2]) if best_lock else None,
            "current_slot": None if cur is None else me.slots[cur], "move_note": move_note,
            "deadline_at": se_entry.deadline_at.isoformat(),
            "deadline_local": fmt_local(se_entry.deadline_at, self.wm.ctx.cfg.tz),
            "values": values,
            "voi_fp": round(voi_keep / res, 2), "voi_status_split_fp": None if voi_rule is None else round(voi_rule / res, 2),
            "conditional_rule": rule, "branch_scenarios": branch_n,
            "next_game": None if game is None else {"game_id": game.game_id, "tip_local": fmt_local(game.tip_at, self.wm.ctx.cfg.tz),
                                                    "p_play": round(float((~ss.out[:, e.p, nxt] & ~ss.scr[:, e.p, nxt]).mean()), 2)},
            "default_if_unattended": ("lock in " + default.split(":")[1]) if default.startswith("lock:") else "pass",
            "delta_vs_runner_up_fp": round(delta / res, 2), "runner_up": runner, "paired_se_fp": round(se / res, 2),
            "flip_score": None if flip is None else round(flip, 1),
            "locked_unknown": se_entry.locked_unknown, "eligibility": se_entry.eligibility,
            "fp_source": se_entry.fp_source, "_choice": choice,
        }
        card["_u"] = {k: v for k, v in oev.items()}
        card["_opts"] = opts
        card["why"] = explain_entry(card)
        return card

    def _band(self, card: dict, bss: SideScenarios, batch: np.ndarray, B: int) -> None:
        opts = card.pop("_opts")
        u = card.pop("_u")
        choice, runner = card.pop("_choice"), card["runner_up"]
        res, cc = self.res, self.scfg.close_call
        if runner is None:
            card["label"], card["band_fp"], card["p_better"] = "clear", None, 1.0
            return
        a = self.sim.run(bss, opts[choice]).total
        b = self.sim.run(bss, opts[runner]).total
        d = np.array([(a[batch == k] - b[batch == k]).mean() for k in range(B)])
        p10, p90 = np.percentile(d, [10, 90])
        p_better = float((d > 0).mean())
        lo, hi = cc.lean_band
        if p10 > 0:
            label = "clear"
        elif lo <= p_better <= hi:
            label = "lean"
        elif p_better > hi:
            label = "clear" if p10 > -0.25 * res else "lean"
        else:
            label = "toss-up"
        if label == "toss-up" and abs(card["delta_vs_runner_up_fp"]) < cc.either_is_fine_fp:
            label = "either is fine"
        card["label"] = label
        card["band_fp"] = [round(p10 / res, 2), round(p90 / res, 2)]
        card["p_better"] = round(p_better, 2)
        _ = u

    def _lineup(self, best: Action, rr, sc) -> dict | None:
        me = self.me
        from lineup_lock.recommend.candidates import next_batch

        nb = next_batch(me)
        if nb is None or rr.first_seated is None:
            return None
        t, pairs = nb
        tip_at = self.wm.state.t + timedelta(minutes=t)
        start, bench, due = [], [], []
        locked_now = {q for q, _ in best.locks}
        holders = {e.p: e for e in me.entries if not e.final}
        for p, k in pairs:
            if p in locked_now:
                continue
            if p in holders and p not in best.passes:
                due.append({"player": me.players[p].name, "player_id": me.players[p].pid,
                            "score": holders[p].units / self.res})
                continue
            ps = float(rr.first_seated[:, p].mean())
            slot = None
            if rr.first_occ is not None and ps > 0:
                rows = rr.first_seated[:, p]
                sl = (rr.first_occ[rows] == p).argmax(1)
                vals, cnt = np.unique(sl, return_counts=True)
                slot = me.slots[int(vals[cnt.argmax()])]
            item = {"player": me.players[p].name, "player_id": me.players[p].pid, "slot": slot,
                    "p_start": round(ps, 2)}
            (start if ps >= 0.5 else bench).append(item)
        held = [{"player": me.players[e.p].name, "score": e.units / self.res} for e in me.entries if not e.final
                and e.p not in {q for q, _ in best.locks} and e.p not in best.passes]
        return {"tip_at": tip_at.isoformat(), "tip_local": fmt_local(tip_at, self.wm.ctx.cfg.tz),
                "start": start, "bench_for_this_game": bench, "entries_due_at_this_tip": due,
                "holding_open_entries": held,
                "locked_before_tip": [{"player": self._name(p), "slot": me.slots[j]} for p, j in best.locks]}

    def _players(self, rr, sc) -> list[dict]:
        me = self.me
        out = []
        for p, pl in enumerate(me.players):
            pc = float(rr.contributed[:, p].mean()) if rr.contributed is not None else 0.0
            i = sc.index.get(pl.pid)
            p_play = [round(float(sc.plays[:, i, k].mean()), 2) for k in range(min(pl.n_games, sc.K))] if i is not None else []
            if me.used[p]:
                label, reason = "locked", ""
            elif pl.n_games == 0 or sum(p_play) < 0.05:
                label, reason = "out this week", "no remaining games he is expected to play"
            elif pc < 0.10:
                label = "unlikely to contribute this week"
                reason = "slots expected to be locked or filled by better options"
            else:
                label, reason = "contributes", ""
            out.append({"player": pl.name, "player_id": pl.pid, "p_contributes": round(pc, 2), "label": label,
                        "reason": reason, "games_left": pl.n_games, "p_play": p_play})
        return out

    def _matchup(self, pw, best, opp_u, ranked) -> dict:
        res = self.res
        st = self.wm.state
        if not opp_u:
            return {"p_win": None, "p_win_reason": "opponent locked total unconfirmed or no opponent this week"}
        by_pol = pw[best.key()]
        best_by_pol = {n_: max(ranked, key=lambda e: pw[e.action.key()][n_]).action.key() for n_ in opp_u}
        sensitive = len(set(best_by_pol.values())) > 1
        return {"my_locked": self.me.locked_units / res, "opp_locked": (self.wm.opp.locked_units / res) if self.wm.opp else None,
                "opp_locked_source": "manual" if st.opp_locked_manual is not None else "inferred from matchup points",
                "p_win": round(by_pol.get("base", np.nan), 3),
                "p_win_by_opponent_policy": {k: round(v, 3) for k, v in by_pol.items()},
                "sensitive_to_opponent_model": sensitive,
                "opp_expected_points": round(float(opp_u["base"].mean()) / res, 2)}

    def _could_change(self, entries: list[dict]) -> list[str]:
        out = []
        for c in entries:
            if c["advice"] == "wait" and c.get("conditional_rule"):
                r = c["conditional_rule"]
                out.append(f"{c['player']}'s status for {c['next_game']['tip_local'] if c.get('next_game') else 'his next game'}"
                           f" decides it: {r['ruled_out']} if ruled out, {r['available']} if cleared")
        st = self.wm.state
        cal = self.wm.ctx.calendar
        live = [g for g in cal.live_games()]
        if live:
            out.append(f"{len(live)} game(s) in progress: new lock decisions when they go final")
        if self.wm.ctx.report_published_at is not None:
            out.append(f"next injury report around {fmt_local(next_report(st.t), self.wm.ctx.cfg.tz)}")
        return out

    def _expires(self) -> datetime:
        st, cal = self.wm.state, self.wm.ctx.calendar
        teams = {st.players[p].team for s in (st.me, st.opp) if s for p in s.active if p in st.players}
        times = [g.tip_at for g in cal.games if g.status == "scheduled" and set(g.teams()) & teams]
        times += [e.deadline_at for e in st.me.entries]
        times += [g.end_at for g in cal.live_games()]
        if cal.is_game_day(st.t.astimezone(cal.week_start.tzinfo).date()):
            times.append(next_report(st.t))
        times = [t for t in times if t > st.t]
        return min(times) if times else cal.week_end


def next_report(t: datetime) -> datetime:
    m = (t.minute // 15 + 1) * 15
    return t.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=m)


def explain_entry(c: dict) -> str:
    v = c["values"]
    adv = c["advice"]
    if adv == "lock_now":
        locks = {k: x for k, x in v.items() if k.startswith("lock_")}
        best = v.get(f"lock_{c['lock_slot']}_now")
        alts = []
        if "keep_open" in v:
            alts.append(f"keeping it open {v['keep_open'] - best:+.1f}")
        if "pass_now" in v:
            alts.append(f"passing {v['pass_now'] - best:+.1f}")
        worse_slots = [f"in {k[5:-4]} {x - best:+.1f}" for k, x in locks.items() if x < best - 0.5]
        s = f"Locking in {c['lock_slot']} is best: " + ", ".join(alts + worse_slots) + " FP for the week."
        if c.get("flip_score") is not None:
            s += f" Locking pays for any score above about {c['flip_score']:.1f}."
        if c.get("move_note"):
            s = s[:-1] + f"; {c['move_note']}."
        return s
    if adv == "pass":
        lk = c.get("lock_slot_if_locking")
        lv = v.get(f"lock_{lk}_now") if lk else None
        s = "His remaining games and the bench are worth more than this score"
        if lv is not None:
            s += f": locking it ({lk}) would cost {lv - v['pass_now']:+.1f} FP"
        if c.get("flip_score") is not None:
            s += f"; locking would pay only for a score above about {c['flip_score']:.1f}"
        return s + "."
    if adv == "wait":
        r = c.get("conditional_rule")
        s = f"News before the deadline is worth {c['voi_fp']:.1f} FP: keep it open and decide by {c['deadline_local']}."
        if r and r["ruled_out"] != r["available"]:
            s += f" Rule: {r['ruled_out']} if he is ruled out of his next game, {r['available']} if he is cleared."
        s += f" If you cannot check, {c['default_if_unattended']}."
        return s
    if adv == "keep_fallback":
        return ("His team's last game this week: the score counts at week end if he stays in a starting slot; "
                "locking adds nothing unless a later starter would take his slot.")
    return ""


def fingerprint(rec: dict) -> str:
    key = {"entries": [(c["entry_id"], c["advice"], c.get("lock_slot")) for c in rec["entries"]],
           "lineup": None if not rec.get("lineup_next_tip") else
           sorted((x["player_id"], x["slot"]) for x in rec["lineup_next_tip"]["start"]),
           "best": rec["best_action"]}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:12]
