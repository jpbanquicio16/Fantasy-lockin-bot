"""Chronological rollout evaluator for recorded/toy weeks.

Re-roots at each base-policy lock decision on revealed state. Fresh planning scenarios are
supplied by the caller, independent of the realised trajectory. Intended for evaluation;
production recommendations use the vectorised candidate engine.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from lockin_core.policy.tables import ContinuationTables
from lockin_core.sim.engine import WeekSim, simulate
from lockin_core.sim.problem import Action, PolicyConfig, RootEntry, SideProblem, SideScenarios
from lockin_core.state.legality import is_legal


class RolloutSim(WeekSim):
    def __init__(self, problem, sample_future, scenarios=200, seed=2026):
        super().__init__(problem, PolicyConfig())
        self.sample_future = sample_future
        self.planning_count = scenarios
        self.rng = np.random.default_rng(seed)

    def _decide(self, p, rows, own, k_next, allowed=None):
        for local, row in enumerate(rows):
            prob, draws = self._root(int(row))
            opts = [Action(label='keep')]
            if not self.efinal[row,p]:
                opts.append(Action(passes=(p,),label='pass'))
            for j in range(prob.S):
                if allowed is not None and not allowed[local,j]:
                    continue
                a = Action(locks=((p,j),),label='lock')
                if is_legal(prob,a): opts.append(a)
            totals = [simulate(prob,draws,a).total.mean() for a in opts]
            best = opts[int(np.argmax(totals))]
            if best.locks:
                self._apply_lock(np.array([row]), p, np.array([best.locks[0][1]]))
            elif best.passes:
                self.eopen[row,p] = False
            else:
                super()._decide(p, np.array([row]), np.array([own[local]]), k_next,
                                None if allowed is None else allowed[local:local+1])

    def _root(self, row):
        t = self.view.t
        original = self.prob
        players, indices = [], []
        for pl in original.players:
            keep = np.flatnonzero(pl.ends > t)
            indices.append(keep)
            tips = pl.tips[keep]-t
            # Current tip statuses are known, seating has not happened yet.
            tips = np.where(tips == 0, 1e-6, tips)
            players.append(replace(pl, tips=tips, ends=pl.ends[keep]-t,
                                   final=pl.final[keep], live=tips<0,
                                   game_ids=[pl.game_ids[k] for k in keep]))
        entries = [RootEntry(p,float(self.eunits[row,p]),bool(self.efinal[row,p]))
                   for p in range(original.P) if self.eopen[row,p] and not self.used[row,p]]
        prob = SideProblem(original.slots, players, self.locked[row].copy(), float(self.total[row]),
                           self.occ[row].copy(), self.used[row].copy(), entries, self.frozen[row].copy())
        P,K=prob.P,prob.K
        W=np.zeros((P,K+1)); Wi=W.copy(); mean=np.zeros((P,K)); tonight=mean.copy()
        prior = self.sample_future(self.planning_count,int(self.rng.integers(0,2**31)))
        out=np.ones((prior.n,P,K),bool); scr=np.zeros_like(out); units=np.zeros((prior.n,P,K))
        for p,keep in enumerate(indices):
            n=len(keep)
            if not n: continue
            W[p,:n]=self.T.W[p,keep]; Wi[p,:n]=self.T.W_in[p,keep]
            mean[p,:n]=self.T.mean[p,keep]; tonight[p,:n]=self.T.tonight[p,keep]
            out[:,p,:n]=prior.out[:,p,keep]; scr[:,p,:n]=prior.scr[:,p,keep]
            units[:,p,:n]=prior.units[:,p,keep]
            if self.known[row,p] != 0:
                out[:,p,0] = self.known[row,p] == -1
            elif players[p].live.any():
                out[:,p,0] = False
        prob.tables=ContinuationTables(W,Wi,mean,tonight,prob.flex)
        return prob,SideScenarios(out,scr,units)
