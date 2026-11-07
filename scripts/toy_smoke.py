import sys, time
import numpy as np
from lockin_core.policy.exact_dp import ExactDP, random_week, toy_problem, toy_scenarios, hindsight
from lockin_core.sim.engine import simulate
from lockin_core.sim.problem import BASELINES, PolicyConfig

n_players = int(sys.argv[1]) if len(sys.argv) > 1 else 4
weeks = int(sys.argv[2]) if len(sys.argv) > 2 else 5
rng = np.random.default_rng(7)
for w in range(weeks):
    wk = random_week(rng, n_players=n_players)
    t = time.time(); dp = ExactDP(wk).value(); t_dp = time.time() - t
    t = time.time(); nw = ExactDP(wk, wait=False).value(); t_nw = time.time() - t
    prob = toy_problem(wk)
    sc = toy_scenarios(wk, 20000, seed=w)
    res = {}
    for name, pol in [("v2", BASELINES["base_v2"]), ("v2_exact", PolicyConfig(assignment="exact")),
                      ("greedy", BASELINES["greedy_nov4"]), ("indep", BASELINES["independent"]),
                      ("above_avg", BASELINES["above_average"]), ("never", BASELINES["never_lock"])]:
        t = time.time(); r = simulate(prob, sc, policy=pol).total; res[name] = (r.mean()/2, r.std()/2/np.sqrt(len(r)), time.time()-t)
    hs = hindsight(wk, sc.subset(np.arange(4000))).mean()/2
    print(f"week {w}: DP {dp/2:.2f} ({t_dp:.1f}s)  no-wait {nw/2:.2f} ({t_nw:.1f}s)  hindsight {hs:.2f}")
    for k, (m, se, tt) in res.items():
        print(f"    {k:10s} {m:7.2f} ±{se:.2f}  regret {dp/2 - m:6.2f}  ({tt*1000:.0f} ms)")
