"""Policies against the exact optimum on small roster-weeks (design sections 12-13).

Each toy week (``exact_dp.random_week``): slots G, F, UTIL; G-only, F-only, G/F or C players;
1-4 games on distinct days; 0-50% ruled-out chance per game; a 2% late scratch; three-point
score distributions. The optimum (waiting allowed) and the best no-wait policy are exact; each
simulated policy is evaluated on the same scenarios (common random numbers); the hindsight
bound uses Monte Carlo. Synthetic: evidence about methods, not about your league.
"""

from __future__ import annotations

import time

import numpy as np

from lockin_core.config import Paths
from lockin_core.policy.exact_dp import ExactDP, hindsight, random_week, toy_problem, toy_scenarios
from lockin_core.sim.engine import simulate
from lockin_core.sim.problem import BASELINES, PolicyConfig

POLICIES = {
    "Base policy v2, exact assignment": PolicyConfig(assignment="exact"),
    "Base policy v2, fast assignment (production)": PolicyConfig(),
    "v2, other open entries valued at continuation only": PolicyConfig(value_open_entries=False),
    "v2, decide each entry when its game ends (no deferral)": PolicyConfig(defer=False),
    "v2, seat by one-player gain": PolicyConfig(seat_by="one_player_gain"),
    "Greedy threshold with assignment (4 Nov)": BASELINES["greedy_nov4"],
    "Independent per-slot thresholds, best slot": PolicyConfig(lock_rule="independent_best", defer=False,
                                                              seat_by="highest_projection"),
    "Independent per-slot thresholds, first eligible slot": BASELINES["independent"],
    "Lock any score >= his 75th percentile": BASELINES["above_p75"],
    "Lock any score >= his average": BASELINES["above_average"],
    "Never lock (final-game fallback)": BASELINES["never_lock"],
}


def main(weeks: int = 200, players: int = 5, scenarios: int = 20000, seed: int = 2026, rollout: int = 0) -> str:
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {k: [] for k in POLICIES}
    rows["Exact DP, must decide right after each final"] = []
    rows["Hindsight oracle (upper bound)"] = []
    exact, t_dp = [], []
    rollout_regret = []
    t0 = time.time()
    for w in range(weeks):
        wk = random_week(rng, n_players=players)
        t = time.time()
        v = ExactDP(wk).value()
        t_dp.append(time.time() - t)
        exact.append(v)
        rows["Exact DP, must decide right after each final"].append(v - ExactDP(wk, wait=False).value())
        prob = toy_problem(wk)
        sc = toy_scenarios(wk, scenarios, seed=seed * 1000 + w)
        for name, pol in POLICIES.items():
            n = sc.n if pol.assignment == "fast" else min(sc.n, 5000)
            r = simulate(prob, sc.subset(np.arange(n)), policy=pol).total
            rows[name].append(v - r.mean())
        rows["Hindsight oracle (upper bound)"].append(v - hindsight(wk, sc.subset(np.arange(min(4000, sc.n)))).mean())
        if w < rollout:
            from lockin_core.policy.rollout import RolloutSim
            # Each outer trajectory gets fresh planning draws; no realised future enters the policy.
            outer = min(scenarios, 64)
            rs = RolloutSim(prob, lambda n, seed: toy_scenarios(wk, n, seed), scenarios=200, seed=seed+w)
            total = rs.run(sc.subset(np.arange(outer))).total
            rollout_regret.append(v - total.mean())
        if (w + 1) % 20 == 0:
            print(f"  {w + 1}/{weeks} weeks ({time.time() - t0:.0f} s)")
    ex = np.array(exact) / 2
    lines = [f"# Policies vs the exact optimum on {weeks} toy roster-weeks ({players} players, slots G/F/UTIL)", "",
             f"Mean exact value {ex.mean():.2f} FP; exact DP {np.mean(t_dp):.2f} s per week (max {np.max(t_dp):.1f}); "
             f"{scenarios} scenarios per week for simulated policies (5,000 for exact assignment). Synthetic.", "",
             "| Policy | Mean regret (FP / week) | SE | 90th pct | Max | % of weeks > 0.5 FP | % of exact value |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in rows.items():
        r = np.array(r) / 2
        lines.append(f"| {name} | {r.mean():.2f} | {r.std(ddof=1) / np.sqrt(len(r)):.2f} | "
                     f"{np.percentile(r, 90):.2f} | {r.max():.2f} | {100 * (r > 0.5).mean():.1f} | "
                     f"{100 * (1 - r.sum() / ex.sum()):.1f} |")
    fast = np.array(rows["Base policy v2, fast assignment (production)"]) / 2
    greedy = np.array(rows["Greedy threshold with assignment (4 Nov)"]) / 2
    d = greedy - fast
    lines += ["", f"v2 (fast) against the 4 Nov greedy policy, paired: {d.mean():.2f} FP better "
                  f"(SE {d.std(ddof=1) / np.sqrt(len(d)):.2f}); better in {100 * (d > 0).mean():.0f}% of weeks.",
              f"Every policy at or below the optimum (within Monte Carlo error): min regret "
              f"{min(np.min(r) for k, r in rows.items() if 'Hindsight' not in k) / 2:.2f} FP."]
    if rollout_regret:
        rr = np.array(rollout_regret) / 2
        lines += ["", f"Chronological rollout: mean regret {rr.mean():.2f} FP across {len(rr)} weeks; "
                  f"{min(scenarios, 64)} outer trajectories, 200 independent planning scenarios per decision. "
                  "Small evaluation sample: this is not a release-gate estimate."]
    report = "\n".join(lines) + "\n"
    out = Paths().ensure().reports / "toy_validation.md"
    out.write_text(report)
    print(report)
    return report
