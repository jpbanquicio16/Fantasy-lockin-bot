"""Full-size benchmark of the simulator (design section 13's table), on a synthetic roster:
14 players with mixed positions; slots PG, SG, G, SF, PF, F, C, C, UTIL, UTIL; 3-4 games each
on six ET tip times; 150-minute games."""

from __future__ import annotations

import time

import numpy as np

from lockin_core.config import Paths
from lockin_core.policy.tables import build_tables, quantile_grid
from lockin_core.sim.engine import WeekSim
from lockin_core.sim.problem import Action, PolicyConfig, RootEntry, SideProblem, SideScenarios, SimPlayer
from lineup_lock.recommend.candidates import generate

SLOTS = ["PG", "SG", "G", "SF", "PF", "F", "C", "C", "UTIL", "UTIL"]
POS = [["PG"], ["PG", "SG"], ["SG"], ["SG", "SF"], ["SF"], ["SF", "PF"], ["PF"], ["PF", "C"], ["C"], ["C"],
       ["PG"], ["SG", "SF"], ["PF"], ["SF", "PF"]]
TIPS_ET = [19 * 60, 19 * 60 + 30, 20 * 60, 20 * 60 + 30, 21 * 60, 22 * 60]


def synthetic_week(seed: int, start_minute: float = 0.0, open_entries: int = 0):
    rng = np.random.default_rng(seed)
    players, raw = [], []
    for i, pos in enumerate(POS):
        n = int(rng.integers(3, 5))
        days = sorted(rng.choice(7, size=n, replace=False))
        tips = np.array([d * 1440 + TIPS_ET[int(rng.integers(len(TIPS_ET)))] for d in days], float) - start_minute
        mu, sd = rng.uniform(18, 42), rng.uniform(7, 12)
        raw.append((mu, sd, rng.uniform(0, 0.2, n)))
        players.append((pos, tips))
    sims, P = [], len(players)
    K = 4
    for i, (pos, tips) in enumerate(players):
        keep = tips + 150 > 0
        t = tips[keep]
        fin = np.zeros(len(t), bool)
        if len(t):
            fin[-1] = True
        sims.append(SimPlayer(str(i), f"P{i}", pos, t, t + 150, fin, t <= 0, [f"g{i}{k}" for k in range(len(t))],
                              raw[i][0] * 2, (raw[i][0] + 0.67 * raw[i][1]) * 2))
    S = len(SLOTS)
    entries = []
    if open_entries:
        cand = [p for p, s in enumerate(sims) if s.n_games >= 1 and s.tips[0] > 0]
        for p in rng.choice(cand, size=min(open_entries, len(cand)), replace=False):
            entries.append(RootEntry(int(p), float(int(rng.normal(raw[p][0], raw[p][1]) * 2)), False, f"e{p}"))
    occ = np.full(S, -1)
    free = list(range(S))
    for e in entries:
        j = next((j for j in free if set(POS[e.p]) & {"PG", "SG", "SF", "PF", "C"} and
                  (SLOTS[j] == "UTIL" or set(POS[e.p]) & {"G": {"PG", "SG"}, "F": {"SF", "PF"}}.get(SLOTS[j], {SLOTS[j]}))),
                 None)
        if j is not None:
            occ[j] = e.p
            free.remove(j)
    prob = SideProblem(SLOTS, sims, np.zeros(S, bool), 0.0, occ, np.zeros(P, bool), entries, np.zeros(P, bool))
    return prob, raw


def scenarios(prob: SideProblem, raw, n: int, seed: int) -> SideScenarios:
    rng = np.random.default_rng(seed)
    P, K = prob.P, prob.K
    out = np.zeros((n, P, K), bool)
    scr = np.zeros((n, P, K), bool)
    units = np.zeros((n, P, K))
    for p, pl in enumerate(prob.players):
        mu, sd, q = raw[p]
        for k in range(pl.n_games):
            out[:, p, k] = rng.random(n) < q[min(k, len(q) - 1)]
            scr[:, p, k] = rng.random(n) < 0.01
            units[:, p, k] = np.round(rng.normal(mu, sd, n) * 2)
    return SideScenarios(out, scr, units)


def attach(prob: SideProblem, raw) -> float:
    t = time.time()
    P, K = prob.P, prob.K
    vals = np.zeros((P, K, 21))
    w = np.full((P, K, 21), 1 / 21)
    q_out = np.zeros((P, K))
    q_scr = np.full((P, K), 0.01)
    final = np.zeros((P, K), bool)
    for p, pl in enumerate(prob.players):
        mu, sd, q = raw[p]
        g, _ = quantile_grid(np.random.default_rng(p).normal(mu, sd, 2000) * 2, 21)
        for k in range(pl.n_games):
            vals[p, k] = g
            q_out[p, k] = q[min(k, len(q) - 1)]
            final[p, k] = pl.final[k]
    prob.tables = build_tables(vals, w, q_out, q_scr, np.array([p.n_games for p in prob.players]), final, prob.flex)
    return time.time() - t


def main(n: int = 2000) -> str:
    lines = [f"# Simulator benchmark (synthetic full-size roster, {n} scenarios)", ""]
    prob, raw = synthetic_week(1)
    tt = attach(prob, raw)
    sc = scenarios(prob, raw, n, 2)
    n_games = sum(p.n_games for p in prob.players)
    n_tips = len({float(t) for p in prob.players for t in p.tips})
    lines.append(f"Roster: 14 players, {n_games} games on {n_tips} distinct tip minutes; slots {', '.join(SLOTS)}.")
    lines.append("")
    lines.append("| Step | Fast variant | Exact variant |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| One-player tables, 14 players | {tt * 1000:.1f} ms | - |")
    res = {}
    for name, pol in (("fast", PolicyConfig()), ("exact", PolicyConfig(assignment="exact"))):
        sim = WeekSim(prob, pol)
        t = time.time()
        sim.run(sc)
        res[name] = time.time() - t
    lines.append(f"| One action, whole week from Monday | {res['fast']:.2f} s | {res['exact']:.1f} s |")
    mid, raw2 = synthetic_week(1, start_minute=2 * 1440 + 12 * 60, open_entries=9)
    attach(mid, raw2)
    sc2 = scenarios(mid, raw2, n, 3)
    acts, _ = generate(mid, 60)
    times = {}
    for name, pol in (("fast", PolicyConfig()), ("exact", PolicyConfig(assignment="exact"))):
        sim = WeekSim(mid, pol)
        ts = []
        for a in acts[:10]:
            t = time.time()
            sim.run(sc2, a)
            ts.append(time.time() - t)
        times[name] = float(np.median(ts))
    lines.append(f"| One action from Wednesday noon ({len(mid.entries)} open entries), median | "
                 f"{times['fast']:.2f} s | {times['exact']:.2f} s |")
    sim = WeekSim(mid, PolicyConfig())
    t = time.time()
    for a in acts:
        sim.run(sc2, a)
    t60 = time.time() - t
    sc500 = sc2.subset(np.arange(min(500, n)))
    t = time.time()
    for a in acts:
        sim.run(sc500, a)
    t500 = time.time() - t
    lines.append(f"| {len(acts)} candidate actions from Wednesday | {t60:.1f} s | - |")
    lines.append(f"| {len(acts)} candidate actions at 500 scenarios (screening) | {t500:.1f} s | - |")
    report = "\n".join(lines) + "\n\nMeasured on this machine; the design's numbers came from a 2-vCPU cloud workspace.\n"
    (Paths().ensure().reports / "benchmark.md").write_text(report)
    print(report)
    return report
