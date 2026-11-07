"""Prospective decision audit. Missing outcomes are pending, never silently scored as zero.

An outcome is a recorded whole-week total and an optional replayed counterfactual whole-week
result. Within-week observations are averaged before bootstrap resampling whole weeks.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from lockin_core.config import Paths
from lockin_core.state.store import Store


def week_interval(values: dict[tuple[int, int], list[float]], seed: int = 2026, draws: int = 4000):
    if not values:
        return None
    means = np.array([np.mean(v) for _, v in sorted(values.items())])
    if len(means) < 2:
        return float(means.mean()), None, None, len(means)
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(0, len(means), (draws, len(means)))].mean(1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return float(means.mean()), float(lo), float(hi), len(means)


def report(store: Store) -> str:
    records = {r["rec_id"]: r for r in store.recommendations()}
    # Corrections are append-only: the latest outcome for a recommendation wins.
    outcomes = {o["rec_id"]: o for o in sorted(store.outcomes(), key=lambda o: o["recorded_at"])}
    matched, scored, pending, deltas = 0, 0, 0, defaultdict(list)
    for rid, r in records.items():
        o = outcomes.get(rid)
        if o is None:
            pending += 1
            continue
        matched += bool(o["matched"])
        if o["realised_week_half"] is None:
            pending += 1
            continue
        scored += 1
        cf = json.loads(o["counterfactual_half"]) if o["counterfactual_half"] else {}
        if "recommended" in cf:
            rec = r["result"]
            units = rec.get("units_per_point", 2)
            deltas[(rec["season"], rec["week"])].append((cf["recommended"] - o["realised_week_half"]) / units)
    waits = sum(c["advice"] == "wait" for r in records.values() for c in r["result"].get("entries", []))
    expired = store.deadlines("expired")
    missed = store.deadlines("missed")
    lines = ["# Prospective shadow report", "",
             f"Recommendations: {len(records)}. Actions recorded: {len(outcomes)}. "
             f"Matched recommended action: {matched}. Scored outcomes: {scored}. Pending: {pending}.",
             f"Wait recommendations: {waits}. Deadlines expired after an alert: {len(expired)}. "
             f"Deadlines with no successful alert: {len(missed)}.", ""]
    interval = week_interval(deltas)
    if interval:
        mean, lo, hi, weeks = interval
        lines.append(f"Recorded recommended-minus-actual difference: {mean:+.2f} FP across {weeks} weeks.")
        lines.append(f"95% whole-week bootstrap interval: [{lo:+.2f}, {hi:+.2f}] FP." if lo is not None
                     else "At least two completed weeks are needed for a bootstrap interval.")
    else:
        lines.append("No paired realised outcomes yet. No claim of improvement can be made.")
    lines += ["", "Counterfactuals must come from a chronological replay, not the highest score seen later. "
              "Repeated decisions within a week do not count as independent weeks. "
              "Expected values and Monte Carlo confidence are not realised performance."]
    return "\n".join(lines) + "\n"


def main(paths: Paths | None = None, store: Store | None = None) -> str:
    paths = (paths or Paths()).ensure()
    store = store or Store(paths)
    score_pending(paths, store)
    text = report(store)
    (paths.reports / "shadow.md").write_text(text)
    print(text)
    return text


def realise_saved_week(paths: Paths, store: Store, record: dict, now=None) -> dict | None:
    """Score a saved recommendation only when all games and actual lineup history are known.

    Missing injury history is treated as information unavailable at tip: a subsequent DNP
    is a late scratch, never an advance OUT signal derived from its box score.
    """
    from datetime import timedelta
    from lockin_core.config import now_utc
    from lockin_core.data import tables as T
    from lockin_core.scoring.engine import score_frame
    from lockin_core.sim.engine import simulate
    from lockin_core.sim.problem import SideScenarios, BASELINES
    from lockin_core.state.ledger import starters_at
    from lineup_lock.replay.artifacts import load, root_action
    import polars as pl

    wm, rec = load(paths, record['rec_id'])
    cal, state, prob = wm.ctx.calendar, wm.state, wm.me
    now = now or now_utc()
    if now <= cal.week_end:
        return None
    box = T.read(paths, 'box', state.season)
    if box is None:
        return None
    box = T.latest_revision(box).filter(pl.col('week') == state.week)
    played = box.filter(pl.col('played') == 'played')
    scored = score_frame(played, wm.ctx.rules.scoring)
    scores = {(r['player_id'], r['game_id']): int(r['fp_units']) for r in scored.iter_rows(named=True)
              if r['fp_units'] is not None}
    rows = {(r['player_id'],r['game_id']): r for r in box.iter_rows(named=True)}
    reports = T.read(paths, 'injury_reports')
    out = np.zeros((1,prob.P,prob.K), bool); scratch = np.zeros_like(out); units = np.zeros(out.shape)
    for p, player in enumerate(prob.players):
        for k, gid in enumerate(player.game_ids):
            row = rows.get((player.pid,gid))
            if row is None:
                return None
            dnp = row['played'] != 'played'
            g = cal.game(gid)
            was_out = False
            if reports is not None and g is not None:
                known = T.report_status_as_of(reports, g.tip_at)
                relevant = known.filter((pl.col('player_id') == player.pid) & (pl.col('game_date') == g.date))
                was_out = relevant.height > 0 and relevant['status'][-1].lower() == 'out'
            out[0,p,k] = was_out and dnp
            scratch[0,p,k] = dnp and not out[0,p,k]
            if not dnp and (player.pid,gid) not in scores:
                return None
            units[0,p,k] = scores.get((player.pid,gid),0)
    # Reconstruct the actual week-end lineup only when polling brackets the boundary.
    observations = store.observations(state.me.roster_id, cal.week_end-timedelta(minutes=15),
                                       cal.week_end+timedelta(minutes=15))
    pre = [o for o in observations if o['poll_ok'] and o['observed_at'] <= cal.week_end]
    post = [o for o in observations if o['poll_ok'] and o['observed_at'] > cal.week_end]
    if not pre or not post or pre[-1]['starters'] != post[0]['starters']:
        return None
    locks = store.locks(state.league_id,state.week,state.me.roster_id)
    actual = sum(l['fp_half'] for l in locks)
    locked_slots = {l['slot_index'] for l in locks}
    used = {l['player_id'] for l in locks}
    for slot,pid in enumerate(pre[-1]['starters']):
        if not pid or slot in locked_slots or pid in used:
            continue
        info = state.players.get(pid)
        if info is None:
            return None
        final = cal.team_final(info.team)
        if final is None:
            continue
        starters,status = starters_at(store,state.me.roster_id,final.tip_at)
        if status not in ('observed','confirmed_manual'):
            return None
        if not starters or pid not in starters:
            continue
        row = rows.get((pid,final.game_id))
        if row is None:
            return None
        if row['played'] == 'played':
            if (pid,final.game_id) not in scores:
                return None
            actual += scores[(pid,final.game_id)]
    real = SideScenarios(out,scratch,units)
    counterfactual = {'recommended': int(round(simulate(prob,real,root_action(rec,prob)).total[0]))}
    for name,policy in BASELINES.items():
        counterfactual[name] = int(round(simulate(prob,real,policy=policy).total[0]))
    return {'actual': actual, 'counterfactual': counterfactual}


def score_pending(paths: Paths, store: Store) -> int:
    existing = {o['rec_id'] for o in store.outcomes() if o['realised_week_half'] is not None}
    count = 0
    for record in store.recommendations():
        if record['rec_id'] in existing:
            continue
        try:
            result = realise_saved_week(paths,store,record)
        except FileNotFoundError:
            continue  # recommendations made before snapshot capture remain explicitly pending
        if result is None:
            continue
        previous = [o for o in store.outcomes() if o['rec_id'] == record['rec_id']]
        last = previous[-1] if previous else None
        action = json.loads(last['action_taken']) if last else {'source':'lineup and lock ledger'}
        store.add_outcome(record['rec_id'], action, bool(last and last['matched']), result['actual'], result['counterfactual'])
        count += 1
    return count
