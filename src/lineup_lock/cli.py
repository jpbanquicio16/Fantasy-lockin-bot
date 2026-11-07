"""``lockin`` command line."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import datetime, timedelta

from lockin_core.config import (REPO_ROOT, Paths, load_alerts, load_league, load_model, load_sim, now_utc)


def _paths() -> Paths:
    return Paths().ensure()


def _client(paths):
    from lockin_core.ingest.sleeper import SleeperClient

    return SleeperClient(paths=paths)


def find_player(state, query: str, side=None) -> str:
    side = side or state.me
    if query in state.players:
        return query
    q = query.lower()
    hits = [p for p in side.active + side.reserve if q in state.name(p).lower()]
    if len(hits) != 1:
        raise SystemExit(f"{query!r} matches {[state.name(h) for h in hits] or 'no player'} on your roster")
    return hits[0]


# --- commands ----------------------------------------------------------------------
def cmd_ingest(a):
    from lockin_core.ingest.collect import ingest

    r = ingest(load_league(), _paths(), history=not a.no_history, refetch_weeks=a.refetch,
               history_tips=a.history_tips)
    print("\n".join(r.lines))
    for i in r.issues:
        print("issue:", i)


def cmd_reconcile(a):
    import polars as pl

    from lockin_core.data import tables as T
    from lockin_core.scoring.reconcile import Reconciliation, against_league, against_pts_std
    from lockin_core.scoring.settings import ScoringSettings

    cfg, paths = load_league(), _paths()
    client = _client(paths)
    lg = client.league(cfg.league_id)
    sc = ScoringSettings.from_sleeper(lg.scoring_settings, cfg.scoring.threshold_bonus_mode, cfg.scoring.tech_definition)
    st = client.state()
    season, week = int(st.season), int(st.week)
    rec = Reconciliation(current_season=season)
    for s in [*cfg.history_seasons, season]:
        b = T.read(paths, "box", s)
        if b is not None:
            against_pts_std(T.latest_revision(b), sc, rec, s)
    box = T.latest_revision(T.read(paths, "box", season))
    wp = {}
    for w in range(1, week + 1):
        pts = {}
        for m in client.matchups(cfg.league_id, w, max_age_s=3600 if w < week else 60):
            pts.update(m.players_points or {})
        wp[w] = pts
    if sc.open_settings():
        print("settle the open settings first; using stack / sleeper_tf for the league comparison")
        sc = sc.model_copy(update={"threshold_bonus_mode": "stack", "tech_definition": "sleeper_tf"})
    against_league(box.filter(pl.col("week") <= week), sc, wp, rec)
    out = paths.reports / "reconciliation.md"
    out.write_text(rec.markdown())
    print(rec.markdown())


def cmd_fit(a):
    from lockin_core.data import tables as T
    from lockin_core.models.availability import fit_status_calibration
    from lockin_core.models.fit import fit
    from lockin_core.scoring.settings import ScoringSettings
    from lockin_core.state.store import Store

    cfg, paths = load_league(), _paths()
    lg = _client(paths).league(cfg.league_id)
    sc = ScoringSettings.from_sleeper(lg.scoring_settings, cfg.scoring.threshold_bonus_mode, cfg.scoring.tech_definition)
    fit(paths, cfg.history_seasons, sc, load_model())
    reps = T.read(paths, "injury_reports")
    st = _client(paths).state()
    box = T.read(paths, "box", int(st.season))
    games = T.read(paths, "games", int(st.season))
    if reps is not None and box is not None:
        cal, cells = fit_status_calibration(reps, T.latest_revision(box), games)
        Store(paths).put("status_calibration", {"|".join(k): v for k, v in cal.items()})
        print(f"status calibration: {len(cal)} cells replace the priors (>= 200 player-games each)")
        if cells.height:
            print(cells)


def cmd_state(a):
    from lockin_core.schedule.calendar import fmt_local
    from lockin_core.state.build import build_state

    cfg = load_league()
    st, ctx = build_state(cfg, _paths(), save=False, strict=False)
    print(f"{st.snapshot_id}  season {st.season} week {st.week}  flags: {st.flags}")
    for reason in st.blocking:
        print("BLOCKED:", reason)
    for side in (st.me, st.opp):
        if side is None:
            continue
        print(f"\n== roster {side.roster_id} ({side.owner})  locked {side.locked_units / st.units_per_point:.1f}")
        for i, s in enumerate(side.slots):
            if i in side.locked:
                l_ = side.locked[i]
                print(f"  {s:4s} LOCKED {st.name(l_.player_id):24s} {l_.fp_half / st.units_per_point:5.1f}  ({l_.source})")
            else:
                pid = side.assignment.get(i)
                print(f"  {s:4s}        {st.name(pid) if pid else '-':24s}")
        for e in side.entries:
            print(f"  open: {st.name(e.player_id):22s} {e.fp_half / st.units_per_point:5.1f}  until "
                  f"{fmt_local(e.deadline_at, cfg.tz)}{'  (team final: counts at week end)' if e.is_team_final else ''}"
                  f"{'  [locked already?]' if e.locked_unknown else ''}")
        print("  bench:", ", ".join(st.name(p) for p in side.bench))
        for n in side.notes:
            print("  note:", n)


def cmd_recommend(a):
    from lineup_lock.recommend.render import render
    from lineup_lock.recommend.run import recommend

    cfg, mcfg, scfg = load_league(), load_model(), load_sim()
    if a.fast:
        scfg.scenarios.start, scfg.scenarios.max = 1000, 4000
        scfg.close_call.model_draws = 10
    rec, *_ = recommend(cfg, mcfg, scfg, _paths(), objective=a.objective, save=not a.dry_run)
    if a.json:
        print(json.dumps(rec, indent=1, default=str))
    else:
        print(render(rec, cfg.tz))


def cmd_record(a):
    from lockin_core.state.build import build_state
    from lockin_core.state.store import Store

    cfg, paths = load_league(), _paths()
    store = Store(paths)
    st, ctx = build_state(cfg, paths, store=store, save=False, strict=False)
    from lockin_core.state.manual import record_entry, record_opponent
    if a.what == "opp-locked":
        record_opponent(store, st, float(a.player))
        print("Opponent locked total recorded")
        return
    pid = find_player(st, a.player)
    entry = next((e for e in st.me.entries if e.player_id == pid), None)
    if a.at:
        from lockin_core.state.manual import record_historical_lock
        if a.what != "lock" or not a.game or a.slot_index is None or a.score is None:
            raise ValueError("Historical lock requires --game, --slot-index, --score and --at")
        print(record_historical_lock(store, st, ctx.calendar, pid, a.game, a.slot_index, a.score,
                                     datetime.fromisoformat(a.at)))
        return
    if entry is None:
        if a.game and a.what in ("pass", "not-locked", "not-eligible", "score"):
            from lockin_core.state.manual import score_units
            key = f"{st.week}:{st.me.roster_id}:{pid}:{a.game}"
            if ctx.calendar.game(a.game) is None:
                raise ValueError("Game is not in this matchup week")
            if a.what == "score" and a.score is None:
                raise ValueError("Supply --score")
            store.add_manual(a.what.replace("-", "_"), key,
                             {"fp_half": score_units(a.score, st.units_per_point)} if a.what == "score" else {"player": pid})
            store.invalidate("manual game correction")
            print("Recorded game correction")
            return
        raise SystemExit("No eligible open score; confirm the tip-off history or specify --game for a correction")
    idx = a.slot_index
    if idx is None and a.slot:
        matches = [i for i, slot in enumerate(st.me.slots) if slot == a.slot and i not in st.me.locked]
        own = st.me.slot_of(pid)
        if own in matches:
            idx = own
        elif len(matches) == 1:
            idx = matches[0]
        else:
            raise SystemExit("Slot is ambiguous or unavailable; use --slot-index (zero based)")
    mid = record_entry(store, st, entry.entry_id, a.what.replace("-", "_"), idx, a.score)
    print(f"Recorded {a.what} for {st.name(pid)} ({mid})")


def cmd_override(a):
    from lockin_core.state.build import build_state
    from lockin_core.state.store import Store

    cfg, paths = load_league(), _paths()
    store = Store(paths)
    st, _ = build_state(cfg, paths, store=store, save=False)
    pid = find_player(st, a.player)
    from lockin_core.state.manual import override_player
    mid = override_player(store, pid, a.p_play, a.minutes, a.out_until, a.hours)
    print(f"Override recorded for {st.name(pid)} ({mid})")


def cmd_confirm_tip(a):
    from lockin_core.state.ledger import tip_batch_id
    from lockin_core.state.store import Store

    cfg, paths = load_league(), _paths()
    from lockin_core.state.manual import confirm_tip
    t = datetime.fromisoformat(a.tip)
    starters = [p.strip() if p.strip() not in ("", "0", "-") else None for p in a.starters.split(",")]
    confirm_tip(Store(paths), cfg.league_id, a.roster or cfg.my_roster_id, t, starters)
    print(f"Tip {t.isoformat()} confirmed")


def cmd_observe(a):
    from lineup_lock.events.service import Service

    print(f"observations saved: {Service().observe(now_utc())}")


def cmd_tick(a):
    from lineup_lock.events.service import Service

    svc = Service()
    if a.force:
        svc.store.put("force_recommend", True)
    r = svc.tick()
    print(f"did {r.did}; errors {r.errors}; recommendation {r.recommended}")


def cmd_run(a):
    from lineup_lock.events.service import Service, run_forever

    run_forever(Service())


def cmd_alerts(a):
    from lockin_core.schedule.calendar import fmt_local
    from lockin_core.state.store import Store

    cfg = load_league()
    store = Store(_paths())
    for d in store.deadlines():
        print(f"{fmt_local(d['due_at'], cfg.tz)}  {d['default_action'].get('player')} "
              f"{d['default_action'].get('score')}  advice {d['default_action'].get('advice')}, "
              f"default {d['default_action'].get('default')}")
    for al in store.alerts(20):
        print(f"sent {al['sent_at'][:16]} {al['kind']:6s} {al['channel']}: {al['body'][:120]}")


def cmd_toy(a):
    from lineup_lock.replay.toy import main as toy_main

    toy_main(a.weeks, a.players, a.scenarios, a.seed, a.rollout)


def cmd_bench(a):
    from lineup_lock.replay.bench import main as bench_main

    bench_main(a.scenarios)


def cmd_replay(a):
    from lineup_lock.replay.synthetic import main as replay_main

    replay_main(a.season, a.rosters, a.weeks, a.seed, a.rollout_weeks)


def cmd_shadow(a):
    from lineup_lock.replay.shadow import main as shadow_main

    shadow_main()


def cmd_audit(a):
    from lockin_core.state.store import Store
    from lineup_lock.replay.artifacts import audit
    paths = _paths()
    records = Store(paths).recommendations()
    ids = [a.rec_id] if a.rec_id else [r["rec_id"] for r in records[-a.limit:]]
    failed = []
    for rid in ids:
        try:
            ok = audit(paths, rid)
        except (OSError, ValueError, KeyError) as error:
            print(f"{rid}: unavailable ({error})")
            failed.append(rid)
            continue
        print(f"{rid}: {'MATCH' if ok else 'MISMATCH'}")
        if not ok:
            failed.append(rid)
    if failed:
        raise SystemExit(1)


def cmd_outcome(a):
    from lockin_core.state.store import Store
    from lockin_core.state.manual import score_units
    store = Store(_paths())
    rec = next((r for r in store.recommendations() if r["rec_id"] == a.rec_id), None)
    if rec is None:
        raise ValueError("Unknown recommendation ID")
    units = rec["result"].get("units_per_point", 2)
    store.add_outcome(a.rec_id, {"description": a.action}, a.matched,
                      score_units(a.realised, units) if a.realised is not None else None,
                      {"recommended": score_units(a.recommended, units)} if a.recommended is not None else None)
    print("Outcome recorded")


def cmd_model(a):
    from lockin_core.models.fit import activate_version
    from lockin_core.state.store import Store
    activate_version(_paths(), a.version)
    Store(_paths()).invalidate("model version changed")
    print(f"Active model: {a.version}")


def cmd_backup(a):
    from pathlib import Path
    from lockin_core.state.store import Store
    store = Store(_paths())
    dest = Path(a.destination)
    store.backup(dest / "app.sqlite")
    store.export_history(dest / "ledger.json.gz")
    print(f"Database and ledger backed up to {dest}")


def cmd_app(a):
    import subprocess

    from lineup_lock.app.access import validate_bind_address
    validate_bind_address(a.address)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(REPO_ROOT / "src/lineup_lock/app/streamlit_app.py"),
                    "--server.address", a.address, "--server.port", str(a.port)], check=False)


def cmd_league_tests(a):
    sys.argv = ["run_league_tests.py", *a.rest]
    runpy.run_path(str(REPO_ROOT / "tools/run_league_tests.py"), run_name="__main__")


def cmd_check_host(a):
    sys.argv = ["cloud_check.py", *a.rest]
    runpy.run_path(str(REPO_ROOT / "tools/cloud_check.py"), run_name="__main__")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lockin", description="Lineup and lock assistant for Sleeper Lock-In")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="stats, schedule, tip times, player map, injury reports")
    p.add_argument("--no-history", action="store_true")
    p.add_argument("--refetch", action="store_true", help="re-fetch every week of this season")
    p.add_argument("--history-tips", action="store_true", help="ESPN tip times for past seasons (replay)")
    p.set_defaults(fn=cmd_ingest)
    sub.add_parser("reconcile", help="scoring engine vs Sleeper's own scores").set_defaults(fn=cmd_reconcile)
    sub.add_parser("fit", help="projection parameters and status calibration").set_defaults(fn=cmd_fit)
    sub.add_parser("state", help="slots, locks, open scores and deadlines").set_defaults(fn=cmd_state)
    p = sub.add_parser("recommend", help="starters, slots, and lock now / wait / pass for every open score")
    p.add_argument("--objective", choices=["points", "win"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="do not store the recommendation")
    p.add_argument("--fast", action="store_true", help="fewer scenarios (a quick look)")
    p.set_defaults(fn=cmd_recommend)
    p = sub.add_parser("record", help="record what you did in Sleeper")
    p.add_argument("what", choices=["lock", "pass", "not-eligible", "not-locked", "score", "opp-locked"])
    p.add_argument("player", help="name (or player id); for opp-locked, the total")
    p.add_argument("--slot")
    p.add_argument("--slot-index", type=int, help="zero-based slot index, for duplicate C/UTIL slots")
    p.add_argument("--score", type=float)
    p.add_argument("--game", help="Sleeper game ID for a history correction")
    p.add_argument("--at", help="actual historical lock time, ISO format with timezone")
    p.set_defaults(fn=cmd_record)
    p = sub.add_parser("override", help="what you know that the data does not")
    p.add_argument("player")
    p.add_argument("--p-play", type=float)
    p.add_argument("--minutes", type=float)
    p.add_argument("--out-until")
    p.add_argument("--hours", type=float, default=48)
    p.set_defaults(fn=cmd_override)
    p = sub.add_parser("confirm-tip", help="correct who was starting at a tip")
    p.add_argument("tip", help="UTC ISO time of the tip")
    p.add_argument("starters", help="comma-separated player ids in slot order")
    p.add_argument("--roster", type=int)
    p.set_defaults(fn=cmd_confirm_tip)
    sub.add_parser("observe", help="one roster poll into the lineup ledger").set_defaults(fn=cmd_observe)
    p = sub.add_parser("tick", help="one pass of the event loop")
    p.add_argument("--force", action="store_true", help="recommend even without a new event")
    p.set_defaults(fn=cmd_tick)
    sub.add_parser("run", help="the event loop (one tick a minute)").set_defaults(fn=cmd_run)
    sub.add_parser("alerts", help="pending deadlines and sent alerts").set_defaults(fn=cmd_alerts)
    p = sub.add_parser("toy", help="policies vs the exact optimum on small roster-weeks")
    p.add_argument("--weeks", type=int, default=200)
    p.add_argument("--players", type=int, default=5)
    p.add_argument("--scenarios", type=int, default=20000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--rollout", type=int, default=0, help="also evaluate the rollout policy on N weeks")
    p.set_defaults(fn=cmd_toy)
    p = sub.add_parser("bench", help="full-size simulator benchmark")
    p.add_argument("--scenarios", type=int, default=2000)
    p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("replay", help="policies on synthetic roster-weeks of a past season's real games")
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--rosters", type=int, default=12)
    p.add_argument("--weeks", type=int, default=0, help="0 = every week")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rollout-weeks", type=int, default=0)
    p.set_defaults(fn=cmd_replay)
    sub.add_parser("shadow", help="score past recommendations against what happened").set_defaults(fn=cmd_shadow)
    p = sub.add_parser("audit", help="recompute saved recommendations without fetching new data")
    p.add_argument("--rec-id")
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(fn=cmd_audit)
    p = sub.add_parser("outcome", help="record realised week totals and replayed counterfactuals")
    p.add_argument("rec_id")
    p.add_argument("--action", required=True)
    p.add_argument("--matched", action="store_true")
    p.add_argument("--realised", type=float)
    p.add_argument("--recommended", type=float, help="counterfactual whole-week points from chronological replay")
    p.set_defaults(fn=cmd_outcome)
    p = sub.add_parser("activate-model", help="activate an evaluated model or roll back to a retained version")
    p.add_argument("version")
    p.set_defaults(fn=cmd_model)
    p = sub.add_parser("backup", help="consistent SQLite backup and compressed ledger export")
    p.add_argument("destination")
    p.set_defaults(fn=cmd_backup)
    p = sub.add_parser("app", help="Streamlit app")
    p.add_argument("--address", default="127.0.0.1", help="bind address (your Tailscale IP to reach it by phone)")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(fn=cmd_app)
    p = sub.add_parser("league-tests", help="Sleeper league API tests T0-T8")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_league_tests)
    p = sub.add_parser("check-host", help="can this host reach the deadline-path sources?")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_check_host)
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("league-tests", "check-host"):
        import argparse as _argparse
        fn = cmd_league_tests if args[0] == "league-tests" else cmd_check_host
        return fn(_argparse.Namespace(rest=args[1:]))
    a = ap.parse_args(args)
    try:
        a.fn(a)
    except (ValueError, RuntimeError) as error:
        ap.exit(2, f"lockin: {error}\n")


if __name__ == "__main__":
    main()
