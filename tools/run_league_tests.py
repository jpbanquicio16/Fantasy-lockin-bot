#!/usr/bin/env python3
"""League API tests T0-T8 (design section 28). Standard library only (Python 3.9+).

Read-only GET requests, 0.2 s apart; every response is saved with its URL and fetch time.

    python run_league_tests.py --username <name>                      # T0, then T1-T6 and T8
    python run_league_tests.py --league-id <id> --user-id <id>        # if T0 cannot list NBA leagues
    python run_league_tests.py --league-id <id> --user-id <id> --poll 20   # T7: rosters every 2 min
    python run_league_tests.py --diff <folder before> <folder after>  # T4: what changed when you locked
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone

BASE = "https://api.sleeper.app/v1"
UA = "lockin-league-tests/0.1 (read-only)"


def get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode() or "null")
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # network
        return None, str(e)


class Run:
    def __init__(self, out: str):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.results = []

    def fetch(self, name: str, path: str):
        url = BASE + path
        status, body = get(url)
        rec = {"test": name, "url": url, "fetched_at": datetime.now(timezone.utc).isoformat(), "status": status,
               "body": body}
        with open(os.path.join(self.out, f"{name}.json"), "w") as f:
            json.dump(rec, f, indent=1)
        time.sleep(0.2)
        return status, body

    def check(self, test: str, what: str, result: str, ok: bool | None = None):
        mark = {True: "PASS", False: "FAIL", None: "INFO"}[ok]
        self.results.append((test, what, result, mark))
        print(f"  [{mark}] {test} {what}: {result}")


def t0(run: Run, username: str, season: str):
    s, u = run.fetch("T0_user", f"/user/{username}")
    if not u:
        run.check("T0", "user", f"HTTP {s}: not found", False)
        return None, None
    uid = u["user_id"]
    run.check("T0", "user_id", uid, True)
    s, leagues = run.fetch("T0_leagues", f"/user/{uid}/leagues/nba/{season}")
    if not leagues:
        run.check("T0", "NBA league listing", f"HTTP {s}, empty (docs say only nfl): take the id from the league URL",
                  False)
        return uid, None
    for lg in leagues:
        run.check("T0", "league", f"{lg.get('league_id')} {lg.get('name')!r} ({lg.get('total_rosters')} teams)", None)
    return uid, leagues[0]["league_id"]


def t1_t8(run: Run, league_id: str, user_id: str | None, week: int):
    print("T1 league")
    s, lg = run.fetch("T1_league", f"/league/{league_id}")
    if not lg:
        run.check("T1", "league", f"HTTP {s}", False)
        return
    run.check("T1", "sport", str(lg.get("sport")), lg.get("sport") == "nba")
    rp = lg.get("roster_positions") or []
    run.check("T1", "roster_positions", ", ".join(rp), bool(rp))
    sc = lg.get("scoring_settings") or {}
    run.check("T1", "scoring keys", ", ".join(f"{k}={v}" for k, v in sorted(sc.items()) if v), bool(sc))
    for k in ("bonus_pt_40p", "bonus_pt_50p", "tf", "ff"):
        run.check("T1", f"scoring {k}", str(sc.get(k)), k in sc)
    st = lg.get("settings") or {}
    for k in ("playoff_teams", "playoff_week_start", "reserve_slots", "num_teams", "waiver_type"):
        run.check("T1", f"settings {k}", str(st.get(k)), None)
    starting = [p for p in rp if p not in ("BN", "IL", "TAXI")]
    print("T2 rosters")
    s, rosters = run.fetch("T2_rosters", f"/league/{league_id}/rosters")
    if rosters:
        lens = Counter(len(r.get("starters") or []) for r in rosters)
        run.check("T2", "starters length", f"{dict(lens)} (starting slots: {len(starting)})",
                  set(lens) == {len(starting)})
        run.check("T2", "reserve (IL) present", str(sum(1 for r in rosters if r.get("reserve"))), None)
        keys = sorted({k for r in rosters for k in r})
        run.check("T2", "roster keys", ", ".join(keys), None)
    print("T3 users")
    s, users = run.fetch("T3_users", f"/league/{league_id}/users")
    if users and rosters and user_id:
        mine = [r["roster_id"] for r in rosters if r.get("owner_id") == user_id or user_id in (r.get("co_owners") or [])]
        run.check("T3", "your roster_id", str(mine), bool(mine))
    print("T4 matchups")
    s, m = run.fetch(f"T4_matchups_w{week}", f"/league/{league_id}/matchups/{week}")
    if m:
        keys = sorted({k for x in m for k in x})
        run.check("T4", "matchup keys", ", ".join(keys), "matchup_id" in keys)
        run.check("T4", "per-slot points (starters_points)", str("starters_points" in keys), "starters_points" in keys)
        run.check("T4", "per-player points (players_points)", str("players_points" in keys), None)
        run.check("T4", "lock field", "none documented; diff before/after a lock with --diff", None)
    print("T5 players (once; about 5 MB)")
    s, pl = run.fetch("T5_players", "/players/nba")
    if pl:
        on = [p for p in pl.values() if p.get("team")]
        run.check("T5", "players on teams", str(len(on)), len(on) > 300)
        run.check("T5", "injury_status values", str(dict(Counter(p.get("injury_status") for p in on))), None)
        run.check("T5", "fantasy_positions example", str(next(iter(on)).get("fantasy_positions")), None)
        run.check("T5", "id format", str(list(pl)[:3]), None)
    print("T6 transactions")
    s, tx = run.fetch(f"T6_transactions_w{week}", f"/league/{league_id}/transactions/{week}")
    if tx is not None:
        run.check("T6", "leg values", str(sorted({t.get("leg") for t in tx})), None)
    print("T8 trending")
    s, tr = run.fetch("T8_trending", "/players/nba/trending/add?lookback_hours=24&limit=25")
    run.check("T8", "trending adds", f"HTTP {s}, {len(tr or [])} players", s == 200)


def poll(run: Run, league_id: str, minutes: int):
    print(f"T7: rosters every 2 minutes for {minutes} minutes")
    n = max(1, minutes // 2 + 1)
    prev = None
    for i in range(n):
        s, r = run.fetch(f"T7_rosters_{i:02d}", f"/league/{league_id}/rosters")
        cur = {x["roster_id"]: x.get("starters") for x in (r or [])}
        if prev is not None:
            changed = [rid for rid in cur if cur[rid] != prev.get(rid)]
            run.check("T7", f"poll {i}", f"starters changed for rosters {changed}" if changed else "no change", None)
        prev = cur
        if i < n - 1:
            time.sleep(120)


def diff(a: str, b: str):
    def load(folder):
        out = {}
        for f in os.listdir(folder):
            if f.endswith(".json"):
                with open(os.path.join(folder, f)) as fh:
                    out[f] = json.load(fh).get("body")
        return out

    A, B = load(a), load(b)
    for name in sorted(set(A) & set(B)):
        if A[name] == B[name]:
            continue
        print(f"== {name}")
        x, y = A[name], B[name]
        if isinstance(x, list) and isinstance(y, list):
            for i, (u, v) in enumerate(zip(x, y)):
                if u != v and isinstance(u, dict):
                    for k in sorted(set(u) | set(v)):
                        if u.get(k) != v.get(k):
                            print(f"  [{i}] roster {u.get('roster_id')} {k}: {json.dumps(u.get(k))[:200]} -> "
                                  f"{json.dumps(v.get(k))[:200]}")
        else:
            print("  changed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--username")
    ap.add_argument("--league-id")
    ap.add_argument("--user-id")
    ap.add_argument("--season", default="2026")
    ap.add_argument("--week", type=int)
    ap.add_argument("--poll", type=int, metavar="MINUTES")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.diff:
        diff(*a.diff)
        return 0
    out = a.out or f"league_tests_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run = Run(out)
    uid, lid = a.user_id, a.league_id
    if a.username:
        print("T0 user and leagues")
        u, l_ = t0(run, a.username, a.season)
        uid = uid or u
        lid = lid or l_
    if not lid:
        print("no league id: pass --league-id (the number in your league's URL)")
        return 2
    week = a.week
    if week is None:
        s, st = get(BASE + "/state/nba")
        week = int((st or {}).get("week") or 1)
    if a.poll:
        poll(run, lid, a.poll)
    else:
        t1_t8(run, lid, uid, week)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(run.results, f, indent=1)
    print(f"\nsaved {len(os.listdir(out))} files to {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
