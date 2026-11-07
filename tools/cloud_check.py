#!/usr/bin/env python3
"""Can this host reach every source on the deadline path? (design section 27). Stdlib only.

The deadline path of this implementation (docs/design_notes.md) is:
  Sleeper v1 (rosters, matchups), Sleeper's stats feed and schedule (api.sleeper.com),
  ESPN's scoreboard (tip times, live status) and the official injury-report PDF.
nba_api's cdn.nba.com / stats.nba.com hosts are no longer needed; ``--nba-cdn`` also probes
them with nba_api's browser-style headers (run that on the candidate host, not from here).

A host qualifies when every deadline-path source passes in two runs: one during a slate and
one between 11:00 and 19:00 ET (02:30-10:30 Adelaide), when fresh report PDFs exist.

    python cloud_check.py [--nba-cdn] [--out cloud_check.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    ET = timezone(timedelta(hours=-5))

PLAIN = {"User-Agent": "lockin-cloud-check/0.1 (read-only)"}
NBA_API_HEADERS = {  # what nba_api sends; only with --nba-cdn
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0",
    "Accept": "application/json, text/plain, */*", "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
}


def probe(url: str, headers: dict, expect: str = "json") -> dict:
    t0 = time.time()
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            body = r.read(400000)
            ok = r.status == 200 and (body[:4] == b"%PDF" if expect == "pdf" else
                                      body[:1] in (b"{", b"[") if expect == "json" else True)
            return {"url": url, "status": r.status, "ok": ok, "bytes": len(body), "ms": int(1000 * (time.time() - t0))}
    except urllib.error.HTTPError as e:
        return {"url": url, "status": e.code, "ok": False, "ms": int(1000 * (time.time() - t0))}
    except Exception as e:
        return {"url": url, "status": None, "ok": False, "error": str(e)[:200], "ms": int(1000 * (time.time() - t0))}


def latest_pdf(headers: dict) -> dict:
    now = datetime.now(timezone.utc).astimezone(ET)
    t = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    tried = []
    for _ in range(16):  # the last four hours of 15-minute slots
        h12 = t.hour % 12 or 12
        name = f"Injury-Report_{t:%Y-%m-%d}_{h12:02d}_{t.minute:02d}{'AM' if t.hour < 12 else 'PM'}.pdf"
        r = probe(f"https://ak-static.cms.nba.com/referee/injury/{name}", headers, "pdf")
        tried.append(name)
        if r["ok"]:
            r["tried"] = len(tried)
            return r
        t -= timedelta(minutes=15)
    return {"url": "ak-static.cms.nba.com/referee/injury/<last 4 h>", "ok": False, "status": None,
            "note": "no report in the last four hours (none are published overnight ET)", "tried": len(tried)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nba-cdn", action="store_true", help="also probe cdn.nba.com and stats.nba.com")
    ap.add_argument("--out", default="cloud_check.json")
    a = ap.parse_args(argv)
    today = datetime.now(timezone.utc).astimezone(ET).strftime("%Y%m%d")
    checks = {
        "sleeper_state (deadline path)": probe("https://api.sleeper.app/v1/state/nba", PLAIN),
        "sleeper_stats_feed (deadline path)": probe("https://api.sleeper.com/schedule/nba/regular/2026", PLAIN),
        "espn_scoreboard (deadline path)": probe(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={today}", PLAIN),
        "injury_report_pdf (deadline path)": latest_pdf(PLAIN),
        "ntfy_health": probe("https://ntfy.sh/v1/health", PLAIN),
    }
    if a.nba_cdn:
        checks["cdn_live_scoreboard"] = probe(
            "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json", NBA_API_HEADERS)
        checks["cdn_static_schedule"] = probe(
            "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", NBA_API_HEADERS)
        checks["stats_schedule"] = probe(
            "https://stats.nba.com/stats/scheduleleaguev2?LeagueID=00&Season=2026-27", NBA_API_HEADERS)
    for k, v in checks.items():
        print(f"{'PASS' if v['ok'] else 'FAIL'}  {k:40s} {v.get('status')}  {v.get('ms', '')} ms  "
              f"{v.get('note', v.get('error', ''))}")
    path = [k for k in checks if "deadline path" in k]
    qualifies = all(checks[k]["ok"] for k in path)
    print(f"\nDeadline path: {'all pass' if qualifies else 'NOT all pass'} "
          f"(run again {'outside' if 11 <= datetime.now(timezone.utc).astimezone(ET).hour < 19 else 'between 11:00 and 19:00 ET'})")
    with open(a.out, "w") as f:
        json.dump({"at": datetime.now(timezone.utc).isoformat(), "checks": checks, "deadline_path_ok": qualifies}, f,
                  indent=1)
    return 0 if qualifies else 1


if __name__ == "__main__":
    sys.exit(main())
