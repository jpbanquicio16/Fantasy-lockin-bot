"""The lineup ledger: who was in a starting slot when each game tipped (design section 7).

Eligibility is a fact about history; the rosters endpoint shows only the lineup now. Every
poll is an immutable observation; for each tip batch the observations just before (T-10,
T-2) and just after (T+2) the tip are compared. If the last one before and the first one
after agree, the starters at that tip are **observed**; if they differ or a poll is missing,
the tip is **uncertain** and the user is asked one question (``lockin confirm-tip``).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from lockin_core.schedule.calendar import WeekCalendar
from lockin_core.state.store import Store

PRE_WINDOW = timedelta(minutes=15)
POST_WINDOW = timedelta(minutes=10)
SETTLE = timedelta(minutes=3)


def tip_batch_id(tip_at: datetime) -> str:
    return f"tip_{tip_at:%Y%m%dT%H%MZ}"


def resolve_tipoffs(store: Store, league_id: str, roster_id: int, calendar: WeekCalendar, now: datetime) -> int:
    """Create or update the tip-off snapshot of every tip this week that has passed (manual
    confirmations are never overwritten). Returns the number of snapshots written."""
    done = {r["tip_batch_id"]: r for r in store.tipoffs(roster_id, calendar.week_start, calendar.week_end)}
    tips = sorted({g.tip_at for g in calendar.games})
    n = 0
    for t in tips:
        if t + SETTLE > now:
            continue
        bid = tip_batch_id(t)
        if bid in done and done[bid]["status"] in ("observed", "confirmed_manual"):
            continue
        obs = store.observations(roster_id, t - PRE_WINDOW, min(t + POST_WINDOW, now))
        pre = [o for o in obs if o["observed_at"] < t and o["poll_ok"]]
        post = [o for o in obs if o["observed_at"] >= t and o["poll_ok"]]
        if not pre and not post:
            continue
        p0 = pre[-1] if pre else None
        p1 = post[0] if post else None
        if p0 and p1 and p0["starters"] == p1["starters"]:
            status, starters = "observed", p0["starters"]
        else:
            status, starters = "uncertain", (p0 or p1)["starters"]
        store.save_tipoff(bid, league_id, roster_id, t, p0["obs_id"] if p0 else None,
                          p1["obs_id"] if p1 else None, starters, status, recorded_at=now)
        n += 1
    return n


def starters_at(store: Store, roster_id: int, tip_at: datetime, as_of: datetime | None = None) -> tuple[list | None, str]:
    """(starters list by slot index, status) at that tip: observed | confirmed_manual |
    uncertain | unknown."""
    if as_of is not None:
        import json
        from lockin_core.state.store import _ts
        with store.conn() as c:
            r = c.execute("SELECT * FROM tipoff_revisions WHERE roster_id=? AND scheduled_tip_at=? "
                          "AND valid_from<=? ORDER BY revision_id DESC LIMIT 1",
                          (roster_id, _ts(tip_at), _ts(as_of))).fetchone()
        if r is not None:
            return json.loads(r["starters_at_tip"]) if r["starters_at_tip"] else None, r["status"]
        # No revision known at this instant: never read a later correction.
        return None, "unknown"
    rows = store.tipoffs(roster_id, tip_at - timedelta(seconds=1), tip_at + timedelta(seconds=1))
    if not rows:
        return None, "unknown"
    r = rows[0]
    return r["starters_at_tip"], r["status"]
