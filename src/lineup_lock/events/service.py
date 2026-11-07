"""The event loop (design section 20): pollers and the clock turn the world into events, and
any event that touches a decision triggers a recommendation.

``Service.tick(now)`` is one pass, deterministic given the clock and the stored state, so it
can be tested with fake clients and survives restarts (everything it remembers is in SQLite):

* rosters - every minute from T-10 to T+5 around each tip batch with your or your opponent's
  players, every 5 minutes on game days, every 30 otherwise -> lineup ledger observations;
* live games - every 2 minutes while one involving a rostered player is in progress: schedule,
  tip times and this week's box scores; a newly final game -> recommendation (post-final);
* injury reports - every 15 minutes on game days; a status change for a relevant player ->
  recommendation (news);
* pre-tip epochs - T-60, T-30, T-10 before each tip batch -> recommendation;
* deadline alerts - T-120, T-30, T-10 minutes before each open entry's deadline, from the latest
  recommendation, or from the stored default if the engine has failed since;
* heartbeat - a ping every 5 minutes on game days, so an external monitor notices if this stops;
* nightly - full ingest; Monday - refit.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import httpx

from lockin_core.config import (ET, AlertsConfig, LeagueConfig, ModelConfig, Paths, SimConfig, load_alerts,
                                load_league, load_model, load_sim, now_utc)
from lockin_core.data import tables as T
from lockin_core.ingest.collect import IngestReport, ingest, ingest_injury_reports, refresh_live
from lockin_core.ingest.espn import EspnClient
from lockin_core.ingest.sleeper import SleeperClient
from lockin_core.schedule.calendar import WeekCalendar, build_calendar, fmt_local
from lockin_core.state.store import Store
from lineup_lock.events.notify import Notifier
from lineup_lock.recommend.render import notification


@dataclass
class TickResult:
    at: datetime
    did: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    recommended: str | None = None


class Service:
    def __init__(self, paths: Paths | None = None, cfg: LeagueConfig | None = None, mcfg: ModelConfig | None = None,
                 scfg: SimConfig | None = None, acfg: AlertsConfig | None = None, client: SleeperClient | None = None,
                 espn: EspnClient | None = None, store: Store | None = None, notifier: Notifier | None = None,
                 recommend_fn=None):
        self.paths = (paths or Paths()).ensure()
        self.cfg = cfg or load_league(self.paths)
        self.mcfg = mcfg or load_model(self.paths)
        self.scfg = scfg or load_sim(self.paths)
        self.acfg = acfg or load_alerts(self.paths)
        self.client = client or SleeperClient(paths=self.paths)
        self.espn = espn or EspnClient(paths=self.paths)
        self.store = store or Store(self.paths)
        self.notifier = notifier or Notifier(self.acfg, self.paths)
        if recommend_fn is None:
            from lineup_lock.recommend.run import recommend as recommend_fn
        self.recommend_fn = recommend_fn

    # --- schedule ------------------------------------------------------------------
    def calendar(self, now: datetime) -> tuple[WeekCalendar, int, int] | None:
        st = self.client.state(max_age_s=3600)
        season, week = int(st.season), int(st.week)
        games = T.read(self.paths, "games", season)
        if games is None:
            return None
        return build_calendar(games, season, week, now, self.cfg.rules.deadline_margin_minutes), season, week

    def _due(self, key: str, every: timedelta, now: datetime) -> bool:
        last = self.store.get(f"last:{key}")
        if last is None or now - datetime.fromisoformat(last) >= every:
            self.store.put(f"last:{key}", now.isoformat())
            return True
        return False

    def _relevant_teams(self) -> set[str]:
        teams = set()
        players = T.read(self.paths, "players")
        rosters = self.client.rosters(self.cfg.league_id, max_age_s=600)
        from lockin_core.state.build import resolve_my_roster
        my_id = resolve_my_roster(self.cfg, self.client, rosters)
        self.store.put("my_roster_id", my_id)
        mine = [r for r in rosters if r.roster_id == my_id]
        opp_ids = set(self.store.get("opp_roster_ids", []))
        pids = {p for r in rosters if r.roster_id in {mine[0].roster_id, *opp_ids} for p in (r.players or [])}
        if players is not None:
            for pid, team in zip(players["player_id"], players["team"]):
                if pid in pids and team:
                    teams.add(team)
        return teams

    # --- one pass ------------------------------------------------------------------
    def tick(self, now: datetime | None = None) -> TickResult:
        now = now or now_utc()
        res = TickResult(now)
        # Persisted deadlines are independent of every upstream feed and model.
        self._step(res, "alerts", lambda: self.alerts(now))
        try:
            self._tick(now, res)
        except Exception as e:
            res.errors.append(f"tick: {e}")
            self.store.log("error", {"step": "tick", "error": str(e)})
        finally:
            self.store.put("last_tick", now.isoformat())
            if self.acfg.heartbeat_url and self._due("heartbeat", timedelta(minutes=5), now):
                self._step(res, "heartbeat", lambda: httpx.get(self.acfg.heartbeat_url, timeout=10).raise_for_status())
        return res

    def _tick(self, now: datetime, res: TickResult) -> None:
        try:
            c = self.calendar(now)
        except Exception as e:
            res.errors.append(f"calendar: {e}")
            c = None
        if c is None:
            self._step(res, "ingest", lambda: ingest(self.cfg, self.paths, self.client, self.espn))
            return res
        cal, season, week = c
        teams = self._relevant_teams()
        today = now.astimezone(ET).date()
        game_day = cal.is_game_day(today)
        tips = sorted({g.tip_at for g in cal.games if set(g.teams()) & teams
                       and g.tip_at >= now - timedelta(minutes=10)})
        near_tip = any(t - timedelta(minutes=10) <= now <= t + timedelta(minutes=5) for t in tips)
        triggers: list[str] = []

        # rosters -> ledger
        every = timedelta(minutes=1) if near_tip else timedelta(minutes=5) if game_day else timedelta(minutes=30)
        if self._due("rosters", every, now):
            self._step(res, "observe", lambda: self.observe(now))
        # Refresh the schedule even between slates; postponements change every deadline.
        if self._due("schedule", timedelta(minutes=15), now):
            before_schedule = [(g.game_id, g.tip_at, g.status) for g in cal.games]
            self._step(res, "schedule", lambda: refresh_live(self.paths, self.client, self.espn, season, week))
            updated = self.calendar(now)
            if updated and before_schedule != [(g.game_id, g.tip_at, g.status) for g in updated[0].games]:
                cal = updated[0]
                triggers.append("schedule_changed")
                self.store.invalidate("schedule changed")
                self.reschedule_deadlines(cal, now)
        # live games
        live = [g for g in cal.live_games() if set(g.teams()) & teams]
        if live and self._due("live", timedelta(minutes=2), now):
            before = {g.game_id for g in cal.games if g.status == "final"}
            self._step(res, "refresh_live", lambda: refresh_live(self.paths, self.client, self.espn, season, week))
            c2 = self.calendar(now)
            if c2:
                newly = [g for g in c2[0].games if g.status == "final" and g.game_id not in before
                         and set(g.teams()) & teams]
                if newly:
                    triggers.append("final:" + ",".join(f"{g.away}@{g.home}" for g in newly))
        # injury reports
        if game_day and self._due("injury", timedelta(minutes=15), now):
            before = self.store.get("report_digest")
            self._step(res, "injury_reports", lambda: ingest_injury_reports(
                self.paths, IngestReport(season, week), lookback_hours=1.0, now=now))
            dig = self._report_digest(teams, now)
            if dig != before:
                self.store.put("report_digest", dig)
                if before is not None:
                    triggers.append("report")
        # pre-tip epochs
        for t in tips:
            for off in (60, 30, 10):
                key = f"pretip:{t.isoformat()}:{off}"
                if t - timedelta(minutes=off) <= now < t - timedelta(minutes=off - 9) and not self.store.get(key):
                    self.store.put(key, now.isoformat())
                    triggers.append(f"pretip-{off}:{fmt_local(t, self.cfg.tz)}")
        # recommendation
        if triggers or self.store.get("force_recommend"):
            self.store.put("force_recommend", True)
            self._step(res, "recommend", lambda: self.recommend(now, ";".join(triggers) or "manual", res))
            if res.recommended:
                self.store.put("force_recommend", False)
        # deadline alerts (after any new recommendation)
        self._step(res, "alerts", lambda: self.alerts(now))
        # nightly ingest (US morning: every game of the night is final) and Monday refit
        if now.astimezone(ET).hour == 9 and self._due("nightly", timedelta(hours=20), now):
            self._step(res, "nightly_ingest", lambda: ingest(self.cfg, self.paths, self.client, self.espn))
            self._step(res, "backup", lambda: self.backup(now))
            self._step(res, "replay_audit", lambda: self.audit(now))
            if now.astimezone(ET).weekday() == 0:
                self._step(res, "weekly_fit", lambda: self.refit())
        self.store.put("last_tick", now.isoformat())
        return res

    def _step(self, res: TickResult, name: str, fn) -> None:
        try:
            fn()
            res.did.append(name)
        except Exception as e:
            res.errors.append(f"{name}: {e}")
            self.store.log("error", {"step": name, "error": str(e), "trace": traceback.format_exc()[-2000:]})

    def _report_digest(self, teams: set[str], now: datetime) -> str:
        import hashlib

        reps = T.read(self.paths, "injury_reports")
        if reps is None:
            return ""
        cur = T.report_status_as_of(reps, now)
        cur = cur.filter(cur["team"].is_in(list(teams)))
        rows = sorted(zip(cur["player_id"].fill_null("").to_list(), cur["game_date"].cast(str).to_list(),
                          cur["status"].fill_null("").to_list()))
        return hashlib.sha256(str(rows).encode()).hexdigest()[:12]

    # --- actions -----------------------------------------------------------------
    def observe(self, now: datetime) -> int:
        rosters = self.client.rosters(self.cfg.league_id, max_age_s=0)
        from lockin_core.state.build import resolve_my_roster
        from lockin_core.state.ledger import resolve_tipoffs
        my_id = resolve_my_roster(self.cfg, self.client, rosters)
        self.store.put("my_roster_id", my_id)
        ids = {my_id, *self.store.get("opp_roster_ids", [])}
        calendar = self.calendar(now)
        n = 0
        for r in rosters:
            if r.roster_id in ids:
                st = [p if p and p != "0" else None for p in (r.starters or [])]
                bench = [p for p in (r.players or []) if p not in set(st) and p not in set(r.reserve or [])]
                previous = self.store.get(f"lineup:{r.roster_id}")
                current = [st, bench, r.reserve or []]
                self.store.add_observation(self.cfg.league_id, r.roster_id, now, st, bench, r.reserve or [])
                self.store.put(f"lineup:{r.roster_id}", current)
                if previous != current:
                    self.store.invalidate("lineup changed")
                if calendar:
                    resolve_tipoffs(self.store, self.cfg.league_id, r.roster_id, calendar[0], now)
                n += 1
        return n

    def recommend(self, now: datetime, trigger: str, res: TickResult) -> None:
        prev = self.store.latest_recommendation()
        try:
            rec, state, ctx, wm = self.recommend_fn(self.cfg, self.mcfg, self.scfg, self.paths, self.store, self.client,
                                                    now, None, "event", trigger)
        except Exception as error:
            from lockin_core.state.build import BlockingInputError
            if isinstance(error, BlockingInputError) and self._due("blocked_notice", timedelta(minutes=30), now):
                self.notifier.send("Lock-In inputs need confirmation", str(error), now)
            raise
        if state.opp is not None:
            self.store.put("opp_roster_ids", [state.opp.roster_id])
        if rec.get("superseded"):
            self.store.put("force_recommend", True)
            return
        res.recommended = rec["rec_id"]
        changed = prev is None or prev.get("fingerprint") != rec["fingerprint"]
        if changed:
            for c in rec["entries"]:
                if c["advice"] in ("lock_now", "wait", "pass"):
                    title, body = notification(c, rec, self.acfg.app_link, prefix="Changed: " if prev else "")
                    self.notifier.send(title, body, now)
            lu = rec.get("lineup_next_tip")
            if lu and lu["start"]:
                self.notifier.send(f"Lineup for {lu['tip_local']}",
                                   " · ".join(f"{x['slot']} {x['player']}" for x in lu["start"]), now)
            self.store.mark_notified(rec["rec_id"])

    def reschedule_deadlines(self, cal: WeekCalendar, now: datetime) -> None:
        players = T.read(self.paths, "players")
        teams = dict(zip(players["player_id"], players["team"])) if players is not None else {}
        for d in self.store.deadlines():
            da = d["default_action"]
            team, gid = teams.get(da.get("player_id")), da.get("game_id")
            if gid in {g.game_id for g in cal.postponed}:
                self.store.set_deadline_status(d["entry_id"], "canceled")
                continue
            if not team or not gid or cal.game(gid) is None:
                continue
            due, _ = cal.deadline(team, gid)
            self.store.upsert_deadline(d["entry_id"], due, da, d["conditional_rule"], self.acfg.offsets_minutes)

    def alerts(self, now: datetime) -> int:
        sent = 0
        latest = self.store.latest_recommendation()
        fresh = latest and datetime.fromisoformat(latest["expires_at"]) > now
        cards = {c["entry_id"]: c for c in latest["result"].get("entries", [])} if fresh else {}
        for d in self.store.deadlines():
            due = d["due_at"]
            key = d["entry_id"] + "@" + due.isoformat()
            if now >= due:
                alerted = any(self.store.alert_sent(key, f"T-{off}") for off in self.acfg.offsets_minutes)
                self.store.set_deadline_status(d["entry_id"], "expired" if alerted else "missed")
                if not alerted:
                    body = f"No successful alert before the deadline for {d['default_action'].get('player', 'open score')}. Check Sleeper."
                    ok = self.notifier.send("Missed deadline alert", body, now)
                    self.store.log_alert(key, "missed", due, self.notifier.channel, body, None, ok)
                    self.store.log("missed_deadline", {"entry_id": d["entry_id"], "due_at": due.isoformat()})
                continue
            offsets = sorted(set(self.acfg.offsets_minutes))
            reached = [off for off in offsets if due - timedelta(minutes=off) <= now]
            if not reached:
                continue
            off = min(reached)  # after a restart, send the most urgent alert, not a stale 120-minute one
            if self.store.alert_sent(key, f"T-{off}"):
                continue
            c = cards.get(d["entry_id"])
            if c is not None:
                title, body = notification(c, latest["result"], self.acfg.app_link,
                                           prefix=f"Due in {max(0, int((due-now).total_seconds()/60))} min: ")
            else:
                da = d["default_action"]
                title = f"Deadline: {da.get('player', 'open score')} {da.get('score', '')}"
                body = (f"Deadline {fmt_local(due, self.cfg.tz)}. No fresh recommendation: stored default "
                        f"is {da.get('default', 'check Sleeper')}. Act in Sleeper; this assistant cannot lock for you.")
            ok = self.notifier.send(title, body, now)
            self.store.log_alert(key, f"T-{off}", due, self.notifier.channel, body,
                                 latest["rec_id"] if fresh else None, ok)
            sent += int(ok)
        return sent

    def backup(self, now: datetime) -> None:
        from pathlib import Path
        import os
        import shutil
        destination = Path(os.environ.get("LOCKIN_BACKUP_DIR", self.paths.data / "backups")) / now.strftime("%Y-%m-%d")
        self.store.backup(destination / "app.sqlite")
        self.store.export_history(destination / "ledger.json.gz")
        shutil.copytree(self.paths.tables, destination / "tables", dirs_exist_ok=True)
        shutil.copytree(self.paths.artifacts, destination / "artifacts", dirs_exist_ok=True)

    def audit(self, now: datetime) -> None:
        import hashlib
        from lineup_lock.replay.artifacts import audit
        for r in self.store.recommendations(since=now - timedelta(days=1)):
            if int(hashlib.sha256(r["rec_id"].encode()).hexdigest()[:8], 16) % 20:
                continue
            ok = audit(self.paths, r["rec_id"])
            self.store.log("replay_audit", {"rec_id": r["rec_id"], "matched": ok})
            if not ok:
                self.notifier.send("Recommendation replay mismatch", f"Audit failed for {r['rec_id']}", now)

    def refit(self) -> None:
        from lockin_core.ingest.sleeper import SleeperClient as _S
        from lockin_core.models.fit import fit, save
        from lockin_core.scoring.settings import ScoringSettings

        lg = _S(paths=self.paths).league(self.cfg.league_id)
        sc = ScoringSettings.from_sleeper(lg.scoring_settings, self.cfg.scoring.threshold_bonus_mode,
                                          self.cfg.scoring.tech_definition)
        params, lib = fit(self.paths, self.cfg.history_seasons, sc, self.mcfg,
                          log=lambda m: self.store.log("fit", m), persist=False, return_lib=True)
        save(self.paths, params, lib, activate=False)
        self.store.log("candidate_model", {"version": params.version,
                       "status": "awaiting evaluation before activation"})


def run_forever(service: Service) -> None:  # pragma: no cover - long-running
    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(lambda: _print(service.tick()), "interval", minutes=1, max_instances=1, coalesce=True,
                  next_run_time=now_utc())
    print("lockin service running: one tick a minute (Ctrl-C to stop)")
    sched.start()


def _print(r: TickResult) -> None:
    if r.did or r.errors:
        print(f"{r.at:%H:%M:%S}Z did={r.did} errors={r.errors} rec={r.recommended}")
