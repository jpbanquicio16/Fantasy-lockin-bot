"""Live application state in SQLite (design section 21): nothing is updated in place except
status columns; every lineup observation, lock, manual entry, alert and recommendation is a
row with its time, so the ledger can be replayed and audited."""

from __future__ import annotations

import gzip
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from lockin_core.config import Paths, now_utc, ensure_utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS lineup_observations(
  obs_id TEXT PRIMARY KEY, league_id TEXT, roster_id INTEGER, observed_at TEXT, source TEXT,
  poll_ok INTEGER, starters TEXT, bench TEXT, reserve TEXT, raw_ref TEXT);
CREATE INDEX IF NOT EXISTS lo_roster_time ON lineup_observations(roster_id, observed_at);
CREATE TABLE IF NOT EXISTS lineup_events(
  event_id TEXT PRIMARY KEY, league_id TEXT, roster_id INTEGER, prev_obs_id TEXT, obs_id TEXT,
  effective_after TEXT, effective_before TEXT, player_id TEXT, from_slot INTEGER, to_slot INTEGER,
  source TEXT, confidence TEXT);
CREATE TABLE IF NOT EXISTS tipoff_snapshots(
  tip_batch_id TEXT, league_id TEXT, roster_id INTEGER, scheduled_tip_at TEXT, pre_obs_id TEXT,
  post_obs_id TEXT, starters_at_tip TEXT, status TEXT, resolved_by TEXT, resolved_at TEXT,
  PRIMARY KEY (tip_batch_id, roster_id));
CREATE TABLE IF NOT EXISTS tipoff_revisions(
  revision_id INTEGER PRIMARY KEY AUTOINCREMENT, tip_batch_id TEXT, league_id TEXT, roster_id INTEGER,
  scheduled_tip_at TEXT, pre_obs_id TEXT, post_obs_id TEXT, starters_at_tip TEXT, status TEXT,
  resolved_by TEXT, valid_from TEXT);
CREATE TABLE IF NOT EXISTS retractions(kind TEXT, object_id TEXT, retracted_at TEXT, entered_by TEXT);
CREATE TABLE IF NOT EXISTS eligible_scores(
  entry_id TEXT PRIMARY KEY, league_id TEXT, roster_id INTEGER, week INTEGER, player_id TEXT,
  game_id TEXT, slot_at_tip INTEGER, fp_half INTEGER, fp_source TEXT, box_revision INTEGER,
  is_team_final INTEGER, deadline_at TEXT, status TEXT, created_at TEXT, resolved_at TEXT,
  resolved_by TEXT, rec_id_followed TEXT);
CREATE TABLE IF NOT EXISTS deadlines(
  deadline_id TEXT PRIMARY KEY, entry_id TEXT, due_at TEXT, default_action TEXT,
  conditional_rule TEXT, alert_120_at TEXT, alert_30_at TEXT, alert_10_at TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS lock_events(
  lock_id TEXT PRIMARY KEY, league_id TEXT, week INTEGER, roster_id INTEGER, slot_index INTEGER,
  slot_type TEXT, player_id TEXT, game_id TEXT, fp_half INTEGER, locked_at TEXT, deadline_at TEXT,
  source TEXT, entered_by TEXT, entered_at TEXT, verified INTEGER, retracted INTEGER DEFAULT 0);
CREATE UNIQUE INDEX IF NOT EXISTS one_lock_per_player_week
  ON lock_events(league_id, week, roster_id, player_id) WHERE retracted = 0;
CREATE TABLE IF NOT EXISTS matchup_snapshots(
  snapshot_id TEXT PRIMARY KEY, league_id TEXT, week INTEGER, roster_id INTEGER, opp_roster_id INTEGER,
  taken_at TEXT, my_locked_half INTEGER, opp_locked_half INTEGER, opp_locked_source TEXT, api_points REAL,
  opp_starters TEXT, opp_open_entries TEXT);
CREATE TABLE IF NOT EXISTS recommendations(
  rec_id TEXT PRIMARY KEY, epoch_type TEXT, trigger_ref TEXT, created_at TEXT, expires_at TEXT,
  superseded_by TEXT, state_snapshot_ids TEXT, projection_version TEXT, model_version TEXT,
  config_hash TEXT, code_version TEXT, tzdata_version TEXT, seed INTEGER, objective TEXT,
  n_scenarios INTEGER, result TEXT, default_if_unattended TEXT, notified_at TEXT, shown_at TEXT,
  fingerprint TEXT);
CREATE TABLE IF NOT EXISTS recommendation_outcomes(
  rec_id TEXT, action_taken TEXT, matched INTEGER, recorded_at TEXT, realised_week_half INTEGER,
  counterfactual_half TEXT);
CREATE TABLE IF NOT EXISTS manual_entries(
  id TEXT PRIMARY KEY, kind TEXT, key TEXT, value TEXT, entered_by TEXT, entered_at TEXT,
  device TEXT, retracted INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS alerts(
  alert_id TEXT PRIMARY KEY, entry_key TEXT, kind TEXT, due_at TEXT, sent_at TEXT, channel TEXT,
  body TEXT, rec_id TEXT, ok INTEGER);
CREATE UNIQUE INDEX IF NOT EXISTS one_alert_per_slot ON alerts(entry_key, kind);
CREATE TABLE IF NOT EXISTS state_snapshots(snapshot_id TEXT PRIMARY KEY, taken_at TEXT, body BLOB);
CREATE TABLE IF NOT EXISTS service_state(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS event_log(id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, kind TEXT, detail TEXT);
"""


def _ts(x: datetime | None) -> str | None:
    return None if x is None else ensure_utc(x).isoformat()


def _dt(s: str | None) -> datetime | None:
    return None if s is None else datetime.fromisoformat(s)


@dataclass
class ManualEntry:
    id: str
    kind: str
    key: str
    value: dict
    entered_by: str
    entered_at: datetime


class Store:
    def __init__(self, paths: Paths | None = None, path: Path | None = None):
        paths = paths or Paths()
        self.path = path or paths.db
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # --- lineup ledger -----------------------------------------------------------
    def add_observation(self, league_id: str, roster_id: int, observed_at: datetime, starters: list,
                        bench: list, reserve: list, source: str = "api_poll", poll_ok: bool = True) -> str:
        oid = f"o_{observed_at:%Y%m%dT%H%M%S%f}_{roster_id}"
        with self.conn() as c:
            prev = c.execute("SELECT obs_id, starters FROM lineup_observations WHERE roster_id=? AND league_id=? "
                             "AND observed_at < ? ORDER BY observed_at DESC LIMIT 1",
                             (roster_id, league_id, _ts(observed_at))).fetchone()
            c.execute("INSERT OR IGNORE INTO lineup_observations VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (oid, league_id, roster_id, _ts(observed_at), source, int(poll_ok), json.dumps(starters),
                       json.dumps(bench), json.dumps(reserve), None))
            if prev is not None:
                before = json.loads(prev["starters"])
                for i, (a, b) in enumerate(zip(before + [None] * len(starters), starters)):
                    if a != b:
                        for pid, frm, to in ((a, i, None), (b, None, i)):
                            if pid:
                                c.execute("INSERT INTO lineup_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                          (uuid.uuid4().hex, league_id, roster_id, prev["obs_id"], oid, None,
                                           _ts(observed_at), pid, frm, to, source, "observed"))
        return oid

    def observations(self, roster_id: int, start: datetime, end: datetime) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM lineup_observations WHERE roster_id=? AND observed_at>=? AND observed_at<=? "
                             "ORDER BY observed_at", (roster_id, _ts(start), _ts(end))).fetchall()
        return [dict(r) | {"starters": json.loads(r["starters"]), "observed_at": _dt(r["observed_at"])} for r in rows]

    def save_tipoff(self, tip_batch_id: str, league_id: str, roster_id: int, tip_at: datetime,
                    pre: str | None, post: str | None, starters: list | None, status: str,
                    resolved_by: str | None = None, recorded_at: datetime | None = None) -> None:
        with self.conn() as c:
            at = _ts(recorded_at or now_utc())
            c.execute("INSERT INTO tipoff_revisions (tip_batch_id, league_id, roster_id, scheduled_tip_at, "
                      "pre_obs_id, post_obs_id, starters_at_tip, status, resolved_by, valid_from) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (tip_batch_id, league_id, roster_id, _ts(tip_at), pre, post,
                       None if starters is None else json.dumps(starters), status, resolved_by, at))
            c.execute("INSERT OR REPLACE INTO tipoff_snapshots VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (tip_batch_id, league_id, roster_id, _ts(tip_at), pre, post,
                       None if starters is None else json.dumps(starters), status, resolved_by,
                       _ts(now_utc()) if resolved_by else None))

    def tipoffs(self, roster_id: int, start: datetime, end: datetime) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM tipoff_snapshots WHERE roster_id=? AND scheduled_tip_at>=? AND "
                             "scheduled_tip_at<=? ORDER BY scheduled_tip_at", (roster_id, _ts(start), _ts(end))).fetchall()
        return [dict(r) | {"starters_at_tip": json.loads(r["starters_at_tip"]) if r["starters_at_tip"] else None,
                           "scheduled_tip_at": _dt(r["scheduled_tip_at"])} for r in rows]

    # --- locks and manual entries ------------------------------------------------
    def record_lock(self, league_id: str, week: int, roster_id: int, slot_index: int, slot_type: str,
                    player_id: str, game_id: str | None, fp_half: int, source: str = "manual",
                    entered_by: str = "you", locked_at: datetime | None = None) -> str:
        lid = uuid.uuid4().hex
        now = now_utc()
        with self.conn() as c:
            if c.execute("SELECT 1 FROM lock_events WHERE league_id=? AND week=? AND roster_id=? "
                         "AND slot_index=? AND retracted=0", (league_id, week, roster_id, slot_index)).fetchone():
                raise ValueError("This slot already has a recorded lock; correct that record first")
            c.execute("INSERT INTO lock_events (lock_id, league_id, week, roster_id, slot_index, slot_type, player_id,"
                      " game_id, fp_half, locked_at, deadline_at, source, entered_by, entered_at, verified) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (lid, league_id, week, roster_id, slot_index, slot_type, player_id, game_id, fp_half,
                       _ts(locked_at or now), None, source, entered_by, _ts(now), 0))
        return lid

    def locks(self, league_id: str, week: int, roster_id: int, as_of: datetime | None = None) -> list[dict]:
        query = "SELECT * FROM lock_events WHERE league_id=? AND week=? AND roster_id=?"
        args = [league_id, week, roster_id]
        if as_of is None:
            query += " AND retracted=0"
        else:
            query += (" AND entered_at<=? AND NOT EXISTS (SELECT 1 FROM retractions r WHERE "
                      "r.kind='lock' AND r.object_id=lock_events.lock_id AND r.retracted_at<=?)")
            args += [_ts(as_of), _ts(as_of)]
        with self.conn() as c:
            rows = c.execute(query, args).fetchall()
        return [dict(r) for r in rows]

    def retract_lock(self, lock_id: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE lock_events SET retracted=1 WHERE lock_id=?", (lock_id,))
            c.execute("INSERT INTO retractions VALUES ('lock',?,?,?)", (lock_id, _ts(now_utc()), "you"))
        self.invalidate("lock correction")

    def add_manual(self, kind: str, key: str, value: dict, entered_by: str = "you", device: str = "cli",
                   entered_at: datetime | None = None) -> str:
        mid = uuid.uuid4().hex
        with self.conn() as c:
            c.execute("INSERT INTO manual_entries VALUES (?,?,?,?,?,?,?,0)",
                      (mid, kind, key, json.dumps(value), entered_by, _ts(entered_at or now_utc()), device))
        return mid

    def manual(self, kind: str, as_of: datetime | None = None, key_prefix: str | None = None) -> list[ManualEntry]:
        q = "SELECT * FROM manual_entries WHERE kind=?"
        args: list[Any] = [kind]
        if as_of is not None:
            q += (" AND entered_at<=? AND NOT EXISTS (SELECT 1 FROM retractions r WHERE r.kind='manual' "
                  "AND r.object_id=manual_entries.id AND r.retracted_at<=?)")
            args.extend([_ts(as_of), _ts(as_of)])
        else:
            q += " AND retracted=0"
        if key_prefix is not None:
            q += " AND key LIKE ?"
            args.append(key_prefix + "%")
        with self.conn() as c:
            rows = c.execute(q + " ORDER BY entered_at", args).fetchall()
        return [ManualEntry(r["id"], r["kind"], r["key"], json.loads(r["value"]), r["entered_by"],
                            _dt(r["entered_at"])) for r in rows]

    def retract_manual(self, mid: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE manual_entries SET retracted=1 WHERE id=?", (mid,))
            c.execute("INSERT INTO retractions VALUES ('manual',?,?,?)", (mid, _ts(now_utc()), "you"))
        self.invalidate("manual correction")

    # --- entries and deadlines ---------------------------------------------------
    def upsert_entry(self, e: dict) -> None:
        cols = ("entry_id", "league_id", "roster_id", "week", "player_id", "game_id", "slot_at_tip", "fp_half",
                "fp_source", "box_revision", "is_team_final", "deadline_at", "status", "created_at", "resolved_at",
                "resolved_by", "rec_id_followed")
        with self.conn() as c:
            old = c.execute("SELECT status, created_at FROM eligible_scores WHERE entry_id=?", (e["entry_id"],)).fetchone()
            row = dict(e)
            row.setdefault("created_at", _ts(now_utc()))
            if old is not None:
                row["created_at"] = old["created_at"]
                if old["status"] != "open" and row.get("status") == "open":
                    row["status"] = old["status"]  # a resolved entry is never reopened by inference
            c.execute(f"INSERT OR REPLACE INTO eligible_scores ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      tuple(row.get(k) for k in cols))

    def entries(self, league_id: str, week: int, roster_id: int) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM eligible_scores WHERE league_id=? AND week=? AND roster_id=?",
                             (league_id, week, roster_id)).fetchall()
        return [dict(r) for r in rows]

    def upsert_deadline(self, entry_id: str, due_at: datetime, default_action: dict, conditional_rule: dict | None,
                        offsets: list[int]) -> None:
        from datetime import timedelta

        al = {o: _ts(due_at - timedelta(minutes=o)) for o in (120, 30, 10)}
        with self.conn() as c:
            old = c.execute("SELECT status, due_at FROM deadlines WHERE deadline_id=?", (entry_id,)).fetchone()
            c.execute("INSERT OR REPLACE INTO deadlines VALUES (?,?,?,?,?,?,?,?,?)",
                      (entry_id, entry_id, _ts(due_at), json.dumps(default_action),
                       None if conditional_rule is None else json.dumps(conditional_rule),
                       al[120], al[30], al[10],
                       old["status"] if old and old["due_at"] == _ts(due_at) else "pending"))

    def deadlines(self, status: str = "pending") -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM deadlines WHERE status=? ORDER BY due_at", (status,)).fetchall()
        return [dict(r) | {"due_at": _dt(r["due_at"]), "default_action": json.loads(r["default_action"] or "{}"),
                           "conditional_rule": json.loads(r["conditional_rule"]) if r["conditional_rule"] else None}
                for r in rows]

    def set_deadline_status(self, entry_id: str, status: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE deadlines SET status=? WHERE deadline_id=?", (status, entry_id))

    # --- alerts ------------------------------------------------------------------
    def alert_sent(self, entry_key: str, kind: str) -> bool:
        with self.conn() as c:
            return c.execute("SELECT 1 FROM alerts WHERE entry_key=? AND kind=? AND ok=1",
                             (entry_key, kind)).fetchone() is not None

    def log_alert(self, entry_key: str, kind: str, due_at: datetime | None, channel: str, body: str,
                  rec_id: str | None, ok: bool) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO alerts VALUES (?,?,?,?,?,?,?,?,?)",
                      (uuid.uuid4().hex, entry_key, kind, _ts(due_at), _ts(now_utc()), channel, body, rec_id, int(ok)))

    def alerts(self, limit: int = 50) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM alerts ORDER BY sent_at DESC LIMIT ?", (limit,))]

    # --- recommendations and snapshots ---------------------------------------------
    def save_snapshot(self, snapshot_id: str, taken_at: datetime, body: dict) -> None:
        blob = gzip.compress(json.dumps(body, default=str).encode())
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO state_snapshots VALUES (?,?,?)", (snapshot_id, _ts(taken_at), blob))

    def snapshot(self, snapshot_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT body FROM state_snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        return json.loads(gzip.decompress(r["body"])) if r else None

    def save_recommendation(self, rec: dict) -> None:
        cols = ("rec_id", "epoch_type", "trigger_ref", "created_at", "expires_at", "superseded_by",
                "state_snapshot_ids", "projection_version", "model_version", "config_hash", "code_version",
                "tzdata_version", "seed", "objective", "n_scenarios", "result", "default_if_unattended",
                "notified_at", "shown_at", "fingerprint")
        with self.conn() as c:
            prev = c.execute("SELECT rec_id FROM recommendations WHERE superseded_by IS NULL AND rec_id != ? "
                             "ORDER BY created_at DESC", (rec["rec_id"],)).fetchall()
            c.execute(f"INSERT INTO recommendations ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                      tuple(json.dumps(rec[k], default=str) if isinstance(rec.get(k), (dict, list)) else rec.get(k)
                            for k in cols))
            if not rec.get("superseded_by"):
                for r in prev:
                    c.execute("UPDATE recommendations SET superseded_by=? WHERE rec_id=? AND created_at<=?",
                              (rec["rec_id"], r["rec_id"], rec["created_at"]))

    def latest_recommendation(self) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM recommendations WHERE superseded_by IS NULL ORDER BY created_at DESC LIMIT 1"
                          ).fetchone()
        if r is None:
            return None
        d = dict(r)
        for k in ("result", "default_if_unattended", "state_snapshot_ids"):
            if d.get(k):
                d[k] = json.loads(d[k])
        return d

    def recommendations(self, since: datetime | None = None) -> list[dict]:
        q, args = "SELECT * FROM recommendations", []
        if since is not None:
            q += " WHERE created_at>=?"
            args.append(_ts(since))
        with self.conn() as c:
            rows = c.execute(q + " ORDER BY created_at", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("result", "default_if_unattended", "state_snapshot_ids"):
                if d.get(k):
                    d[k] = json.loads(d[k])
            out.append(d)
        return out

    def mark_notified(self, rec_id: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE recommendations SET notified_at=? WHERE rec_id=?", (_ts(now_utc()), rec_id))

    def add_outcome(self, rec_id: str, action_taken: dict, matched: bool, realised: int | None,
                    counterfactual: dict | None) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO recommendation_outcomes VALUES (?,?,?,?,?,?)",
                      (rec_id, json.dumps(action_taken), int(matched), _ts(now_utc()), realised,
                       None if counterfactual is None else json.dumps(counterfactual)))

    def outcomes(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM recommendation_outcomes")]

    # --- service bookkeeping -------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with self.conn() as c:
            r = c.execute("SELECT value FROM service_state WHERE key=?", (key,)).fetchone()
        return json.loads(r["value"]) if r else default

    def put(self, key: str, value: Any) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO service_state VALUES (?,?,?)",
                      (key, json.dumps(value, default=str), _ts(now_utc())))

    def log(self, kind: str, detail: Any) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO event_log (at, kind, detail) VALUES (?,?,?)",
                      (_ts(now_utc()), kind, json.dumps(detail, default=str)))

    def events(self, limit: int = 100) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM event_log ORDER BY id DESC LIMIT ?", (limit,))]

    def invalidate(self, reason: str) -> None:
        """Hide advice immediately after a user or feed changes its inputs."""
        with self.conn() as c:
            c.execute("UPDATE recommendations SET superseded_by=? WHERE superseded_by IS NULL",
                      ("input:" + reason,))
            c.execute("INSERT OR REPLACE INTO service_state VALUES (?,?,?)",
                      ("force_recommend", "true", _ts(now_utc())))
            row = c.execute("SELECT value FROM service_state WHERE key='input_revision'").fetchone()
            revision = int(json.loads(row[0])) + 1 if row else 1
            c.execute("INSERT OR REPLACE INTO service_state VALUES (?,?,?)",
                      ("input_revision", json.dumps(revision), _ts(now_utc())))

    def resolve_entry(self, entry_id: str, status: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE eligible_scores SET status=?, resolved_at=?, resolved_by=? WHERE entry_id=?",
                      (status, _ts(now_utc()), "you", entry_id))
            c.execute("UPDATE deadlines SET status='met' WHERE entry_id=?", (entry_id,))
        self.invalidate(status)

    def mark_shown(self, rec_id: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE recommendations SET shown_at=COALESCE(shown_at,?) WHERE rec_id=?",
                      (_ts(now_utc()), rec_id))

    def backup(self, destination: Path) -> Path:
        """Online, transactionally consistent SQLite backup (safe with a running service)."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.resolve() == self.path.resolve():
            raise ValueError("Backup destination must differ from the live database")
        with self.conn() as src, sqlite3.connect(destination) as dst:
            src.backup(dst)
        return destination

    def export_history(self, destination: Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        names = ("lineup_observations", "lineup_events", "tipoff_revisions", "lock_events", "manual_entries",
                 "retractions", "recommendations", "recommendation_outcomes", "alerts")
        with self.conn() as c:
            data = {name: [dict(r) for r in c.execute(f"SELECT * FROM {name}")] for name in names}
        with gzip.open(destination, "wt", encoding="utf-8") as f:
            json.dump(data, f)
        return destination
