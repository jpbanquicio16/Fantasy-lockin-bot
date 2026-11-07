"""One recommendation, end to end: validated state -> week model -> rollout -> stored record."""

from __future__ import annotations

from datetime import datetime

from lockin_core.config import LeagueConfig, ModelConfig, Paths, SimConfig, now_utc
from lockin_core.ingest.sleeper import SleeperClient
from lockin_core.models.fit import load as load_params
from lockin_core.sim.build import WeekModel, build_week_model
from lockin_core.state.build import Context, build_state
from lockin_core.state.models import ValidatedState
from lockin_core.state.store import Store
from lineup_lock.recommend.engine import Recommender


def manual_overrides(store: Store, now: datetime) -> dict[str, dict]:
    """Availability / minutes overrides entered by you, unexpired at ``now``."""
    out: dict[str, dict] = {}
    for e in store.manual("override", as_of=now):
        exp = e.value.get("expires_at")
        if exp is None or datetime.fromisoformat(exp) > now:
            out.setdefault(e.key, {}).update({k: v for k, v in e.value.items() if k != "expires_at"})
    return out


def recommend(cfg: LeagueConfig, mcfg: ModelConfig, scfg: SimConfig, paths: Paths | None = None,
              store: Store | None = None, client: SleeperClient | None = None, now: datetime | None = None,
              objective: str | None = None, epoch_type: str = "manual", trigger: str = "",
              save: bool = True) -> tuple[dict, ValidatedState, Context, WeekModel]:
    paths = (paths or Paths()).ensure()
    store = store or Store(paths)
    now = now or now_utc()
    state, ctx = build_state(cfg, paths, client, store, now, objective or scfg.objective, save=save)
    revision = store.get("input_revision", 0)
    params, lib = load_params(paths)
    cal = store.get("status_calibration")
    calibration = {tuple(k.split("|")): v for k, v in (cal or {}).items()} or None
    wm = build_week_model(state, ctx, paths, params, lib, mcfg, scfg, manual_overrides(store, now), calibration)
    rec = Recommender(wm, scfg).run(objective or scfg.objective, epoch_type, trigger)
    if save:
        import uuid
        rec["rec_id"] += "_" + uuid.uuid4().hex[:8]
        rec["units_per_point"] = state.units_per_point
        rec["superseded"] = store.get("input_revision", 0) != revision or now_utc() >= datetime.fromisoformat(rec["expires_at"])
        from lineup_lock.replay.artifacts import save as save_artifact
        save_artifact(paths, rec, wm)
        store.save_recommendation({
            "rec_id": rec["rec_id"], "epoch_type": epoch_type, "trigger_ref": trigger,
            "created_at": rec["created_at"], "expires_at": rec["expires_at"], "superseded_by": "input_changed" if rec["superseded"] else None,
            "state_snapshot_ids": [state.snapshot_id], "projection_version": params.version,
            "model_version": params.version, "config_hash": rec["versions"]["config"],
            "code_version": rec["versions"]["code"], "tzdata_version": rec["versions"]["tzdata"],
            "seed": scfg.seed, "objective": rec["objective"], "n_scenarios": rec["n_scenarios"], "result": rec,
            "default_if_unattended": {c["entry_id"]: c["default_if_unattended"] for c in rec["entries"]},
            "fingerprint": rec["fingerprint"]})
        if rec["superseded"]:
            store.put("force_recommend", True)
            return rec, state, ctx, wm
        active_entries = {c["entry_id"] for c in rec["entries"]}
        for d in store.deadlines():
            if d["entry_id"] not in active_entries:
                store.set_deadline_status(d["entry_id"], "resolved")
        for c in rec["entries"]:
            from datetime import datetime as _dt

            store.upsert_deadline(c["entry_id"], _dt.fromisoformat(c["deadline_at"]),
                                  {"advice": c["advice"], "default": c["default_if_unattended"],
                                   "player": c["player"], "player_id": c["player_id"], "game_id": c["game_id"],
                                   "score": c["score"], "lock_slot": c["lock_slot_if_locking"]},
                                  c.get("conditional_rule") if c["advice"] == "wait" else None, [120, 30, 10])
    return rec, state, ctx, wm
