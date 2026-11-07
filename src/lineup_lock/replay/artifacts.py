"""Safe JSON snapshots of every model input needed to reproduce a recommendation.

Only explicitly registered application types can be decoded; no pickle or executable imports.
"""
from __future__ import annotations

import dataclasses
import gzip
import json
from datetime import date, datetime, timedelta

import numpy as np
from pydantic import BaseModel

from lockin_core.config import Paths, LeagueConfig, ModelConfig, SimConfig, RuleSwitches
from lockin_core.models.fit import Params
from lockin_core.models.project import PlayerProjection
from lockin_core.policy.tables import ContinuationTables
from lockin_core.schedule.calendar import Game, WeekCalendar
from lockin_core.scoring.settings import ScoringSettings
from lockin_core.sim.build import WeekModel
from lockin_core.sim.problem import SideProblem, SimPlayer, RootEntry, Action
from lockin_core.state.build import Context, WeekGameScore
from lockin_core.state.league import LeagueRules
from lockin_core.state.models import ValidatedState, SideState, PlayerInfo, EligibleScore, LockedSlot

TYPES = {c.__name__: c for c in (LeagueConfig, ModelConfig, SimConfig, RuleSwitches, Params, PlayerProjection,
         ContinuationTables, Game, WeekCalendar, ScoringSettings, WeekModel, SideProblem, SimPlayer, RootEntry,
         Context, WeekGameScore, LeagueRules, ValidatedState, SideState, PlayerInfo, EligibleScore, LockedSlot)}


def encode(value):
    if isinstance(value, np.ndarray):
        return {"type": "array", "dtype": str(value.dtype), "data": value.tolist(), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return {"type": "datetime", "data": value.isoformat()}
    if isinstance(value, date):
        return {"type": "date", "data": value.isoformat()}
    if isinstance(value, timedelta):
        return {"type": "timedelta", "data": value.total_seconds()}
    if isinstance(value, BaseModel):
        return {"type": type(value).__name__, "data": value.model_dump(mode="json")}
    if dataclasses.is_dataclass(value):
        return {"type": type(value).__name__, "data": {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, dict):
        return {"type": "dict", "data": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (tuple, set, frozenset)):
        return {"type": type(value).__name__, "data": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind, data = value["type"], value["data"]
    if kind == "array":
        return np.array(data, dtype=value["dtype"]).reshape(value["shape"])
    if kind == "datetime":
        return datetime.fromisoformat(data)
    if kind == "date":
        return date.fromisoformat(data)
    if kind == "timedelta":
        return timedelta(seconds=data)
    if kind == "dict":
        return {decode(k): decode(v) for k, v in data}
    if kind in ("tuple", "set", "frozenset"):
        return {"tuple": tuple, "set": set, "frozenset": frozenset}[kind](decode(v) for v in data)
    cls = TYPES[kind]
    if issubclass(cls, BaseModel):
        return cls.model_validate(data)
    return cls(**{k: decode(v) for k, v in data.items()})


def artifact_path(paths: Paths, rec_id: str):
    if '/' in rec_id or '\\' in rec_id or rec_id in ('.', '..'):
        raise ValueError("Invalid recommendation ID")
    return paths.artifacts / "recommendations" / (rec_id + ".json.gz")


def save(paths: Paths, rec: dict, model: WeekModel):
    path = artifact_path(paths, rec["rec_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"format": 1, "model": encode(model), "recommendation": rec}, f, allow_nan=False)


def load(paths: Paths, rec_id: str):
    with gzip.open(artifact_path(paths, rec_id), "rt", encoding="utf-8") as f:
        artifact = json.load(f)
    if artifact["format"] != 1:
        raise ValueError("Unsupported recommendation snapshot version")
    return decode(artifact["model"]), artifact["recommendation"]


def canonical(rec: dict) -> str:
    # Runtime/identity are not decisions. Compare the actual advice and all numerical evidence.
    skip = {"rec_id", "runtime_s", "timings", "versions", "superseded", "units_per_point", "flags"}
    body = {k: v for k, v in rec.items() if k not in skip}
    body["flags"] = [f for f in rec.get("flags", []) if f != "monte_carlo_limited"]
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)


def audit(paths: Paths, rec_id: str) -> bool:
    from lineup_lock.recommend.engine import Recommender
    wm, rec = load(paths, rec_id)
    replayed = Recommender(wm, wm.scfg).run(rec["objective"], rec["epoch_type"], rec["trigger"],
                                         fixed_scenarios=rec["n_scenarios"])
    return canonical(replayed) == canonical(rec)


def root_action(rec: dict, problem: SideProblem) -> Action:
    action = rec["best_action"]
    return Action(locks=tuple((problem.index(x["player_id"]), x["slot_index"]) for x in action["locks"]),
                  passes=tuple(problem.index(x["player_id"]) for x in action["passes"]),
                  lineup=None if action.get("lineup") is None else tuple(
                      (problem.index(x["player_id"]), x["slot_index"]) for x in action["lineup"]))
