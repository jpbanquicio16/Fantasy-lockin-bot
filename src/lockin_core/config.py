"""Configuration: paths, the YAML configs (league, model, sim, alerts) and time helpers.

Times are stored as UTC instants. NBA game dates and Sleeper's Monday-Sunday weeks are
US Eastern dates (the scheduled ET date decides the week, switch ``week_assignment``);
the user sees Adelaide time, converted only when rendering. Seasons are named by their
start year: 2026 means 2026-27.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")
UTC = timezone.utc


def now_utc() -> datetime:
    return datetime.now(UTC)


def et_date(ts: datetime) -> date:
    return ts.astimezone(ET).date()


def from_ms(ms: int | float | None) -> datetime | None:
    return None if ms is None else datetime.fromtimestamp(ms / 1000, UTC)


def ensure_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


class Paths(BaseModel):
    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def tables(self) -> Path:
        return self.data / "tables"

    @property
    def artifacts(self) -> Path:
        return self.data / "artifacts"

    @property
    def reports(self) -> Path:
        return self.data / "reports"

    @property
    def db(self) -> Path:
        return self.data / "app.sqlite"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.tables, self.artifacts, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


# --- league ----------------------------------------------------------------------
class ScoringConfig(BaseModel):
    # Open tests in the design (section 4). None = not settled: exact scoring refuses to run.
    threshold_bonus_mode: Literal["stack", "exclusive"] | None = None
    tech_definition: Literal["sleeper_tf"] | None = None


class RuleSwitches(BaseModel):
    """Undocumented Lock-In behaviour (design section 1). Each default is the conservative
    reading; each is settled by an in-app Open test and then frozen in league.yaml."""

    final_game_pending_mode: Literal["auto_until_week_end"] = "auto_until_week_end"
    move_during_game: bool = False
    bench_final_slot_rule: Literal["zero"] = "zero"
    deadline_on_inactive: Literal["team_tip"] = "team_tip"
    deadline_margin_minutes: int = 10
    postponement_rule: Literal["recompute_from_schedule"] = "recompute_from_schedule"
    locked_points_survive_il: Literal["unknown", "yes", "no"] = "unknown"
    empty_slot_scores_zero: bool = False
    tie_value: float = 0.5


class LeagueConfig(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    league_id: str | None = None
    username: str | None = None
    my_roster_id: int | None = None
    display_tz: str = "Australia/Adelaide"
    schedule_tz: str = "America/New_York"
    history_seasons: list[int] = Field(default_factory=lambda: [2024, 2025])
    week_assignment: Literal["scheduled_et_date"] = "scheduled_et_date"
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    rules: RuleSwitches = Field(default_factory=RuleSwitches)

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.display_tz)


# --- model -----------------------------------------------------------------------
class AvailabilityConfig(BaseModel):
    status_prior: dict[str, float] = Field(default_factory=lambda: {
        "out": 0.0, "doubtful": 0.10, "questionable": 0.55, "probable": 0.93, "available": 0.99})
    late_scratch: float = 0.01
    questionable_late_share: float = 0.35
    out_return_hazard: float = 0.25
    ir_return_hazard: float = 0.03
    prior_uncertainty_logit_sd: float = 0.4
    healthy_floor_same_day: float = 0.97
    healthy_floor_next_day: float = 0.95


class ModelConfig(BaseModel):
    half_life_games: dict[str, float] = Field(default_factory=lambda: {
        "min": 5, "pts": 10, "reb": 10, "ast": 10, "tov": 10, "fg3m": 10, "stl": 20, "blk": 20})
    previous_season_weight: float = 0.6
    min_games_own_residuals: int = 15
    min_minutes_residual_game: float = 8.0
    availability: AvailabilityConfig = Field(default_factory=AvailabilityConfig)


# --- simulation ------------------------------------------------------------------
class ScenarioConfig(BaseModel):
    start: int = 2000
    max: int = 16000
    target_paired_se_fp: float = 0.2
    wall_clock_s: float = 20.0


class CandidateConfig(BaseModel):
    max_actions: int = 60
    top_lineups: int = 5
    screen_scenarios: int = 500
    finalists: int = 12
    keep_within_paired_se: float = 2.0


class BasePolicyConfig(BaseModel):
    assignment: Literal["fast", "exact"] = "fast"
    seat_by: Literal["tonight_mean", "one_player_gain"] = "tonight_mean"
    value_open_entries: bool = True
    defer_to_deadline: bool = True
    quantile_grid: int = 21


class CloseCallConfig(BaseModel):
    model_draws: int = 30
    scenarios_per_draw: int = 300
    lean_band: tuple[float, float] = (0.65, 0.90)
    either_is_fine_fp: float = 1.0


class WaitConfig(BaseModel):
    min_voi_fp: float = 0.5
    min_branch_scenarios: int = 500


class SimConfig(BaseModel):
    objective: Literal["points", "win"] = "points"
    scenarios: ScenarioConfig = Field(default_factory=ScenarioConfig)
    candidates: CandidateConfig = Field(default_factory=CandidateConfig)
    base_policy: BasePolicyConfig = Field(default_factory=BasePolicyConfig)
    planning_scenarios: int = 2000
    close_call: CloseCallConfig = Field(default_factory=CloseCallConfig)
    wait: WaitConfig = Field(default_factory=WaitConfig)
    within_week_rho: float = 0.0
    game_minutes: float = 150.0
    opponent_policies: list[str] = Field(default_factory=lambda: ["base", "never_lock", "lock_above_average"])
    seed: int | None = None


class AlertsConfig(BaseModel):
    urls: list[str] = Field(default_factory=list)
    offsets_minutes: list[int] = Field(default_factory=lambda: [120, 30, 10])
    heartbeat_url: str | None = None
    app_link: str | None = None


# --- loading ---------------------------------------------------------------------
def _load(path: Path, model: type[BaseModel]):
    if not path.exists():
        return model()
    with path.open() as f:
        return model(**(yaml.safe_load(f) or {}))


def load_league(paths: Paths | None = None) -> LeagueConfig:
    paths = paths or Paths()
    p = paths.configs / "league.yaml"
    return _load(p if p.exists() else paths.configs / "league.example.yaml", LeagueConfig)


def load_model(paths: Paths | None = None) -> ModelConfig:
    return _load((paths or Paths()).configs / "model.yaml", ModelConfig)


def load_sim(paths: Paths | None = None) -> SimConfig:
    return _load((paths or Paths()).configs / "sim.yaml", SimConfig)


def load_alerts(paths: Paths | None = None) -> AlertsConfig:
    paths = paths or Paths()
    p = paths.configs / "alerts.yaml"
    cfg = _load(p if p.exists() else paths.configs / "alerts.example.yaml", AlertsConfig)
    import os
    if os.environ.get("LOCKIN_ALERT_URLS"):
        cfg.urls = json.loads(os.environ["LOCKIN_ALERT_URLS"])
    if os.environ.get("LOCKIN_HEARTBEAT_URL"):
        cfg.heartbeat_url = os.environ["LOCKIN_HEARTBEAT_URL"]
    return cfg


def config_hash(*models: BaseModel) -> str:
    blob = json.dumps([m.model_dump(mode="json") for m in models], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def tzdata_version() -> str:
    try:
        import tzdata

        return getattr(tzdata, "IANA_VERSION", None) or tzdata.__version__
    except Exception:  # system zoneinfo
        return "system"


def code_version() -> str:
    """Git commit of the working tree (with '+dirty' if modified), else 'unknown'."""
    import subprocess

    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
                               text=True, timeout=5).stdout.strip()
        return (sha or "unknown") + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"
