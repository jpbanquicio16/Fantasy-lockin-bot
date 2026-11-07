"""League scoring settings, built from the Sleeper league's ``scoring_settings``.

Two rules Sleeper does not document are explicit settings with no silent default:

* ``threshold_bonus_mode`` - does a 50-point game earn both the 40+ and the 50+ bonus?
  Settled as "stack" by Sleeper's own scores of 2025-26 fifty-point games
  (tests/fixtures/sleeper_std_2025.json).
* ``tech_definition`` - which technical fouls count. "sleeper_tf" means the ``tf``
  count in Sleeper's own stats feed; on 30 Oct 2026 the league's matchup points
  matched that count (and ``ff``) at -2 each in every foul game checked.
"""

from __future__ import annotations

import math
import warnings
from typing import Literal, Mapping

from pydantic import BaseModel, Field, model_validator

# Sleeper stat key -> box-score column.
STAT_KEYS: dict[str, str] = {
    "pts": "pts", "reb": "reb", "oreb": "oreb", "dreb": "dreb", "ast": "ast", "stl": "stl",
    "blk": "blk", "to": "tov", "tpm": "fg3m", "tpa": "fg3a", "fgm": "fgm", "fga": "fga",
    "ftm": "ftm", "fta": "fta", "pf": "pf", "tf": "tf", "ff": "ff",
}
MISS_KEYS: dict[str, tuple[str, str]] = {
    "fgmi": ("fga", "fgm"), "ftmi": ("fta", "ftm"), "tpmi": ("fg3a", "fg3m"),
}
BONUS_KEYS = ("dd", "td", "bonus_pt_40p", "bonus_pt_50p", "bonus_ast_15p", "bonus_reb_20p")
DD_CATEGORIES = ("pts", "reb", "ast", "stl", "blk")
FOUL_COLUMNS = ("tf", "ff")
COMBO_PARTS = {"pts", "reb", "ast", "stl", "blk"}
NON_SCORING_KEYS = {"gp", "gs", "sp", "min", "plus_minus", "pts_std", "pts_std_dfs", "rank_std",
                    "pos_rank_std"}

SLEEPER_DEFAULT: dict[str, float] = {
    "pts": 0.5, "reb": 1.0, "ast": 1.0, "stl": 2.0, "blk": 2.0, "to": -1.0, "tpm": 0.5,
    "dd": 1.0, "td": 2.0, "bonus_pt_40p": 2.0, "bonus_pt_50p": 2.0, "tf": -2.0, "ff": -2.0,
}

BOX_COLUMNS = (
    "pts", "reb", "oreb", "dreb", "ast", "stl", "blk", "tov",
    "fg3m", "fg3a", "fgm", "fga", "ftm", "fta", "pf", "tf", "ff",
)


def _is_combo(key: str) -> bool:
    parts = key.split("_")
    return len(parts) >= 2 and all(p in COMBO_PARTS for p in parts)


class OpenSettingError(RuntimeError):
    """Exact scoring needs a setting that has not been settled by reconciliation."""


class ScoringSettings(BaseModel):
    weights: dict[str, float] = Field(default_factory=lambda: dict(SLEEPER_DEFAULT))
    threshold_bonus_mode: Literal["stack", "exclusive"] | None = None
    tech_definition: Literal["sleeper_tf"] | None = None
    unknown_keys: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _split_unknown(self) -> "ScoringSettings":
        known, unknown = {}, dict(self.unknown_keys)
        for k, v in self.weights.items():
            if v == 0:
                continue
            if k in STAT_KEYS or k in MISS_KEYS or k in BONUS_KEYS or _is_combo(k):
                known[k] = float(v)
            elif k not in NON_SCORING_KEYS:
                unknown[k] = float(v)
        self.weights, self.unknown_keys = known, unknown
        return self

    @classmethod
    def from_sleeper(cls, scoring_settings: Mapping[str, float] | None,
                     threshold_bonus_mode: str | None = None,
                     tech_definition: str | None = None) -> "ScoringSettings":
        s = cls(weights=dict(scoring_settings or SLEEPER_DEFAULT),
                threshold_bonus_mode=threshold_bonus_mode, tech_definition=tech_definition)
        if s.unknown_keys:
            warnings.warn(f"Unrecognised Sleeper scoring keys ignored: {sorted(s.unknown_keys)}",
                          stacklevel=2)
        return s

    def without_fouls(self) -> "ScoringSettings":
        """Sleeper's ``pts_std`` ignores fouls; reconciliation compares against this."""
        w = {k: v for k, v in self.weights.items() if k not in FOUL_COLUMNS}
        return self.model_copy(update={"weights": w})

    @property
    def units_per_point(self) -> int:
        """Smallest integer resolution that represents every weight exactly (2 = half-points)."""
        for res in (1, 2, 10, 100):
            if all(math.isclose(w * res, round(w * res), abs_tol=1e-9) for w in self.weights.values()):
                return res
        raise ValueError(f"Scoring weights need finer than 0.01 resolution: {self.weights}")

    def unit_weights(self) -> dict[str, int]:
        res = self.units_per_point
        return {k: int(round(v * res)) for k, v in self.weights.items()}

    @property
    def scores_fouls(self) -> bool:
        return any(self.weights.get(k) for k in FOUL_COLUMNS)

    @property
    def needs_bonus_mode(self) -> bool:
        return bool(self.weights.get("bonus_pt_40p") and self.weights.get("bonus_pt_50p"))

    def open_settings(self) -> list[str]:
        out = []
        if self.needs_bonus_mode and self.threshold_bonus_mode is None:
            out.append("threshold_bonus_mode")
        if self.scores_fouls and self.tech_definition is None:
            out.append("tech_definition")
        return out

    def require_exact(self) -> None:
        missing = self.open_settings()
        if missing:
            raise OpenSettingError(
                f"exact scoring needs {missing}; run `lockin reconcile` or set them in configs/league.yaml"
            )

    def stacking(self, default: bool = True) -> bool:
        if self.threshold_bonus_mode is None:
            return default
        return self.threshold_bonus_mode == "stack"
