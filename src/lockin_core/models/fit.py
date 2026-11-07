"""Fit the projection model's parameters on past seasons (``lockin fit``).

Everything here is estimated from complete past seasons only, so it can never leak the
current season into a projection:

* **Cohort rate priors** per position group x minutes band, and **shrinkage strength**
  by empirical Bayes (gamma-Poisson with overdispersion): the posterior per-minute rate
  is (X + K mu) / (M + K) with K = phi mu / tau^2, where phi is the stat's dispersion
  (points come in twos and threes, so phi > 1) and tau^2 the spread of true rates
  between players.
* **Availability**: P(plays the game k team games ahead) given whether he played his
  last listed game and his minutes band - from the explicit did-not-play rows.
* **Minutes**: spread around the recent average by band; the ramp after 3+ missed games.
* **Residual vectors**: each real game's per-minute production relative to that
  player-season's rates, pooled by cohort, so simulated box scores keep real
  correlations (points with threes, rebounds with blocks) and real tails.
"""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import polars as pl
from pydantic import BaseModel

from lockin_core.config import ModelConfig, Paths, now_utc
from lockin_core.data import tables as T
from lockin_core.models.data import (
    AVAIL_BANDS,
    GROUPS,
    MINUTE_BANDS,
    RATE_STATS,
    band,
    ew_minutes_asof,
    history,
    with_groups,
)
from lockin_core.scoring.engine import score_frame
from lockin_core.scoring.settings import ScoringSettings

K_AHEAD = 12


class Params(BaseModel):
    version: str
    fitted_at: datetime
    seasons: list[int]
    rate_prior: dict[str, list[dict[str, float]]]  # group -> band -> stat -> per-minute rate
    tau2: dict[str, dict[str, float]]  # group -> stat -> between-player variance of rates
    phi: dict[str, float]  # stat -> dispersion
    avail_played: list[list[float]]  # band -> k-1 -> P(play) after a played game
    avail_dnp: list[list[float]]  # band -> k-1 -> P(play) after a did-not-play
    b2b_factor: float
    minutes_sd: list[float]  # by minutes band
    ramp: list[float]  # minutes multiplier for games 1..3 after 3+ missed
    fp_cv: list[float]  # FP coefficient of variation per game, by minutes band
    foul_mean: float  # technical + flagrant fouls per game, league
    foul_k: float  # shrinkage strength in games
    ot_rate: float
    residual_keys: list[str]
    n_player_seasons: int
    # isotonic map raw play rate -> observed P(plays next listed game): (x knots, y knots)
    play_calibration: tuple[list[float], list[float]] = ([0.0, 1.0], [0.0, 1.0])

    def calibrate_play(self, raw: float) -> float:
        x, y = self.play_calibration
        return float(np.interp(raw, x, y))

    def shrink_k(self, group: str, b: int, stat: str) -> float:
        mu = self.rate_prior[group][b][stat]
        return self.phi[stat] * mu / max(self.tau2[group][stat], 1e-10)


def player_seasons(df: pl.DataFrame) -> pl.DataFrame:
    p = df.filter(pl.col("played") == "played")
    return p.group_by("player_id", "season", "group").agg(
        pl.len().alias("games"), pl.col("min").sum().alias("M"),
        *[pl.col(c).sum().alias(f"X_{c}") for c in RATE_STATS],
        (pl.col("tf") + pl.col("ff")).sum().alias("fouls"),
    ).with_columns((pl.col("M") / pl.col("games")).alias("mpg")).with_columns(
        pl.col("mpg").map_elements(band, return_dtype=pl.Int64).alias("band"))


def fit_rates(ps: pl.DataFrame, df: pl.DataFrame) -> tuple[dict, dict, dict]:
    ps = ps.filter(pl.col("M") >= 100)
    prior: dict[str, list[dict[str, float]]] = {}
    for g in GROUPS:
        prior[g] = []
        for b in range(len(MINUTE_BANDS)):
            sub = ps.filter((pl.col("group") == g) & (pl.col("band") == b))
            if sub["M"].sum() < 500:
                sub = ps.filter(pl.col("group") == g)
            prior[g].append({c: float(sub[f"X_{c}"].sum() / sub["M"].sum()) for c in RATE_STATS})
    # dispersion: Pearson ratio of sums within player-seasons with enough games
    big = ps.filter((pl.col("games") >= 20) & (pl.col("M") >= 300))
    played = df.filter(pl.col("played") == "played").join(
        big.select("player_id", "season", "M", *[f"X_{c}" for c in RATE_STATS]),
        on=["player_id", "season"])
    phi = {}
    for c in RATE_STATS:
        e = played["min"] * played[f"X_{c}"] / played["M"]
        phi[c] = float(max(1.0, ((played[c] - e) ** 2).sum() / max(e.sum(), 1e-9)))
    tau2: dict[str, dict[str, float]] = {}
    big5 = ps.filter(pl.col("M") >= 500)
    for g in GROUPS:
        sub = big5.filter(pl.col("group") == g)
        tau2[g] = {}
        for c in RATE_STATS:
            mu = np.array([prior[g][b][c] for b in sub["band"].to_list()])
            r = (sub[f"X_{c}"] / sub["M"]).to_numpy()
            m = sub["M"].to_numpy()
            w = m / m.sum()
            between = float(np.sum(w * (r - mu) ** 2) - np.sum(w * phi[c] * mu / m))
            tau2[g][c] = max(between, 0.05 * float(np.mean(mu)) ** 2 / 100)
    return prior, phi, tau2


def fit_availability(df: pl.DataFrame) -> tuple[list[list[float]], list[list[float]], float]:
    d = df.with_columns(
        pl.col("ew_min").fill_null(0.0).map_elements(lambda m: band(m, AVAIL_BANDS),
                                                     return_dtype=pl.Int64).alias("ab"),
        (pl.col("played") == "played").alias("p"),
    )
    over = ["player_id", "season"]
    cols = []
    for k in range(1, K_AHEAD + 1):
        cols += [pl.col("p").shift(-k).over(over).alias(f"p{k}"),
                 (pl.col("team").shift(-k).over(over) == pl.col("team")).alias(f"same{k}")]
    d = d.with_columns(cols).with_columns(
        (pl.col("date").shift(-1).over(over) - pl.col("date")).dt.total_days().alias("gap1"))
    d = d.filter(pl.col("ew_min").is_not_null())
    tables = {}
    for state in (True, False):
        rows = []
        for b in range(len(AVAIL_BANDS)):
            sub = d.filter((pl.col("p") == state) & (pl.col("ab") == b))
            row = []
            for k in range(1, K_AHEAD + 1):
                s = sub.filter(pl.col(f"same{k}").fill_null(False))
                n, x = s.height, int(s[f"p{k}"].sum() or 0)
                prior = 0.9 if state else 0.5
                row.append((x + 20 * prior) / (n + 20))
            rows.append(row)
        tables[state] = rows
    rot = d.filter(pl.col("p") & (pl.col("ab") >= 2) & pl.col("same1").fill_null(False))
    b2b = rot.filter(pl.col("gap1") == 1)["p1"].mean()
    rest = rot.filter(pl.col("gap1") > 1)["p1"].mean()
    factor = float(b2b / rest) if b2b and rest else 1.0
    return tables[True], tables[False], min(1.0, factor)


def fit_minutes(df: pl.DataFrame) -> tuple[list[float], list[float]]:
    p = df.filter((pl.col("played") == "played") & pl.col("ew_min").is_not_null())
    sds = []
    for b in range(len(MINUTE_BANDS)):
        lo = MINUTE_BANDS[b]
        hi = MINUTE_BANDS[b + 1] if b + 1 < len(MINUTE_BANDS) else 99
        r = p.filter((pl.col("ew_min") >= lo) & (pl.col("ew_min") < hi))
        sds.append(float((r["min"] - r["ew_min"]).std() or 5.0))
    # ramp: k-th game back after a run of 3+ listed did-not-play games
    ratios: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for (_pid, _s), g in df.group_by("player_id", "season", maintain_order=True):
        played = (g["played"] == "played").to_list()
        mins, ew = g["min"].to_list(), g["ew_min"].to_list()
        run, since = 0, None
        for i, pl_ in enumerate(played):
            if not pl_:
                run += 1
                since = None
                continue
            if run >= 3:
                since = 1
            elif since is not None:
                since += 1
            run = 0
            if since in ratios and ew[i] and ew[i] >= 15:
                ratios[since].append(mins[i] / ew[i])
            if since is not None and since >= 3:
                since = None
    ramp = [float(np.median(ratios[k])) if len(ratios[k]) >= 30 else 1.0 for k in (1, 2, 3)]
    return sds, ramp


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: the nondecreasing fit to ``y`` with weights ``w``."""
    vals, wts, sizes = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi), wts.append(wi), sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wt, sz = wts[-2] + wts[-1], sizes[-2] + sizes[-1]
            vals[-2:], wts[-2:], sizes[-2:] = [v], [wt], [sz]
    return np.repeat(vals, sizes)


def fit_play_calibration(df: pl.DataFrame, avail_played: list[list[float]], bins: int = 25
                         ) -> tuple[list[float], list[float]]:
    """Raw play rate as it stood before each listed game vs whether he played it."""
    from lockin_core.models.project import play_rate_seq

    raws, ys = [], []
    for (_pid,), g in df.group_by("player_id", maintain_order=True):
        seq = (g["played"] == "played").to_list()
        seasons, ew = g["season"].to_list(), g["ew_min"].to_list()
        for i in range(1, len(seq)):
            prior = avail_played[band(ew[i] or 0.0, AVAIL_BANDS)][0]
            prev = [s < seasons[i] for s in seasons[:i]]
            raws.append(play_rate_seq(seq[max(0, i - 40):i], prev[max(0, i - 40):i], prior))
            ys.append(seq[i])
    r, y = np.asarray(raws), np.asarray(ys, float)
    order = np.argsort(r)
    r, y = r[order], y[order]
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, len(edges) - 2)
    xs, ms, ns = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            xs.append(float(r[m].mean())), ms.append(float(y[m].mean())), ns.append(int(m.sum()))
    fitted = pava(np.array(ms), np.array(ns, float))
    return [0.0, *xs, 1.0], [float(fitted[0]), *map(float, fitted), float(fitted[-1])]


def residual_library(df: pl.DataFrame, ps: pl.DataFrame, min_minutes: float) -> dict[str, np.ndarray]:
    big = ps.filter(pl.col("M") >= 300)
    p = df.filter((pl.col("played") == "played") & (pl.col("min") >= min_minutes)).join(
        big.select("player_id", "season", "band", "M", *[f"X_{c}" for c in RATE_STATS]),
        on=["player_id", "season"])
    u = []
    for c in RATE_STATS:
        rate = p[f"X_{c}"] / p["M"]
        u.append(np.where(rate.to_numpy() > 0, (p[c] / p["min"]).to_numpy() / np.maximum(rate.to_numpy(), 1e-12), 1.0))
    U = np.column_stack(u).astype(np.float32)
    groups, bands, mins = p["group"].to_numpy(), p["band"].to_numpy(), p["min"].to_numpy()
    lib = {}
    for g in GROUPS:
        for b in range(len(MINUTE_BANDS)):
            m = (groups == g) & (bands == b)
            if m.sum() < 200:
                m = groups == g
            arr = U[m]
            w = mins[m] / mins[m].sum()
            lib[f"{g}{b}"] = (arr / np.maximum((w[:, None] * arr).sum(0), 1e-6)).astype(np.float32)
    return lib


def fit(paths: Paths, seasons: list[int], scoring: ScoringSettings, cfg: ModelConfig,
        log=print, persist: bool = True, return_lib: bool = False):
    df = history(paths, seasons)
    df = with_groups(df, T.read(paths, "players"))
    df = ew_minutes_asof(df, cfg.half_life_games["min"])
    ps = player_seasons(df)
    prior, phi, tau2 = fit_rates(ps, df)
    avail_p, avail_d, b2b = fit_availability(df)
    calib = fit_play_calibration(df, avail_p)
    msd, ramp = fit_minutes(df)
    lib = residual_library(df, ps, cfg.min_minutes_residual_game)

    played = df.filter(pl.col("played") == "played")
    fp = score_frame(played, scoring)
    ps_fp = fp.group_by("player_id", "season").agg(
        pl.col("fp").mean().alias("m"), pl.col("fp").std().alias("s"), pl.len().alias("n"),
        pl.col("min").mean().alias("mpg")).filter(pl.col("n") >= 20)
    fp_cv = []
    for b in range(len(MINUTE_BANDS)):
        lo = MINUTE_BANDS[b]
        hi = MINUTE_BANDS[b + 1] if b + 1 < len(MINUTE_BANDS) else 99
        s = ps_fp.filter((pl.col("mpg") >= lo) & (pl.col("mpg") < hi))
        fp_cv.append(float((s["s"] / s["m"]).median() or 0.45))
    fouls = ps.filter(pl.col("games") >= 20).with_columns((pl.col("fouls") / pl.col("games")).alias("r"))
    fm = float(fouls["fouls"].sum() / fouls["games"].sum())
    fvar = float(fouls["r"].var() or 0) - float((fm / fouls["games"]).mean())
    foul_k = fm / fvar if fvar > 0 else 200.0
    team_min = played.group_by("game_id", "team").agg(pl.col("min").sum())
    ot = float((team_min["min"] > 250).mean())

    version = f"p_{now_utc():%Y%m%dT%H%M%S}"
    params = Params(
        version=version, fitted_at=now_utc(), seasons=seasons, rate_prior=prior, tau2=tau2,
        phi=phi, avail_played=avail_p, avail_dnp=avail_d, b2b_factor=b2b, minutes_sd=msd,
        ramp=ramp, fp_cv=fp_cv, foul_mean=fm, foul_k=foul_k, ot_rate=ot,
        residual_keys=sorted(lib), n_player_seasons=ps.height, play_calibration=calib)
    if persist:
        save(paths, params, lib)
    log(f"fitted on {seasons}: {ps.height} player-seasons, {played.height} games; "
        f"play calibration {[round(v, 2) for v in calib[1][::5]]}, phi pts {phi['pts']:.2f}, "
        f"ramp {[round(r, 2) for r in ramp]}, b2b {b2b:.3f}, OT {ot:.3f}")
    return (params, lib) if return_lib else params


def save(paths: Paths, params: Params, lib: dict[str, np.ndarray], activate: bool = True) -> None:
    """Keep every model version; activation is an atomic pointer change."""
    import uuid
    d = paths.artifacts / "models" / params.version
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.json").write_text(params.model_dump_json(indent=1))
    np.savez_compressed(d / "residuals.npz", **lib)
    if activate:
        activate_version(paths, params.version)


def activate_version(paths: Paths, version: str) -> None:
    if "/" in version or "\\" in version or version in (".", ".."):
        raise ValueError("Invalid model version")
    d = paths.artifacts / "models" / version
    if not (d / "params.json").exists() or not (d / "residuals.npz").exists():
        raise FileNotFoundError(f"Model version {version} is incomplete or missing")
    import uuid
    tmp = paths.artifacts / ("active_model." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(version)
    tmp.replace(paths.artifacts / "active_model")


def load(paths: Paths) -> tuple[Params, dict[str, np.ndarray]]:
    d = paths.artifacts
    pointer = d / "active_model"
    if pointer.exists():
        version = pointer.read_text().strip()
        if "/" in version or "\\" in version or version in (".", ".."):
            raise ValueError("Invalid active model pointer")
        d = d / "models" / version
    if not (d / "params.json").exists():
        raise FileNotFoundError("no fitted parameters; run `lockin fit`")
    params = Params(**json.loads((d / "params.json").read_text()))
    with np.load(d / "residuals.npz", allow_pickle=False) as z:
        lib = {k: z[k] for k in z.files}
    return params, lib
