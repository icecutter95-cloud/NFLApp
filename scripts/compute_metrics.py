"""
Phase 1 — Step 1: Pull nfl_data_py PBP data and compute rolling team efficiency metrics.

Run once per season during the offseason build, or weekly in-season to update the
most recent week. Output is one row per (team, season, week) representing metrics
*going into* that week (i.e., the current game is excluded from its own rolling window).

Usage:
    python compute_metrics.py              # all historical seasons
    python compute_metrics.py 2024         # single season
    python compute_metrics.py --force      # recompute all seasons (ignores cache)
    python compute_metrics.py 2024 --force # recompute single season
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import nfl_data_py as nfl
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import DATA_DIR, ALL_HISTORICAL_SEASONS


# ---------------------------------------------------------------------------
# Per-game metric computation
# ---------------------------------------------------------------------------

# Garbage-time filter. Plays run at win probabilities outside this band are
# systematically distorted -- prevent defense, clock-killing runs, backups in --
# and describe game state rather than team quality. Restricting to competitive
# situations is standard practice for efficiency metrics.
# `wp` is pure in-game win probability (no betting line as an input), so this
# introduces no dependence on the market.
# DEFAULT OFF -- measured, not assumed. Filtering to competitive plays is
# standard advice, but on held-out 2023-2025 it made margin prediction WORSE
# (corr 0.360 -> 0.337, MAE 10.42 -> 10.48). It discards ~23% of plays, and an
# NFL rolling window is only 4-8 games; the added variance in each game's EPA
# estimate outweighs the bias it removes. Set METRICS_GARBAGE_FILTER=1 to
# re-test (e.g. with a wider band via METRICS_WP_BAND).
GARBAGE_TIME_ENABLED = os.getenv("METRICS_GARBAGE_FILTER", "0") == "1"
_WP_BAND = float(os.getenv("METRICS_WP_BAND", "0.10"))
GARBAGE_TIME_WP_LOW = _WP_BAND
GARBAGE_TIME_WP_HIGH = 1.0 - _WP_BAND


def _plays(pbp: pd.DataFrame, competitive_only: bool = True) -> pd.DataFrame:
    """Scoreable pass/run plays with valid EPA, optionally competitive-only.

    Plays with a missing win probability are KEPT: absent evidence that a snap
    was garbage time, dropping it would throw away real data.
    """
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    if not (competitive_only and GARBAGE_TIME_ENABLED) or "wp" not in p.columns:
        return p

    wp = p["wp"]
    keep = wp.isna() | ((wp >= GARBAGE_TIME_WP_LOW) & (wp <= GARBAGE_TIME_WP_HIGH))
    return p[keep].copy()


def _matchups(pbp: pd.DataFrame) -> pd.DataFrame:
    """(game_id, team, opponent) for every team-game -- needed to adjust a
    team's efficiency for the quality of the units it actually faced."""
    m = (pbp[["game_id", "posteam", "defteam"]]
         .dropna()
         .drop_duplicates()
         .rename(columns={"posteam": "team", "defteam": "opponent"}))
    return m[m["team"] != m["opponent"]]


def compute_epa_metrics(pbp: pd.DataFrame) -> pd.DataFrame:
    """EPA/play (off + def), pass EPA, rush EPA, success rate per team per game."""
    p = _plays(pbp)

    # Offensive EPA — all plays
    off = (p.groupby(["game_id", "posteam", "season", "week"])
           .agg(epa_sum=("epa", "sum"), n=("epa", "count"), success=("success", "sum"))
           .reset_index()
           .rename(columns={"posteam": "team"}))
    off["epa_per_play_off"] = off["epa_sum"] / off["n"]
    off["success_rate_off"] = off["success"] / off["n"]

    # Pass EPA
    pass_p = p[p["play_type"] == "pass"]
    pass_off = (pass_p.groupby(["game_id", "posteam", "season", "week"])
                .agg(epa_pass_sum=("epa", "sum"), n_pass=("epa", "count"))
                .reset_index()
                .rename(columns={"posteam": "team"}))
    pass_off["epa_pass_off"] = pass_off["epa_pass_sum"] / pass_off["n_pass"]

    # Rush EPA
    run_p = p[p["play_type"] == "run"]
    rush_off = (run_p.groupby(["game_id", "posteam", "season", "week"])
                .agg(epa_rush_sum=("epa", "sum"), n_rush=("epa", "count"))
                .reset_index()
                .rename(columns={"posteam": "team"}))
    rush_off["epa_rush_off"] = rush_off["epa_rush_sum"] / rush_off["n_rush"]

    # Defensive EPA (what was allowed — higher = worse defense)
    def_epa = (p.groupby(["game_id", "defteam", "season", "week"])
               .agg(epa_def_sum=("epa", "sum"), n_def=("epa", "count"), success_def=("success", "sum"))
               .reset_index()
               .rename(columns={"defteam": "team"}))
    def_epa["epa_per_play_def"] = def_epa["epa_def_sum"] / def_epa["n_def"]
    def_epa["success_rate_def"] = def_epa["success_def"] / def_epa["n_def"]

    result = (off[["game_id", "team", "season", "week", "epa_per_play_off", "success_rate_off"]]
              .merge(pass_off[["game_id", "team", "season", "week", "epa_pass_off"]],
                     on=["game_id", "team", "season", "week"], how="left")
              .merge(rush_off[["game_id", "team", "season", "week", "epa_rush_off"]],
                     on=["game_id", "team", "season", "week"], how="left")
              .merge(def_epa[["game_id", "team", "season", "week", "epa_per_play_def", "success_rate_def"]],
                     on=["game_id", "team", "season", "week"], how="left"))
    return result


def compute_cpoe(pbp: pd.DataFrame) -> pd.DataFrame:
    """Completion % over expected per team per game (passing plays only, no sacks)."""
    pass_p = pbp[
        (pbp["play_type"] == "pass") &
        pbp["cpoe"].notna() &
        (pbp.get("sack", pd.Series(0, index=pbp.index)) == 0)
    ]
    cpoe = (pass_p.groupby(["game_id", "posteam", "season", "week"])
            .agg(cpoe=("cpoe", "mean"))
            .reset_index()
            .rename(columns={"posteam": "team"}))
    return cpoe


def compute_third_down(pbp: pd.DataFrame) -> pd.DataFrame:
    """Third-down conversion rate (off) and stop rate (def) per team per game."""
    td = pbp[pbp["down"] == 3].copy()
    if "first_down" not in td.columns:
        td["first_down"] = (td["first_down_rush"].fillna(0) + td["first_down_pass"].fillna(0)).clip(upper=1)

    off = (td.groupby(["game_id", "posteam", "season", "week"])
           .agg(attempts=("first_down", "count"), conv=("first_down", "sum"))
           .reset_index()
           .rename(columns={"posteam": "team"}))
    off["third_down_conv_off"] = off["conv"] / off["attempts"].clip(lower=1)

    def_ = (td.groupby(["game_id", "defteam", "season", "week"])
            .agg(attempts_d=("first_down", "count"), conv_d=("first_down", "sum"))
            .reset_index()
            .rename(columns={"defteam": "team"}))
    def_["third_down_stop_def"] = 1 - (def_["conv_d"] / def_["attempts_d"].clip(lower=1))

    return (off[["game_id", "team", "season", "week", "third_down_conv_off"]]
            .merge(def_[["game_id", "team", "season", "week", "third_down_stop_def"]],
                   on=["game_id", "team", "season", "week"], how="outer"))


def compute_redzone(pbp: pd.DataFrame) -> pd.DataFrame:
    """Red zone TD% per team per game (plays inside opponent 20)."""
    rz = pbp[pbp["yardline_100"] <= 20].copy()
    result = (rz.groupby(["game_id", "posteam", "season", "week"])
              .agg(rz_plays=("play_type", "count"), rz_tds=("touchdown", "sum"))
              .reset_index()
              .rename(columns={"posteam": "team"}))
    result["rz_td_pct_off"] = result["rz_tds"] / result["rz_plays"].clip(lower=1)
    return result[["game_id", "team", "season", "week", "rz_td_pct_off"]]


def compute_pace(pbp: pd.DataFrame) -> pd.DataFrame:
    """Plays per game (pace) per team."""
    pace = (pbp[pbp["play_type"].isin(["pass", "run"])]
            .groupby(["game_id", "posteam", "season", "week"])
            .agg(plays_per_game=("play_type", "count"))
            .reset_index()
            .rename(columns={"posteam": "team"}))
    return pace


def compute_pressure(pbp: pd.DataFrame) -> pd.DataFrame:
    """Sack rate and QB-hit rate, both generated (defense) and allowed (offense).

    Trench play is a strong, relatively stable team signal we previously had no
    coverage of at all. Rates are per DROPBACK (qb_dropback), not per play, so
    they aren't confounded by how pass-heavy a team is.
    """
    for c in ["qb_dropback", "sack", "qb_hit"]:
        if c not in pbp.columns:
            return pd.DataFrame()

    db = pbp[pbp["qb_dropback"] == 1].copy()
    db["sack"] = db["sack"].fillna(0)
    db["qb_hit"] = db["qb_hit"].fillna(0)

    off = (db.groupby(["game_id", "posteam", "season", "week"])
           .agg(dropbacks=("qb_dropback", "count"),
                sacks_allowed=("sack", "sum"),
                hits_allowed=("qb_hit", "sum"))
           .reset_index()
           .rename(columns={"posteam": "team"}))
    off["sack_rate_off"] = off["sacks_allowed"] / off["dropbacks"].clip(lower=1)
    off["qb_hit_rate_off"] = off["hits_allowed"] / off["dropbacks"].clip(lower=1)

    dfn = (db.groupby(["game_id", "defteam", "season", "week"])
           .agg(dropbacks_faced=("qb_dropback", "count"),
                sacks_made=("sack", "sum"),
                hits_made=("qb_hit", "sum"))
           .reset_index()
           .rename(columns={"defteam": "team"}))
    dfn["sack_rate_def"] = dfn["sacks_made"] / dfn["dropbacks_faced"].clip(lower=1)
    dfn["qb_hit_rate_def"] = dfn["hits_made"] / dfn["dropbacks_faced"].clip(lower=1)

    return (off[["game_id", "team", "season", "week", "sack_rate_off", "qb_hit_rate_off"]]
            .merge(dfn[["game_id", "team", "season", "week", "sack_rate_def", "qb_hit_rate_def"]],
                   on=["game_id", "team", "season", "week"], how="outer"))


def compute_expected_turnovers(pbp: pd.DataFrame) -> pd.DataFrame:
    """Turnover margin with fumble-recovery LUCK stripped out.

    How often a team fumbles (and how often its defense forces fumbles) is a
    repeatable team trait. WHO FALLS ON THE BALL is close to a coin flip -- in
    2024, offenses retained 57% of 663 fumbles. Raw turnover margin therefore
    mixes a real signal with a large amount of noise, and `turnover_margin` has
    been a model feature all along.

    So instead of counting fumbles actually lost, we count fumbles OCCURRED and
    multiply by the league-wide loss rate:

        expected_committed = own_fumbles   * league_loss_rate + interceptions_thrown
        expected_forced    = forced_fumbles * league_loss_rate + interceptions_made

    Interceptions are left as-is: they carry far more skill (pressure, coverage,
    ball skills) than fumble recoveries, though they are not noise-free either.

    Also emits `turnover_luck` = actual - expected, which is itself a
    regression signal: a team well above its expected margin has been getting
    bounces it should not count on repeating.

    The league loss rate is a single leaguewide constant (~0.43) computed over
    hundreds of fumbles, so deriving it within-season carries no meaningful
    team-level leakage.
    """
    needed = ["fumble", "fumble_lost", "interception", "posteam", "defteam"]
    if any(c not in pbp.columns for c in needed):
        return pd.DataFrame()

    fum = pbp[pbp["fumble"] == 1]
    if fum.empty:
        return pd.DataFrame()
    league_loss_rate = float(fum["fumble_lost"].sum() / len(fum))

    keys = ["game_id", "season", "week"]

    off_fum = (fum.groupby(keys + ["posteam"]).size()
               .reset_index(name="own_fumbles").rename(columns={"posteam": "team"}))
    def_fum = (fum.groupby(keys + ["defteam"]).size()
               .reset_index(name="forced_fumbles").rename(columns={"defteam": "team"}))

    ints = pbp[pbp["interception"] == 1]
    int_thrown = (ints.groupby(keys + ["posteam"]).size()
                  .reset_index(name="ints_thrown").rename(columns={"posteam": "team"}))
    int_made = (ints.groupby(keys + ["defteam"]).size()
                .reset_index(name="ints_made").rename(columns={"defteam": "team"}))

    out = off_fum
    for frame in (def_fum, int_thrown, int_made):
        out = out.merge(frame, on=keys + ["team"], how="outer")
    out = out.fillna(0)

    out["exp_to_committed"] = out["own_fumbles"] * league_loss_rate + out["ints_thrown"]
    out["exp_to_forced"] = out["forced_fumbles"] * league_loss_rate + out["ints_made"]
    out["expected_turnover_margin"] = out["exp_to_forced"] - out["exp_to_committed"]
    return out[keys + ["team", "expected_turnover_margin"]]


def compute_turnovers(pbp: pd.DataFrame) -> pd.DataFrame:
    """Turnover margin per team per game."""
    cols = ["game_id", "posteam", "defteam", "season", "week"]
    for c in ["fumble_lost", "interception"]:
        if c not in pbp.columns:
            pbp[c] = 0

    off_to = (pbp.groupby(["game_id", "posteam", "season", "week"])
              .agg(fumbles_lost=("fumble_lost", "sum"), ints_thrown=("interception", "sum"))
              .reset_index()
              .rename(columns={"posteam": "team"}))
    off_to["turnovers_committed"] = off_to["fumbles_lost"] + off_to["ints_thrown"]

    def_to = (pbp.groupby(["game_id", "defteam", "season", "week"])
              .agg(fumbles_forced=("fumble_lost", "sum"), ints_forced=("interception", "sum"))
              .reset_index()
              .rename(columns={"defteam": "team"}))
    def_to["turnovers_forced"] = def_to["fumbles_forced"] + def_to["ints_forced"]

    merged = (off_to[["game_id", "team", "season", "week", "turnovers_committed"]]
              .merge(def_to[["game_id", "team", "season", "week", "turnovers_forced"]],
                     on=["game_id", "team", "season", "week"], how="outer")
              .fillna(0))
    merged["turnover_margin"] = merged["turnovers_forced"] - merged["turnovers_committed"]
    return merged[["game_id", "team", "season", "week", "turnover_margin"]]


def compute_ngs(season: int) -> pd.DataFrame:
    """Next Gen Stats aggregated from player-week to team-week.

    These are tracking-derived signals EPA cannot express: how long the QB
    holds the ball, how much separation receivers create, how many yards a back
    gains above what the blocking/box justified. Player rows are volume-weighted
    (attempts / targets) so a backup's small sample doesn't swing the team mean.

    NGS covers 2018+ — unlike FTN charting, which starts in 2022 and therefore
    can't be used with a 2018-2022 training window.
    """
    cache = DATA_DIR / f"ngs_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    def agg(kind, weight_col, spec):
        try:
            d = nfl.import_ngs_data(kind, [season])
        except Exception as exc:
            print(f"  WARNING: NGS {kind} unavailable for {season}: {exc}")
            return pd.DataFrame()
        d = d[(d["week"] > 0) & (d.get("season_type", "REG") == "REG")].copy()
        if d.empty:
            return pd.DataFrame()
        d = d.rename(columns={"team_abbr": "team"})
        d[weight_col] = pd.to_numeric(d[weight_col], errors="coerce").fillna(0)
        out = []
        for (team, week), g in d.groupby(["team", "week"]):
            w = g[weight_col]
            row = {"team": team, "season": season, "week": int(week)}
            for src, dest in spec.items():
                if src not in g.columns:
                    continue
                v = pd.to_numeric(g[src], errors="coerce")
                m = v.notna() & (w > 0)
                row[dest] = float(np.average(v[m], weights=w[m])) if m.any() else np.nan
            out.append(row)
        return pd.DataFrame(out)

    p = agg("passing", "attempts", {
        "avg_time_to_throw": "ngs_time_to_throw",
        "aggressiveness": "ngs_aggressiveness",
        "avg_air_yards_differential": "ngs_air_yards_diff",
    })
    r = agg("rushing", "rush_attempts", {
        "rush_yards_over_expected_per_att": "ngs_ryoe_per_att",
        "percent_attempts_gte_eight_defenders": "ngs_eight_def_pct",
    })
    c = agg("receiving", "targets", {
        "avg_separation": "ngs_separation",
        "avg_yac_above_expectation": "ngs_yac_oe",
    })

    frames = [f for f in (p, r, c) if not f.empty]
    if not frames:
        return pd.DataFrame()
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["team", "season", "week"], how="outer")
    merged.to_parquet(cache, index=False)
    print(f"  NGS: {len(merged)} team-week rows cached")
    return merged


def compute_scoring_from_schedule(season: int) -> pd.DataFrame:
    """
    Points scored and allowed per team per game, pulled from schedule final scores.
    These are the most direct predictors of combined scoring (totals) that EPA doesn't capture:
    a team can have high EPA/play but low scoring volume (e.g., slow pace, missed FGs).
    """
    try:
        sched = nfl.import_schedules([season])
        needed = ["game_id", "home_team", "away_team", "home_score", "away_score", "season", "week"]
        sched = sched[[c for c in needed if c in sched.columns]].dropna(subset=["home_score", "away_score"])

        home_rows = sched[["game_id", "home_team", "home_score", "away_score", "season", "week"]].copy()
        home_rows.rename(columns={"home_team": "team"}, inplace=True)
        home_rows["points_scored_off"] = home_rows["home_score"].astype(float)
        home_rows["points_allowed_def"] = home_rows["away_score"].astype(float)

        away_rows = sched[["game_id", "away_team", "away_score", "home_score", "season", "week"]].copy()
        away_rows.rename(columns={"away_team": "team"}, inplace=True)
        away_rows["points_scored_off"] = away_rows["away_score"].astype(float)
        away_rows["points_allowed_def"] = away_rows["home_score"].astype(float)

        result = pd.concat([
            home_rows[["game_id", "team", "season", "week", "points_scored_off", "points_allowed_def"]],
            away_rows[["game_id", "team", "season", "week", "points_scored_off", "points_allowed_def"]],
        ], ignore_index=True)
        print(f"  Scoring metrics: {len(result)} team-game rows")
        return result
    except Exception as e:
        print(f"  WARNING: Could not compute scoring metrics: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Rolling window aggregation
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "epa_per_play_off", "epa_per_play_def",
    "epa_pass_off", "epa_rush_off",
    "success_rate_off", "success_rate_def",
    "cpoe",
    "third_down_conv_off", "third_down_stop_def",
    "rz_td_pct_off",
    "plays_per_game",
    "turnover_margin",
    "expected_turnover_margin", "turnover_luck",
    # Trench play (per dropback) — previously no coverage at all
    "sack_rate_off", "sack_rate_def",
    "qb_hit_rate_off", "qb_hit_rate_def",
    # Next Gen Stats (tracking-derived, offense only)
    "ngs_time_to_throw", "ngs_aggressiveness", "ngs_air_yards_diff",
    "ngs_ryoe_per_att", "ngs_eight_def_pct",
    "ngs_separation", "ngs_yac_oe",
    # Scoring volume — critical for totals model
    "points_scored_off", "points_allowed_def",
]

# Opponent-adjusted counterparts, produced inside build_rolling_metrics.
ADJUSTED_COLS = [
    "epa_per_play_off_adj", "epa_per_play_def_adj",
    "success_rate_off_adj", "success_rate_def_adj",
]

# Recency decay inside a rolling window. A flat L4 average treats a game four
# weeks ago exactly like last week; with a 2-game half-life the most recent
# game carries ~4x the weight of the 4th-most-recent.
# DEFAULT OFF (0 = flat unweighted mean) -- measured, not assumed. Recency
# weighting sounds obviously right but was the single most harmful of the three
# changes tried: corr 0.360 -> 0.329, MAE 10.42 -> 10.54. Same reason as the
# garbage-time filter: concentrating weight on the newest 1-2 games shrinks the
# effective sample of an already-tiny window, and the extra variance costs more
# than the staleness it fixes. Set METRICS_DECAY_HALFLIFE=2.0 to re-test.
DECAY_HALFLIFE_GAMES = float(os.getenv("METRICS_DECAY_HALFLIFE", "0"))


def _decay_mean(values: np.ndarray) -> float | None:
    """Exponentially recency-weighted mean. `values` must be oldest-first."""
    n = len(values)
    if n == 0:
        return None
    if DECAY_HALFLIFE_GAMES <= 0 or not np.isfinite(DECAY_HALFLIFE_GAMES):
        return float(np.mean(values))
    ages = np.arange(n - 1, -1, -1, dtype=float)   # newest gets age 0
    weights = 0.5 ** (ages / DECAY_HALFLIFE_GAMES)
    return float(np.average(values, weights=weights))


def _apply_opponent_adjustment(history: pd.DataFrame) -> pd.DataFrame:
    """Adjust each team-game's efficiency for the strength of the unit faced.

    Raw EPA treats +0.15/play against the worst defense in the league the same
    as +0.15 against the best. This subtracts the opponent's established
    tendency (relative to league average) from each observation:

        adj_off = raw_off - (opponent defense's EPA allowed vs league avg)
        adj_def = raw_def - (opponent offense's EPA gained vs league avg)

    Strengths are derived ONLY from `history`, which the caller has already
    restricted to weeks strictly before the week being computed -- so this
    stays leakage-free. It is a single-pass adjustment (opponent strengths are
    themselves unadjusted), which captures most of the effect; a full iterative
    or ridge-regression solve would be the next refinement.
    """
    h = history.copy()
    if "opponent" not in h.columns:
        for c in ADJUSTED_COLS:
            h[c] = np.nan
        return h

    # EPA allowed league-wide equals EPA gained league-wide (same plays, other
    # side of the ball), so one baseline serves both directions.
    league_epa = h["epa_per_play_off"].mean()
    league_sr = h["success_rate_off"].mean()

    def_epa_str = (h.groupby("team")["epa_per_play_def"].mean() - league_epa)
    off_epa_str = (h.groupby("team")["epa_per_play_off"].mean() - league_epa)
    def_sr_str = (h.groupby("team")["success_rate_def"].mean() - league_sr)
    off_sr_str = (h.groupby("team")["success_rate_off"].mean() - league_sr)

    opp = h["opponent"]
    h["epa_per_play_off_adj"] = h["epa_per_play_off"] - opp.map(def_epa_str).fillna(0.0)
    h["epa_per_play_def_adj"] = h["epa_per_play_def"] - opp.map(off_epa_str).fillna(0.0)
    h["success_rate_off_adj"] = h["success_rate_off"] - opp.map(def_sr_str).fillna(0.0)
    h["success_rate_def_adj"] = h["success_rate_def"] - opp.map(off_sr_str).fillna(0.0)
    return h


def build_rolling_metrics(game_level: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    For each (team, week), compute rolling L4 / L8 / season averages
    using only games *before* that week (no data leakage).

    Windows are recency-weighted (see _decay_mean) and EPA/success-rate values
    are additionally produced in opponent-adjusted form.

    Loops week-outer so the opponent adjustment can be fit once per week on the
    whole league's prior games, then applied to every team.
    """
    rows = []
    teams = sorted(game_level["team"].unique())
    weeks = sorted(game_level["week"].unique())
    all_cols = METRIC_COLS + ADJUSTED_COLS

    for week in tqdm(weeks, desc=f"{season} weeks", leave=False):
        history_all = game_level[game_level["week"] < week]
        if history_all.empty:
            continue

        history_all = _apply_opponent_adjustment(history_all)

        for team in teams:
            history = history_all[history_all["team"] == team].sort_values("week")
            if history.empty:
                continue

            row: dict = {"team": team, "season": season, "week": week}
            for col in all_cols:
                if col not in history.columns:
                    continue
                series = history[col].dropna()
                if len(series) == 0:
                    row[f"{col}_L4"] = None
                    row[f"{col}_L8"] = None
                    row[f"{col}_season"] = None
                    continue
                row[f"{col}_L4"] = _decay_mean(series.tail(4).values)
                row[f"{col}_L8"] = _decay_mean(series.tail(8).values)
                row[f"{col}_season"] = float(series.mean())
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main per-season function
# ---------------------------------------------------------------------------

def _load_pbp(season: int) -> pd.DataFrame:
    """Download PBP for a season and cache it locally as Parquet."""
    cache_path = DATA_DIR / f"pbp_{season}.parquet"
    if cache_path.exists():
        print(f"  Loading {season} PBP from local cache...")
        return pd.read_parquet(cache_path)
    print(f"  Downloading {season} PBP from nfl_data_py (first run, this takes a few minutes)...")
    pbp = nfl.import_pbp_data([season], downcast=True, cache=False, include_participation=False)
    pbp.to_parquet(cache_path, index=False)
    print(f"  Cached -> {cache_path.name}")
    return pbp


def build_team_metrics_for_season(season: int) -> pd.DataFrame:
    print(f"\n=== Season {season} ===")
    pbp = _load_pbp(season)
    print(f"  {len(pbp):,} plays loaded")

    # Compute per-game metrics and merge
    epa = compute_epa_metrics(pbp)
    cpoe = compute_cpoe(pbp)
    td = compute_third_down(pbp)
    rz = compute_redzone(pbp)
    pace = compute_pace(pbp)
    to = compute_turnovers(pbp)
    exp_to = compute_expected_turnovers(pbp)
    press = compute_pressure(pbp)
    ngs = compute_ngs(season)
    scoring = compute_scoring_from_schedule(season)

    game_level = (epa
                  .merge(cpoe,    on=["game_id", "team", "season", "week"], how="left")
                  .merge(td,      on=["game_id", "team", "season", "week"], how="left")
                  .merge(rz,      on=["game_id", "team", "season", "week"], how="left")
                  .merge(pace,    on=["game_id", "team", "season", "week"], how="left")
                  .merge(to,      on=["game_id", "team", "season", "week"], how="left"))

    if not press.empty:
        game_level = game_level.merge(press, on=["game_id", "team", "season", "week"], how="left")

    if not exp_to.empty:
        game_level = game_level.merge(exp_to, on=["game_id", "team", "season", "week"], how="left")
        # Luck = what you got minus what you deserved. Positive means bounces
        # went your way and should not be projected forward.
        game_level["turnover_luck"] = (
            game_level["turnover_margin"] - game_level["expected_turnover_margin"])

    # NGS is keyed team-week (one game per team per week), not game_id.
    if not ngs.empty:
        game_level = game_level.merge(ngs, on=["team", "season", "week"], how="left")

    if not scoring.empty:
        game_level = game_level.merge(
            scoring, on=["game_id", "team", "season", "week"], how="left")

    # Who each team actually played — required for opponent adjustment.
    game_level = game_level.merge(_matchups(pbp), on=["game_id", "team"], how="left")
    missing_opp = game_level["opponent"].isna().sum()
    if missing_opp:
        print(f"  WARNING: {missing_opp} team-game rows have no opponent mapped")

    print(f"  {len(game_level)} team-game rows. Computing rolling windows...")
    weekly = build_rolling_metrics(game_level, season)
    print(f"  {len(weekly)} team-week rows produced")
    return weekly


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv

    seasons = [int(args[0])] if args else ALL_HISTORICAL_SEASONS
    all_frames = []

    for season in seasons:
        out_path = DATA_DIR / f"team_metrics_{season}.parquet"
        if out_path.exists() and not force:
            print(f"Skipping {season} — already cached (use --force to recompute)")
            all_frames.append(pd.read_parquet(out_path))
            continue
        elif out_path.exists() and force:
            print(f"--force: recomputing {season} (deleting cached file)")
            out_path.unlink()

        df = build_team_metrics_for_season(season)
        df.to_parquet(out_path, index=False)
        print(f"  Saved -> {out_path}")
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = DATA_DIR / "team_metrics_all.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"\nAll seasons combined: {len(combined):,} rows -> {combined_path}")

    # Quick sanity check
    print("\nSample (KC, 2023, week 5):")
    sample = combined[(combined["team"] == "KC") & (combined["season"] == 2023) & (combined["week"] == 5)]
    if not sample.empty:
        print(sample[["team", "season", "week", "epa_per_play_off_L4", "epa_per_play_def_L4"]].to_string(index=False))


if __name__ == "__main__":
    main()
