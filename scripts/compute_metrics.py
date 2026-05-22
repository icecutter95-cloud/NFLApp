"""
Phase 1 — Step 1: Pull nfl_data_py PBP data and compute rolling team efficiency metrics.

Run once per season during the offseason build, or weekly in-season to update the
most recent week. Output is one row per (team, season, week) representing metrics
*going into* that week (i.e., the current game is excluded from its own rolling window).

Usage:
    python compute_metrics.py              # all historical seasons
    python compute_metrics.py 2024         # single season
"""

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

def _plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Filter to scoreable pass/run plays with valid EPA."""
    return pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()


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
]


def build_rolling_metrics(game_level: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    For each (team, week), compute rolling L4 / L8 / season averages
    using only games *before* that week (no data leakage).
    """
    rows = []
    teams = sorted(game_level["team"].unique())
    weeks = sorted(game_level["week"].unique())

    for team in tqdm(teams, desc=f"{season} teams", leave=False):
        team_df = game_level[game_level["team"] == team].sort_values("week")

        for week in weeks:
            history = team_df[team_df["week"] < week]
            if history.empty:
                continue

            row: dict = {"team": team, "season": season, "week": week}
            for col in METRIC_COLS:
                if col not in history.columns:
                    continue
                series = history[col].dropna()
                row[f"{col}_L4"] = float(series.tail(4).mean()) if len(series) > 0 else None
                row[f"{col}_L8"] = float(series.tail(8).mean()) if len(series) > 0 else None
                row[f"{col}_season"] = float(series.mean()) if len(series) > 0 else None
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
    print(f"  Cached → {cache_path.name}")
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

    game_level = (epa
                  .merge(cpoe, on=["game_id", "team", "season", "week"], how="left")
                  .merge(td, on=["game_id", "team", "season", "week"], how="left")
                  .merge(rz, on=["game_id", "team", "season", "week"], how="left")
                  .merge(pace, on=["game_id", "team", "season", "week"], how="left")
                  .merge(to, on=["game_id", "team", "season", "week"], how="left"))

    print(f"  {len(game_level)} team-game rows. Computing rolling windows...")
    weekly = build_rolling_metrics(game_level, season)
    print(f"  {len(weekly)} team-week rows produced")
    return weekly


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    seasons = [int(sys.argv[1])] if len(sys.argv) > 1 else ALL_HISTORICAL_SEASONS
    all_frames = []

    for season in seasons:
        out_path = DATA_DIR / f"team_metrics_{season}.parquet"
        if out_path.exists():
            print(f"Skipping {season} — already cached at {out_path}")
            all_frames.append(pd.read_parquet(out_path))
            continue

        df = build_team_metrics_for_season(season)
        df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}")
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = DATA_DIR / "team_metrics_all.parquet"
    combined.to_parquet(combined_path, index=False)
    print(f"\nAll seasons combined: {len(combined):,} rows → {combined_path}")

    # Quick sanity check
    print("\nSample (KC, 2023, week 5):")
    sample = combined[(combined["team"] == "KC") & (combined["season"] == 2023) & (combined["week"] == 5)]
    if not sample.empty:
        print(sample[["team", "season", "week", "epa_per_play_off_L4", "epa_per_play_def_L4"]].to_string(index=False))


if __name__ == "__main__":
    main()
