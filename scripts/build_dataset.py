"""
Phase 1 — Step 3: Merge team metrics + schedule lines into a unified training dataset.

Primary line source: nfl_data_py schedules (spread_line, total_line columns from nflverse).
Fallback: PFR scraped lines (pfr_lines_all.parquet).

Output: data/historical_dataset.parquet
  - One row per game
  - All team efficiency features for home + away (with _home / _away suffix)
  - Target columns: home_margin, combined_score
  - Closing lines: closing_spread_home, closing_total

Usage:
    python build_dataset.py
"""

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from config import DATA_DIR, ALL_HISTORICAL_SEASONS, HFA_OVERRIDES, HFA_DEFAULT, DOME_TEAMS


# ---------------------------------------------------------------------------
# Schedule loading
# ---------------------------------------------------------------------------

def load_schedules(seasons: list) -> pd.DataFrame:
    print("Loading schedules via nfl_data_py...")
    sched = nfl.import_schedules(seasons)

    keep = [
        "season", "week", "game_id", "home_team", "away_team", "game_type",
        "gameday", "gametime", "home_score", "away_score",
        "spread_line", "total_line", "div_game", "roof",
    ]
    cols = [c for c in keep if c in sched.columns]
    sched = sched[cols].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    print(f"  {len(sched)} games loaded (seasons {min(seasons)}–{max(seasons)})")
    return sched


# ---------------------------------------------------------------------------
# Rest days / bye week features
# ---------------------------------------------------------------------------

def add_rest_features(sched: pd.DataFrame) -> pd.DataFrame:
    """Compute rest_days and had_bye for home and away teams."""
    home = sched[["season", "week", "home_team", "gameday"]].rename(
        columns={"home_team": "team", "gameday": "game_date"})
    away = sched[["season", "week", "away_team", "gameday"]].rename(
        columns={"away_team": "team", "gameday": "game_date"})

    all_games = (pd.concat([home, away])
                 .sort_values(["season", "team", "week"])
                 .reset_index(drop=True))
    all_games["prev_date"] = all_games.groupby(["season", "team"])["game_date"].shift(1)
    all_games["rest_days"] = (all_games["game_date"] - all_games["prev_date"]).dt.days.fillna(7)
    all_games["had_bye"] = (all_games["rest_days"] >= 13).astype(int)

    home_rest = (all_games.rename(columns={"team": "home_team", "rest_days": "rest_days_home",
                                            "had_bye": "had_bye_home"})
                 [["season", "week", "home_team", "rest_days_home", "had_bye_home"]])
    away_rest = (all_games.rename(columns={"team": "away_team", "rest_days": "rest_days_away",
                                            "had_bye": "had_bye_away"})
                 [["season", "week", "away_team", "rest_days_away", "had_bye_away"]])

    sched = (sched
             .merge(home_rest, on=["season", "week", "home_team"], how="left")
             .merge(away_rest, on=["season", "week", "away_team"], how="left"))
    sched["rest_diff"] = sched["rest_days_home"] - sched["rest_days_away"]
    sched["is_short_week_home"] = (sched["rest_days_home"] <= 5).astype(int)
    sched["is_short_week_away"] = (sched["rest_days_away"] <= 5).astype(int)
    return sched


# ---------------------------------------------------------------------------
# Stadium / game context features
# ---------------------------------------------------------------------------

def add_game_context(sched: pd.DataFrame) -> pd.DataFrame:
    sched["home_field_advantage"] = sched["home_team"].map(
        lambda t: HFA_OVERRIDES.get(t, HFA_DEFAULT))
    sched["is_dome"] = sched["home_team"].isin(DOME_TEAMS).astype(int)
    sched["is_divisional"] = sched.get("div_game", pd.Series(0, index=sched.index)).fillna(0).astype(int)
    sched["week_number"] = sched["week"].astype(int)
    sched["is_playoffs"] = (sched["game_type"] != "REG").astype(int) if "game_type" in sched.columns else 0

    # Neutral-site HFA reduction (London, Mexico City, Super Bowl)
    if "game_type" in sched.columns:
        neutral = sched["game_type"].isin(["SB", "CON"])
        sched.loc[neutral, "home_field_advantage"] -= 0.5

    return sched


# ---------------------------------------------------------------------------
# Team metrics merge
# ---------------------------------------------------------------------------

def merge_team_metrics(sched: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in metrics.columns if c not in ("team", "season", "week")]

    home_m = (metrics.rename(columns={c: f"{c}_home" for c in metric_cols})
              .rename(columns={"team": "home_team"}))
    away_m = (metrics.rename(columns={c: f"{c}_away" for c in metric_cols})
              .rename(columns={"team": "away_team"}))

    df = (sched
          .merge(home_m, on=["season", "week", "home_team"], how="left")
          .merge(away_m, on=["season", "week", "away_team"], how="left"))
    return df


# ---------------------------------------------------------------------------
# Lines (primary = nfl_data_py schedule, fallback = PFR)
# ---------------------------------------------------------------------------

def add_closing_lines(df: pd.DataFrame) -> pd.DataFrame:
    pfr_path = DATA_DIR / "pfr_lines_all.parquet"
    if pfr_path.exists():
        pfr = pd.read_parquet(pfr_path)[["season", "week", "home_team", "away_team",
                                          "pfr_spread_home", "pfr_total"]]
        df = df.merge(pfr, on=["season", "week", "home_team", "away_team"], how="left")
    else:
        print("  WARNING: pfr_lines_all.parquet not found — run scrape_pfr_lines.py first")
        df["pfr_spread_home"] = np.nan
        df["pfr_total"] = np.nan

    # Primary source from nfl_data_py schedule, fallback to PFR
    df["closing_spread_home"] = df["spread_line"].fillna(df["pfr_spread_home"]) if "spread_line" in df.columns else df["pfr_spread_home"]
    df["closing_total"] = df["total_line"].fillna(df["pfr_total"]) if "total_line" in df.columns else df["pfr_total"]
    return df


# ---------------------------------------------------------------------------
# Target variables
# ---------------------------------------------------------------------------

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["combined_score"] = df["home_score"] + df["away_score"]

    # ATS / O/U result labels (for validation reporting)
    df["spread_result"] = np.where(
        df["home_margin"] > df["closing_spread_home"].abs(), "home_covered",
        np.where(df["home_margin"] < -df["closing_spread_home"].abs(), "away_covered", "push")
    )
    df["total_result"] = np.where(
        df["combined_score"] > df["closing_total"], "over",
        np.where(df["combined_score"] < df["closing_total"], "under", "push")
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load schedules
    sched = load_schedules(ALL_HISTORICAL_SEASONS)
    sched = add_rest_features(sched)
    sched = add_game_context(sched)

    # 2. Load team metrics (must run compute_metrics.py first)
    metrics_path = DATA_DIR / "team_metrics_all.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError("Run compute_metrics.py first to generate team_metrics_all.parquet")
    print("Loading team metrics...")
    metrics = pd.read_parquet(metrics_path)

    # 3. Merge
    print("Merging schedules + metrics...")
    df = merge_team_metrics(sched, metrics)

    # 4. Lines
    print("Adding closing lines...")
    df = add_closing_lines(df)

    # 5. Expose market lines as model features
    # The closing line is the market's best estimate — giving it to the model
    # lets it learn to adjust *off* the market rather than predict from scratch.
    df["market_spread_home"] = df["closing_spread_home"]
    df["market_total"]       = df["closing_total"]

    # 6. Targets
    df = add_targets(df)

    # 6. Drop rows without scores or lines (can't train/evaluate on them)
    before = len(df)
    df = df.dropna(subset=["home_score", "closing_spread_home"])
    print(f"Dropped {before - len(df)} rows with missing scores or lines ({len(df)} remain)")

    # 7. Regular season only for training (keep playoffs separate)
    regular = df[df["is_playoffs"] == 0].copy()
    print(f"Regular season games: {len(regular)}")

    # 8. Save
    out = DATA_DIR / "historical_dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"\nFull dataset ({len(df)} games) → {out}")

    out_reg = DATA_DIR / "historical_dataset_regular.parquet"
    regular.to_parquet(out_reg, index=False)
    print(f"Regular season ({len(regular)} games) → {out_reg}")

    # Quick coverage summary
    print("\nLine coverage by season:")
    cov = (df.groupby("season")["closing_spread_home"]
           .apply(lambda x: f"{x.notna().sum()}/{len(x)}")
           .reset_index()
           .rename(columns={"closing_spread_home": "spread_coverage"}))
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
