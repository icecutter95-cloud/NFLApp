"""
Phase 1 — Step 2: Pull historical closing lines from nfl_data_py schedules.

nfl_data_py.import_schedules() includes spread_line and total_line columns sourced
from nflverse, which contains DraftKings-anchored closing lines for 2018–present.
This is a cleaner and more reliable source than scraping PFR (which blocks bots).

Output: data/pfr_lines_all.parquet  (same filename so build_dataset.py finds it)

Usage:
    python scrape_pfr_lines.py              # all historical seasons
    python scrape_pfr_lines.py 2022         # single year
"""

import sys
import pandas as pd
import nfl_data_py as nfl

from config import DATA_DIR, ALL_HISTORICAL_SEASONS

# nfl_data_py team abbrevs already match what the rest of the pipeline uses
# No mapping needed.


def pull_lines_for_seasons(seasons: list) -> pd.DataFrame:
    print(f"Pulling schedules + lines for seasons: {seasons}")
    sched = nfl.import_schedules(seasons)

    keep = [
        "season", "week", "game_id", "home_team", "away_team",
        "gameday", "game_type",
        "spread_line", "total_line",
        "home_score", "away_score",
    ]
    cols = [c for c in keep if c in sched.columns]
    df = sched[cols].copy()

    # Rename to match the column names build_dataset.py expects from this file
    df = df.rename(columns={
        "spread_line": "pfr_spread_home",
        "total_line":  "pfr_total",
    })

    # Regular season only (weeks 1–18)
    # Keep playoffs too so build_dataset can choose
    df = df[df["week"].notna()].copy()

    coverage = df["pfr_spread_home"].notna().mean() * 100
    print(f"  {len(df)} games loaded, spread coverage: {coverage:.1f}%")
    return df


def main():
    seasons = [int(sys.argv[1])] if len(sys.argv) > 1 else ALL_HISTORICAL_SEASONS

    out_all = DATA_DIR / "pfr_lines_all.parquet"
    all_frames = []

    for season in seasons:
        out_path = DATA_DIR / f"pfr_lines_{season}.parquet"
        if out_path.exists():
            print(f"Skipping {season} — cached")
            all_frames.append(pd.read_parquet(out_path))
            continue

        df = pull_lines_for_seasons([season])
        df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path.name}")
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_parquet(out_all, index=False)
    print(f"\nAll seasons: {len(combined)} games → {out_all.name}")

    # Coverage by season
    print("\nSpread coverage by season:")
    cov = (combined.groupby("season")["pfr_spread_home"]
           .apply(lambda x: f"{x.notna().sum()}/{len(x)}")
           .reset_index()
           .rename(columns={"pfr_spread_home": "coverage"}))
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
