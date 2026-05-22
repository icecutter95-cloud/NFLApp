"""
Upload locally computed team metrics to the Supabase team_metrics table.

Run this after compute_metrics.py has finished and produced team_metrics_all.parquet.
Uses the Supabase service role key — never run from the frontend.

Usage:
    python upload_metrics_to_supabase.py            # all seasons
    python upload_metrics_to_supabase.py 2024       # single season
    python upload_metrics_to_supabase.py 2024 10    # specific season + week
"""

import sys
import math
import pandas as pd
from supabase import create_client

from config import DATA_DIR, SUPABASE_URL, SUPABASE_SERVICE_KEY

BATCH_SIZE = 200  # Supabase upsert batch size

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# Supabase team_metrics column names (must match schema.sql)
# Keys: parquet column names from team_metrics_all.parquet
# Values: actual PostgreSQL column names (lowercase — Postgres folds unquoted identifiers)
SUPABASE_COL_MAP = {
    "epa_per_play_off_L4": "epa_off_l4",
    "epa_per_play_off_L8": "epa_off_l8",
    "epa_per_play_def_L4": "epa_def_l4",
    "epa_per_play_def_L8": "epa_def_l8",
    "epa_pass_off_L4":     "epa_pass_off_l4",
    "epa_rush_off_L4":     "epa_rush_off_l4",
    "success_rate_off_L4": "success_rate_off_l4",
    "success_rate_def_L4": "success_rate_def_l4",
    "cpoe_L4":             "cpoe_l4",
    "cpoe_L8":             "cpoe_l8",
    "third_down_conv_off_season": "third_down_conv_off",
    "third_down_stop_def_season": "third_down_stop_def",
    "rz_td_pct_off_season":       "rz_td_pct_off",
    "plays_per_game_L4":          "pace_plays_per_game",
    "turnover_margin_L8":         "turnover_luck_adj",
}


def prep_rows(df: pd.DataFrame) -> list[dict]:
    """Convert parquet rows to Supabase-compatible dicts, replacing NaN with None."""
    rows = []
    for _, row in df.iterrows():
        record: dict = {
            "team": row["team"],
            "season": int(row["season"]),
            "week": int(row["week"]),
        }
        for parquet_col, db_col in SUPABASE_COL_MAP.items():
            val = row.get(parquet_col)
            record[db_col] = None if (val is None or (isinstance(val, float) and math.isnan(val))) else float(val)
        rows.append(record)
    return rows


def upload(df: pd.DataFrame):
    rows = prep_rows(df)
    total = len(rows)
    uploaded = 0

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        supabase.table("team_metrics").upsert(batch, on_conflict="team,season,week").execute()
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{total} rows...")

    print(f"Done. {total} rows upserted to team_metrics.")


def main():
    path = DATA_DIR / "team_metrics_all.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run compute_metrics.py first")

    df = pd.read_parquet(path)

    # Optional filters from CLI args
    if len(sys.argv) >= 2:
        season = int(sys.argv[1])
        df = df[df["season"] == season]
        print(f"Filtering to season {season}: {len(df)} rows")

    if len(sys.argv) >= 3:
        week = int(sys.argv[2])
        df = df[df["week"] == week]
        print(f"Filtering to week {week}: {len(df)} rows")

    if df.empty:
        print("No rows to upload.")
        return

    print(f"Uploading {len(df)} rows to Supabase team_metrics...")
    upload(df)


if __name__ == "__main__":
    main()
