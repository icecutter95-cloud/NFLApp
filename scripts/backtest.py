"""
Backtest the trained XGBoost models against historical closing lines.
Evaluates out-of-sample seasons (2023–2025) — model was trained on 2018–2022.

Simulates flat 1-unit betting whenever |model_line - closing_line| >= edge threshold.
Results are uploaded to Supabase `backtest_results` table for the frontend.

Usage:
    python backtest.py                       # backtest 2023-2025, upload
    python backtest.py --seasons 2024 2025   # specific seasons
    python backtest.py --no-upload           # print only, skip Supabase upload
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, MODELS_DIR, SPREAD_FEATURES, TOTAL_FEATURES,
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    VALIDATE_SEASONS, TEST_SEASON,
)
from supabase import create_client

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Bet simulation
# ---------------------------------------------------------------------------

def simulate_bets(preds, actuals, closing_lines, game_df, bet_type, min_edge=0.5):
    """
    For each game where |model_line - closing_line| >= min_edge, simulate a
    flat 1-unit bet and record win/loss/push.
    """
    results = []
    payout = 100 / 110  # -110 vig

    for pred, actual, line, (_, row) in zip(preds, actuals, closing_lines, game_df.iterrows()):
        if np.isnan(line) or np.isnan(actual) or np.isnan(pred):
            continue

        edge = pred - line
        abs_edge = abs(edge)
        if abs_edge < min_edge:
            continue

        if bet_type == "spread":
            bet_home = edge > 0
            side = row["home_team"] if bet_home else row["away_team"]
            win = actual > line if bet_home else actual < line
            push = abs(actual - line) < 0.001
        else:  # total
            side = "over" if edge > 0 else "under"
            win = (actual > line) if edge > 0 else (actual < line)
            push = abs(actual - line) < 0.001

        if push:
            result, units = "push", 0.0
        elif win:
            result, units = "win", round(payout, 4)
        else:
            result, units = "loss", -1.0

        # Simulated EV at this edge
        win_prob = min(0.5238 + abs_edge * 0.03, 0.85)
        ev_pct = round(win_prob * payout - (1 - win_prob), 4)

        results.append({
            "season":        int(row["season"]),
            "week":          int(row["week"]),
            "game_id":       row.get("game_id", f"{row['home_team']}_{row['away_team']}_{row['week']}"),
            "home_team":     row["home_team"],
            "away_team":     row["away_team"],
            "bet_type":      bet_type,
            "side":          side,
            "model_line":    round(float(pred), 2),
            "closing_line":  float(line),
            "actual_result": float(actual),
            "edge_points":   round(abs_edge, 2),
            "ev_pct":        ev_pct,
            "result":        result,
            "units":         units,
        })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(df):
    print("\n" + "=" * 72)
    print("BACKTEST SUMMARY  (out-of-sample, flat 1u)")
    print("=" * 72)

    for bet_type in ["spread", "total"]:
        sub = df[df["bet_type"] == bet_type]
        if sub.empty:
            continue
        print(f"\n{'─'*72}")
        print(f"  {bet_type.upper()} MODEL")
        print(f"  {'Season':<8} {'Edge≥':<8} {'W-L-P':<14} {'ROI':>8}  {'Units':>8}  {'Bets':>6}")
        print(f"  {'─'*60}")
        for season in sorted(sub["season"].unique()):
            s = sub[sub["season"] == season]
            for thresh in [1.0, 1.5, 2.0, 2.5]:
                t = s[s["edge_points"] >= thresh]
                if len(t) == 0:
                    continue
                w = (t["result"] == "win").sum()
                l = (t["result"] == "loss").sum()
                p = (t["result"] == "push").sum()
                u = t["units"].sum()
                roi = (u / max(w + l, 1)) * 100
                print(f"  {season:<8} {thresh:<8.1f} {w}-{l}-{p:<10} {roi:>+7.1f}%  {u:>+7.1f}u  {len(t):>6}")

    print(f"\n{'─'*72}")
    print("  OVERALL (edge ≥ 1.5, both models combined)")
    t = df[df["edge_points"] >= 1.5]
    w = (t["result"] == "win").sum()
    l = (t["result"] == "loss").sum()
    p = (t["result"] == "push").sum()
    u = t["units"].sum()
    roi = (u / max(w + l, 1)) * 100
    print(f"  {w}-{l}-{p}  |  ROI: {roi:+.1f}%  |  {u:+.1f} units  |  {len(t)} bets")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=None)
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    parquet = DATA_DIR / "historical_dataset_regular.parquet"
    if not parquet.exists():
        raise FileNotFoundError("Run build_dataset.py first to generate historical_dataset_regular.parquet")

    df = pd.read_parquet(parquet)

    eval_seasons = args.seasons or (VALIDATE_SEASONS + [TEST_SEASON])
    print(f"Backtesting seasons: {eval_seasons}")
    df = df[df["season"].isin(eval_seasons)].copy()
    print(f"  {len(df)} games loaded")

    # Load models
    spread_model = joblib.load(MODELS_DIR / "spread_model.joblib")
    total_model  = joblib.load(MODELS_DIR / "total_model.joblib")

    spread_cols = [c for c in SPREAD_FEATURES if c in df.columns]
    total_cols  = [c for c in TOTAL_FEATURES  if c in df.columns]

    spread_preds = spread_model.predict(df[spread_cols].fillna(0))
    total_preds  = total_model.predict( df[total_cols].fillna(0))

    # Simulate bets (record all edge >= 0.5 so frontend can filter)
    all_results = (
        simulate_bets(spread_preds, df["home_margin"].values,    df["closing_spread_home"].values, df, "spread") +
        simulate_bets(total_preds,  df["combined_score"].values, df["closing_total"].values,       df, "total")
    )

    results_df = pd.DataFrame(all_results)
    print(f"\n{len(results_df)} simulated bets (edge ≥ 0.5)")
    print_summary(results_df)

    if not args.no_upload:
        print(f"\nUploading to Supabase backtest_results...")
        for season in eval_seasons:
            sb.table("backtest_results").delete().eq("season", season).execute()

        batch = 500
        for i in range(0, len(all_results), batch):
            sb.table("backtest_results").insert(all_results[i:i+batch]).execute()
            print(f"  {min(i+batch, len(all_results))}/{len(all_results)}")

        print("Done.")


if __name__ == "__main__":
    main()
