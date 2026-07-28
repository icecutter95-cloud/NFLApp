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

warnings.filterwarnings("ignore")

from config import (
    DATA_DIR, MODELS_DIR, SPREAD_FEATURES, TOTAL_FEATURES,
    SUPABASE_URL, SUPABASE_SERVICE_KEY,
    VALIDATE_SEASONS, TEST_SEASON, EDGE_PER_WIN_PCT_POINT,
)
from train_models import load_split, train_model
from supabase import create_client

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Bet simulation
# ---------------------------------------------------------------------------

def simulate_bets(surplus_preds, actual_surpluses, closing_lines, game_df, bet_type, min_edge=0.5):
    """
    Simulate bets using COVER SURPLUS predictions.

    surplus_preds    – model output: home_cover_surplus (spread) or ou_surplus (total)
                        positive → bet home/over; negative → bet away/under
    actual_surpluses – actual cover surplus from historical results
    closing_lines    – closing_spread_home or closing_total (used for display only)

    We bet when |surplus_pred| >= min_edge.
    We win when sign(pred) == sign(actual).
    """
    results = []
    payout = 100 / 110  # -110 vig

    for _pred, _actual, _line, (_, row) in zip(surplus_preds, actual_surpluses, closing_lines, game_df.iterrows()):
        # Cast numpy scalars → Python floats so json.dumps() doesn't choke on float32
        pred   = float(_pred)
        actual = float(_actual)
        line   = float(_line)

        if np.isnan(pred) or np.isnan(actual):
            continue

        abs_edge = abs(pred)
        if abs_edge < min_edge:
            continue

        push = abs(actual) < 0.01
        won  = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)

        if bet_type == "spread":
            side = row["home_team"] if pred > 0 else row["away_team"]
            # For display: reconstruct the approximate projected home margin
            # projected_margin ≈ pred - closing_spread_home  (since surplus = margin + line)
            display_line = round(pred - line if not np.isnan(line) else pred, 2)
        else:
            side = "over" if pred > 0 else "under"
            # For display: projected combined score ≈ pred + closing_total
            display_line = round(pred + line if not np.isnan(line) else pred, 2)

        if push:
            result, units = "push", 0.0
        elif won:
            result, units = "win", round(payout, 4)
        else:
            result, units = "loss", -1.0

        win_prob = min(0.5238 + abs_edge * EDGE_PER_WIN_PCT_POINT, 0.85)
        ev_pct   = round(win_prob * payout - (1 - win_prob), 4)

        results.append({
            "season":        int(row["season"]),
            "week":          int(row["week"]),
            "game_id":       row.get("game_id", f"{row['home_team']}_{row['away_team']}_{row['week']}"),
            "home_team":     row["home_team"],
            "away_team":     row["away_team"],
            "bet_type":      bet_type,
            "side":          side,
            "model_line":    display_line,
            "closing_line":  round(line, 2),
            "actual_result": round(actual, 2),
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

    # A backtest is only meaningful if the model never saw the seasons being
    # evaluated. The production joblib models (spread_model.joblib /
    # total_model.joblib) are trained on TRAIN_SEASONS from config.py, which
    # is deliberately "all available history" for the best live 2026
    # predictions — that means they may already include the very seasons
    # being backtested here. Loading them would silently turn this into an
    # in-sample sanity check, not a backtest, and would inflate results
    # (this happened: totals ROI jumped to +70%+ despite validated
    # true-holdout correlation of only ~0.01-0.07).
    #
    # So: always train fresh, backtest-only models using strictly the
    # seasons before the earliest one being evaluated, independent of
    # whatever config.py's TRAIN_SEASONS is set to for production.
    backtest_train_seasons = list(range(2018, min(eval_seasons)))
    if not backtest_train_seasons:
        raise ValueError(
            f"No seasons available to train on before {min(eval_seasons)} — "
            f"can't backtest a season with no prior history."
        )
    print(f"  Training honest backtest models on {backtest_train_seasons} "
          f"(strictly prior to eval seasons — independent of production models)")

    X_tr_s, y_tr_s, df_tr_s = load_split(backtest_train_seasons, SPREAD_FEATURES, "home_cover_surplus")
    spread_model = train_model(X_tr_s, y_tr_s, df_tr_s, "spread_model_backtest")

    X_tr_t, y_tr_t, df_tr_t = load_split(backtest_train_seasons, TOTAL_FEATURES, "ou_surplus")
    total_model = train_model(X_tr_t, y_tr_t, df_tr_t, "total_model_backtest")

    # Use model's own feature list (not config) to avoid column mismatches
    spread_model_cols = spread_model.get_booster().feature_names
    total_model_cols  = total_model.get_booster().feature_names

    for col in spread_model_cols + total_model_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Models now predict cover surplus directly:
    #   spread_model → home_cover_surplus = home_margin + closing_spread_home
    #   total_model  → ou_surplus = combined_score - closing_total
    spread_surplus_preds = spread_model.predict(df[spread_model_cols].fillna(0))
    total_surplus_preds  = total_model.predict( df[total_model_cols].fillna(0))

    # Compute actual surpluses for evaluation
    actual_cover_surplus = (df["home_margin"] + df["closing_spread_home"]).values
    actual_ou_surplus    = (df["combined_score"] - df["closing_total"]).values

    # Simulate bets (record all |pred| >= 0.5 so frontend can filter)
    all_results = (
        simulate_bets(spread_surplus_preds, actual_cover_surplus, df["closing_spread_home"].values, df, "spread") +
        simulate_bets(total_surplus_preds,  actual_ou_surplus,    df["closing_total"].values,       df, "total")
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
