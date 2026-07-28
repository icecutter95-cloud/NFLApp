"""
Diagnostic: how much does spread-model accuracy degrade in early-season weeks?

Motivation
----------
compute_metrics.py builds each team's rolling window from strictly prior weeks
(`history = team_df[team_df.week < week]`), so WEEK 1 GAMES HAVE NO METRICS AT
ALL -- every team feature is NaN, and train_models.py / score_week.py fill NaN
with 0. So the model was trained on week-1 games whose team features were all
zeros.

But score_week.py's live fallback chain does something completely different:
when no current-season metrics exist yet, it substitutes END OF PRIOR SEASON
metrics -- real, non-zero values. That is a train/serve skew: at week 1 the
model is fed an input distribution it never saw during training.

This script measures the size of that problem, three ways:
  1. Out-of-sample accuracy bucketed by week (is early season worse?)
  2. What week-1 predictions look like under the AS-TRAINED (zeros) regime
  3. What week-1 predictions look like under the LIVE (prior-season-end) regime

Usage:
    python diagnose_preseason.py
"""

import numpy as np
import pandas as pd

from config import DATA_DIR, SPREAD_FEATURES, VALIDATE_SEASONS, TEST_SEASON, _TEAM_METRIC_COLS
from train_models import load_split, train_model, ats_roi


def evaluate(preds, actuals, label: str, edge_threshold: float = 1.5):
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = (~np.isnan(preds)) & (~np.isnan(actuals))
    preds, actuals = preds[mask], actuals[mask]

    if len(preds) < 2:
        print(f"  {label:<34} (too few games: {len(preds)})")
        return

    corr = np.corrcoef(preds, actuals)[0, 1]
    r = ats_roi(preds, actuals, edge_threshold=edge_threshold)
    wins, losses = r["wins"], r["losses"]
    wr = wins / max(wins + losses, 1) * 100
    print(f"  {label:<34} n={len(preds):>4}  corr={corr:>+6.3f}  "
          f"{wins:>3}-{losses:<3} ({wr:>4.1f}%)  ROI={r['roi_pct']:>+6.1f}%  "
          f"meanEdge={np.abs(preds).mean():>5.1f}")


def prior_season_end_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season): that team's metrics as of the LAST week
    available in that season -- i.e. end-of-season form."""
    idx = metrics.groupby(["team", "season"])["week"].idxmax()
    return metrics.loc[idx].copy()


def simulate_live_week1(df: pd.DataFrame, metrics: pd.DataFrame,
                        weeks: list) -> pd.DataFrame:
    """Replace team metric features with END OF PRIOR SEASON values for the
    given weeks -- reproducing exactly what score_week.py's fallback does live."""
    out = df.copy()
    end = prior_season_end_metrics(metrics)

    # lookup: (team, season_that_just_ended) -> metric row
    lookup = {(r["team"], r["season"]): r for _, r in end.iterrows()}

    target = out["week"].isin(weeks)
    for side in ("home", "away"):
        team_col = f"{side}_team"
        for i in out.index[target]:
            team = out.at[i, team_col]
            prior_season = int(out.at[i, "season"]) - 1
            src = lookup.get((team, prior_season))
            if src is None:
                continue
            for base in _TEAM_METRIC_COLS:
                col = f"{base}_{side}"
                if col in out.columns and base in src:
                    out.at[i, col] = src[base]
    return out


def main():
    eval_seasons = VALIDATE_SEASONS + [TEST_SEASON]
    train_seasons = list(range(2018, min(eval_seasons)))

    print("=" * 78)
    print("PRESEASON / EARLY-SEASON DEGRADATION DIAGNOSTIC (spread model)")
    print(f"  train={train_seasons}  eval={eval_seasons}")
    print("=" * 78)

    X_tr, y_tr, df_tr = load_split(train_seasons, SPREAD_FEATURES, "home_cover_surplus")
    model = train_model(X_tr, y_tr, df_tr, "preseason_diagnostic")
    feat_cols = model.get_booster().feature_names

    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    ev = full[full["season"].isin(eval_seasons)].copy()
    ev = ev[ev["home_cover_surplus"].notna()]

    # ---- 1. Accuracy by week bucket, as the pipeline currently behaves ----
    print("\n1. OUT-OF-SAMPLE ACCURACY BY WEEK BUCKET (as-is)")
    print("   (week 1 features are all-NaN -> filled to 0, exactly as trained)")
    buckets = [("Week 1 only", [1]),
               ("Weeks 2-4", [2, 3, 4]),
               ("Weeks 5-9", list(range(5, 10))),
               ("Weeks 10+", list(range(10, 23)))]
    for label, weeks in buckets:
        sub = ev[ev["week"].isin(weeks)]
        if sub.empty:
            continue
        preds = model.predict(sub[feat_cols].fillna(0))
        evaluate(preds, sub["home_cover_surplus"].values, label)

    all_preds = model.predict(ev[feat_cols].fillna(0))
    evaluate(all_preds, ev["home_cover_surplus"].values, "ALL WEEKS (reference)")

    # ---- 2. Week 1 under the two competing input regimes ----
    print("\n2. WEEK 1 -- AS-TRAINED (zeros) vs LIVE FALLBACK (prior-season-end)")
    metrics = pd.read_parquet(DATA_DIR / "team_metrics_all.parquet")
    wk1 = ev[ev["week"] == 1]

    if wk1.empty:
        print("   (no week 1 games in eval seasons)")
    else:
        preds_zero = model.predict(wk1[feat_cols].fillna(0))
        evaluate(preds_zero, wk1["home_cover_surplus"].values, "Week 1 as-trained (zeros)")

        wk1_live = simulate_live_week1(ev, metrics, [1])
        wk1_live = wk1_live[wk1_live["week"] == 1]
        preds_live = model.predict(wk1_live[feat_cols].fillna(0))
        evaluate(preds_live, wk1_live["home_cover_surplus"].values, "Week 1 live fallback (prior end)")

        print(f"\n   Prediction spread (how much the model differentiates games):")
        print(f"     as-trained  : min={preds_zero.min():+6.1f}  max={preds_zero.max():+6.1f}  "
              f"std={preds_zero.std():5.2f}")
        print(f"     live fallback: min={preds_live.min():+6.1f}  max={preds_live.max():+6.1f}  "
              f"std={preds_live.std():5.2f}")

    # ---- 3. Same comparison across weeks 1-4 ----
    print("\n3. WEEKS 1-4 -- AS-IS vs LIVE FALLBACK SUBSTITUTION")
    early = ev[ev["week"].isin([1, 2, 3, 4])]
    preds_asis = model.predict(early[feat_cols].fillna(0))
    evaluate(preds_asis, early["home_cover_surplus"].values, "Weeks 1-4 as-is")

    early_live = simulate_live_week1(ev, metrics, [1, 2, 3, 4])
    early_live = early_live[early_live["week"].isin([1, 2, 3, 4])]
    preds_early_live = model.predict(early_live[feat_cols].fillna(0))
    evaluate(preds_early_live, early_live["home_cover_surplus"].values, "Weeks 1-4 prior-season-end")


if __name__ == "__main__":
    main()
