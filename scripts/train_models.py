"""
Phase 2 — Train XGBoost spread and total models.

Train/validate/test split (from spec):
  Train:    2018–2021
  Validate: 2022–2023
  Test:     2024  ← SEALED — do NOT evaluate until model is finalized

Minimum targets before going live:
  - Spread MAE ≤ 8 pts on validation
  - ATS ROI > +3% at 1.5+ pt edge threshold on 2022–2023 validation
  - ATS ROI > 0% on 2024 holdout at same threshold

Usage:
    python train_models.py              # train both models
    python train_models.py --validate   # also print full validation report
    python train_models.py --test       # ⚠ run holdout (only when ready to ship)
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

from config import (DATA_DIR, MODELS_DIR, SPREAD_FEATURES, TOTAL_FEATURES,
                    TRAIN_SEASONS, VALIDATE_SEASONS, TEST_SEASON, EDGE_PER_WIN_PCT_POINT)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(seasons: list, feature_cols: list, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    df = df[df["season"].isin(seasons)].copy()

    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        print(f"  WARNING: {len(missing)} feature columns not in dataset: {sorted(missing)[:5]}...")

    X = df[available].fillna(0)
    y = df[target_col]
    mask = y.notna()
    return X[mask], y[mask], df[mask]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

XGBOOST_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.03,       # slower learning → less overfitting
    "max_depth": 3,              # shallower trees → simpler model
    "subsample": 0.6,            # more aggressive row sampling
    "colsample_bytree": 0.6,     # more aggressive column sampling
    "min_child_weight": 10,      # require more samples per leaf
    "reg_alpha": 1.0,            # stronger L1 regularization
    "reg_lambda": 5.0,           # stronger L2 regularization
    "random_state": 42,
    "n_jobs": -1,
}


def recency_weights(seasons: pd.Series) -> np.ndarray:
    """Exponential recency weighting: each additional season back gets weight * 0.75.
    Most recent season gets weight 1.0; 5 seasons back gets ~0.24."""
    max_season = seasons.max()
    return (0.75 ** (max_season - seasons)).values


def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                df_train: pd.DataFrame, model_name: str) -> XGBRegressor:
    print(f"\nTraining {model_name} on {len(X_train)} rows, {X_train.shape[1]} features...")
    weights = recency_weights(df_train["season"])
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights,
              eval_set=[(X_train, y_train)], verbose=False)
    return model


# ---------------------------------------------------------------------------
# Validation / evaluation
# ---------------------------------------------------------------------------

def ats_roi(preds: np.ndarray, actuals: np.ndarray,
            edge_threshold: float = 1.5) -> dict:
    """
    Simulate ATS / O-U betting using COVER SURPLUS predictions.

    Both preds and actuals are already expressed as cover surplus:
      - Spread model: home_cover_surplus = home_margin + closing_spread_home
          positive → home covered; negative → away covered
      - Total model:  ou_surplus = combined_score - closing_total
          positive → over hit; negative → under hit

    We bet when |pred| >= edge_threshold (model is confident enough).
    We win when sign(pred) == sign(actual).

    This eliminates any edge-formula sign ambiguity: the model directly
    predicts the ATS outcome, so the win condition is simply:
        pred > 0 AND actual > 0  →  win (both say home/over)
        pred < 0 AND actual < 0  →  win (both say away/under)
    """
    payout = 100 / 110  # -110 vig
    wins, losses, pushes = 0, 0, 0
    profit = 0.0

    for pred, actual in zip(preds, actuals):
        if np.isnan(actual) or np.isnan(pred):
            continue
        if abs(pred) < edge_threshold:
            continue

        push_flag = abs(actual) < 0.01
        won = (pred > 0 and actual > 0) or (pred < 0 and actual < 0)

        if push_flag:
            pushes += 1
        elif won:
            wins += 1
            profit += payout
        else:
            losses += 1
            profit -= 1.0

    total_bets = wins + losses + pushes
    roi = (profit / max(wins + losses, 1)) * 100
    return {"bets": total_bets, "wins": wins, "losses": losses, "pushes": pushes,
            "roi_pct": round(roi, 2), "profit_units": round(profit, 2)}


def evaluate_model(model: XGBRegressor, X: pd.DataFrame, y: pd.Series,
                   df_raw: pd.DataFrame, target: str, label: str):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    corr = np.corrcoef(y, preds)[0, 1]

    print(f"\n{label} ({target})")
    print(f"  MAE:  {mae:.3f} pts")
    print(f"  RMSE: {rmse:.3f} pts")
    print(f"  Corr: {corr:.3f}")

    # ATS / O-U ROI — preds and y are already cover surplus
    bet_label = "ATS ROI" if target == "home_cover_surplus" else "AOU ROI"
    print(f"  {bet_label} by |pred| threshold:")
    for thresh in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        r = ats_roi(preds, y.values, edge_threshold=thresh)
        print(f"    edge ≥ {thresh:.1f}: {r['wins']}-{r['losses']} | "
              f"ROI: {r['roi_pct']:+.1f}% | {r['bets']} bets")

    # Feature importance (top 10)
    feat_names = X.columns.tolist()
    importance = sorted(zip(feat_names, model.feature_importances_),
                        key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 features:")
    for name, imp in importance:
        print(f"    {name}: {imp:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Run full validation report")
    parser.add_argument("--test", action="store_true",
                        help="Run 2024 holdout (WARNING: only do this once, at ship time)")
    args = parser.parse_args()

    # --- Spread model ---
    print("=" * 60)
    print("SPREAD MODEL (predicts: home_cover_surplus)")
    print("  home_cover_surplus = home_margin + closing_spread_home")
    print("  positive → home covers; negative → away covers")
    print("=" * 60)
    X_train_s, y_train_s, df_train_s = load_split(TRAIN_SEASONS, SPREAD_FEATURES, "home_cover_surplus")
    spread_model = train_model(X_train_s, y_train_s, df_train_s, "spread_model")

    spread_path = MODELS_DIR / "spread_model.joblib"
    joblib.dump(spread_model, spread_path)
    print(f"Saved → {spread_path}")

    evaluate_model(spread_model, X_train_s, y_train_s, df_train_s, "home_cover_surplus", "Training set")

    if args.validate:
        X_val_s, y_val_s, df_val_s = load_split(VALIDATE_SEASONS, SPREAD_FEATURES, "home_cover_surplus")
        evaluate_model(spread_model, X_val_s, y_val_s, df_val_s, "home_cover_surplus",
                       f"Validation ({VALIDATE_SEASONS})")

    if args.test:
        print("\n⚠  RUNNING HOLDOUT — seal broken")
        X_test_s, y_test_s, df_test_s = load_split([TEST_SEASON], SPREAD_FEATURES, "home_cover_surplus")
        evaluate_model(spread_model, X_test_s, y_test_s, df_test_s, "home_cover_surplus",
                       f"TEST HOLDOUT ({TEST_SEASON})")

    # --- Total model ---
    print("\n" + "=" * 60)
    print("TOTAL MODEL (predicts: ou_surplus)")
    print("  ou_surplus = combined_score - closing_total")
    print("  positive → over hits; negative → under hits")
    print("=" * 60)
    X_train_t, y_train_t, df_train_t = load_split(TRAIN_SEASONS, TOTAL_FEATURES, "ou_surplus")
    total_model = train_model(X_train_t, y_train_t, df_train_t, "total_model")

    total_path = MODELS_DIR / "total_model.joblib"
    joblib.dump(total_model, total_path)
    print(f"Saved → {total_path}")

    evaluate_model(total_model, X_train_t, y_train_t, df_train_t, "ou_surplus", "Training set")

    if args.validate:
        X_val_t, y_val_t, df_val_t = load_split(VALIDATE_SEASONS, TOTAL_FEATURES, "ou_surplus")
        evaluate_model(total_model, X_val_t, y_val_t, df_val_t, "ou_surplus",
                       f"Validation ({VALIDATE_SEASONS})")

    if args.test:
        X_test_t, y_test_t, df_test_t = load_split([TEST_SEASON], TOTAL_FEATURES, "ou_surplus")
        evaluate_model(total_model, X_test_t, y_test_t, df_test_t, "ou_surplus",
                       f"TEST HOLDOUT ({TEST_SEASON})")

    print("\nDone. Run with --validate to see full validation metrics.")


if __name__ == "__main__":
    main()
