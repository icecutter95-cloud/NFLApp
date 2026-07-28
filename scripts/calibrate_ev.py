"""
Calibrate the edge -> win-probability mapping used to compute displayed EV%.

Previously this was a guessed linear heuristic (EDGE_PER_WIN_PCT_POINT = 0.03,
i.e. "each point of edge ~ 3% win prob", capped at 85%) that was never actually
tuned against real results -- any edge >= ~10.9 points hit the cap and produced
an identical +62.3% EV regardless of whether the real edge was 11 or 30 points.

This script fits an isotonic regression directly on real, honest out-of-sample
backtest outcomes -- exactly the same train/eval season split methodology as
backtest.py (always train strictly on seasons before the evaluation window,
independent of whatever config.py's TRAIN_SEASONS is set to for production).
Isotonic regression only assumes win probability is non-decreasing in edge
size (a safe assumption) and won't extrapolate wildly into thin-sample tails
the way a fitted linear/polynomial curve could.

Usage:
    python calibrate_ev.py
"""

import numpy as np
import joblib

from config import MODELS_DIR, SPREAD_FEATURES, TOTAL_FEATURES, VALIDATE_SEASONS, TEST_SEASON
from train_models import load_split, train_model

from sklearn.isotonic import IsotonicRegression

MIN_WIN_PROB = 110 / 210   # can never be below the break-even implied probability at -110
MAX_WIN_PROB = 0.90        # avoid overconfident extrapolation into the sparse high-edge tail

BUCKET_EDGES = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7, 10, 15, 20, 25, 999]


def honest_predictions(feature_cols: list, target_col: str, model_label: str):
    """Train strictly on seasons before the eval window, predict on the eval
    window. Same decoupling as backtest.py -- never contaminated by whatever
    TRAIN_SEASONS is set to for the production model."""
    eval_seasons = VALIDATE_SEASONS + [TEST_SEASON]
    train_seasons = list(range(2018, min(eval_seasons)))

    X_tr, y_tr, df_tr = load_split(train_seasons, feature_cols, target_col)
    model = train_model(X_tr, y_tr, df_tr, model_label)

    X_val, y_val, df_val = load_split(eval_seasons, feature_cols, target_col)
    preds = model.predict(X_val)
    return preds, y_val.values


def fit_calibration(preds, actuals, min_edge: float = 0.5, label: str = "") -> IsotonicRegression:
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    mask = (~np.isnan(preds)) & (~np.isnan(actuals))
    preds, actuals = preds[mask], actuals[mask]

    edge = np.abs(preds)
    push = np.abs(actuals) < 0.01
    won = ((preds > 0) & (actuals > 0)) | ((preds < 0) & (actuals < 0))

    keep = (edge >= min_edge) & (~push)
    edge, won = edge[keep], won[keep].astype(float)

    print(f"\n{label} calibration sample: {len(edge)} bets (edge >= {min_edge}, pushes excluded)")
    print(f"  {'Edge range':<16} {'N':>6} {'Win Rate':>10}")
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        m = (edge >= lo) & (edge < hi)
        n = int(m.sum())
        if n == 0:
            continue
        hi_label = f"{hi}" if hi < 999 else "inf"
        print(f"  [{lo:>5}, {hi_label:>5})  {n:>6}  {won[m].mean()*100:>9.1f}%")

    iso = IsotonicRegression(y_min=MIN_WIN_PROB, y_max=MAX_WIN_PROB,
                             out_of_bounds="clip", increasing=True)
    iso.fit(edge, won)
    return iso


def main():
    print("=" * 60)
    print("SPREAD EV calibration")
    print("=" * 60)
    spread_preds, spread_actuals = honest_predictions(
        SPREAD_FEATURES, "home_cover_surplus", "spread_calib_diagnostic")
    spread_iso = fit_calibration(spread_preds, spread_actuals, label="SPREAD")
    spread_path = MODELS_DIR / "spread_calibration.joblib"
    joblib.dump(spread_iso, spread_path)
    print(f"\nSaved -> {spread_path}")

    # Sanity check: show what the OLD guessed formula vs NEW calibrated curve
    # produce at a few representative edge sizes.
    print("\n  Old guess vs new calibration (win prob):")
    implied = 110 / 210
    for e in [1, 2, 3, 5, 8, 11, 15, 20, 25, 30]:
        old = min(implied + e * 0.03, 0.85)
        new = float(spread_iso.predict([e])[0])
        print(f"    edge={e:>4}: old={old*100:5.1f}%   new={new*100:5.1f}%")

    print("\n" + "=" * 60)
    print("TOTAL EV calibration (for future use once totals model has signal)")
    print("=" * 60)
    total_preds, total_actuals = honest_predictions(
        TOTAL_FEATURES, "ou_surplus", "total_calib_diagnostic")
    total_iso = fit_calibration(total_preds, total_actuals, label="TOTAL")
    total_path = MODELS_DIR / "total_calibration.joblib"
    joblib.dump(total_iso, total_path)
    print(f"\nSaved -> {total_path}")


if __name__ == "__main__":
    main()
