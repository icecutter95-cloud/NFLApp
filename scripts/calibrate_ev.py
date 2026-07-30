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

# The curve is FIT on one set of seasons and VALIDATED on another it never saw.
# Fitting and reporting on the same seasons is how a meaningless curve ends up
# displayed as a confident number: isotonic regression chooses which buckets to
# pool by looking at those very outcomes, so a flattering pooled bucket is
# selection on the test set, not a finding.
#
# Measured (fit 2023-2024, applied to 2025): corr(predicted win prob, actual
# win) = +0.082, and two of five edge bands landed on the OPPOSITE side of
# break-even from their prediction -- e.g. the 2.5-3.5 band was fitted at 57.6%
# and actually went 48.6%. The curve had been driving a displayed "+13.4% EV".
CALIBRATION_FIT_SEASONS = [2023, 2024]
CALIBRATION_HOLDOUT_SEASONS = [2025]

# Held-out correlation the curve must clear to be shipped as-is. Below this we
# ship a FLAT break-even curve, so the UI reports 0% EV rather than a number
# that cannot be justified out of sample.
MIN_HOLDOUT_CORR = 0.15


def flat_calibrator() -> IsotonicRegression:
    """A curve that always returns break-even -> EV 0% at every edge.

    Used when the fitted curve fails out-of-sample validation. Displaying 0%
    is the honest output when edge magnitude demonstrably carries no
    information about win probability.
    """
    iso = IsotonicRegression(y_min=MIN_WIN_PROB, y_max=MIN_WIN_PROB,
                             out_of_bounds="clip", increasing=True)
    iso.fit([0.0, 100.0], [MIN_WIN_PROB, MIN_WIN_PROB])
    return iso


def validate_calibration(iso: IsotonicRegression, preds, actuals, label: str) -> bool:
    """Does the curve predict outcomes on seasons it was never fit on?"""
    preds = np.asarray(preds, float)
    actuals = np.asarray(actuals, float)
    mask = (~np.isnan(preds)) & (~np.isnan(actuals)) & (np.abs(actuals) >= 0.01)
    edge = np.abs(preds[mask])
    won = (((preds > 0) & (actuals > 0)) | ((preds < 0) & (actuals < 0)))[mask].astype(float)
    edge, won = edge[edge >= 0.5], won[edge >= 0.5]

    if len(edge) < 50:
        print(f"  {label}: only {len(edge)} holdout bets — cannot validate, shipping FLAT")
        return False

    predicted = iso.predict(edge)
    if predicted.std() < 1e-9:
        print(f"  {label}: fitted curve is already flat")
        return False

    corr = float(np.corrcoef(predicted, won)[0, 1])
    ok = corr >= MIN_HOLDOUT_CORR
    print(f"  {label}: holdout corr(predicted, actual) = {corr:+.3f} on {len(edge)} bets "
          f"-> {'PASS, shipping fitted curve' if ok else 'FAIL, shipping FLAT break-even curve'}")
    return ok


def honest_predictions(feature_cols: list, target_col: str, model_label: str,
                       seasons: list | None = None):
    """Train strictly on seasons before the eval window, predict on `seasons`.

    Same decoupling as backtest.py -- never contaminated by whatever
    TRAIN_SEASONS is set to for the production model. `seasons` lets the caller
    request the calibration-fit slice and the holdout slice separately, so the
    curve is never validated on the data it was fit to.
    """
    eval_seasons = VALIDATE_SEASONS + [TEST_SEASON]
    train_seasons = list(range(2018, min(eval_seasons)))

    X_tr, y_tr, df_tr = load_split(train_seasons, feature_cols, target_col)
    model = train_model(X_tr, y_tr, df_tr, model_label)

    X_val, y_val, df_val = load_split(seasons or eval_seasons, feature_cols, target_col)
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
    for label, feats, target, out_name in [
        ("SPREAD", SPREAD_FEATURES, "home_cover_surplus", "spread_calibration.joblib"),
        ("TOTAL",  TOTAL_FEATURES,  "ou_surplus",         "total_calibration.joblib"),
    ]:
        print("=" * 64)
        print(f"{label} EV calibration")
        print(f"  fit on {CALIBRATION_FIT_SEASONS}, validated on {CALIBRATION_HOLDOUT_SEASONS}")
        print("=" * 64)

        fit_p, fit_a = honest_predictions(feats, target, f"{label.lower()}_cal_fit",
                                          CALIBRATION_FIT_SEASONS)
        iso = fit_calibration(fit_p, fit_a, label=label)

        ho_p, ho_a = honest_predictions(feats, target, f"{label.lower()}_cal_ho",
                                        CALIBRATION_HOLDOUT_SEASONS)
        print()
        passed = validate_calibration(iso, ho_p, ho_a, label)
        final = iso if passed else flat_calibrator()

        path = MODELS_DIR / out_name
        joblib.dump(final, path)
        print(f"  Saved -> {path.name}")

        print("  edge -> win prob -> EV as shipped:")
        payout = 100 / 110
        for e in [1, 2, 3, 5, 8, 12, 20]:
            wp = float(final.predict([e])[0])
            print(f"    edge {e:>4}: winprob {wp*100:5.2f}%   EV {(wp*payout-(1-wp))*100:+6.1f}%")
        print()


if __name__ == "__main__":
    main()
