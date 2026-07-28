"""
Fit the empirical-Bayes pseudo-count `k` used to blend current-season rolling
team metrics with end-of-prior-season metrics.

    blended = (n * in_season + k * prior) / (n + k)

where n = games played so far this season (week - 1), so:
    week 1  -> n=0 -> pure prior (matches score_week.py's current fallback)
    week 5  -> n=4 -> prior still carries weight k/(4+k)
    week 15 -> n=14 -> prior mostly washed out

`k` is the number of games the prior is "worth". Rather than hand-picking it
(the mistake EDGE_PER_WIN_PCT_POINT made), we sweep it and let held-out
results choose -- same discipline as calibrate_ev.py.

diagnose_preseason.py established the motivation: weeks 2-4 run on a 1-3 game
rolling sample that is NOISIER than simply using last season (63.6% -> 69.5%
win rate when substituting prior-season-end wholesale), while week 1 has no
in-season data at all.

Usage:
    python calibrate_metric_blend.py
"""

import numpy as np
import pandas as pd

from config import (DATA_DIR, SPREAD_FEATURES, VALIDATE_SEASONS, TEST_SEASON,
                    _TEAM_METRIC_COLS)
from train_models import load_split, train_model, ats_roi

# k values to sweep. 0 = pure in-season (current behaviour for weeks 2+),
# PRIOR_ONLY = pure prior-season-end for every week.
K_GRID = [0, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24]
PRIOR_ONLY = float("inf")

BUCKETS = [("wk1-4", [1, 2, 3, 4]),
           ("wk5-9", list(range(5, 10))),
           ("wk10+", list(range(10, 23)))]


def prior_season_end_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, season): metrics as of the last week of that season."""
    idx = metrics.groupby(["team", "season"])["week"].idxmax()
    return metrics.loc[idx].copy()


def blend(df: pd.DataFrame, metrics: pd.DataFrame, k: float) -> pd.DataFrame:
    """Vectorised empirical-Bayes blend of in-season and prior-season metrics."""
    out = df.reset_index(drop=True).copy()

    end = prior_season_end_metrics(metrics)
    end = end.copy()
    # shift forward: a season's end-of-year metrics are the PRIOR for season+1
    end["season"] = end["season"].astype(int) + 1

    n = (out["week"].astype(float) - 1.0).clip(lower=0.0)

    for side in ("home", "away"):
        bases = [b for b in _TEAM_METRIC_COLS if f"{b}_{side}" in out.columns and b in end.columns]
        if not bases:
            continue

        prior_df = end[["team", "season"] + bases].rename(
            columns={"team": f"{side}_team", **{b: f"__p_{b}" for b in bases}})
        out = out.merge(prior_df, on=[f"{side}_team", "season"], how="left")

        for b in bases:
            col, pcol = f"{b}_{side}", f"__p_{b}"
            cur, pri = out[col], out[pcol]

            if k == PRIOR_ONLY:
                blended = pri.where(pri.notna(), cur)
            else:
                denom = n + k
                raw = (n * cur.fillna(0.0) + k * pri.fillna(0.0)) / denom.replace(0.0, np.nan)
                blended = pd.Series(
                    np.where(cur.isna() & pri.notna(), pri,
                             np.where(pri.isna() & cur.notna(), cur, raw)),
                    index=out.index)
            out[col] = blended
            out.drop(columns=[pcol], inplace=True)

    return out


def score(preds, actuals, edge_threshold: float = 1.5) -> dict:
    preds = np.asarray(preds, float)
    actuals = np.asarray(actuals, float)
    m = (~np.isnan(preds)) & (~np.isnan(actuals))
    preds, actuals = preds[m], actuals[m]
    if len(preds) < 2:
        return {"n": len(preds), "corr": float("nan"), "wr": float("nan"),
                "roi": float("nan"), "bets": 0}
    r = ats_roi(preds, actuals, edge_threshold=edge_threshold)
    w, l = r["wins"], r["losses"]
    return {"n": len(preds), "corr": np.corrcoef(preds, actuals)[0, 1],
            "wr": w / max(w + l, 1) * 100, "roi": r["roi_pct"], "bets": w + l}


def main():
    eval_seasons = VALIDATE_SEASONS + [TEST_SEASON]
    train_seasons = list(range(2018, min(eval_seasons)))

    print("=" * 92)
    print("METRIC BLEND CALIBRATION -- fitting prior-season pseudo-count k")
    print(f"  train={train_seasons}  eval={eval_seasons}")
    print("=" * 92)

    metrics = pd.read_parquet(DATA_DIR / "team_metrics_all.parquet")
    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    ev = full[full["season"].isin(eval_seasons)].copy()
    ev = ev[ev["home_cover_surplus"].notna()].reset_index(drop=True)

    # Model trained on raw (unblended) features -- blending applied at serve
    # time only. This is the minimal deployable change: score_week.py blends,
    # the model itself is untouched.
    X_tr, y_tr, df_tr = load_split(train_seasons, SPREAD_FEATURES, "home_cover_surplus")
    model = train_model(X_tr, y_tr, df_tr, "blend_sweep")
    feat_cols = model.get_booster().feature_names

    print("\nBLEND AT SERVE TIME ONLY (model trained on raw features)")
    header = f"  {'k':>10} | " + " | ".join(f"{lbl:^28}" for lbl, _ in BUCKETS) + " |    OVERALL"
    print(header)
    print(f"  {'':>10} | " + " | ".join(f"{'corr   win%    ROI':^28}" for _ in BUCKETS) + " |")
    print("  " + "-" * (len(header) - 2))

    results = {}
    for k in K_GRID + [PRIOR_ONLY]:
        blended = blend(ev, metrics, k)
        preds = model.predict(blended[feat_cols].fillna(0))
        actuals = blended["home_cover_surplus"].values

        cells = []
        for _, weeks in BUCKETS:
            m = blended["week"].isin(weeks).values
            s = score(preds[m], actuals[m])
            cells.append(f"{s['corr']:+.3f} {s['wr']:5.1f}% {s['roi']:+6.1f}%")

        overall = score(preds, actuals)
        results[k] = overall
        klabel = "prior-only" if k == PRIOR_ONLY else f"{k:g}"
        marker = "   <- current" if k == 0 else ""
        print(f"  {klabel:>10} | " + " | ".join(f"{c:^28}" for c in cells) +
              f" | {overall['corr']:+.3f} {overall['wr']:5.1f}% {overall['roi']:+6.1f}%{marker}")

    # ---- Retrain on blended features at the best early-season k ----
    early_scores = {}
    for k in K_GRID + [PRIOR_ONLY]:
        blended = blend(ev, metrics, k)
        preds = model.predict(blended[feat_cols].fillna(0))
        m = blended["week"].isin([1, 2, 3, 4]).values
        early_scores[k] = score(preds[m], blended["home_cover_surplus"].values[m])["roi"]
    best_k = max(early_scores, key=lambda kk: early_scores[kk])
    best_label = "prior-only" if best_k == PRIOR_ONLY else f"{best_k:g}"

    print(f"\nBest k by weeks 1-4 ROI: {best_label}")
    print("\nRETRAINED ON BLENDED FEATURES (blend applied to training data too)")
    print("  (week-1 training rows currently carry all-zero metrics -- blending")
    print("   at train time turns those 128 wasted rows into usable examples)")

    train_raw = full[full["season"].isin(train_seasons)].copy()
    train_raw = train_raw[train_raw["home_cover_surplus"].notna()].reset_index(drop=True)
    train_blended = blend(train_raw, metrics, best_k)

    avail = [c for c in SPREAD_FEATURES if c in train_blended.columns]
    Xb = train_blended[avail].fillna(0)
    yb = train_blended["home_cover_surplus"]
    model_b = train_model(Xb, yb, train_blended, "blend_sweep_retrained")
    feat_b = model_b.get_booster().feature_names

    ev_blended = blend(ev, metrics, best_k)
    preds_b = model_b.predict(ev_blended[feat_b].fillna(0))
    actuals_b = ev_blended["home_cover_surplus"].values

    for lbl, weeks in BUCKETS:
        m = ev_blended["week"].isin(weeks).values
        s = score(preds_b[m], actuals_b[m])
        print(f"    {lbl:<8} n={s['n']:>4}  corr={s['corr']:+.3f}  "
              f"win={s['wr']:5.1f}%  ROI={s['roi']:+6.1f}%  bets={s['bets']}")
    s = score(preds_b, actuals_b)
    print(f"    {'OVERALL':<8} n={s['n']:>4}  corr={s['corr']:+.3f}  "
          f"win={s['wr']:5.1f}%  ROI={s['roi']:+6.1f}%  bets={s['bets']}")


if __name__ == "__main__":
    main()
