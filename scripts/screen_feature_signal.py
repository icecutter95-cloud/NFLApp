"""
Screen every team metric for how much of it is signal vs noise, BEFORE adding
any more features.

Two independent measures, because they answer different questions:

1. SPLIT-HALF REPEATABILITY (r)
   Correlate each team's first-half-of-season average against its second half,
   pooled across seasons. If a metric doesn't predict *itself*, it is unlikely
   to predict anything else. Run on RAW per-game values, never the rolled L4/L8
   columns -- rolling smooths the series and inflates autocorrelation, which
   would make noise look like signal.

2. ABLATION (delta corr / MAE)
   Drop the metric from the live feature set, retrain, and measure held-out
   margin prediction. A metric can be repeatable but redundant, or barely
   repeatable but still carry something; only ablation settles its net worth.

Motivation: turnover_margin sat in the model from day one and was never
questioned. It scored r=0.196 here, and removing it improved held-out margin
correlation from 0.374 to 0.383 -- it had been actively harmful. This screens
the rest of the feature set for the same problem.

Usage:
    python screen_feature_signal.py             # repeatability only (fast)
    python screen_feature_signal.py --ablate    # also run ablation (slow)
"""

import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES, ALL_HISTORICAL_SEASONS
from compute_metrics import build_game_level, METRIC_COLS
from train_models import train_model

TRAIN_SEASONS = list(range(2018, 2023))
EVAL_SEASONS = [2023, 2024, 2025]
SPLIT_WEEK = 9          # weeks <= 9 vs > 9
MIN_GAMES_PER_HALF = 3


def split_half_repeatability(seasons: list) -> pd.DataFrame:
    """Per-metric correlation between a team's first- and second-half means."""
    pairs = {m: [[], []] for m in METRIC_COLS}

    for season in seasons:
        try:
            gl = build_game_level(season, verbose=False)
        except Exception as exc:
            print(f"  skipping {season}: {exc}")
            continue

        first = gl[gl["week"] <= SPLIT_WEEK]
        second = gl[gl["week"] > SPLIT_WEEK]
        for metric in METRIC_COLS:
            if metric not in gl.columns:
                continue
            a = first.groupby("team")[metric].agg(["mean", "count"])
            b = second.groupby("team")[metric].agg(["mean", "count"])
            a = a[a["count"] >= MIN_GAMES_PER_HALF]
            b = b[b["count"] >= MIN_GAMES_PER_HALF]
            common = a.index.intersection(b.index)
            if len(common) < 10:
                continue
            pairs[metric][0].extend(a.loc[common, "mean"].tolist())
            pairs[metric][1].extend(b.loc[common, "mean"].tolist())

    rows = []
    for metric, (x, y) in pairs.items():
        if len(x) < 30:
            continue
        x, y = np.array(x, float), np.array(y, float)
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 30:
            continue
        rows.append({"metric": metric,
                     "split_half_r": float(np.corrcoef(x[ok], y[ok])[0, 1]),
                     "n_team_seasons": int(ok.sum())})
    return pd.DataFrame(rows).sort_values("split_half_r")


def ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Drop each metric's home/away columns; measure held-out margin change."""
    base_feats = [c for c in SPREAD_FEATURES if c in df.columns]
    tr = df[df.season.isin(TRAIN_SEASONS)].dropna(subset=["home_margin"])
    va = df[df.season.isin(EVAL_SEASONS)].dropna(subset=["home_margin"])
    actual = va["home_margin"].values

    def score(feats, label):
        m = train_model(tr[feats].fillna(0), tr["home_margin"], tr, label)
        p = m.predict(va[feats].fillna(0))
        return float(np.corrcoef(p, actual)[0, 1]), float(mean_absolute_error(actual, p))

    base_corr, base_mae = score(base_feats, "base")
    print(f"\n  baseline: corr={base_corr:+.3f}  MAE={base_mae:.2f}  ({len(base_feats)} feats)")

    # Group feature columns by the underlying metric they came from.
    groups: dict = {}
    for col in base_feats:
        if not (col.endswith("_home") or col.endswith("_away")):
            continue
        stem = col.rsplit("_", 1)[0]
        for suffix in ("_L4", "_L8", "_season"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        groups.setdefault(stem, []).append(col)

    rows = []
    for stem, cols in sorted(groups.items()):
        kept = [c for c in base_feats if c not in cols]
        c, m = score(kept, f"drop_{stem}")
        rows.append({"metric": stem, "dropped_cols": len(cols),
                     "corr_without": c, "d_corr": c - base_corr,
                     "mae_without": m, "d_mae": m - base_mae})
    return pd.DataFrame(rows).sort_values("d_corr", ascending=False)


def main():
    print("=" * 78)
    print("FEATURE SIGNAL SCREEN")
    print("=" * 78)
    print("\n1. SPLIT-HALF REPEATABILITY (raw per-game values, all seasons)")
    print("   Does the metric predict ITSELF from one half of a season to the next?\n")
    rep = split_half_repeatability(ALL_HISTORICAL_SEASONS)
    print(f"   {'metric':<34}{'r':>8}{'n':>7}   verdict")
    for _, r in rep.iterrows():
        v = ("NOISE — barely repeats" if r.split_half_r < 0.25 else
             "weak" if r.split_half_r < 0.45 else
             "solid" if r.split_half_r < 0.65 else "strong")
        print(f"   {r.metric:<34}{r.split_half_r:>+8.3f}{r.n_team_seasons:>7}   {v}")

    if "--ablate" not in sys.argv:
        print("\n(run with --ablate to also measure each metric's net contribution)")
        return

    print("\n\n2. ABLATION — drop each metric, retrain, measure held-out margin")
    print("   d_corr > 0 means the model got BETTER without it.\n")
    df = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    abl = ablation(df)
    print(f"\n   {'metric':<34}{'d_corr':>9}{'d_MAE':>9}   verdict")
    for _, r in abl.iterrows():
        v = "HARMFUL — drop it" if r.d_corr > 0.004 else ("neutral" if r.d_corr > -0.004 else "keep")
        print(f"   {r.metric:<34}{r.d_corr:>+9.4f}{r.d_mae:>+9.3f}   {v}")


if __name__ == "__main__":
    main()
