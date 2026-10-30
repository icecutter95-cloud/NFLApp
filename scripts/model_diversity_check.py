"""
Would a third model help? No -- the two we have already make the same errors.

THE QUESTION
Asked whether a Monte Carlo game simulator -- the kind sold publicly as
"10,000 simulations per game" -- is worth building as a third model for spreads
and totals, alongside the residual and margin models.

The ensemble argument for a third model requires that its errors DECORRELATE
from the existing ones. This measures whether that premise holds, using the two
models already in production as the test case: if a second model adds nothing
to the first, a third built from the same play-by-play will not either.

WHAT IT FOUND (2026-10-30, walk-forward 2022-2025, 1,716 games)

    error correlation, residual vs margin model    +0.949
    agreement on side                              72.1% of 1,138 games

    at |edge| >= 1.5
      residual alone      430-359   54.5%
      margin alone        479-408   54.0%
      50/50 blend         447-368   54.8%

+0.949 means they are not two views of a game, they are one view with two
heads. The blend gains 0.3 points on the better of them, which is noise at
n=815. A simulator parameterised from the same inputs -- EPA, pace, success
rate, red-zone conversion -- would land in the same place, because the
information is the same and only the functional form differs.

TWO MORE REASONS, INDEPENDENT OF THE FIRST

1. A final margin is a sum over many plays, so by the central limit theorem it
   is near-normal with a well-known sigma (~13.5 points). For a single scalar
   market that makes P(cover) = Phi((E[margin] + line) / sigma): a 10,000-run
   simulation is an expensive normal CDF. A distribution only becomes
   irreducible for NON-LINEAR questions about game state -- a named player
   scoring, an alternate line, a correlated parlay -- which is why the same
   argument does NOT rule a simulator out for props.

2. Our edge is barely graded. Win rate by bar runs 54.7 / 54.5 / 54.9 at
   1.0 / 1.5 / 2.5 -- flat -- then 56.7% at 3.5 on n=434 (p=0.006). One raised
   bucket is not a gradient, and the isotonic calibrator fitted on honest
   out-of-sample outcomes still returns 0.5238 at EVERY edge size, having found
   no monotone relationship at all. The headline product of a simulator is
   graded probability, and we have very little resolution to grade with --
   though "none" would overstate it, and the 3.5 bar is worth watching.

WHAT WOULD CHANGE THE ANSWER
A simulator is not merely a different functional form if its INPUTS differ --
injury-adjusted unit strength propagated through drive outcomes, score-
dependent pace, a backup quarterback reshaping the whole distribution rather
than nudging a coefficient. That would be new information. But the market
prices a quarterback's absence to within 5% (see scripts/build_qb_quality.py),
so that door is open and narrow.

Usage:
    python scripts/model_diversity_check.py
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

from config import DATA_DIR, MODELS_DIR

SEASONS = (2022, 2023, 2024, 2025)
BAR = 1.5


def frame() -> pd.DataFrame:
    d = pd.read_parquet(DATA_DIR / "historical_dataset.parquet")
    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    kd = pd.to_datetime(oc["kick_date"])
    oc["season"] = kd.dt.year
    # January and February games belong to the PRIOR NFL season.
    oc.loc[kd.dt.month <= 2, "season"] -= 1
    m = d.merge(oc[["season", "home_team", "away_team", "week_open_spread_home"]],
                on=["season", "home_team", "away_team"], how="inner")
    m["resid_open"] = m.home_margin + m.week_open_spread_home
    return m.dropna(subset=["resid_open"])


def walk(m, feats, target):
    """Train on all prior seasons, predict the held-out one."""
    import xgboost as xgb
    out = []
    for s in SEASONS:
        tr, te = m[m.season < s], m[m.season == s]
        if not len(te):
            continue
        mod = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03,
                               subsample=0.8, colsample_bytree=0.8, random_state=0)
        mod.fit(tr[feats].fillna(0), tr[target])
        out.append(pd.DataFrame(
            {"i": te.index, "pred": mod.predict(te[feats].fillna(0)),
             "resid_open": te.resid_open.values,
             "open": te.week_open_spread_home.values}))
    return pd.concat(out).set_index("i")


def rate(j, col, label, bar=BAR):
    s = j[j[col].abs() >= bar]
    hit = np.sign(s[col]) == np.sign(s.resid_open)
    w, n = int(hit.sum()), len(s)
    print(f"  {label:22} {w:3d}-{n - w:<3d} {w / n:5.1%} "
          f"(p={binomtest(w, n).pvalue:.3f}) n={n}")


def main():
    m = frame()
    rf = [c for c in joblib.load(MODELS_DIR / "nfl_residual_features.joblib")
          if c in m.columns]
    mf = [c for c in joblib.load(MODELS_DIR / "margin_features.joblib")
          if c in m.columns]
    print(f"{len(m)} games | residual {len(rf)} features | margin {len(mf)}")

    A = walk(m, rf, "resid_open")
    B = walk(m, mf, "home_margin")
    j = A.join(B[["pred"]], rsuffix="_m").dropna()
    # The margin model's edge against the opener, so both are on one scale.
    j["edge_resid"] = j.pred
    j["edge_margin"] = j.pred_m + j.open
    j = j[j.resid_open != 0]

    err_r = j.edge_resid - j.resid_open
    err_m = j.edge_margin - j.resid_open
    corr = float(np.corrcoef(err_r, err_m)[0, 1])
    agree = float((np.sign(j.edge_resid) == np.sign(j.edge_margin)).mean())
    print(f"\nerror correlation between the two models: {corr:+.3f}")
    print(f"agreement on side: {agree:.1%} of {len(j)} games")
    if corr > 0.8:
        print("  -> near-duplicates. The ensemble case for a THIRD model built")
        print("     from the same inputs fails here, before it is built.")

    print(f"\nat |edge| >= {BAR}")
    rate(j, "edge_resid", "residual alone")
    rate(j, "edge_margin", "margin alone")
    j["blend"] = (j.edge_resid + j.edge_margin) / 2
    rate(j, "blend", "50/50 blend")

    print("\nis the edge GRADED? win rate by bar, residual model")
    for bar in (1.0, 1.5, 2.5, 3.5):
        rate(j, "edge_resid", f"|edge| >= {bar}", bar)
    print("  -> flat from 1.0 to 2.5 (54.7 / 54.5 / 54.9), then 56.7% at 3.5")
    print("     on n=434. So the edge is mostly UNGRADED with a possible lift")
    print("     in the top bar -- one bucket, not a gradient. A simulator sells")
    print("     graded probability; we have little resolution to grade with,")
    print("     but 'none' would overstate it.")

    cal = MODELS_DIR / "spread_calibration.joblib"
    if cal.exists():
        c = joblib.load(cal)
        vals = [float(c.predict([e])[0]) for e in (1.0, 2.5, 4.0, 6.0)]
        print(f"\nisotonic calibrator at edges 1.0/2.5/4.0/6.0: "
              f"{', '.join(f'{v:.4f}' for v in vals)}")
        if max(vals) - min(vals) < 1e-6:
            print("  -> flat at every edge size; it found no monotone")
            print("     relationship between edge magnitude and win rate.")


if __name__ == "__main__":
    main()
