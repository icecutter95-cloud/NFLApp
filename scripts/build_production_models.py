"""
Train and save the production artifacts for all four markets.

One script so the four models cannot drift apart in how they are fitted, and so
what ships is traceable to what was tested.

  NFL spreads   RESIDUAL model -- predict home_margin + week_open_spread_home.
                Best evidenced thing we have: beat the previous approach at
                every threshold, 713 bets at bar 1.5, cluster CI [53.4, 60.8]
                clears break-even, 0/25 permutations reached it (+3.5 sd).
  NFL totals    movement model, |predicted movement| >= 1.25. Unchanged. Shown
                as qualifying by request, but it FAILS permutation (p = 0.231)
                and its noise floor sits above break-even, so the caveat in the
                UI is doing real work.
  CFB spreads   movement + margin, direction agreement. 54.7% walk-forward but
                p = 0.077. Projections shown, nothing flagged.
  CFB totals    residual model. The better of two poor options -- the movement
                version scored BELOW its own noise floor. p = 0.192. Projections
                shown, nothing flagged.

Also saved: the previous NFL spread rule's margin model, so the frozen
production rule can run as a SHADOW arm all season. It costs nothing to log
both, and by midseason there is a real answer about which was right rather than
a backtest.

Every model is trained on all seasons with line coverage. Evaluation used
strict walk-forward; the shipped artifact uses everything, which is the same
pattern the NFL models already follow.

Usage:
    python build_production_models.py
"""

import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, MODELS_DIR, SPREAD_FEATURES, TOTAL_FEATURES
from train_models import train_model

OPEN_S = "week_open_spread_home"
OPEN_T = "week_open_total"


def save(model, feats, name):
    joblib.dump(model, MODELS_DIR / f"{name}_model.joblib")
    joblib.dump(feats, MODELS_DIR / f"{name}_features.joblib")
    print(f"  saved {name}_model.joblib  ({len(feats)} features)")


def nfl():
    from model_line_movement import load_joined
    d = load_joined().dropna(subset=[OPEN_S, "home_margin"]).copy()
    d["resid_open"] = d["home_margin"] + d[OPEN_S]
    feats = [c for c in SPREAD_FEATURES if c in d.columns] + [OPEN_S]

    print(f"\nNFL SPREADS (residual) — {len(d)} games")
    m = train_model(d[feats], d["resid_open"], d, "nfl_residual")
    save(m, feats, "nfl_residual")

    # Shadow arm: the frozen rule's margin model, market-free by design.
    mf = [c for c in SPREAD_FEATURES if c in d.columns]
    print(f"NFL SPREADS (shadow: frozen rule's margin model)")
    save(train_model(d[mf], d["home_margin"], d, "nfl_margin_shadow"), mf,
         "nfl_margin_shadow")

    # Totals unchanged — the live rule stays exactly as it is.
    t = load_joined()
    t = t.dropna(subset=[OPEN_T, "week_total_movement"])
    tf = [c for c in TOTAL_FEATURES if c in t.columns] + [OPEN_T]
    print(f"NFL TOTALS (movement) — {len(t)} games")
    save(train_model(t[tf], t["week_total_movement"], t, "nfl_total_movement"),
         tf, "nfl_total_movement")


def cfb():
    from build_cfb_dataset import FEATURE_COLS, PRESEASON_COLS, INSEASON_COLS
    base = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")
    feats = ([f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS + INSEASON_COLS
              + ["games_played", "rest_days"]]
             + ["neutral_site", "conference_game", "travel_miles",
                "elev_change", "is_dome"])
    feats = [c for c in feats if c in base.columns]

    d = base.dropna(subset=[OPEN_S, "week_spread_movement", "home_margin"])
    print(f"\nCFB SPREADS — {len(d)} games")
    save(train_model(d[feats + [OPEN_S]], d["week_spread_movement"], d, "cfb_movement"),
         feats + [OPEN_S], "cfb_movement")
    save(train_model(d[feats], d["home_margin"], d, "cfb_margin"), feats, "cfb_margin")

    # Totals: join the separately-backfilled totals lines.
    oc = []
    for s in range(2020, 2026):
        p = DATA_DIR / f"cfb_open_close_totals_{s}.parquet"
        if p.exists():
            o = pd.read_parquet(p)
            o["season"] = s
            oc.append(o)
    oc = pd.concat(oc, ignore_index=True)
    oc = oc[oc.n_snapshots_week >= 2].rename(
        columns={"week_open_spread_home": OPEN_T, "closing_spread_home": "closing_total"})
    t = base.merge(oc[["season", "home_team", "away_team", "kick_date", OPEN_T]],
                   on=["season", "home_team", "away_team", "kick_date"], how="inner")
    t["resid_open"] = t["total_points"] - t[OPEN_T]
    t = t.dropna(subset=[OPEN_T, "resid_open"])
    print(f"CFB TOTALS (residual) — {len(t)} games")
    save(train_model(t[feats + [OPEN_T]], t["resid_open"], t, "cfb_total_residual"),
         feats + [OPEN_T], "cfb_total_residual")


def main():
    nfl()
    cfb()
    print("\nnext: python upload_models_to_storage.py")


if __name__ == "__main__":
    main()
