"""
Is the DUAL-signal structure actually earning its keep in college football?

The NFL filter fires only when two independent things agree: the margin model
disagrees with the opener, AND the movement model expects drift the same way.
That combination held up where either signal alone was weaker, which is the
entire reason the rule has that shape.

walk_forward.py already applies the same rule to CFB, so 47.6% is the dual-model
number. What has never been checked is whether the second signal CONTRIBUTES
there, or whether it is inherited NFL structure being carried along.

Three rules, identical walk-forward, identical bars:

  movement only        bet the side the line is predicted to move toward
  disagreement only    bet the side the margin model likes vs the opener
  both agree           the shipped NFL rule

If "both agree" beats each single signal in CFB, the structure is doing work.
If it does not, the rule is NFL furniture in a college house.

Bars are fixed rather than swept, so the three rules are compared on identical
game sets and the comparison is not contaminated by threshold selection.

Usage:
    python test_dual_signal_value.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model

OPEN, TGT = "week_open_spread_home", "week_spread_movement"


def load_cfb():
    from build_cfb_dataset import FEATURE_COLS, PRESEASON_COLS, INSEASON_COLS
    df = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")
    f = ([f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS + INSEASON_COLS
          + ["games_played", "rest_days"]]
         + ["neutral_site", "conference_game", "travel_miles", "elev_change", "is_dome"])
    return df, [c for c in f if c in df.columns]


def load_nfl():
    from model_line_movement import load_joined
    df = load_joined()
    return df, [c for c in SPREAD_FEATURES if c in df.columns]


def grade(s, bet_home):
    sur = s["home_margin"].values + s[OPEN].values
    live = np.abs(sur) > 1e-9
    won = np.where(bet_home, sur > 0, sur < 0)[live]
    w, l = int(won.sum()), int((~won).sum())
    clv = np.where(bet_home, -s[TGT].values, s[TGT].values)
    return w, l, float(np.mean(clv)) if len(clv) else 0.0


def run(name, df, feats, dbar, mbar):
    acc = {k: [0, 0, []] for k in ("movement", "disagreement", "both")}
    for S in sorted(s for s in df.season.unique() if s >= 2022):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        mv = train_model(tr[feats + [OPEN]], tr[TGT], tr, f"ds_{name}_{S}_mv")
        mm = train_model(tr[feats], tr["home_margin"], tr, f"ds_{name}_{S}_mg")
        t = te.copy()
        t["mv"] = mv.predict(t[feats + [OPEN]].fillna(0))
        t["dis"] = mm.predict(t[feats].fillna(0)) + t[OPEN].values

        agree = (t["dis"] > 0) == (t["mv"] < 0)
        rules = {
            # Each single signal picks its OWN side, so it is a fair standalone.
            "movement":     (t["mv"].abs() >= mbar, (t["mv"] < 0).values),
            "disagreement": (t["dis"].abs() >= dbar, (t["dis"] > 0).values),
            "both":         ((t["dis"].abs() >= dbar) & (t["mv"].abs() >= mbar) & agree,
                             (t["mv"] < 0).values),
        }
        for k, (mask, side) in rules.items():
            w, l, clv = grade(t[mask], side[mask.values])
            acc[k][0] += w
            acc[k][1] += l
            acc[k][2].append(clv * (w + l))

    print(f"\n{name}  (d>={dbar}, m>={mbar})")
    print(f"  {'rule':<16}{'n':>6}{'W-L':>10}{'win%':>8}{'ROI':>8}{'CLV':>8}")
    for k, (w, l, clvs) in acc.items():
        n = max(w + l, 1)
        print(f"  {k:<16}{w+l:>6}{f'{w}-{l}':>10}{w/n*100:>7.1f}%"
              f"{(w*(100/110)-l)/n*100:>+7.1f}%{sum(clvs)/n:>+8.2f}")


def main():
    nfl, nf = load_nfl()
    cfb, cf = load_cfb()
    # The NFL's shipped bars, and the bar the CFB sweep keeps selecting.
    for dbar, mbar in [(3.0, 0.5), (7.0, 0.5)]:
        run("NFL", nfl, nf, dbar, mbar)
        run("CFB", cfb, cf, dbar, mbar)
    print("\nbreak-even at -110 is 52.38%")


if __name__ == "__main__":
    main()
