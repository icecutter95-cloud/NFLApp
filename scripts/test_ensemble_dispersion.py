"""
Confidence from model AGREEMENT, not prediction size.

We measured that the size of a prediction is a bad way to rank bets: as the
movement model's predicted magnitude rose, CLV rose and win rate FELL. Yet
magnitude is still what every threshold in this project uses.

This tests a different confidence signal. Train many versions of the SAME
thesis -- bootstrap resamples of the training rows, random feature subsets,
different seeds -- and ask whether they agree. Two games can both show a
2-point edge while twenty models unanimously back one and split 11-9 on the
other. Those are not equally good bets.

Note this is NOT the existing "both signals agree" rule. That asks whether two
DIFFERENT models (margin and movement) point the same way. This asks whether
independently trained versions of ONE model do -- which is a measure of
estimation stability rather than corroboration.

Built on the residual model (predict margin + opener), since that now
outperforms the production approach at every threshold.

The comparison is deliberately volume-matched. Any selector looks better if it
simply bets less, so each is asked for the SAME number of bets and judged on
what it does with them:

    magnitude    rank by |mean prediction|          (what we do today)
    agreement    rank by share of models on one side
    t-ratio      rank by |mean| / dispersion        (both at once)

Usage:
    python test_ensemble_dispersion.py [n_models]
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import SPREAD_FEATURES
from train_models import train_model
from test_residual_model import load, grade, OPEN


def ensemble_walk(df, feats, n_models=20, seed=5):
    """Walk-forward, but fit an ensemble per fold and keep the spread of it."""
    rng = np.random.default_rng(seed)
    out = []
    for S in sorted(s for s in df.season.unique() if s >= 2022):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        f_all = feats + [OPEN]
        preds = np.zeros((len(te), n_models))
        for i in range(n_models):
            # Two independent sources of variation: which games the model sees,
            # and which features it may use. Seed alone would understate
            # disagreement, since XGBoost is close to deterministic here.
            rows = rng.integers(0, len(tr), len(tr))
            k = max(8, int(len(feats) * 0.7))
            sub = list(rng.choice(feats, size=k, replace=False)) + [OPEN]
            b = tr.iloc[rows]
            m = train_model(b[sub], b["resid_open"], b, f"ens_{S}_{i}")
            preds[:, i] = m.predict(te[sub].fillna(0))

        t = te.copy()
        t["mean_pred"] = preds.mean(axis=1)
        t["disp"] = preds.std(axis=1)
        # Share of the ensemble on the majority side: 0.5 = a coin flip among
        # the models, 1.0 = unanimous.
        share_home = (preds > 0).mean(axis=1)
        t["agree"] = np.maximum(share_home, 1 - share_home)
        t["side_home"] = share_home > 0.5
        # Mean over dispersion -- how far from zero relative to the ensemble's
        # own uncertainty.
        t["t_ratio"] = t["mean_pred"] / t["disp"].replace(0, np.nan)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def topn(ev, col, n, use_abs=True):
    v = ev[col].abs() if use_abs else ev[col]
    return ev.loc[v.nlargest(n).index]


def main():
    n_models = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    df, feats = load()
    print(f"NFL {len(df)} games | ensemble of {n_models} residual models per fold")
    ev = ensemble_walk(df, feats, n_models)

    print(f"\nensemble behaviour: mean dispersion {ev.disp.mean():.2f} pts, "
          f"unanimous on {(ev.agree == 1).mean()*100:.0f}% of games")
    print(f"  corr(|mean pred|, agreement) = "
          f"{np.corrcoef(ev.mean_pred.abs(), ev.agree)[0,1]:+.3f}")

    print(f"\nVOLUME-MATCHED SELECTOR COMPARISON")
    print(f"  {'bets':<8}{'magnitude':>22}{'agreement':>22}{'t-ratio':>22}")
    for n in (250, 400, 550, 700):
        if n > len(ev):
            continue
        cells = []
        for col, ab in [("mean_pred", True), ("agree", False), ("t_ratio", True)]:
            s = topn(ev, col, n, ab)
            w, l = grade(s, s["side_home"].values)
            tot = max(w + l, 1)
            cells.append(f"{w}-{l}  {w/tot*100:.1f}%")
        print(f"  {n:<8}{cells[0]:>22}{cells[1]:>22}{cells[2]:>22}")

    # Does agreement add anything ON TOP of magnitude? Hold magnitude fixed and
    # split by agreement -- the only way to see incremental value.
    print(f"\nAGREEMENT WITHIN A FIXED MAGNITUDE BAND")
    print(f"  {'|pred| band':<14}{'low agreement':>24}{'high agreement':>24}")
    for lo, hi in [(0.5, 1.5), (1.5, 3.0), (3.0, 99)]:
        band = ev[(ev.mean_pred.abs() >= lo) & (ev.mean_pred.abs() < hi)]
        if len(band) < 80:
            continue
        cut = band.agree.median()
        cells = []
        for m in (band.agree <= cut, band.agree > cut):
            s = band[m]
            w, l = grade(s, s["side_home"].values)
            tot = max(w + l, 1)
            cells.append(f"{w}-{l}  {w/tot*100:.1f}%  (n={len(s)})")
        lbl = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        print(f"  {lbl:<14}{cells[0]:>24}{cells[1]:>24}")

    print(f"\nUNANIMITY")
    for thr in (0.8, 0.9, 1.0):
        s = ev[ev.agree >= thr]
        if len(s) < 60:
            continue
        w, l = grade(s, s["side_home"].values)
        tot = max(w + l, 1)
        print(f"  agreement >= {thr:<5}{w}-{l}  {w/tot*100:.1f}%  "
              f"ROI {(w*(100/110)-l)/tot*100:+.1f}%  (n={len(s)})")


if __name__ == "__main__":
    main()
