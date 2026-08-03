"""
How good is the model at predicting the GAME, not the line?

Everything else in this project measures line movement and CLV. This asks the
older, blunter question: given our features, how well can we predict the final
margin -- and does that beat the number the market has already posted?

The market's closing spread is a forecast of the margin. So is our model. Put
them side by side:

    corr(prediction, actual margin)     higher is better
    MAE                                 lower is better
    ATS record betting our disagreement WITH THE CLOSE, not the opener

That last line is the real test. Beating the opener is a timing edge; beating
the close means genuinely knowing something the market does not. The NFL answer
was no -- 0.374 against the market's 0.477 -- which is why the whole app pivoted
to predicting movement instead.

Walk-forward so nothing is scored by a model that saw it: train on seasons < S,
predict S, never revisit.

Usage:
    python test_margin_efficacy.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model


def load_nfl():
    from model_line_movement import load_joined
    d = load_joined().dropna(subset=["home_margin", "closing_spread_home"])
    return d, [c for c in SPREAD_FEATURES if c in d.columns]


def load_cfb():
    from build_cfb_dataset import FEATURE_COLS, PRESEASON_COLS
    d = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")
    d = d.dropna(subset=["home_margin", "closing_spread_home"])
    f = ([f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS
          + ["games_played", "rest_days"]]
         + ["neutral_site", "conference_game", "travel_miles", "elev_change", "is_dome"])
    return d, [c for c in f if c in d.columns]


def run(name, df, feats, first_test=2022):
    preds, actual, market = [], [], []
    for S in sorted(s for s in df.season.unique() if s >= first_test):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        m = train_model(tr[feats], tr["home_margin"], tr, f"me_{name}_{S}")
        preds.append(m.predict(te[feats].fillna(0)))
        actual.append(te["home_margin"].values)
        # The market's implied home margin is the negated home spread.
        market.append(-te["closing_spread_home"].values)

    p, a, mk = np.concatenate(preds), np.concatenate(actual), np.concatenate(market)
    print(f"\n{name}   n={len(a)} games (walk-forward)")
    print(f"  {'':<22}{'corr':>8}{'MAE':>8}")
    print(f"  {'our model':<22}{np.corrcoef(p, a)[0,1]:>+8.3f}{np.abs(p-a).mean():>8.2f}")
    print(f"  {'market close':<22}{np.corrcoef(mk, a)[0,1]:>+8.3f}{np.abs(mk-a).mean():>8.2f}")
    gap = np.corrcoef(p, a)[0, 1] - np.corrcoef(mk, a)[0, 1]
    print(f"  gap: {gap:+.3f}  ->  {'WE BEAT THE MARKET' if gap > 0 else 'market wins'}")

    # Blend test: does the model add anything ON TOP of the market number?
    # If the market already contains our information, the best blend weight on
    # our model is ~0 and the blend does not beat the market alone.
    best = max(((np.corrcoef(w * p + (1 - w) * mk, a)[0, 1], w)
                for w in np.arange(0, 1.01, 0.05)))
    print(f"  best blend: {best[1]*100:.0f}% model / {(1-best[1])*100:.0f}% market "
          f"-> corr {best[0]:+.3f}")

    # ATS: bet our disagreement with the CLOSE.
    print(f"  {'ATS vs close':<16}{'n':>7}{'W-L':>11}{'win%':>8}{'ROI':>8}")
    edge = p - mk
    cover = a + (-mk)          # >0 means home covered the closing number
    for thr in [0, 3, 6, 10]:
        s = np.abs(edge) >= thr
        if s.sum() < 50:
            continue
        bet_home = edge[s] > 0
        live = np.abs(cover[s]) > 1e-9
        won = np.where(bet_home, cover[s] > 0, cover[s] < 0)[live]
        w, l = int(won.sum()), int((~won).sum())
        n = max(w + l, 1)
        print(f"  |edge|>={thr:<10}{int(s.sum()):>7}{f'{w}-{l}':>11}"
              f"{w/n*100:>7.1f}%{(w*(100/110)-l)/n*100:>+7.1f}%")


def main():
    n, nf = load_nfl()
    c, cf = load_cfb()
    run("NFL", n, nf)
    run("CFB", c, cf)
    print("\nbreak-even at -110 is 52.38%")


if __name__ == "__main__":
    main()
