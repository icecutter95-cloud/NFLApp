"""
Does the residual reframing help where the current approach FAILS?

Predicting the market's error instead of the game lifted NFL spreads at every
threshold. That was one market. The interesting question is whether it is a
general improvement or something specific to NFL spread pricing, and the way to
find out is to point it at the two models that failed label permutation:

    NFL totals    p = 0.231   (movement-only >= 1.25, live but not validated)
    CFB spreads   p = 0.077   (not in the app)

Targets, all "how wrong was the opener":

    NFL/CFB spreads   home_margin + week_open_spread_home
    NFL totals        total_points - week_open_total

Positive means the home side, or the over, beat the number. The opener is a
FEATURE in every case, which is safe for the same reason as before: an efficient
market implies E[target | opener] = 0, so echoing the line back earns nothing.

Each market gets the identical treatment -- walk-forward, a threshold sweep, and
a label permutation on a fixed bar -- so the three are comparable.

Usage:
    python test_residual_all.py [n_permutations]
"""

import contextlib
import io as _io
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES, TOTAL_FEATURES
from train_models import train_model

BARS = [0.5, 1.0, 1.5, 2.0, 3.0]
PERM_BAR = 1.0


def load_nfl_spreads():
    from model_line_movement import load_joined
    d = load_joined().dropna(subset=["week_open_spread_home", "home_margin"]).copy()
    d["target"] = d["home_margin"] + d["week_open_spread_home"]
    return d, [c for c in SPREAD_FEATURES if c in d.columns], "week_open_spread_home"


def load_nfl_totals():
    from model_line_movement import load_joined
    d = load_joined().copy()
    d["total_points"] = d["home_score"] + d["away_score"]
    d = d.dropna(subset=["week_open_total", "total_points"])
    # Over wins when the game outscores the opener.
    d["target"] = d["total_points"] - d["week_open_total"]
    return d, [c for c in TOTAL_FEATURES if c in d.columns], "week_open_total"


def load_cfb_spreads():
    from build_cfb_dataset import FEATURE_COLS, PRESEASON_COLS, INSEASON_COLS
    d = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")
    d = d.dropna(subset=["week_open_spread_home", "home_margin"]).copy()
    d["target"] = d["home_margin"] + d["week_open_spread_home"]
    f = ([f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS + INSEASON_COLS
          + ["games_played", "rest_days"]]
         + ["neutral_site", "conference_game", "travel_miles", "elev_change", "is_dome"])
    return d, [c for c in f if c in d.columns], "week_open_spread_home"


def walk(df, feats, line_col, first_test=2022, permute_rng=None):
    f = feats + [line_col]
    out = []
    for S in sorted(s for s in df.season.unique() if s >= first_test):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        if permute_rng is not None:
            tr = tr.copy()
            for s in tr.season.unique():
                idx = tr.index[tr.season == s].to_numpy()
                tr.loc[idx, feats] = tr.loc[permute_rng.permutation(idx), feats].to_numpy()
        m = train_model(tr[f], tr["target"], tr, f"ra_{line_col}_{S}")
        t = te.copy()
        t["pred"] = m.predict(te[f].fillna(0))
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def grade(s):
    """Bet the side the residual points to; graded at the opener."""
    tgt = s["target"].values
    live = np.abs(tgt) > 1e-9
    won = np.where(s["pred"].values > 0, tgt > 0, tgt < 0)[live]
    return int(won.sum()), int((~won).sum())


def analyse(name, loader, n_perm, first_test=2022):
    df, feats, line_col = loader()
    print(f"\n{'='*64}\n{name}   {len(df)} games, {len(feats)} features\n{'='*64}")
    ev = walk(df, feats, line_col, first_test)
    if ev.empty:
        print("  no folds")
        return

    corr = np.corrcoef(ev["pred"], ev["target"])[0, 1]
    print(f"  corr(pred, opener error) = {corr:+.4f}   "
          f"corr(pred, line) = {np.corrcoef(ev['pred'], ev[line_col])[0,1]:+.3f}")
    print(f"  {'bar':<8}{'n':>7}{'W-L':>12}{'win%':>8}{'ROI':>8}")
    for bar in BARS:
        q = ev["pred"].abs() >= bar
        if q.sum() < 60:
            continue
        w, l = grade(ev[q])
        n = max(w + l, 1)
        print(f"  {bar:<8}{w+l:>7}{f'{w}-{l}':>12}{w/n*100:>7.1f}%"
              f"{(w*(100/110)-l)/n*100:>+7.1f}%")

    q = ev["pred"].abs() >= PERM_BAR
    w, l = grade(ev[q])
    real = w / max(w + l, 1) * 100
    print(f"\n  permutation at bar {PERM_BAR} ({n_perm} runs): real {w}-{l} = {real:.1f}%")
    rng = np.random.default_rng(31)
    out = []
    for _ in range(n_perm):
        with contextlib.redirect_stdout(_io.StringIO()):
            pe = walk(df, feats, line_col, first_test, permute_rng=rng)
        if pe.empty:
            continue
        pq = pe["pred"].abs() >= PERM_BAR
        if pq.sum() < 60:
            continue
        pw, pl = grade(pe[pq])
        out.append(pw / max(pw + pl, 1) * 100)
    if not out:
        return
    a = np.array(out)
    beat = int((a >= real).sum())
    print(f"  permuted mean {a.mean():.1f}%  sd {a.std():.1f}  max {a.max():.1f}")
    print(f"  {beat}/{len(a)} reached it -> p = {(beat+1)/(len(a)+1):.3f}   "
          f"{(real - a.mean())/a.std():+.1f} sd")


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    analyse("NFL SPREADS (reference)", load_nfl_spreads, n_perm)
    analyse("NFL TOTALS", load_nfl_totals, n_perm)
    analyse("CFB SPREADS", load_cfb_spreads, n_perm)
    print("\nbreak-even at -110 is 52.38%")


if __name__ == "__main__":
    main()
