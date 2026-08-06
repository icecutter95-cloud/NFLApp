"""
Predict the MARKET'S ERROR instead of the game.

Our margin model predicts home_margin from football features with no market
input, and disagreement is then computed as (predicted_margin + opener). That
asks the model to explain the entire game from scratch and only afterwards
compares it to the line.

This reframes the task:

    target = home_margin + week_open_spread_home

which is the opener's error, and also exactly the quantity that decides whether
a bet at the opener wins. Positive means the home side beat the number.

Why the line belongs IN the features here
-----------------------------------------
It looks circular and is not. If the market is efficient then
E[margin | open] = -open, so E[target | open] = 0. An efficient market implies
the correct prediction is ZERO everywhere, whatever the line is. So the model
cannot profit by echoing the number back; it can only profit by finding places
where the market is SYSTEMATICALLY wrong.

This is a different shape from the sign bug that once inflated this project's
results. There the TARGET secretly contained 2*spread, so a model that
reconstructed the line was rewarded for it. Here the target is the real betting
outcome and reconstruction earns nothing. The guard below checks it anyway.

Two targets are fitted, because the difference between them is diagnostic:

    opener error   margin + week_open_spread_home
    close error    margin + closing_spread_home

A feature that predicts opener error but NOT close error means we are getting
there before the market does -- a timing edge. One that predicts close error
means we evaluate football differently from the market, which is a deeper and
rarer thing.

Walk-forward throughout: train on seasons < S, predict S, never revisit.

Usage:
    python test_residual_model.py [n_permutations]
"""

import contextlib
import io as _io
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import SPREAD_FEATURES
from train_models import train_model
from walk_forward import wilson_lcb

OPEN = "week_open_spread_home"
CLOSE = "closing_spread_home"
BARS = [0.5, 1.0, 1.5, 2.0, 3.0]
MIN_BETS = 12


def load():
    from model_line_movement import load_joined
    d = load_joined().dropna(subset=[OPEN, CLOSE, "home_margin"]).copy()
    d["resid_open"] = d["home_margin"] + d[OPEN]
    d["resid_close"] = d["home_margin"] + d[CLOSE]
    feats = [c for c in SPREAD_FEATURES if c in d.columns]
    return d, feats


def grade(s, bet_home):
    sur = s["resid_open"].values
    live = np.abs(sur) > 1e-9
    won = np.where(bet_home, sur > 0, sur < 0)[live]
    return int(won.sum()), int((~won).sum())


def walk(df, feats, target, use_line=True, quiet=True, permute_rng=None):
    """Walk-forward the residual model; returns per-game predictions."""
    f = feats + ([OPEN] if use_line else [])
    out = []
    for S in sorted(s for s in df.season.unique() if s >= 2022):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        if permute_rng is not None:
            tr = tr.copy()
            for s in tr.season.unique():
                idx = tr.index[tr.season == s].to_numpy()
                tr.loc[idx, feats] = tr.loc[permute_rng.permutation(idx), feats].to_numpy()
        m = train_model(tr[f], tr[target], tr, f"res_{target}_{S}")
        t = te.copy()
        t["pred"] = m.predict(te[f].fillna(0))
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def report(name, ev, target):
    p, a = ev["pred"].values, ev[target].values
    corr = np.corrcoef(p, a)[0, 1]
    print(f"\n  {name}")
    print(f"    corr(pred, actual {target}) = {corr:+.4f}   "
          f"mean |pred| {np.abs(p).mean():.2f}")
    # An efficient market means the right answer is ~0 everywhere. How far from
    # zero the model is willing to go is a measure of claimed bias.
    print(f"    pred vs the line: corr {np.corrcoef(p, ev[OPEN])[0,1]:+.3f}  "
          f"(high = the model is mostly re-expressing the spread)")
    print(f"    {'bar':<8}{'n':>6}{'W-L':>10}{'win%':>8}{'ROI':>8}")
    for bar in BARS:
        q = np.abs(p) >= bar
        if q.sum() < 40:
            continue
        w, l = grade(ev[q], (p[q] > 0))
        n = max(w + l, 1)
        print(f"    {bar:<8}{w+l:>6}{f'{w}-{l}':>10}{w/n*100:>7.1f}%"
              f"{(w*(100/110)-l)/n*100:>+7.1f}%")
    return corr


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    df, feats = load()
    print(f"NFL {len(df)} games, {len(feats)} football features")

    # --- the reframing, head to head with the current approach
    print("\n=== 1. OPENER ERROR (the thing we actually bet) ===")
    ev = walk(df, feats, "resid_open", use_line=True)
    report("residual model  (line IS a feature)", ev, "resid_open")

    ev_nl = walk(df, feats, "resid_open", use_line=False)
    report("residual model  (no line)", ev_nl, "resid_open")

    # Current production approach, for comparison on the same folds.
    cur = walk(df, feats, "home_margin", use_line=False)
    cur["pred"] = cur["pred"] + cur[OPEN]          # = the disagreement signal
    report("CURRENT: margin model + opener", cur, "resid_open")

    # --- is the edge timing, or football?
    print("\n=== 2. OPENER ERROR vs CLOSE ERROR (timing or genuine?) ===")
    evc = walk(df, feats, "resid_close", use_line=True)
    print("    If the model predicts opener error but NOT close error, the edge")
    print("    is getting there early. If it predicts close error too, it is a")
    print("    genuine difference of football opinion.")
    report("residual model on CLOSE error", evc, "resid_close")

    # --- permutation on the winning variant
    print(f"\n=== 3. PERMUTATION ({n_perm} runs, opener-error model) ===")
    best_bar = 1.0
    q = np.abs(ev["pred"].values) >= best_bar
    w, l = grade(ev[q], (ev["pred"].values[q] > 0))
    real = w / max(w + l, 1) * 100
    print(f"    real at bar {best_bar}: {w}-{l} = {real:.1f}%")

    rng = np.random.default_rng(23)
    out = []
    for _ in range(n_perm):
        with contextlib.redirect_stdout(_io.StringIO()):
            pe = walk(df, feats, "resid_open", use_line=True, permute_rng=rng)
        if pe.empty:
            continue
        pq = np.abs(pe["pred"].values) >= best_bar
        if pq.sum() < 40:
            continue
        pw, pl = grade(pe[pq], (pe["pred"].values[pq] > 0))
        out.append(pw / max(pw + pl, 1) * 100)
    a = np.array(out)
    beat = int((a >= real).sum())
    print(f"    permuted mean {a.mean():.1f}%  sd {a.std():.1f}  max {a.max():.1f}")
    print(f"    {beat}/{len(a)} reached it -> p = {(beat+1)/(len(a)+1):.3f}   "
          f"{(real-a.mean())/a.std():+.1f} sd")


if __name__ == "__main__":
    main()
