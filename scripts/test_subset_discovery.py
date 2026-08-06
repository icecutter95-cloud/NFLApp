"""
Honest subset discovery: are there game states where the model is unusually good?

Unrestricted subset searching is the easiest way in this entire project to
manufacture a 60% system, so the discipline matters more than the method:

  for each test season S
      discover subsets using ONLY out-of-fold predictions from seasons < S
      freeze the subset definition
      grade it on S
      never revisit

and then, because searching many subsets guarantees some look good by chance, a
MAX-STATISTIC permutation: shuffle the outcomes, run the identical search, and
record the best subset it finds. Our best real subset has to beat the
distribution of best NULL subsets, not merely its own naive p-value.

Splitting on game characteristics, not model conviction
-------------------------------------------------------
The ensemble test showed our per-game confidence is not real: dispersion among
twenty models (3.01 pts) exceeds the mean prediction (2.83), and they are
unanimous on 4% of games. So subsets defined by how sure the model claims to be
are unlikely to hold. Subsets defined by observable game STATE -- spread region,
week, rest, key-number proximity -- are the plausible ones, since those describe
where a market might be systematically slow rather than where a model feels
confident.

Shallow trees only: depth 2, large minimum leaf. A deep tree on 2,700 candidate
bets will find something every time.

Usage:
    python test_subset_discovery.py [n_permutations]
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")

from test_residual_model import load, walk, grade, OPEN

MIN_LEAF = 120          # a leaf smaller than this is not a strategy
MAX_DEPTH = 2
BET_BAR = 1.0           # candidate bets: |residual prediction| >= this


def game_state(ev: pd.DataFrame) -> pd.DataFrame:
    """Observable, pre-game characteristics only. No model conviction."""
    x = pd.DataFrame(index=ev.index)
    x["abs_spread"] = ev[OPEN].abs()
    x["home_favored"] = (ev[OPEN] < 0).astype(int)
    x["week"] = ev["week"]
    x["near_key"] = (((ev[OPEN].abs() - 3).abs() <= 0.5)
                     | ((ev[OPEN].abs() - 7).abs() <= 0.5)).astype(int)
    x["total"] = ev["week_open_total"] if "week_open_total" in ev else np.nan
    x["is_div"] = ev["is_divisional"] if "is_divisional" in ev else 0
    x["rest_diff"] = ev["rest_diff"] if "rest_diff" in ev else 0
    x["bet_home"] = ev["bet_home"].astype(int)
    return x.fillna(x.median())


def discover(train_x, train_y, seed=0):
    """Shallow tree; return the best leaf as a boolean mask function."""
    t = DecisionTreeClassifier(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF,
                               random_state=seed)
    t.fit(train_x, train_y)
    leaf_ids = t.apply(train_x)
    best, best_rate = None, -1
    for lid in np.unique(leaf_ids):
        m = leaf_ids == lid
        if m.sum() < MIN_LEAF:
            continue
        rate = train_y[m].mean()
        if rate > best_rate:
            best_rate, best = rate, lid
    return t, best, best_rate


def run_search(ev, y, seasons, permute_rng=None):
    """Walk-forward subset discovery. Returns pooled out-of-sample record."""
    tot_w = tot_l = 0
    picked = []
    for S in seasons:
        tr_m = ev.season < S
        te_m = ev.season == S
        if tr_m.sum() < 400 or te_m.sum() < 50:
            continue
        ytr = y[tr_m.values].copy()
        if permute_rng is not None:
            ytr = permute_rng.permutation(ytr)
        xtr = game_state(ev[tr_m])
        tree, leaf, rate = discover(xtr, ytr)
        if leaf is None:
            continue
        xte = game_state(ev[te_m])
        sel = tree.apply(xte) == leaf
        if sel.sum() < 10:
            continue
        yte = y[te_m.values][sel]
        tot_w += int(yte.sum())
        tot_l += int((~yte.astype(bool)).sum())
        picked.append((S, int(sel.sum()), rate, tree, leaf, xtr.columns))
    return tot_w, tot_l, picked


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    df, feats = load()
    ev = walk(df, feats, "resid_open", use_line=True)
    ev = ev[ev["pred"].abs() >= BET_BAR].copy()
    ev["bet_home"] = ev["pred"] > 0
    sur = ev["resid_open"].values
    ev = ev[np.abs(sur) > 1e-9]
    y = np.where(ev["bet_home"], ev["resid_open"] > 0, ev["resid_open"] < 0).astype(int)

    base_w, base_l = int(y.sum()), int((1 - y).sum())
    print(f"candidate bets (|pred| >= {BET_BAR}): {len(ev)}  "
          f"baseline {base_w}-{base_l} = {base_w/(base_w+base_l)*100:.1f}%")

    seasons = sorted(s for s in ev.season.unique() if s >= 2023)
    w, l, picked = run_search(ev, y, seasons)
    n = max(w + l, 1)
    print(f"\nWALK-FORWARD SUBSET (discovered on prior seasons, graded on S):")
    for S, cnt, rate, *_ in picked:
        print(f"  {S}: leaf selected {cnt} bets  (looked {rate*100:.1f}% in discovery)")
    print(f"  POOLED {w}-{l} = {w/n*100:.1f}%   vs betting everything "
          f"{base_w/(base_w+base_l)*100:.1f}%")
    print(f"  lift: {w/n*100 - base_w/(base_w+base_l)*100:+.1f} points")

    if picked:
        S, cnt, rate, tree, leaf, cols = picked[-1]
        print(f"\n  most recent tree ({S}), for interpretability:")
        for line in export_text(tree, feature_names=list(cols)).split("\n")[:14]:
            print(f"    {line}")

    print(f"\nMAX-STATISTIC PERMUTATION ({n_perm} runs)")
    print("  Shuffles outcomes and reruns the IDENTICAL search, recording the")
    print("  best subset it finds. Tests our best against the best noise can do.")
    rng = np.random.default_rng(17)
    out = []
    for _ in range(n_perm):
        pw, pl, _ = run_search(ev, y, seasons, permute_rng=rng)
        if pw + pl >= 30:
            out.append(pw / (pw + pl) * 100)
    a = np.array(out)
    real = w / n * 100
    beat = int((a >= real).sum())
    print(f"  permuted best-subset: mean {a.mean():.1f}%  sd {a.std():.1f}  "
          f"max {a.max():.1f}")
    print(f"  {beat}/{len(a)} reached the real subset -> p = "
          f"{(beat+1)/(len(a)+1):.3f}")
    if a.std() > 0:
        print(f"  real sits {(real - a.mean())/a.std():+.1f} sd above the null search")


if __name__ == "__main__":
    main()
