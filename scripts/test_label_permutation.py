"""
How often does this pipeline manufacture an edge from nothing?

Every result in this project is the end of a long research process: many feature
families, several target formulations, threshold sweeps, rule shapes, bug fixes
and re-runs. Each individual decision was defensible. The concern is cumulative
-- with enough forks, a pipeline can produce a convincing-looking edge from pure
noise, and no single step looks wrong.

This measures that directly. Break the link between features and outcomes, then
run the ENTIRE pipeline unchanged -- nested out-of-fold threshold selection,
rule-shape choice, evidence-weighted objective, walk-forward grading -- and see
what win rate comes out. Repeat.

The permutation shuffles the TEAM FEATURE rows within each season, leaving the
lines, movement and margins correctly paired with each other. So the market
structure, base rates and the movement/margin relationship all survive intact;
only the model's ability to know anything is destroyed. Any edge the pipeline
reports under that condition is manufactured.

Read the output as: the real result should sit far out in the tail of the
permuted distribution. If it sits near the middle, the pipeline is the source.

Usage:
    python test_label_permutation.py [n_permutations]
"""

import contextlib
import io as _io
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import walk_forward as W


def permute_features(df: pd.DataFrame, feats: list, rng) -> pd.DataFrame:
    """Shuffle feature rows within season; leave lines and outcomes alone."""
    d = df.copy()
    for s in d.season.unique():
        m = d.season == s
        idx = d.index[m].to_numpy()
        shuffled = rng.permutation(idx)
        d.loc[idx, feats] = d.loc[shuffled, feats].to_numpy()
    return d


def one_run(name, df, feats) -> tuple:
    W.BETS.clear()
    with contextlib.redirect_stdout(_io.StringIO()):
        W.run(name, df, feats, first_test=2022)
    if not W.BETS:
        return None
    b = pd.concat(W.BETS, ignore_index=True)
    b = b[~b["push"]]
    if len(b) == 0:
        return None
    w = int(b.won.sum())
    return w / len(b) * 100, len(b)


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    rng = np.random.default_rng(7)

    for name, load in [("NFL", W.load_nfl), ("CFB", W.load_cfb)]:
        df, feats = load()
        real = one_run(name, df, feats)
        print(f"\n{name}: REAL result {real[0]:.1f}% on {real[1]} bets")
        print(f"  running {n_perm} permutations of the full pipeline...")

        out = []
        for i in range(n_perm):
            p = permute_features(df, feats, rng)
            r = one_run(name, p, feats)
            if r:
                out.append(r[0])
        if not out:
            print("  no permutation produced bets")
            continue

        a = np.array(out)
        # One-sided: how often does noise reach the real result?
        beat = int((a >= real[0]).sum())
        print(f"  permuted win rate: mean {a.mean():.1f}%  sd {a.std():.1f}  "
              f"min {a.min():.1f}  max {a.max():.1f}")
        print(f"  90th pct {np.percentile(a, 90):.1f}%   "
              f"95th pct {np.percentile(a, 95):.1f}%")
        print(f"  permutations reaching the real result: {beat}/{len(a)}  "
              f"-> empirical p = {(beat + 1) / (len(a) + 1):.3f}")
        z = (real[0] - a.mean()) / a.std() if a.std() > 0 else float("nan")
        print(f"  real result sits {z:+.1f} sd above the noise distribution")


if __name__ == "__main__":
    main()
