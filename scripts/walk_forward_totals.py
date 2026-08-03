"""
The same gauntlet the spread rule went through, applied to TOTALS.

Everything validated on 2026-08-03 -- the nested-selection fix, the
evidence-weighted selector, the adverse-selection test, the label permutation --
operated on spreads. The live totals rule (|predicted movement| >= 1.25) came
from an earlier select/holdout sweep: never walk-forward, never nested, never
permutation-tested.

That matters because the permutation test measured this pipeline's noise floor
at about 3.3 points of win rate. The totals rule's holdout was 56.6% on 53 bets,
roughly 2 sd above a coin flip, which is exactly the zone where a threshold
picked by looking at a sweep is most likely to be an artefact.

So: same procedure, same objective, same permutation check.

  target      week_total_movement = closing_total - week_open_total
  side        over if the number is predicted to rise, under if to fall
  grading     at the OPENING total, -110
  disagreement  predicted_total_points - week_open_total
                (>0 means the model projects more scoring than the opener)

Direction agreement flips relative to spreads: projecting MORE points means
wanting the number to RISE, so agreement is (disagreement > 0) == (movement > 0).
For spreads, rating home higher means wanting the line to FALL.

NFL only. The CFB line backfill fetched spreads exclusively, so college totals
cannot be tested without new API pulls.

Usage:
    python walk_forward_totals.py [n_permutations]
"""

import contextlib
import io as _io
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, TOTAL_FEATURES
from train_models import train_model
from walk_forward import wilson_lcb

OPEN, TGT = "week_open_total", "week_total_movement"
DBARS = [0.0, 2.0, 3.0, 5.0]
MBARS = [0.0, 0.5, 1.0, 1.25]
RULES = ("both", "disagreement_only", "movement_only")
MIN_BETS = 12


def load():
    from model_line_movement import load_joined
    df = load_joined()
    df["total_points"] = df["home_score"] + df["away_score"]
    df = df.dropna(subset=[OPEN, TGT, "total_points"])
    return df, [c for c in TOTAL_FEATURES if c in df.columns]


def grade(s, over):
    pts, line = s["total_points"].values, s[OPEN].values
    live = pts != line
    won = np.where(over, pts > line, pts < line)[live]
    w, l = int(won.sum()), int((~won).sum())
    clv = np.where(over, s[TGT].values, -s[TGT].values)
    return w, l, float(np.mean(clv)) if len(clv) else 0.0


def apply_rule(d, kind, dbar, mbar):
    # Over wants the number to RISE, so agreement is same-sign here, unlike
    # spreads where rating home higher means wanting the line to fall.
    agree = (d["dis"] > 0) == (d["mv"] > 0)
    if kind == "both":
        mask = (d["dis"].abs() >= dbar) & (d["mv"].abs() >= mbar) & agree
        over = (d["mv"] > 0).values
    elif kind == "disagreement_only":
        mask = d["dis"].abs() >= dbar
        over = (d["dis"] > 0).values
    else:
        mask = d["mv"].abs() >= mbar
        over = (d["mv"] > 0).values
    return mask, over


def run(df, feats, quiet=False):
    tot_w = tot_l = 0
    rows = []
    for S in sorted(s for s in df.season.unique() if s >= 2022):
        tr, te = df[df.season < S], df[df.season == S]
        if len(tr) < 300:
            continue
        mv = train_model(tr[feats + [OPEN]], tr[TGT], tr, f"t_{S}_mv")
        pm = train_model(tr[feats], tr["total_points"], tr, f"t_{S}_pm")

        def prep(d):
            d = d.copy()
            d["mv"] = mv.predict(d[feats + [OPEN]].fillna(0))
            d["dis"] = pm.predict(d[feats].fillna(0)) - d[OPEN].values
            return d

        # Out-of-fold selection, same discipline as the spread version.
        oof = []
        for s in sorted(x for x in df.season.unique() if x < S):
            itr = df[df.season < s]
            if len(itr) < 300:
                continue
            i_mv = train_model(itr[feats + [OPEN]], itr[TGT], itr, f"t_{S}_{s}_mv")
            i_pm = train_model(itr[feats], itr["total_points"], itr, f"t_{S}_{s}_pm")
            d = df[df.season == s].copy()
            d["mv"] = i_mv.predict(d[feats + [OPEN]].fillna(0))
            d["dis"] = i_pm.predict(d[feats].fillna(0)) - d[OPEN].values
            oof.append(d)
        if not oof:
            continue
        sel = pd.concat(oof, ignore_index=True)

        best, best_score = None, -1e9
        for kind in RULES:
            for dbar in DBARS:
                for mbar in MBARS:
                    if kind == "disagreement_only" and mbar != MBARS[0]:
                        continue
                    if kind == "movement_only" and dbar != DBARS[0]:
                        continue
                    mask, over = apply_rule(sel, kind, dbar, mbar)
                    if mask.sum() < MIN_BETS * 2:
                        continue
                    w, l, _ = grade(sel[mask], over[mask.values])
                    sc = wilson_lcb(w, l)
                    if sc > best_score:
                        best_score, best = sc, (kind, dbar, mbar)
        if best is None:
            continue

        kind, dbar, mbar = best
        t = prep(te)
        q, over = apply_rule(t, kind, dbar, mbar)
        w, l, clv = grade(t[q], over[q.values])
        tot_w += w
        tot_l += l
        tag = {"both": f"both d>={dbar} m>={mbar}",
               "disagreement_only": f"dis-only d>={dbar}",
               "movement_only": f"mv-only m>={mbar}"}[kind]
        rows.append((S, tag, w, l, clv))

    if not quiet:
        print(f"  {'season':<8}{'rule chosen':<22}{'W-L':>10}{'win%':>8}{'ROI':>9}{'CLV':>8}")
        for S, tag, w, l, clv in rows:
            n = max(w + l, 1)
            print(f"  {S:<8}{tag:<22}{f'{w}-{l}':>10}{w/n*100:>7.1f}%"
                  f"{(w*(100/110)-l)/n*100:>+8.1f}%{clv:>+8.2f}")
        n = max(tot_w + tot_l, 1)
        wr = tot_w / n * 100
        se = np.sqrt(wr / 100 * (1 - wr / 100) / n) * 100
        print(f"  {'-'*58}")
        print(f"  {'TOTAL':<8}{'':<22}{f'{tot_w}-{tot_l}':>10}{wr:>7.1f}%"
              f"{(tot_w*(100/110)-tot_l)/n*100:>+8.1f}%")
        print(f"  95% CI [{wr-1.96*se:.1f}, {wr+1.96*se:.1f}]   break-even 52.38%")
    return (tot_w / max(tot_w + tot_l, 1) * 100, tot_w + tot_l)


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    df, feats = load()
    print(f"NFL TOTALS  {len(df)} games, {len(feats)} features")
    real = run(df, feats)

    print(f"\n  running {n_perm} permutations of the full totals pipeline...")
    rng = np.random.default_rng(11)
    out = []
    for _ in range(n_perm):
        d = df.copy()
        for s in d.season.unique():
            idx = d.index[d.season == s].to_numpy()
            d.loc[idx, feats] = d.loc[rng.permutation(idx), feats].to_numpy()
        with contextlib.redirect_stdout(_io.StringIO()):
            r = run(d, feats, quiet=True)
        if r[1] > 0:
            out.append(r[0])
    a = np.array(out)
    beat = int((a >= real[0]).sum())
    print(f"  permuted: mean {a.mean():.1f}%  sd {a.std():.1f}  "
          f"min {a.min():.1f}  max {a.max():.1f}")
    print(f"  reaching the real result: {beat}/{len(a)}  "
          f"-> p = {(beat + 1) / (len(a) + 1):.3f}")
    z = (real[0] - a.mean()) / a.std() if a.std() else float("nan")
    print(f"  real sits {z:+.1f} sd above the noise distribution")


if __name__ == "__main__":
    main()
