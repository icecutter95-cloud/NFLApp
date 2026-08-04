"""
CFB totals, brought up to parity with every other model in one pass.

College totals were the one empty cell in the standing table: the line backfill
had requested spreads only, so there were no totals lines, no model and no
tests. This builds it and puts it through the identical battery the other three
have been through, so the comparison is like for like rather than a new model
flattered by a gentler examination.

  1. movement model + evidence-weighted walk-forward   (as NFL/CFB spreads)
  2. label permutation                                  (the multiplicity check)
  3. residual model                                     (predict the opener's
                                                         error, which beat the
                                                         production approach on
                                                         NFL spreads)

Targets:
    movement   closing_total - week_open_total
    residual   total_points  - week_open_total     (how wrong the opener was)

Bets are graded at the OPENING total, -110, over if the signal is positive.

Prior expectation, stated up front: NFL totals showed no predictable opener
error at all (corr +0.029, every bar on 50%, 14/25 permutations matched it), and
CFB spreads showed a real but economically worthless one (52.4% against a 52.38%
break-even). CFB totals sit at the intersection of the two weakest cells, so the
honest prior is that this finds nothing.

Usage:
    python test_cfb_totals.py [n_permutations]
"""

import contextlib
import io as _io
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR
from train_models import train_model
from walk_forward import wilson_lcb

OPEN = "week_open_total"
MOVE = "week_total_movement"
BARS = [0.5, 1.0, 1.5, 2.0, 3.0]
PERM_BAR = 1.0


def load():
    """CFB features joined to the totals lines and actual points scored."""
    from build_cfb_dataset import FEATURE_COLS, PRESEASON_COLS, INSEASON_COLS
    df = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")

    oc = []
    for s in range(2020, 2026):
        p = DATA_DIR / f"cfb_open_close_totals_{s}.parquet"
        if not p.exists():
            continue
        o = pd.read_parquet(p)
        o["season"] = s
        oc.append(o)
    if not oc:
        raise SystemExit("No cfb_open_close_totals_*.parquet — run the backfill "
                         "with CFB_MARKET=totals first")
    oc = pd.concat(oc, ignore_index=True)
    oc = oc[oc.n_snapshots_week >= 2]
    # The fetcher stores whatever market it pulled in the spread columns, so
    # rename them to what they actually are here.
    oc = oc.rename(columns={"week_open_spread_home": OPEN,
                            "closing_spread_home": "closing_total",
                            "week_spread_movement": MOVE})

    before = len(df)
    df = df.merge(oc[["season", "home_team", "away_team", "kick_date",
                      OPEN, "closing_total", MOVE]],
                  on=["season", "home_team", "away_team", "kick_date"], how="inner")
    assert df.duplicated(subset=["season", "home_team", "away_team",
                                 "kick_date"]).sum() == 0, "totals join fanned out"
    df["resid_open"] = df["total_points"] - df[OPEN]

    feats = ([f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS + INSEASON_COLS
              + ["games_played", "rest_days"]]
             + ["neutral_site", "conference_game", "travel_miles",
                "elev_change", "is_dome"])
    print(f"  {before} CFB games -> {len(df)} with a totals line")
    return df, [c for c in feats if c in df.columns]


def grade(s, over):
    pts, line = s["total_points"].values, s[OPEN].values
    live = pts != line
    won = np.where(over, pts > line, pts < line)[live]
    return int(won.sum()), int((~won).sum())


def walk(df, feats, target, extra_col, permute_rng=None):
    f = feats + [extra_col]
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
        m = train_model(tr[f], tr[target], tr, f"ct_{target}_{S}")
        t = te.copy()
        t["pred"] = m.predict(te[f].fillna(0))
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sweep(label, ev):
    print(f"\n  {label}")
    print(f"    {'bar':<8}{'n':>7}{'W-L':>12}{'win%':>8}{'ROI':>8}")
    for bar in BARS:
        q = ev["pred"].abs() >= bar
        if q.sum() < 60:
            continue
        w, l = grade(ev[q], (ev[q]["pred"] > 0).values)
        n = max(w + l, 1)
        print(f"    {bar:<8}{w+l:>7}{f'{w}-{l}':>12}{w/n*100:>7.1f}%"
              f"{(w*(100/110)-l)/n*100:>+7.1f}%")


def permute(df, feats, target, extra_col, n_perm, real):
    rng = np.random.default_rng(41)
    out = []
    for _ in range(n_perm):
        with contextlib.redirect_stdout(_io.StringIO()):
            pe = walk(df, feats, target, extra_col, permute_rng=rng)
        if pe.empty:
            continue
        q = pe["pred"].abs() >= PERM_BAR
        if q.sum() < 60:
            continue
        w, l = grade(pe[q], (pe[q]["pred"] > 0).values)
        out.append(w / max(w + l, 1) * 100)
    a = np.array(out)
    beat = int((a >= real).sum())
    print(f"    permuted mean {a.mean():.1f}%  sd {a.std():.1f}  max {a.max():.1f}")
    print(f"    {beat}/{len(a)} reached it -> p = {(beat+1)/(len(a)+1):.3f}   "
          f"{(real - a.mean())/a.std():+.1f} sd")


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    print("CFB TOTALS")
    df, feats = load()
    print(f"  {len(feats)} features, seasons {sorted(df.season.unique())}")

    # 1. movement model — the approach production uses for NFL totals
    ev_mv = walk(df, feats, MOVE, OPEN)
    sweep("MOVEMENT model (predict where the total goes)", ev_mv)

    # 2. residual model — the approach that beat production on NFL spreads
    ev_rs = walk(df, feats, "resid_open", OPEN)
    sweep("RESIDUAL model (predict the opener's error)", ev_rs)
    print(f"    corr(pred, opener error) = "
          f"{np.corrcoef(ev_rs['pred'], ev_rs['resid_open'])[0,1]:+.4f}")

    # 3. permutation on both, at a fixed bar
    for label, ev, target in [("MOVEMENT", ev_mv, MOVE),
                              ("RESIDUAL", ev_rs, "resid_open")]:
        q = ev["pred"].abs() >= PERM_BAR
        w, l = grade(ev[q], (ev[q]["pred"] > 0).values)
        real = w / max(w + l, 1) * 100
        print(f"\n  PERMUTATION — {label} at bar {PERM_BAR}: real {w}-{l} = {real:.1f}%")
        permute(df, feats, target, OPEN, n_perm, real)

    print("\n  break-even at -110 is 52.38%")


if __name__ == "__main__":
    main()
