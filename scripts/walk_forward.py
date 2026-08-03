"""
Walk-forward evaluation for BOTH sports, with nested threshold selection.

Why
---
The 2025 holdout has been consulted three times across CFB iterations, so it is
no longer a holdout in any meaningful sense. Rather than burn another season
waiting for a clean one, this evaluates every season as if it were the future:

    for each test season S:
        fit the models on seasons < S only
        choose the qualifying threshold on seasons < S only
        apply that threshold to S, record the result, never look back

No season is ever both the thing a threshold was chosen on and the thing it is
scored against. The threshold is allowed to change from year to year, which is
what someone actually running this would do.

Both sports go through the identical procedure. Comparing the NFL's
once-looked-at holdout against CFB's thrice-looked-at one was never fair, and
the question -- can college rival the NFL model? -- deserves a like-for-like
answer.

Usage:
    python walk_forward.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model

OPEN, TGT = "week_open_spread_home", "week_spread_movement"
# Candidate bars, identical for both sports so neither gets a bespoke advantage.
DBARS = [2.0, 3.0, 5.0, 7.0]
MBARS = [0.5, 1.0]
MIN_BETS = 12          # a fold that picks fewer than this is not a strategy


def load_nfl():
    from model_line_movement import load_joined
    df = load_joined()
    feats = [c for c in SPREAD_FEATURES if c in df.columns]
    return df, feats


def load_cfb():
    from build_cfb_dataset import FEATURE_COLS, PRESEASON_COLS, INSEASON_COLS
    df = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")
    feats = ([f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS + INSEASON_COLS
              + ["games_played", "rest_days"]]
             + ["neutral_site", "conference_game", "travel_miles",
                "elev_change", "is_dome"])
    return df, [c for c in feats if c in df.columns]


def grade(s, bet_home):
    surplus = s["home_margin"].values + s[OPEN].values
    live = np.abs(surplus) > 1e-9
    won = np.where(bet_home, surplus > 0, surplus < 0)[live]
    w, l = int(won.sum()), int((~won).sum())
    clv = np.where(bet_home, -s[TGT].values, s[TGT].values)
    return w, l, float(np.mean(clv)) if len(clv) else 0.0


def run(name, df, feats, first_test):
    seasons = sorted(df.season.unique())
    tests = [s for s in seasons if s >= first_test]
    print(f"\n{'='*66}\n{name}   {len(df)} games, {len(feats)} features, "
          f"testing {tests}\n{'='*66}")
    print(f"  {'season':<8}{'bar chosen':<16}{'W-L':>10}{'win%':>8}{'ROI':>9}{'CLV':>8}")

    tot_w = tot_l = 0
    all_clv = []
    for S in tests:
        tr = df[df.season < S]
        te = df[df.season == S]
        if len(tr) < 300:
            continue

        mv = train_model(tr[feats + [OPEN]], tr[TGT], tr, f"wf_{name}_{S}_mv")
        mm = train_model(tr[feats], tr["home_margin"], tr, f"wf_{name}_{S}_mg")

        def prep(d):
            d = d.copy()
            d["mv"] = mv.predict(d[feats + [OPEN]].fillna(0))
            d["dis"] = mm.predict(d[feats].fillna(0)) + d[OPEN].values
            return d

        # Threshold chosen on PRIOR seasons only. The most recent two are used
        # so the choice reflects current market behaviour rather than 2020.
        sel = prep(df[(df.season < S) & (df.season >= S - 2)])
        best, best_wr = None, -1
        for dbar in DBARS:
            for mbar in MBARS:
                q = ((sel["dis"].abs() >= dbar) & (sel["mv"].abs() >= mbar)
                     & ((sel["dis"] > 0) == (sel["mv"] < 0)))
                if q.sum() < MIN_BETS * 2:
                    continue
                w, l, _ = grade(sel[q], (sel[q]["mv"] < 0).values)
                wr = w / max(w + l, 1)
                if wr > best_wr:
                    best_wr, best = wr, (dbar, mbar)
        if best is None:
            print(f"  {S:<8}{'no bar qualified':<16}")
            continue

        dbar, mbar = best
        t = prep(te)
        q = ((t["dis"].abs() >= dbar) & (t["mv"].abs() >= mbar)
             & ((t["dis"] > 0) == (t["mv"] < 0)))
        w, l, clv = grade(t[q], (t[q]["mv"] < 0).values)
        n = max(w + l, 1)
        tot_w += w
        tot_l += l
        all_clv.append(clv * n)
        print(f"  {S:<8}{f'd>={dbar} m>={mbar}':<16}{f'{w}-{l}':>10}"
              f"{w/n*100:>7.1f}%{(w*(100/110)-l)/n*100:>+8.1f}%{clv:>+8.2f}")

    n = max(tot_w + tot_l, 1)
    wr = tot_w / n * 100
    roi = (tot_w * (100 / 110) - tot_l) / n * 100
    se = np.sqrt((wr / 100) * (1 - wr / 100) / n) * 100
    print(f"  {'-'*58}")
    print(f"  {'TOTAL':<8}{'':<16}{f'{tot_w}-{tot_l}':>10}{wr:>7.1f}%{roi:>+8.1f}%"
          f"{sum(all_clv)/n:>+8.2f}")
    print(f"  95% CI [{wr-1.96*se:.1f}, {wr+1.96*se:.1f}]   "
          f"break-even 52.38%   {'CLEARS' if wr-1.96*se > 52.38 else 'contains break-even'}")
    return tot_w, tot_l


def main():
    nfl, nf = load_nfl()
    cfb, cf = load_cfb()
    # NFL has line coverage from 2020; first testable season needs 2 prior.
    run("NFL", nfl, nf, first_test=2022)
    run("CFB", cfb, cf, first_test=2022)


if __name__ == "__main__":
    main()
