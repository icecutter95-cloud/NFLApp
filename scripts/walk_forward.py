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
BETS = []              # every individual walk-forward bet, for the tier breakdown


def tier_table(name):
    """Break the pooled record down by conviction.

    This is a DIAGNOSTIC, not a strategy. The tiers are read off the same test
    results the record came from, so picking the best-looking one and betting it
    would be selection on the test set. What it can honestly show is SHAPE: a
    real edge should strengthen as the model's disagreement grows. A flat or
    inverted profile means the number is coming from somewhere other than skill.
    """
    if not BETS:
        return
    b = pd.concat(BETS, ignore_index=True)
    b = b[~b["push"]]
    print(f"\n  {name} — {len(b)} bets broken down by conviction")

    for label, col, bands in [
        ("|disagreement|", "abs_dis", [(7, 10), (10, 14), (14, 99)]),
        ("|pred movement|", "abs_mv", [(0.5, 1.0), (1.0, 1.5), (1.5, 99)]),
    ]:
        print(f"    {label:<16}{'n':>6}{'W-L':>10}{'win%':>8}{'ROI':>8}{'CLV':>8}")
        for lo, hi in bands:
            s = b[(b[col] >= lo) & (b[col] < hi)]
            if len(s) < 8:
                continue
            w = int(s.won.sum())
            l = len(s) - w
            band = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
            print(f"    {band:<16}{len(s):>6}{f'{w}-{l}':>10}{w/len(s)*100:>7.1f}%"
                  f"{(w*(100/110)-l)/len(s)*100:>+7.1f}%{s.clv_pts.mean():>+8.2f}")


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

        # Threshold selection must use OUT-OF-FOLD predictions.
        #
        # This previously scored df[season < S] with `mv`/`mm`, which were
        # trained on exactly those games. Choosing a bar against a model's own
        # training fit is in-sample optimisation wearing a nested costume: the
        # fit is systematically too good, and it biases selection toward bars
        # that look strong in-sample. Caught in outside review.
        #
        # Now each selection season s is predicted by a model trained only on
        # seasons < s, so the threshold never sees a prediction from a model
        # that trained on the game being scored.
        oof = []
        for s in sorted(x for x in df.season.unique() if x < S):
            inner_tr = df[df.season < s]
            if len(inner_tr) < 300:
                continue
            i_mv = train_model(inner_tr[feats + [OPEN]], inner_tr[TGT], inner_tr,
                               f"wf_{name}_{S}_{s}_mv")
            i_mm = train_model(inner_tr[feats], inner_tr["home_margin"], inner_tr,
                               f"wf_{name}_{S}_{s}_mg")
            d = df[df.season == s].copy()
            d["mv"] = i_mv.predict(d[feats + [OPEN]].fillna(0))
            d["dis"] = i_mm.predict(d[feats].fillna(0)) + d[OPEN].values
            oof.append(d)
        if not oof:
            print(f"  {S:<8}{'no out-of-fold data':<16}")
            continue
        sel = pd.concat(oof, ignore_index=True)
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

        # Keep every individual bet so the pooled record can be broken down by
        # conviction afterwards. A record is only interesting if you can see
        # whether it improves as the model gets more confident.
        b = t[q].copy()
        b["bet_home"] = b["mv"] < 0
        sur = b["home_margin"].values + b[OPEN].values
        b["won"] = np.where(b["bet_home"], sur > 0, sur < 0)
        b["push"] = np.abs(sur) < 1e-9
        b["abs_dis"] = b["dis"].abs()
        b["abs_mv"] = b["mv"].abs()
        b["clv_pts"] = np.where(b["bet_home"], -b[TGT].values, b[TGT].values)
        b["test_season"] = S
        BETS.append(b[["test_season", "abs_dis", "abs_mv", "won", "push", "clv_pts"]])
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
    BETS.clear()
    run("NFL", nfl, nf, first_test=2022)
    tier_table("NFL")
    BETS.clear()
    run("CFB", cfb, cf, first_test=2022)
    tier_table("CFB")


if __name__ == "__main__":
    main()
