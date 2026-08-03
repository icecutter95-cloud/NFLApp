"""
Is our CLV toxic?

The uncomfortable pattern across this project: as the model's confidence rises,
CLV rises and win rate FALLS. Two explanations fit and they have opposite
consequences.

  BENIGN       CLV points are not economically uniform. Low-movement games sit
               near key numbers (67% within half a point of 3 or 7, against 36%
               of high-movement games), so their fewer points are worth more.
               The dissociation is then a measurement artefact.

  TOXIC        We earn CLV precisely in games where the closing move carries
               real information -- injury, weather, late news. We get the points
               and the wrong side. If so, CLV is a bad optimisation target and
               the entire project premise is weaker than assumed.

The test that separates them, in one sentence: compare our model's bets against
GENERIC bets that earned the same CLV.

  generic bet   for every game, back the side the line moved toward, at the
                opener. By construction its CLV is |movement|, and it is not
                selected by any model.

If a generic bet earning +2 points of CLV covers 58%, and OUR bets earning +2
points cover 49%, the CLV we earn is worse than the CLV anyone else earns, and
that is adverse selection. If the two match, the low win rate in high-confidence
buckets is composition or noise.

The comparison is also run controlling for key-number proximity, since that is
the known compositional difference.

Usage:
    python test_adverse_selection.py
"""

import contextlib
import io as _io
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import walk_forward as W

OPEN, TGT = "week_open_spread_home", "week_spread_movement"
BANDS = [(0.01, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99)]


def generic_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Back the side the line moved toward, in every game. No model involved."""
    d = df.dropna(subset=[OPEN, TGT, "home_margin"]).copy()
    mv = d[TGT]
    d["bet_home"] = mv < 0
    d["clv"] = mv.abs()                       # positive by construction
    sur = d["home_margin"] + d[OPEN]
    d["won"] = np.where(d["bet_home"], sur > 0, sur < 0)
    d = d[sur.abs() > 1e-9]
    d["near_key"] = (((d[OPEN].abs() - 3).abs() <= 0.5)
                     | ((d[OPEN].abs() - 7).abs() <= 0.5))
    return d


def model_bets(name, load) -> pd.DataFrame:
    W.BETS.clear()
    df, feats = load()
    with contextlib.redirect_stdout(_io.StringIO()):
        W.run(name, df, feats, first_test=2022)
    b = pd.concat(W.BETS, ignore_index=True)
    return b[~b["push"]]


def compare(name, generic, ours):
    # Restrict the generic pool to the seasons actually tested, so the baseline
    # is not built from a different era than the bets it is grading.
    seasons = sorted(ours["test_season"].unique())
    g = generic[generic.season.isin(seasons)]
    print(f"\n{name}  —  {len(ours)} model bets vs {len(g)} generic bets "
          f"(seasons {seasons})")
    print(f"  {'CLV earned':<12}{'generic n':>11}{'generic%':>10}"
          f"{'ours n':>8}{'ours%':>8}{'gap':>8}")

    tot_exp = tot_act = tot_n = 0
    for lo, hi in BANDS:
        gb = g[(g.clv >= lo) & (g.clv < hi)]
        ob = ours[(ours.clv_pts >= lo) & (ours.clv_pts < hi)]
        if len(gb) < 40 or len(ob) < 15:
            continue
        gp, op = gb.won.mean() * 100, ob.won.mean() * 100
        band = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        print(f"  {band:<12}{len(gb):>11}{gp:>9.1f}%{len(ob):>8}{op:>7.1f}%"
              f"{op - gp:>+8.1f}")
        tot_exp += gp / 100 * len(ob)
        tot_act += ob.won.sum()
        tot_n += len(ob)

    if tot_n:
        exp = tot_exp / tot_n * 100
        act = tot_act / tot_n * 100
        se = np.sqrt(exp / 100 * (1 - exp / 100) / tot_n) * 100
        z = (act - exp) / se if se else 0
        print(f"  {'POOLED':<12}{'':>11}{exp:>9.1f}%{tot_n:>8}{act:>7.1f}%"
              f"{act - exp:>+8.1f}   z={z:+.2f}")
        verdict = ("ADVERSE SELECTION" if z < -1.96 else
                   "our CLV is BETTER than generic" if z > 1.96 else
                   "no detectable difference — CLV is not toxic")
        print(f"  -> {verdict}")

    # Same comparison, but only where the opener sits away from a key number,
    # removing the known compositional difference.
    go = g[~g.near_key]
    print(f"  away from key numbers: generic {go.won.mean()*100:.1f}% "
          f"({len(go)} bets, CLV>0)")


def main():
    for name, load in [("NFL", W.load_nfl), ("CFB", W.load_cfb)]:
        df, _ = load()
        compare(name, generic_universe(df), model_bets(name, load))
    print("\nbreak-even at -110 is 52.38%")


if __name__ == "__main__":
    main()
