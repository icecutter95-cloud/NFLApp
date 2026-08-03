"""
Probability-weighted CLV: convert points of line value into win probability.

Why points are the wrong unit
-----------------------------
A point of spread is not economically uniform. Moving 2.5 -> 3.5 crosses the
single densest margin in football; moving 8 -> 9 crosses almost nothing. Raw
point CLV treats those identically, which flatters large moves through empty
regions of the margin distribution and understates half-point moves across a
key number.

This project already has evidence that matters: low-movement games sit near key
numbers (67% within half a point of 3 or 7, against 36% of high-movement games),
so the cheap-versus-expensive distinction lines up exactly with the bets whose
win rate diverged from their CLV.

Method
------
For a bet held at home-line h_bet on a game that closed at h_close:

    P(home covers at h) = P(margin > -h)
    prob_CLV(home bet)  = P(margin > -h_bet) - P(margin > -h_close)
    prob_CLV(away bet)  = the negative of that

The margin distribution is taken EMPIRICALLY and conditioned on the closing
line, not assumed normal. A normal approximation would smooth away the very
spikes this is meant to measure -- the whole point is that P(margin > -h) jumps
as h crosses 3.

Conditioning matters: for a game closing at -3, the residual sits exactly on the
densest margin, while a game closing at -5.5 can never land on it. So games are
bucketed by closing line and the empirical distribution is built within a
bucket.

Usage:
    python prob_clv.py
"""

import contextlib
import io as _io
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import walk_forward as W

OPEN, TGT = "week_open_spread_home", "week_spread_movement"
BUCKET = 2.0        # closing lines within +/- this are pooled to build the CDF
MIN_IN_BUCKET = 120


class CoverCurve:
    """Empirical P(home covers at line h), conditioned on the closing line."""

    def __init__(self, df: pd.DataFrame):
        d = df.dropna(subset=["home_margin", "closing_spread_home"])
        self.close = d["closing_spread_home"].to_numpy()
        self.margin = d["home_margin"].to_numpy()

    def p_home_covers(self, h: float, h_close: float) -> float:
        # Neighbouring closing lines only, widening until the sample is usable.
        w = BUCKET
        while True:
            m = self.margin[np.abs(self.close - h_close) <= w]
            if len(m) >= MIN_IN_BUCKET or w > 12:
                break
            w += 1.0
        if len(m) == 0:
            return 0.5
        # Pushes split. A margin exactly on the number is neither a win nor a
        # loss, and counting it either way biases half-point comparisons.
        wins = (m > -h).sum()
        push = (m == -h).sum()
        return float((wins + 0.5 * push) / len(m))

    def prob_clv(self, h_bet, h_close, bet_home) -> float:
        p_bet = self.p_home_covers(h_bet, h_close)
        p_cls = self.p_home_covers(h_close, h_close)
        d = p_bet - p_cls
        return d if bet_home else -d


def demo_curve(curve: CoverCurve):
    """Show that a point is worth different amounts in different places."""
    print("\nWHAT ONE POINT IS ACTUALLY WORTH (home side, points of win prob)")
    print(f"  {'close':>7}{'move':>14}{'gain':>9}")
    for h_close, lo, hi, label in [
        (-3.0, -3.5, -2.5, "-3.5 -> -2.5  (across 3)"),
        (-3.0, -3.0, -2.5, "-3.0 -> -2.5  (off 3)"),
        (-7.0, -7.5, -6.5, "-7.5 -> -6.5  (across 7)"),
        (-8.5, -9.0, -8.0, "-9.0 -> -8.0  (empty)"),
        (-5.5, -6.0, -5.0, "-6.0 -> -5.0  (empty)"),
        (-10.0, -10.5, -9.5, "-10.5 -> -9.5 (across 10)"),
    ]:
        g = (curve.p_home_covers(hi, h_close) - curve.p_home_covers(lo, h_close)) * 100
        print(f"  {h_close:>7.1f}{label:>26}{g:>+8.1f}pp")


def analyse(name, load):
    df, feats = load()
    curve = CoverCurve(df)
    if name == "NFL":
        demo_curve(curve)

    W.BETS.clear()
    with contextlib.redirect_stdout(_io.StringIO()):
        W.run(name, df, feats, first_test=2022)
    b = pd.concat(W.BETS, ignore_index=True)
    b = b[~b["push"]].copy()

    # Rebuild each bet's lines from what the tier table stored plus the source.
    src = df.set_index(["season", "home_team", "away_team"]) if "home_team" in df else None
    # clv_pts is signed already; recover the bet/close lines from it.
    # h_bet - h_close = clv for a home bet, and the negative for an away bet.
    b["h_close"] = b["close_line"]
    b["h_bet"] = b["open_line"]
    b["prob_clv"] = [curve.prob_clv(hb, hc, bh) * 100
                     for hb, hc, bh in zip(b.h_bet, b.h_close, b.bet_home)]

    print(f"\n{name}  —  {len(b)} bets")
    print(f"  mean point CLV {b.clv_pts.mean():+.2f}   "
          f"mean probability CLV {b.prob_clv.mean():+.2f}pp")

    print(f"\n  {'by POINT CLV':<14}{'n':>6}{'win%':>8}{'pts':>8}{'prob':>9}")
    for lo, hi in [(-99, 0.01), (0.01, 1.0), (1.0, 2.0), (2.0, 99)]:
        s = b[(b.clv_pts >= lo) & (b.clv_pts < hi)]
        if len(s) < 20:
            continue
        band = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        print(f"  {band:<14}{len(s):>6}{s.won.mean()*100:>7.1f}%"
              f"{s.clv_pts.mean():>+8.2f}{s.prob_clv.mean():>+8.2f}pp")

    print(f"\n  {'by PROB CLV':<14}{'n':>6}{'win%':>8}{'pts':>8}{'prob':>9}")
    for lo, hi in [(-99, 0.0), (0.0, 2.0), (2.0, 5.0), (5.0, 99)]:
        s = b[(b.prob_clv >= lo) & (b.prob_clv < hi)]
        if len(s) < 20:
            continue
        band = f"{lo}-{hi}pp" if hi < 99 else f"{lo}+pp"
        print(f"  {band:<14}{len(s):>6}{s.won.mean()*100:>7.1f}%"
              f"{s.clv_pts.mean():>+8.2f}{s.prob_clv.mean():>+8.2f}pp")

    # Which unit better predicts whether the bet actually won?
    for unit in ("clv_pts", "prob_clv"):
        r = np.corrcoef(b[unit], b.won.astype(float))[0, 1]
        print(f"  corr({unit:<9}, won) = {r:+.4f}")


def main():
    for name, load in [("NFL", W.load_nfl), ("CFB", W.load_cfb)]:
        analyse(name, load)


if __name__ == "__main__":
    main()
