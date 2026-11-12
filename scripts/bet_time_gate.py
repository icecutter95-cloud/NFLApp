"""
Is rule 1 usable at BET time? Yes -- and it is cheaper than it looks.

THE QUESTION
nfl_threshold_sweep.py asked whether the gates can be loosened for volume and
concluded no. This file asks the prior question, which nearly killed the whole
idea: whether rule 1 can be applied before a bet is placed at all.
clv_points is open -> CLOSING line, the close is the kickoff number, and the
75% was graded at the OPENER -- a price obtainable only by betting early. The
worry was that "bet the opener" and "gate on movement" are mutually exclusive.

WHAT IT FOUND (2026-11-12)
They are not, because the movement finishes long before kickoff:

    at        have line   mean move   share of final   corr w/ final
    T-48h          88       +0.39          94%             0.97
    T-24h          88       +0.37          98%             0.98
    T-3h           88       +0.38          99%             0.99
    close          88       +0.40         100%             1.00

94% of it is in place two days out. The pick freezing a median 320 hours
before kickoff (clv_tracking.line_frozen_at) turned out to be irrelevant --
nothing forces the bet to be placed when the pick is generated.

The gate fires on the SAME 16 picks at every horizon from T-48h to T-3h. What
waiting costs is the price, not the signal:

    gated 16, graded at the opener        12-4   75.0%
    gated 16, graded at the real T-48h    11-5   68.8%
    average number surrendered by waiting        3.44 pts

So one win, and ~3.4 points of line value. Which makes the operable question a
staking question, and there are three answers, priced on 2026's 88 picks:

    A  flat 1u on all 88 at the opener                       -4.0u  (50.0%)
    B  wait to T-48h on everything, 1u all + 0.5u on gated   -6.2u  (47.1%)
    C  1u all at the OPENER + 0.5u top-up at T-48h on gated  -1.5u

C wins and B is the trap. Waiting on everything to gain the right to size 16
picks degrades all 88 -- 50.0% -> 47.1% -- and pays for the privilege. Taking
the base unit early and topping up only the gated increment keeps the opener
on every pick and pays the worse number on 0.5u of 16 bets. All three are
negative here because 2026 is a 50% season; the ORDERING is the finding.

WHAT THIS DOES NOT SHOW
That the gate is real. The 16 picks are the same cell the permutation test in
nfl_threshold_sweep.py priced at p=0.048, re-graded at a different number.
This file measures whether the gate is OPERABLE. Whether it is TRUE is still
answered only by the out-of-sample movement sweep, which says no.

THE JOIN, AND TWO TRAPS IN IT
line_history is keyed on odds-API hashes and clv_tracking on 2026_1_TB_CIN
synthetic keys -- zero overlap, which is why none of this had been measured.
Matched on (home_team, away_team) instead, which is unique per NFL season.

1. An exact kickoff-DATE join dropped 20 of 78 pick-games, every one a Sunday
   or Monday night game: 8:20pm ET is the next day in UTC and the two tables
   do not render it the same way. A systematic loss of the most heavily bet
   slate, not a random 26%.
2. line_history.commence_time DRIFTS. GB-CHI carries four values for one game
   (17:00, 17:03:41, 17:05:00, and an earlier-scheduled 20:25); 136 of 272
   pairs do this. Cutting a horizon against a kickoff wrong by 3.5 hours
   silently changes which snapshot is "the line at T-3h", so every cut here
   is taken against clv_tracking's commence_time and line_history's is never
   used as the kickoff.

It also keeps recording THROUGH kickoff -- some games' last snapshot is hours
after it, a live in-game number that encodes the result. Filtered and
asserted, the same hazard audited for cfb_open_close in
two_rule_filter_check.py.

Usage:
    python scripts/bet_time_gate.py
    python scripts/bet_time_gate.py --horizons 24,3
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

from score_week import fetch_all
from nfl_threshold_sweep import (nfl_frame, rate, show, MOVE_BARS,
                                 TICKET_BARS, VIG_RETURN)
from log_gate_readings import toward_us as _toward_us

HORIZONS = [48, 24, 6, 3]
MOVE_BAR = 1.5
TICKET_BAR = 50.0


def ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def toward_us(d, line_col, open_col="open_line"):
    """Frame-shaped wrapper over the canonical sign convention.

    The definition itself lives in log_gate_readings.toward_us, which is the
    production path, so there is exactly one copy of it. This project has
    been bitten by a restated sign convention before.
    """
    return _toward_us(d.predicted_side, d[open_col], d[line_col])


def grade_at(d, line_col):
    """Grade every pick against an arbitrary line. Vectorised grade_at()."""
    spread = d.bet_type == "spread"
    v = np.where(spread,
                 (d.home_score - d.away_score) + d[line_col],
                 (d.home_score + d.away_score) - d[line_col])
    ours = np.where(spread, d.predicted_side == "home",
                    d.predicted_side == "over")
    out = np.where((v > 0) == ours, "win", "loss")
    return pd.Series(np.where(np.isclose(v, 0), "push", out), index=d.index)


def bet_time_lines(d, horizons):
    """Attach the DraftKings line as of H hours before each kickoff.

    One pass over line_history per horizon. The snapshot taken is the LAST one
    strictly before the cutoff, which is what a bettor deciding at that moment
    would see.
    """
    lh = pd.DataFrame(fetch_all(
        "line_history",
        "game_id, recorded_at, spread_home, total, home_team, away_team, "
        "commence_time",
        book="draftkings"))
    lh["rec"] = ts(lh.recorded_at)
    lh["kick"] = ts(lh.commence_time)
    lh["day"] = lh.kick.dt.date
    print(f"line_history: {len(lh):,} draftkings snapshots, "
          f"{lh.game_id.nunique()} games")

    d = d.copy()
    d["kick_ct"] = ts(d.commence_time)

    # Matched on (home_team, away_team) alone, and every cut below is taken
    # against clv_tracking's kickoff -- line_history's own commence_time is
    # never used as the kickoff.
    #
    # Two reasons. An exact-date join dropped 20 of 78 pick-games, every one a
    # Sunday or Monday night game, because a 8:20pm ET kickoff is the next day
    # in UTC and the two tables do not render that the same way -- a
    # systematic loss of the most heavily bet slate, not a random 26%.
    # And line_history's commence_time DRIFTS: GB-CHI carries four of them
    # (17:00, 17:03:41, 17:05:00, and an earlier-scheduled 20:25), one real
    # game, and 136 of 272 pairs do this. Cutting a horizon against a kickoff
    # that is wrong by 3.5 hours silently shifts which snapshot is "the line
    # at T-3h", so the authoritative time has to come from the pick row.
    #
    # A pair meets at most once at a given venue in a season, and a playoff
    # rematch would fall AFTER this kickoff, so `rec < kick_ct` excludes it.
    key = ["home_team", "away_team"]
    m = d[key + ["game_id", "bet_type", "kick_ct"]].merge(
        lh[key + ["rec", "spread_home", "total"]], on=key, how="inner")
    m = m[m.rec < m.kick_ct]
    hit = m.game_id.nunique()
    print(f"  matched {hit}/{d.game_id.nunique()} pick-games on (home, away)")
    if hit < d.game_id.nunique():
        miss = sorted(set(d.game_id) - set(m.game_id))
        print(f"  unmatched ({len(miss)}): {', '.join(miss[:6])}")

    # The snapshot line in the units of the market we bet.
    m["snap"] = np.where(m.bet_type == "spread", m.spread_home, m.total)

    # line_history keeps recording through kickoff -- some games' last
    # snapshot lands hours AFTER it, a live in-game number that encodes the
    # result. The `rec < kick_ct` filter above removed those; each horizon cut
    # is stricter still. Asserted rather than assumed, because this is the
    # exact hazard audited for cfb_open_close in two_rule_filter_check.py.
    for h in horizons:
        cut = m.kick_ct - pd.Timedelta(hours=h)
        s = m[m.rec < cut]
        assert (s.rec < s.kick_ct - pd.Timedelta(hours=h)).all(), \
            f"snapshot inside the T-{h}h cut"
        last = (s.sort_values("rec").groupby(["game_id", "bet_type"]).tail(1)
                [["game_id", "bet_type", "snap"]]
                .rename(columns={"snap": f"line_t{h}"}))
        d = d.merge(last, on=["game_id", "bet_type"], how="left")
        d[f"move_t{h}"] = toward_us(d, f"line_t{h}")
        d[f"result_t{h}"] = grade_at(d, f"line_t{h}")
    return d


def accumulation(d, horizons):
    """How much of the final movement has happened by each horizon."""
    print("\n" + "=" * 74)
    print("HOW MUCH OF THE MOVEMENT EXISTS YET")
    print("=" * 74)
    print(f"  {'at':<10}{'have line':>11}{'mean move':>11}"
          f"{'share of final':>16}{'corr w/ final':>15}")
    fin = d.clv_points
    for h in horizons:
        mv = d[f"move_t{h}"]
        ok = mv.notna() & fin.notna()
        share = mv[ok].abs().sum() / fin[ok].abs().sum() if ok.any() else None
        corr = mv[ok].corr(fin[ok]) if ok.sum() > 2 else None
        print(f"  T-{h:<8}{int(ok.sum()):>11}{mv[ok].mean():>+11.2f}"
              f"{share:>15.0%}{corr:>15.2f}")
    print(f"  {'final':<10}{int(fin.notna().sum()):>11}"
          f"{fin.mean():>+11.2f}{1.0:>15.0%}{1.0:>15.2f}")
    print("\n  The movement is DONE well before kickoff: 94% of it by T-48h,")
    print("  98% by T-24h, correlation 0.97+ with the final figure. The")
    print("  freeze time being 320h out does not matter, because nothing")
    print("  forces the bet to be placed when the pick is generated. The")
    print("  gate is knowable two days early -- the cost is the price.")


def the_tradeoff(d, horizons):
    """The decisive table: the opener you give up to learn the gate fired.

    Column A grades the gated picks at the opener -- the published 75%, and a
    price you can only have by betting before the gate is knowable. Column B
    grades the SAME picks at the line actually on offer at that horizon, which
    is what waiting for the gate really buys.
    """
    print("\n" + "=" * 74)
    print("WHAT WAITING FOR THE GATE COSTS")
    print("=" * 74)
    full = d[(d.clv_points >= MOVE_BAR) & (d.our_tix < TICKET_BAR)]
    print(f"  the gate as published (move>={MOVE_BAR} by CLOSE, "
          f"tix<{TICKET_BAR:.0f}), graded at the opener:")
    show(full, "both rules, at open_line", indent=4)

    print(f"\n  evaluated at each horizon instead -- gate read from the line")
    print(f"  available then, and graded at that same line:")
    print(f"\n  {'at':<8}{'fires':>7}{'at opener':>22}{'at the real line':>24}")
    for h in horizons:
        g = d[(d[f"move_t{h}"] >= MOVE_BAR) & (d.our_tix < TICKET_BAR)]
        if g.empty:
            print(f"  T-{h:<6}{0:>7}{'--':>22}{'--':>24}")
            continue
        aw, al, ar = rate(g)
        bw = int((g[f"result_t{h}"] == "win").sum())
        bl = int((g[f"result_t{h}"] == "loss").sum())
        br = bw / (bw + bl) if bw + bl else None
        a = f"{aw}-{al}  {ar:.1%}"
        b = "--" if br is None else f"{bw}-{bl}  {br:.1%}"
        print(f"  T-{h:<6}{len(g):>7}{a:>22}{b:>24}")

    print("\n  the two columns are the same picks at two prices.")
    print("  The opener cannot be had by waiting, so the gap between them")
    print("  is what reading the gate actually costs -- one win, not the")
    print("  whole edge.")

    # And the cost in points, which is the whole argument in one line.
    g = d[(d.clv_points >= MOVE_BAR) & (d.our_tix < TICKET_BAR)]
    for h in horizons:
        col = f"line_t{h}"
        if g[col].notna().sum() == 0:
            continue
        give = np.abs(toward_us(g, col))
        print(f"  waiting to T-{h}h on the gated picks surrenders "
              f"{np.nanmean(give):.2f} pts of the number on average "
              f"(n={int(np.isfinite(give).sum())})")


def strategies(d, h=48, top_up=0.5):
    """The actual choice, priced. Three ways to play the same 88 picks.

    The gate cannot be read at the opener and the opener cannot be had by
    waiting, so "bet the opener AND size on the gate" is not available. But a
    THIRD option is, and it falls out of the horizon measurement: take the
    base unit at the opener on everything, then top up the gated picks later
    at whatever the number has become. The base keeps the good price on all 88
    and only the 0.5u increment pays the worse one.
    """
    print("\n" + "=" * 74)
    print(f"THE THREE WAYS TO PLAY IT  (top-up {top_up}u, read at T-{h}h)")
    print("=" * 74)

    def units(x, col, stake):
        w = int((x[col] == "win").sum())
        l = int((x[col] == "loss").sum())
        return stake * (w * VIG_RETURN - l), w, l

    res_h = f"result_t{h}"
    fires = (d[f"move_t{h}"] >= MOVE_BAR) & (d.our_tix < TICKET_BAR)
    g = d[fires]

    a, aw, al = units(d, "result", 1.0)
    print(f"  A  flat 1u on all {len(d)}, every bet at the OPENER")
    print(f"     {aw}-{al}  {aw / (aw + al):.1%}   {a:+.1f}u")

    b1, bw, bl = units(d, res_h, 1.0)
    b2, _, _ = units(g, res_h, top_up)
    print(f"\n  B  wait to T-{h}h on everything, 1u all + {top_up}u on the "
          f"{len(g)} gated")
    print(f"     {bw}-{bl}  {bw / (bw + bl):.1%}   {b1:+.1f}u base "
          f"{b2:+.1f}u top-up = {b1 + b2:+.1f}u")

    c2, cw, cl = units(g, res_h, top_up)
    print(f"\n  C  1u on all {len(d)} at the OPENER, then {top_up}u top-up at "
          f"T-{h}h on the {len(g)} gated")
    print(f"     base {a:+.1f}u  +  top-up {cw}-{cl} {c2:+.1f}u "
          f"= {a + c2:+.1f}u")

    print(f"\n  B gives up the opener on all {len(d)} picks to size up {len(g)}"
          f" of them.")
    print("  C keeps it on every base unit and pays the worse number only on")
    print("  the increment. C dominates B whenever the opener is the better")
    print("  price, which the CLV figures say it is by construction.")
    print("\n  All of this is contingent: the gated 16 are the same 16 cells")
    print("  the permutation test priced at p=0.048. This measures whether")
    print("  the gate is OPERABLE, not whether it is REAL.")


def bet_time_grid(d, h):
    """The nfl_threshold_sweep grid, rebuilt on movement knowable at T-h."""
    print("\n" + "=" * 74)
    print(f"THE GRID ON BET-TIME MOVEMENT (T-{h}h), graded at that line")
    print("=" * 74)
    g = d[d[f"move_t{h}"].notna()].copy()
    res = g[f"result_t{h}"]
    head = "".join(f"{('any' if t is None else '<' + str(int(t))):>12}"
                   for t in TICKET_BARS)
    print(f"  {'move >=':<9}{head}")
    for mv in MOVE_BARS:
        row = f"  {('any' if mv is None else str(mv)):<9}"
        for tx in TICKET_BARS:
            m = pd.Series(True, index=g.index)
            if mv is not None:
                m &= g[f"move_t{h}"] >= mv
            if tx is not None:
                m &= g.our_tix < tx
            w = int((res[m] == "win").sum())
            l = int((res[m] == "loss").sum())
            cell = "--" if w + l == 0 else f"{w / (w + l):.0%}"
            row += f"{cell + ' (' + str(w + l) + ')':>12}"
        print(row)
    print("\n  This is the only version of the gate that could ever be")
    print("  applied prospectively. Compare it to the 75% headline.")


def main():
    horizons = HORIZONS
    if "--horizons" in sys.argv:
        horizons = [int(x) for x in
                    sys.argv[sys.argv.index("--horizons") + 1].split(",")]

    d = nfl_frame()
    d = d[d.home_score.notna() & d.open_line.notna()]
    print(f"{len(d)} graded NFL qualifying picks with an opener and a score")
    d = bet_time_lines(d, horizons)
    accumulation(d, horizons)
    the_tradeoff(d, horizons)
    strategies(d, h=max(h for h in horizons if h >= 24) if any(h >= 24 for h in horizons) else max(horizons))
    bet_time_grid(d, min(horizons))


if __name__ == "__main__":
    main()
