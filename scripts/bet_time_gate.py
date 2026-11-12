"""
What is the record at a price that was actually available? Not the opener.

THE CORRECTION THIS FILE EXISTS TO MAKE
Every NFL number published before it -- the 75% gated record, the 12-4, the
50% season -- is graded at open_line. That is a lookahead price: measured
2026-11-12, the opener was same-day through week 4 and has been 43 to 71 days
stale since week 7, because the model freezes a pick ~13 days out while the
odds feed carries no line for that game until ~10 days out. The user does not
bet lookahead lines ("a week away, 6 days away, absolutely, but not
necessarily that far away"), so the opener is not a price, it is a scoring
convention. Everything here is therefore regraded at the line on offer 7, 6,
5, 3, 2 and 1 days before kickoff. All 88 graded picks have a line at every
one of those horizons.

THE RECORD AT A REAL PRICE
                         all picks         rule 1       both rules
    the OPENER        44-44  50% (88)   17-13 57% (30)  12-4 75% (16)
    T-168h (7d)       43-43  50% (86)   14-13 52% (27)   7-4 64% (11)
    T-144h (6d)       43-43  50% (86)   14-13 52% (27)   9-4 69% (13)
    T-72h  (3d)       41-46  47% (87)   13-14 48% (27)  10-5 67% (15)
    T-24h  (1d)       40-46  47% (86)   14-14 50% (28)  11-5 69% (16)

The ungated set barely moves: ~50% at the opener, ~47-50% at every real
price. What collapses is the gate. Rule 1 goes 57% -> 52%, and both rules
75% -> 64-69% on fewer picks.

WHERE THE GATE'S EDGE ACTUALLY LIVES
Split by opener staleness, graded at T-144h:

                       all picks        rule 1        both rules
    fresh  (7d avg)  31-32 49% (63)   6-9  40% (15)  4-3 57% (7)
    stale (53d avg)  12-11 52% (23)   8-4  67% (12)  5-1 83% (6)

    fresh openers: mean CLV -0.02, mean move by T-144h +0.08
    stale openers: mean CLV +1.57, mean move by T-144h +1.61

Rule 1 does nothing on a fresh opener -- 6-9 at a real price. Its whole record
is the stale half. And the stale half's CLV is not the model beating the
market: +1.57 of CLV against +1.61 already in the line by T-144h means the
entire lookahead correction has happened before the bet is placed. The gate is
largely a detector of "this pick was priced against a stale number".

That matters more each week, because from week 7 every pick is in the stale
regime. The gated stale subset is still 5-1 at a real price, which is either
something real or six games.

WHAT THIS INVERTS
The earlier recommendation here was: bet all 88 flat at the opener, top up the
gated ones. At real prices that is wrong, because the ungated base is ~50% --
below the 52.38% break-even -- so the flat base loses money and the gate stops
being a refinement and becomes the only part with a positive record. At
T-144h: flat 1u on all 88 is -3.9u, adding a 0.5u top-up is -1.8u, and the 13
gated picks alone are +4.2u. The thin thing being leaned on is 11-16 picks,
the same cell nfl_threshold_sweep.py priced at p=0.048.

A LIMIT WORTH STATING
The 2023-25 holdout that reported 55.6% is also graded at openers, and
movement_history stores only the open and the close -- no intermediate lines
-- so it cannot be regraded this way. The one out-of-sample number available
is measured on the convention this file just discredited.

THE JOIN, AND THREE TRAPS IN IT
line_history is keyed on odds-API hashes and clv_tracking on 2026_1_TB_CIN
synthetic keys -- zero overlap, which is why none of this had been measured.
Matched on (home_team, away_team), unique per NFL season.

1. An exact kickoff-DATE join dropped 20 of 78 pick-games, every one a Sunday
   or Monday night game: 8:20pm ET is the next day in UTC and the two tables
   do not render it the same way. A systematic loss of the most heavily bet
   slate, not a random 26%.
2. line_history.commence_time DRIFTS -- GB-CHI carries four values for one
   game and 136 of 272 pairs do this -- so every cut is taken against
   clv_tracking's commence_time instead.
3. It also records THROUGH kickoff, some games' last snapshot landing hours
   after it, which encodes the result. Filtered and asserted, the same hazard
   audited for cfb_open_close in two_rule_filter_check.py.

Usage:
    python scripts/bet_time_gate.py
    python scripts/bet_time_gate.py --horizons 168,144
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

# Horizons a bet would actually be placed at. The opener is NOT one of
# them: it is a lookahead price up to 10 weeks before kickoff, and the
# user's stated window is 'a week away, 6 days away, absolutely, but not
# necessarily that far away'. 168h/144h is that window.
HORIZONS = [168, 144, 120, 72, 48, 24]
BET_HORIZON = 144   # 6 days: the default realistic decision point
STALE_DAYS = 45     # opener older than this is a lookahead price
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
    print()
    print("  By T-48h the movement is essentially complete -- 94%, corr")
    print("  0.97. At a REALISTIC bet time it is not: 77% by T-168h and 76%")
    print("  by T-144h, correlation 0.78 and 0.89. Betting a week out means")
    print("  reading the gate on about three quarters of the movement, and")
    print("  waiting for the rest means paying for it in the price.")


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

    print()
    print("  Both columns are the same picks at two prices, and the opener",
          "column")
    print("  is the unbettable one: it is a lookahead number up to ten weeks")
    print("  out. The gap is not what waiting costs, it is how much of the")
    print("  published record came from a price that was never on offer.")

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


def opener_era(d):
    """Tag each pick by how stale the line open_line was measured against.

    The single most important split in this file. Through week 4 the opener
    was same-day; from week 7 it has been 43-71 days old, because the model
    freezes ~13 days out while the odds feed carries no line for that game
    until ~10 days out. A large move_toward_us on a stale opener is a
    lookahead correction that already happened.
    """
    a = pd.DataFrame(fetch_all("nfl_pick_opener_age",
                               "game_id, bet_type, opener_age_days"))
    d = d.merge(a, on=["game_id", "bet_type"], how="left")
    d["era"] = np.where(d.opener_age_days > STALE_DAYS, "stale", "fresh")
    return d


def wl(x, rescol):
    """(wins, losses, rate) at an arbitrary result column."""
    w = int((x[rescol] == "win").sum())
    l = int((x[rescol] == "loss").sum())
    return w, l, (w / (w + l) if w + l else None)


def cell(x, rescol):
    w, l, r = wl(x, rescol)
    return "--" if r is None else f"{w}-{l} {r:.0%} (n={w + l})"


def realistic_record(d, horizons):
    """The record at prices that were actually available to take.

    Everything published before this graded at open_line. That is a lookahead
    number, not a price this bettor takes, so the opener row here is the
    fantasy baseline and every row below it is the real one.
    """
    print()
    print("=" * 78)
    print("THE RECORD AT A PRICE YOU WOULD ACTUALLY BET")
    print("=" * 78)
    print(f"  {'graded at':<16}{'all picks':>21}{'rule 1':>21}{'both rules':>21}")

    rows = [("the OPENER", "result", "clv_points")]
    rows += [(f"T-{h}h ({h // 24}d)", f"result_t{h}", f"move_t{h}")
             for h in horizons]
    for label, rescol, movecol in rows:
        g = d[d[rescol].notna()]
        r1 = g[g[movecol] >= MOVE_BAR]
        both = r1[r1.our_tix < TICKET_BAR]
        print(f"  {label:<16}{cell(g, rescol):>21}{cell(r1, rescol):>21}"
              f"{cell(both, rescol):>21}")
    print()
    print("  The opener row is not a result, it is a scoring convention. Read")
    print("  the T-168h and T-144h rows as the record.")


def by_era(d, h=None):
    """The same picks split by whether the opener was fresh or stale.

    This is where the gate's apparent edge turns out to live. If rule 1 only
    works on stale openers it is not reading sharp money, it is reading the
    gap between a lookahead price and the real market.
    """
    h = h or BET_HORIZON
    rescol, movecol = f"result_t{h}", f"move_t{h}"
    print()
    print("=" * 78)
    print(f"SPLIT BY OPENER STALENESS  (graded at T-{h}h, the real price)")
    print("=" * 78)
    print(f"  {'opener':<22}{'all picks':>21}{'rule 1':>21}{'both rules':>21}")
    for era in ("fresh", "stale"):
        g = d[(d.era == era) & d[rescol].notna()]
        r1 = g[g[movecol] >= MOVE_BAR]
        both = r1[r1.our_tix < TICKET_BAR]
        lbl = f"{era} ({g.opener_age_days.mean():.0f}d avg)" if len(g) else era
        print(f"  {lbl:<22}{cell(g, rescol):>21}{cell(r1, rescol):>21}"
              f"{cell(both, rescol):>21}")

    print()
    print("  and the same split graded at the OPENER, for contrast:")
    for era in ("fresh", "stale"):
        g = d[(d.era == era) & d.result.notna()]
        r1 = g[g.clv_points >= MOVE_BAR]
        both = r1[r1.our_tix < TICKET_BAR]
        print(f"  {era:<22}{cell(g, 'result'):>21}{cell(r1, 'result'):>21}"
              f"{cell(both, 'result'):>21}")

    print()
    for era in ("fresh", "stale"):
        g = d[d.era == era]
        if g.empty:
            continue
        print(f"  {era:<6} openers: mean CLV {g.clv_points.mean():+.2f}, "
              f"mean move by T-{h}h {g[movecol].mean():+.2f}, n={len(g)}")


def strategies(d, h=BET_HORIZON, top_up=0.5):
    """The staking choice, priced at a line that was actually available.

    An earlier version of this function compared everything against betting
    the opener. That has been dropped: the opener is a lookahead price up to
    ten weeks before kickoff and is not a number this bettor takes, so an
    option built on it is not an option. Every row below is placed and graded
    at T-{h}h.
    """
    print()
    print("=" * 78)
    print(f"THE STAKING CHOICE AT T-{h}h  (top-up {top_up}u)")
    print("=" * 78)
    rescol, movecol = f"result_t{h}", f"move_t{h}"
    g = d[d[rescol].notna()]

    def units(x, stake):
        w, l, _ = wl(x, rescol)
        return stake * (w * VIG_RETURN - l), w, l

    gated = g[(g[movecol] >= MOVE_BAR) & (g.our_tix < TICKET_BAR)]

    a, aw, al = units(g, 1.0)
    print(f"  A  flat 1u on all {len(g)}")
    print(f"     {aw}-{al}  {aw / (aw + al):.1%}   {a:+.1f}u")

    b2, bw, bl = units(gated, top_up)
    print()
    print(f"  B  1u on all {len(g)}, plus {top_up}u on the {len(gated)} gated")
    print(f"     base {a:+.1f}u  +  top-up {bw}-{bl} {b2:+.1f}u = {a + b2:+.1f}u")

    c, cw, cl = units(gated, 1.0)
    print()
    print(f"  C  1u on the {len(gated)} gated picks ONLY, nothing else")
    print(f"     {cw}-{cl}  {cw / (cw + cl):.1%}   {c:+.1f}u")

    print()
    print("  This inverts the earlier recommendation. Graded at the opener the")
    print("  ungated set looked like a 50% base worth betting flat; at a price")
    print("  actually available it is still ~50%, which is BELOW the 52.38%")
    print("  break-even, so A loses and the gate stops being a sizing")
    print("  refinement and becomes the only part carrying a positive record.")
    print("  On 11-16 picks that is a thin thing to lean on -- and it is the")
    print("  same cell the permutation test priced at p=0.048.")


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
    d = opener_era(bet_time_lines(d, horizons))
    accumulation(d, horizons)
    realistic_record(d, horizons)
    by_era(d)
    the_tradeoff(d, horizons)
    strategies(d)
    bet_time_grid(d, min(horizons))


if __name__ == "__main__":
    main()
