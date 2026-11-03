"""
The two-rule filter: strong in college, absent in the NFL. Measured, not argued.

THE RULES
Applied by hand to the college board since roughly week 6:
  rule 1  the line moved >= 1.5 points TOWARD the side the model likes
  rule 2  fewer than 50% of tickets are on that side
Together they are a sharp-money proxy. Neither asks whether the model is
right; both ask whether other bettors agree, and whether it was the public
who agreed. The question this answers is whether they transfer to the NFL.

WHAT IT FOUND (2026-11-03)

                           CFB (verified pre-kick)      NFL 2026
    no filter              52.2%  (n=322)               48.1%  (n=77)
    rule 1 only            66.0%  (n=103, p=0.001)      53.8%  (n=26)
    both rules             76.0%  (n=50,  p<0.001)      71.4%  (n=14, p=0.180)

The NFL cell is 14 picks and proves nothing either way. The informative
number is the out-of-sample one, from movement_history's select/holdout
split (2023-25):

    rule 1, NFL holdout    55.4% (n=65)   vs   55.8% (n=77) for everything else

Zero separation. In-sample the same gate showed 68.3% vs 57.0%, so the
apparent NFL edge is a selection artefact of the period it was found in.

WHY THE ASYMMETRY IS STRUCTURAL AND NOT AN ACCIDENT OF SAMPLE
1.5 points of NFL spread is 1.5/13.5 = 0.11 sigma, worth ~4.4 points of win
probability. The NFL holdout delivers 55.4% against a ~55% base rate: that is
the CLV arithmetic and nothing on top of it. The gate confirms a slightly
better number, which we already knew we had.

College sigma is wider (~16), so 1.5 points buys only ~3.7 points there --
yet the gate moves CFB from 52.2% to 66.0%. Fourteen points cannot come from
CLV arithmetic. It is selecting PICKS, not better numbers, and that only
works where the market holds information the model lacks. 130 FBS teams on
unequal attention with soft openers on small games, against an NFL board
priced by every sharp shop on earth. Our CFB model sits at 52.2% and our NFL
model at ~55% on its honest holdout, so in the NFL the model and the sharp
market are two careful reads of the same well-covered data -- correlated, and
the gate is largely redundant with the edge calculation it filters.

Rule 2 should transfer worse for a less obvious reason: NFL public money is
large and predictable enough that books shade openers for it, so "the public
is on the other side" is partly in the price already. Nobody shades 60
college games a week with that care. It cannot be backtested either way --
nfl_public_splits begins September 2026.

THREE THINGS THAT COULD HAVE FAKED THE COLLEGE RESULT, AND DID NOT
1. Look-ahead in the prediction. cfb_predictions upserts, so finished games
   were re-stamped with later inputs (see restore_cfb_prekickoff.py). Split
   on whether a pre-kickoff cfb_prediction_log entry exists: the filter is
   76.0% on the 322 VERIFIED picks and 60.9% on the 464 that cannot be
   verified. The clean half is the stronger half, which is the opposite of
   what contamination produces.
2. A post-kickoff closing line. clv_points keys off cfb_open_close's close,
   which was once the newest snapshot of any kind -- a live in-game number on
   a played game, which would encode the result. Audited here: of the
   qualifying picks, 0 have zero pre-kickoff snapshots, 0 close at or after
   kickoff, minimum 12 pre-kick snapshots, average close 2.3h before kick.
3. A knife-edge threshold. The cutoffs were chosen by eye after seeing
   college data, so the sweep exists to show whether the neighbourhood is
   uniformly raised or the result sits on one lucky cell. It is uniformly
   raised and monotone in both gates -- which then raised a FOURTH worry.
4. A bad opener. The sweep reaches 100% (n=22) at move >= 2.5 / tix < 45,
   and no true 60-65% edge produces 22-0. Both the grade and clv_points are
   computed from the same open_line, so one mis-keyed opener manufactures
   large CLV and an easy win at once -- a perfectly correlated artefact, and
   cfb_open_close reconciles duplicate game_ids through a lineage/canon join
   that could plausibly mis-assign one. Tested by re-grading the same picks
   at the CLOSING line, where an opener error cannot help:

       both rules      at open 76.0%   at close 68.0%   (n=50)
       tight 2.5/<45   at open 100.0%  at close 86.4%   (n=22)

   The edge survives, and the 8-point gap is what the arithmetic predicts:
   qualifying picks move 2.82 points on average, which at sigma 16 is 0.176
   sigma, or ~7 points of win probability. Movement sizes are plausible too
   (62% of qualifying picks move 2-4 points, none more than 7), so the opens
   are real numbers rather than typos.

   So the 76% decomposes: a 52.2% base rate, plus ~16 points of pick
   selection that is still there at the close, plus ~8 points of owning a
   better number than the market closed at. The first component is the one
   that cannot be an opener artefact, and it is the larger of the two.

WHAT IS STILL WRONG WITH THE COLLEGE NUMBER
One season, one ten-week window, no out-of-sample split -- the same thing
that made the NFL in-sample figure look good. 76% on 50 picks is not a 76%
edge; the 95% interval runs roughly 62-87%, and the honest reading is "well
clear of break-even", not "three in four". The tight cells are worse on this
count, not better: 86% on 22 picks at thresholds chosen after seeing these
same picks is the most over-fitted number on the page, which is why the sweep
prints sample sizes next to every rate. It is still the best-evidenced result
in this project, and the only one that has survived a pre-kickoff audit, a
closing-line audit and an opener audit.

WHAT TO DO WITH IT
College: keep it. NFL: the half with NFL evidence behind it is the ADVERSE
direction, not the favourable one -- line running 4+ points away from us went
4-15 (21%). So the NFL version belongs as a veto (skip when the market runs
away) rather than a requirement (demand it come to us), which also keeps far
more volume. Judge the NFL two-rule set at ~40 picks; at ~3 a week that is
most of a season, which is itself the argument for trusting the college
result first.

Usage:
    python scripts/two_rule_filter_check.py
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

warnings.filterwarnings("ignore")

from score_week import fetch_all

MOVE_BAR = 1.5     # rule 1: points the line must come toward us
TICKET_BAR = 50.0  # rule 2: max % of tickets on our side

# Spread sigma by sport, for turning points of CLV into win probability.
SIGMA = {"NFL": 13.5, "CFB": 16.0}


def ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def rec(x, label, indent=4):
    """Print W-L, rate and a two-sided binomial p-value. Returns the rate."""
    w = int((x.result == "win").sum())
    lo = int((x.result == "loss").sum())
    pad = " " * indent
    if w + lo == 0:
        print(f"{pad}{label:<34} n=0")
        return None
    r = w / (w + lo)
    print(f"{pad}{label:<34} {w:3d}-{lo:<3d} {r:5.1%} "
          f"(p={binomtest(w, w + lo).pvalue:.3f}) n={w + lo}")
    return r


def our_ticket_pct(d):
    """Ticket % on the side we like. home/over read home_bets_pct."""
    return np.where(d.predicted_side.isin(["home", "over"]),
                    d.home_bets_pct, d.away_bets_pct)


def last_pre_kick_splits(splits, games):
    """The final splits capture BEFORE kickoff for each game-market.

    Captures after kickoff exist because the splits job does not stop at
    kick; using the newest row would read a line that moved on live play.
    """
    s = splits.copy()
    s["c"] = ts(s.captured_at)
    m = s.merge(games[["game_id", "bet_type", "kick"]],
                on=["game_id", "bet_type"], how="inner")
    m = m[m.c < m.kick]
    return (m.sort_values("c")
             .groupby(["game_id", "bet_type"]).tail(1)
             [["game_id", "bet_type", "home_bets_pct", "away_bets_pct"]])


def report(d, tag):
    r1 = d.clv_points >= MOVE_BAR
    r2 = d.our_tix < TICKET_BAR
    print(f"\n  {tag}  ({len(d)} graded)")
    rec(d, "no filter")
    rec(d[r1], "rule 1 only (moved toward us)")
    rec(d[r1 & r2], "BOTH rules")
    for bt in ("spread", "total"):
        rec(d[(d.bet_type == bt) & r1 & r2], f"  both, {bt}")
    rec(d[~(r1 & r2)], "everything the filter rejects")


def cfb():
    print("=" * 72)
    print("COLLEGE")
    print("=" * 72)
    t = pd.DataFrame(fetch_all("cfb_tracking"))
    t = t[t.bet_type.isin(["spread", "total"])].copy()
    t["kick"] = ts(t.commence_time)

    sp = last_pre_kick_splits(pd.DataFrame(fetch_all("cfb_public_splits")), t)
    d = t.merge(sp, on=["game_id", "bet_type"], how="left")
    d["our_tix"] = our_ticket_pct(d)

    # Does a pre-kickoff log entry exist? Without one, the current
    # cfb_predictions row may have been re-stamped after the game finished.
    lg = pd.DataFrame(fetch_all("cfb_prediction_log"))
    lg["la"], lg["k"] = ts(lg.logged_at), ts(lg.commence_time)
    ok = set(map(tuple, lg[lg.la < lg.k][["game_id", "bet_type"]]
                 .drop_duplicates().values))
    d["verified"] = [(r.game_id, r.bet_type) in ok for r in d.itertuples()]

    g = d[d.result.isin(["win", "loss"]) & d.predicted_side.notna()].copy()
    print(f"\n{len(g)} graded picks with a side | "
          f"{int(g.our_tix.notna().sum())} have splits | "
          f"{int(g.verified.sum())} verifiable pre-kickoff")

    report(g[g.verified], "VERIFIED pre-kickoff -- the trustworthy half")
    report(g[~g.verified], "CANNOT verify -- reported, not relied on")

    audit_close(g[g.verified])
    sweep(g[g.verified])
    regrade_at_close(d[d.home_score.notna() & d.predicted_side.notna()
                       & d.verified & d.closing_line.notna()])


def audit_close(g):
    """Rule 1 only means anything if the closing line predates kickoff."""
    oc = pd.DataFrame(fetch_all("cfb_open_close"))
    oc["closed"], oc["ct"] = ts(oc.closed_at), ts(oc.commence_time)
    # cfb_tracking already exposes n_snapshots, so take it from cfb_open_close
    # under a distinct name rather than letting the merge suffix both copies.
    q = g[g.clv_points >= MOVE_BAR].merge(
        oc[["game_id", "closed", "ct", "n_snapshots"]]
        .rename(columns={"n_snapshots": "snaps"}),
        on="game_id", how="left")
    after = int((q.closed >= q.ct).sum())
    zero = int((q.snaps == 0).sum())
    hrs = ((q.ct - q.closed).dt.total_seconds() / 3600).mean()
    print(f"\n  closing-line audit over {len(q)} qualifying picks")
    print(f"    closes at/after kickoff {after} | zero pre-kick snapshots "
          f"{zero} | min snapshots {int(q.snaps.min())}")
    print(f"    average close {hrs:.1f}h before kickoff")
    if after or zero:
        print("    -> LEAK: clv_points partly encodes the result. Stop here.")
    else:
        print("    -> clean. The movement was information available pre-kick.")


def sweep(g):
    """A raised neighbourhood, or one lucky cell?"""
    print("\n  threshold sweep -- both rules, VERIFIED picks only")
    head = "".join(f"{'tix <' + str(t):>14}" for t in (60, 55, 50, 45, 40))
    print(f"    {'move >=':>8}{head}")
    for mv in (1.0, 1.5, 2.0, 2.5):
        cells = []
        for tx in (60, 55, 50, 45, 40):
            s = g[(g.clv_points >= mv) & (g.our_tix < tx)]
            w = int((s.result == "win").sum())
            n = w + int((s.result == "loss").sum())
            cells.append(f"{w / n:.0%} (n={n})" if n >= 10 else f"-- (n={n})")
        print(f"    {mv:>8.1f}" + "".join(f"{c:>14}" for c in cells))
    print("    cells under n=10 print as -- rather than as a rate")


def grade_at(row, line):
    """Grade a pick against an arbitrary line. Mirrors cfb_tracking's CASE."""
    if pd.isna(line):
        return None
    if row.bet_type == "spread":
        v = (row.home_score - row.away_score) + line
        if v == 0:
            return "push"
        return "win" if (v > 0) == (row.predicted_side == "home") else "loss"
    v = (row.home_score + row.away_score) - line
    if v == 0:
        return "push"
    return "win" if (v > 0) == (row.predicted_side == "over") else "loss"


def regrade_at_close(g):
    """Could a mis-keyed opener have produced the whole thing?

    cfb_tracking grades at open_line and clv_points is also derived from
    open_line, so a wrong opener inflates both at once. Re-grading at the
    close removes that channel: an opener error cannot help a pick that is
    scored against a number taken hours before kickoff.
    """
    g = g.copy()
    g["at_open"] = [grade_at(r, r.open_line) for r in g.itertuples()]
    g["at_close"] = [grade_at(r, r.closing_line) for r in g.itertuples()]
    print(f"\n  opener audit -- re-graded at the CLOSE ({len(g)} picks "
          f"with both numbers)")
    for mv, tx, tag in ((MOVE_BAR, TICKET_BAR, "both rules"),
                        (2.5, 45, "tight 2.5 / <45")):
        s = g[(g.clv_points >= mv) & (g.our_tix < tx)]
        if s.empty:
            continue
        print(f"    {tag}  n={len(s)}, mean |movement| "
              f"{s.actual_movement.abs().mean():.2f} pts "
              f"(all verified {g.actual_movement.abs().mean():.2f})")
        for col, label in (("at_open", "at the open"),
                           ("at_close", "at the close")):
            w = int((s[col] == "win").sum())
            n = w + int((s[col] == "loss").sum())
            print(f"      {label:<14} {w:3d}-{n - w:<3d} {w / n:6.1%} "
                  f"(p={binomtest(w, n).pvalue:.3f})")
    q = g[(g.clv_points >= MOVE_BAR) & (g.our_tix < TICKET_BAR)]
    print("    movement sizes, qualifying vs all verified:")
    for lo, hi in ((0, 2), (2, 4), (4, 7), (7, 99)):
        a = ((q.actual_movement.abs() >= lo)
             & (q.actual_movement.abs() < hi)).mean()
        b = ((g.actual_movement.abs() >= lo)
             & (g.actual_movement.abs() < hi)).mean()
        print(f"      |move| {lo}-{hi:<3} qualifying {a:5.0%}   all {b:5.0%}")
    print("    -> an edge that survives at the close is pick selection, not a")
    print("       typo in the opener. The open-to-close gap is the CLV value")
    print("       of the better number, and it matches the arithmetic.")


def nfl():
    print("\n" + "=" * 72)
    print("NFL -- the same two rules")
    print("=" * 72)
    t = pd.DataFrame(fetch_all("clv_tracking"))
    t = t[t.bet_type.isin(["spread", "total"])].copy()
    t["kick"] = ts(t.commence_time)

    raw = pd.DataFrame(fetch_all("nfl_public_splits"))
    sp = last_pre_kick_splits(raw, t)
    d = t.merge(sp, on=["game_id", "bet_type"], how="left")
    d["our_tix"] = our_ticket_pct(d)

    g = d[(d.qualifies == True) & d.result.isin(["win", "loss"])  # noqa: E712
          & d.predicted_side.notna()].copy()
    report(g, "2026 live, qualifying picks only")
    print(f"\n  nfl_public_splits begins {ts(raw.captured_at).min():%Y-%m-%d}"
          f" -- rule 2 cannot be backtested at all.")


def nfl_out_of_sample():
    """The only honest test available: rule 1 across movement_history's split."""
    print("\n" + "=" * 72)
    print("NFL OUT OF SAMPLE -- rule 1 on movement_history (2023-25)")
    print("=" * 72)
    m = pd.DataFrame(fetch_all("movement_history"))
    m = m[(m.qualifies == True) & m.result.isin(["win", "loss"])  # noqa: E712
          & m.clv_points.notna()].copy()
    for period in ("select", "holdout"):
        s = m[m.period == period]
        print(f"\n  {period}  ({len(s)} graded)")
        a = rec(s[s.clv_points >= MOVE_BAR], f"rule 1 (moved >= {MOVE_BAR})")
        b = rec(s[s.clv_points < MOVE_BAR], "everything else")
        if a is not None and b is not None:
            print(f"    separation {a - b:+.1%}")
    print("\n  A large in-sample gap and a zero holdout gap is the signature")
    print("  of a threshold fitted to the period it was found in.")


def arithmetic():
    """How much of the gain is just owning a better number?"""
    print("\n" + "=" * 72)
    print("HOW MUCH CLV ARITHMETIC EXPLAINS")
    print("=" * 72)
    for sport, sd in SIGMA.items():
        pp = (norm.cdf(MOVE_BAR / sd) - 0.5) * 100
        print(f"  {sport}: {MOVE_BAR} pts / sigma {sd} = {MOVE_BAR / sd:.3f} "
              f"sigma -> ~{pp:.1f} points of win probability")
    print("\n  NFL holdout gained nothing over its comparison group against a")
    print("  ~55% base rate: the arithmetic, and nothing beyond it.")
    print("  CFB gained ~14 points over a 52.2% base rate on a WIDER sigma,")
    print("  where the arithmetic allows ~3.7. The excess is pick selection,")
    print("  which needs a market that knows things the model does not.")


def main():
    cfb()
    nfl()
    nfl_out_of_sample()
    arithmetic()


if __name__ == "__main__":
    main()
