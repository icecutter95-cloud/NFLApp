"""
Where both models actually stand: full record, then the record under the gates.

Answers the question asked every few weeks -- "how are we doing, spreads versus
totals, and what would it be with the filter" -- without hand-written SQL that
quietly picks a different denominator each time.

WHAT COUNTS AS A PICK
NFL: qualifies = true, which is the set the app presents as actionable. The
full unfiltered set is printed beside it, because the gap between them is the
only evidence that the qualifying bar does anything.

CFB: restricted to picks with a verifiable PRE-KICKOFF entry in
cfb_prediction_log. cfb_predictions upserts, so finished games were re-stamped
with later inputs until 2026-10-20 (see restore_cfb_prekickoff.py); a record
built over the unverifiable rows is the model grading its own homework. The
unverifiable half is printed separately and should not be added to the total.

Spreads carry a no-pick case and totals do not: a spread pick requires the
residual and margin models to agree on a side, which they fail to do in ~37% of
college games. Those rows are excluded, not counted as losses.

THE GATES
  rule 1  the line moved >= 1.5 points TOWARD our side between open and close
  rule 2  fewer than 50% of tickets are on our side at the last pre-kick capture
See two_rule_filter_check.py for whether they are real -- short version: well
evidenced in college, no out-of-sample separation in the NFL. This script
reports the record; that one argues about whether to believe it.

GRADING AND PRICE
Everything is graded at the OPENER (open_line), which is the number the model
was frozen against -- not necessarily the number that got bet. ROI assumes -110
on every side, so it is a floor for anyone shopping for a better price and a
ceiling for anyone taking -115. Break-even at -110 is 52.38%.

READ bet_time_gate.py BEFORE QUOTING ANY NFL NUMBER FROM HERE. That caveat is
far larger than it sounds. From week 7 the NFL opener has been a lookahead
price 43-71 days before kickoff, and regraded at a line actually on offer six
days out the gated record falls from 12-4 (75%) to 9-4 (69%) and rule 1 from
57% to 52%.

The college side is much less exposed but not clean: cfb_open_close openers
average 13.7 days before kickoff against the NFL's 43-71 from week 7, and 47
of 632 (7.4%) are more than 45 days stale. Whether the CFB gated record leans
on that 7.4% has NOT been measured and is the obvious next check.

Usage:
    python scripts/season_record.py
    python scripts/season_record.py --league nfl
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

from score_week import fetch_all

MOVE_BAR = 1.5
TICKET_BAR = 50.0
VIG_RETURN = 100 / 110  # win 0.909 units per unit risked at -110
BREAK_EVEN = 110 / 210  # 0.5238


def ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601")


def line(x, label, indent=2, show_p=True):
    """One record row: W-L, rate, ROI, p-value."""
    w = int((x.result == "win").sum())
    l = int((x.result == "loss").sum())
    p = int((x.result == "push").sum())
    pad = " " * indent
    if w + l == 0:
        print(f"{pad}{label:<30} --")
        return
    n = w + l
    rate = w / n
    roi = (w * VIG_RETURN - l) / n
    units = w * VIG_RETURN - l
    tail = f"  p={binomtest(w, n).pvalue:.3f}" if show_p else ""
    print(f"{pad}{label:<30} {w:3d}-{l:<3d}{('-' + str(p)) if p else '  '} "
          f"{rate:6.1%}  {roi:+6.1%} ROI  {units:+6.1f}u{tail}")


def splits_joined(trk_table, splits_table):
    """Tracking rows with the last PRE-kickoff ticket split attached."""
    t = pd.DataFrame(fetch_all(trk_table))
    t = t[t.bet_type.isin(["spread", "total"])].copy()
    t["kick"] = ts(t.commence_time)

    sp = pd.DataFrame(fetch_all(splits_table))
    sp["c"] = ts(sp.captured_at)
    m = sp.merge(t[["game_id", "bet_type", "kick"]],
                 on=["game_id", "bet_type"], how="inner")
    m = m[m.c < m.kick]
    last = (m.sort_values("c").groupby(["game_id", "bet_type"]).tail(1)
            [["game_id", "bet_type", "home_bets_pct", "away_bets_pct"]])
    d = t.merge(last, on=["game_id", "bet_type"], how="left")
    d["our_tix"] = np.where(d.predicted_side.isin(["home", "over"]),
                            d.home_bets_pct, d.away_bets_pct)
    return d


def report(g, title, weeks=True):
    """Full record then gated record for one graded frame."""
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")
    print(f"  {'':<30} {'W-L':>7} {'rate':>7} {'ROI':>11} {'units':>8}")
    line(g, "ALL PICKS")
    for bt in ("spread", "total"):
        line(g[g.bet_type == bt], f"  {bt}s")

    if weeks and g.week.notna().any():
        print("\n  by week")
        for w, s in g.groupby(g.week.astype("Int64")):
            line(s, f"  week {w}", indent=4, show_p=False)

    r1 = g.clv_points >= MOVE_BAR
    r2 = g.our_tix < TICKET_BAR
    cov = g.our_tix.notna().mean()
    print(f"\n  THE GATES   (ticket data on {cov:.0%} of these picks)")
    line(g[r1], f"rule 1 only (moved >={MOVE_BAR})")
    line(g[r2], "rule 2 only (tickets <50%)")
    line(g[r1 & r2], "BOTH RULES")
    for bt in ("spread", "total"):
        line(g[(g.bet_type == bt) & r1 & r2], f"  both, {bt}s")
    line(g[~(r1 & r2)], "everything rejected")
    kept = (r1 & r2).sum()
    print(f"  -> the gates keep {kept}/{len(g)} picks ({kept / len(g):.0%})")


def nfl():
    d = splits_joined("clv_tracking", "nfl_public_splits")
    graded = d.result.isin(["win", "loss", "push"]) & d.predicted_side.notna()
    q = d[graded & (d.qualifies == True)]           # noqa: E712
    report(q, "NFL 2026 -- QUALIFYING PICKS (what the app presents)")
    nq = d[graded & (d.qualifies != True)]          # noqa: E712
    print("\n  for contrast, picks that did NOT qualify:")
    line(nq, "non-qualifying", indent=4)
    line(d[graded], "every pick regardless", indent=4)


def cfb():
    d = splits_joined("cfb_tracking", "cfb_public_splits")
    lg = pd.DataFrame(fetch_all("cfb_prediction_log"))
    lg["la"], lg["k"] = ts(lg.logged_at), ts(lg.commence_time)
    ok = set(map(tuple, lg[lg.la < lg.k][["game_id", "bet_type"]]
                 .drop_duplicates().values))
    d["verified"] = [(r.game_id, r.bet_type) in ok for r in d.itertuples()]

    graded = d.result.isin(["win", "loss", "push"]) & d.predicted_side.notna()
    report(d[graded & d.verified],
           "CFB 2026 -- VERIFIED PRE-KICKOFF PICKS ONLY")
    print("\n  the unverifiable half, NOT to be added to the above "
          "(upserted after kickoff):")
    line(d[graded & ~d.verified], "cannot verify", indent=4)
    nopick = d[d.predicted_side.isna() & d.result.isna()]
    print(f"\n  excluded: {len(nopick)} game-markets where the two models "
          f"disagreed on a side")


def main():
    which = "both"
    if "--league" in sys.argv:
        which = sys.argv[sys.argv.index("--league") + 1].lower()
    print(f"break-even at -110 is {BREAK_EVEN:.2%}; "
          f"everything below is graded at the OPENER")
    if which in ("both", "nfl"):
        nfl()
    if which in ("both", "cfb"):
        cfb()


if __name__ == "__main__":
    main()
