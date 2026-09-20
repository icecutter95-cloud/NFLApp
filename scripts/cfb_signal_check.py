"""
What, if anything, is predictive on the college board so far?

Read-only. Pulls cfb_tracking (model leans, open/close, results), the public
splits, and asks the same questions each week as the sample grows:

  1. Model lean ATS at the opener and at the close; CLV; direction accuracy.
  2. Each model alone, on every game, not just the ones they agree on.
  3. Line movement as a signal: does following the move cover at the open
     (market moving toward the truth), and at the close (the move overshot).
  4. Public splits at the last capture before kickoff: tickets, money, fading
     the heavy side, money-vs-tickets gap, reverse line movement.
  5. From the FIRST capture (a few days out): which side did the line move
     toward by kickoff? This is the timing question -- bet now or wait.

Every test prints a record, a rate, and a two-sided binomial p-value. Under
~150 games nothing here can reach significance; treat p < 0.1 as "keep
watching", not "bet it". CLV and movement are measured against the last
PRE-kickoff snapshot -- see migration cfb_open_close_close_is_pre_kickoff for
why that matters.

Usage:
    python scripts/cfb_signal_check.py
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from score_week import fetch_all  # noqa: E402


def ats(df, side, line):
    """W-L for betting `side` at `line` (home spread). Pushes dropped."""
    d = df[df[side].notna() & df[line].notna()]
    hc = d.margin + d[line]
    m = hc != 0
    w = int(((hc[m] > 0) == (d[side][m] == "home")).sum())
    n = int(m.sum())
    if not n:
        return "n=0"
    return f"{w}-{n - w} ({w / n:.1%}, p={binomtest(w, n).pvalue:.2f})"


def toward(df, side):
    """Did the line move toward `side` between capture and close?"""
    d = df[df[side].notna()]
    s = np.where(d[side] == "home", -1, 1) * d.after
    return (f"n={len(d):3d}  toward {int((s > 0).sum()):2d} / away "
            f"{int((s < 0).sum()):2d} / flat {int((s == 0).sum()):2d}   "
            f"mean {s.mean():+.2f}")


def main():
    t = pd.DataFrame(fetch_all("cfb_tracking"))
    g = t[t.home_score.notna() & (t.bet_type == "spread")].copy()
    g["margin"] = g.home_score - g.away_score
    print(f"{len(g)} graded spread games, {g.predicted_side.notna().sum()} with a lean\n")

    print("== MODEL LEAN (both models agree) ==")
    print("  at open ", ats(g, "predicted_side", "open_line"),
          "| at close", ats(g, "predicted_side", "closing_line"))
    for w, d in g.groupby("week"):
        print(f"  week {int(w):2d}", ats(d, "predicted_side", "open_line"))
    g["md"] = g.margin_disagreement.abs()
    for lo, hi in [(0, 3), (3, 6), (6, 99)]:
        print(f"  |margin disagreement| {lo}-{hi:<2}",
              ats(g[(g.md >= lo) & (g.md < hi)], "predicted_side", "open_line"))
    c = g.clv_points.dropna()
    print(f"  CLV mean {c.mean():+.2f} median {c.median():+.2f} n={len(c)}  "
          f"+/-/0 = {(c > 0).sum()}/{(c < 0).sum()}/{(c == 0).sum()}  "
          f"t-test p={ttest_1samp(c, 0).pvalue:.2f}")
    dc = g.direction_correct.dropna()
    print(f"  movement direction right {int(dc.sum())}/{len(dc)} "
          f"({dc.mean():.1%}, p={binomtest(int(dc.sum()), len(dc)).pvalue:.2f})")
    g["margin_side"] = np.where(g.margin_disagreement > 0, "home",
                                np.where(g.margin_disagreement < 0, "away", None))
    g["move_side"] = np.where(g.predicted_movement < 0, "home",
                              np.where(g.predicted_movement > 0, "away", None))
    print("  margin model alone  ", ats(g, "margin_side", "open_line"))
    print("  movement model alone", ats(g, "move_side", "open_line"))

    print("\n== LINE MOVEMENT, opener to pre-kickoff close ==")
    print(f"  avg |move| {g.actual_movement.abs().mean():.2f} pts, "
          f">=2 pts in {(g.actual_movement.abs() >= 2).mean():.0%} of games")
    mv = g[g.actual_movement.abs() >= 0.5].copy()
    mv["to"] = np.where(mv.actual_movement < 0, "home", "away")
    big = mv[mv.actual_movement.abs() >= 2]
    print("  follow any move : at open", ats(mv, "to", "open_line"),
          "| at close", ats(mv, "to", "closing_line"))
    print("  follow >=2pt    : at open", ats(big, "to", "open_line"),
          "| at close", ats(big, "to", "closing_line"))
    g["dog"] = np.where(g.open_line > 0, "home", np.where(g.open_line < 0, "away", None))
    print("  every dog       : at open", ats(g, "dog", "open_line"),
          "| at close", ats(g, "dog", "closing_line"))

    sp = pd.DataFrame(fetch_all("cfb_public_splits"))
    sp = sp[sp.bet_type == "spread"].copy()
    if sp.empty:
        print("\nno splits captured"); return
    sp["captured_at"] = pd.to_datetime(sp.captured_at, utc=True, format="ISO8601")
    sp["lc"] = pd.to_numeric(sp.line_at_capture, errors="coerce")
    g["ct"] = pd.to_datetime(g.commence_time, utc=True, format="ISO8601")
    x = sp.merge(g, on="game_id")
    x = x[x.captured_at < x.ct].copy()
    x["hrs"] = (x.ct - x.captured_at).dt.total_seconds() / 3600
    x["pub"] = np.where(x.home_bets_pct > 50, "home", np.where(x.home_bets_pct < 50, "away", None))
    x["fade"] = np.where(x.pub == "home", "away", np.where(x.pub == "away", "home", None))
    x["heavy_fade"] = np.where(x.home_bets_pct >= 65, "away",
                               np.where(x.home_bets_pct <= 35, "home", None))
    x["money"] = np.where(x.home_money_pct > 50, "home", np.where(x.home_money_pct < 50, "away", None))
    gap = x.home_money_pct - x.home_bets_pct
    x["sharp"] = np.where(gap >= 10, "home", np.where(gap <= -10, "away", None))
    so_far = x.lc - x.open_line
    x["rlm"] = np.where((x.pub == "home") & (so_far > 0), "away",
                        np.where((x.pub == "away") & (so_far < 0), "home", None))
    x["dog_now"] = np.where(x.lc > 0, "home", np.where(x.lc < 0, "away", None))

    d = x.sort_values("captured_at").groupby("game_id").tail(1)
    print(f"\n== SPLITS at the last capture before kickoff "
          f"(n={len(d)}, median {d.hrs.median():.0f}h out), bet at the captured line ==")
    for k, label in [("pub", "ticket majority"), ("fade", "fade tickets"),
                     ("heavy_fade", "fade >=65% tickets"), ("money", "money majority"),
                     ("sharp", "money-tickets gap >=10"), ("rlm", "reverse line move")]:
        print(f"  {label:24}", ats(d, k, "lc"))
    m = d[d.predicted_side.notna()]
    for k, dd in m.groupby(m.predicted_side == m.pub):
        print("  lean", "with tickets    " if k else "against tickets ", ats(dd, "predicted_side", "lc"))
    for k, dd in m.groupby(m.predicted_side == m.money):
        print("  lean", "with money      " if k else "against money   ", ats(dd, "predicted_side", "lc"))

    e = x[x.hrs >= 24].sort_values("captured_at").groupby("game_id").head(1).copy()
    e["after"] = e.closing_line - e.lc
    print(f"\n== TIMING: from the first capture (n={len(e)}, median {e.hrs.median():.0f}h out) "
          f"which side did the line move toward by kickoff? ==")
    for k, label in [("predicted_side", "model lean"), ("pub", "ticket majority"),
                     ("heavy_fade", "side AGAINST >=65% tickets"), ("sharp", "money side"),
                     ("rlm", "RLM side"), ("dog_now", "the dog")]:
        print(f"  {label:28}", toward(e, k))
    print("  RLM side ATS at that early line", ats(e, "rlm", "lc"))
    print("  dog ATS at that early line     ", ats(e, "dog_now", "lc"))


if __name__ == "__main__":
    main()
