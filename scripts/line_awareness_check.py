"""
Should the spread edge respond to the line that is actually available?

THE STRUCTURAL WORRY
score_week shows edge_points from spread_model.joblib next to dk_line, which
reads as "the edge at this number". It is not. spread_model has 59 features and
NONE of them is a line (verified: market_spread_home is absent), and its target
is home_cover_surplus = margin + CLOSING spread. So its output is an estimate of
cover surplus against an unobserved closing line, and it is identical whether
the market says BAL -4.5 or BAL +3.5. On 2026 week 5 it read +4.12 for
Baltimore after the market had moved 7.5 points away from that side.

TWO CANDIDATE FIXES, BOTH TESTED, BOTH REJECTED

1. "Add the line as a feature."
   Already done, in the model that matters. nfl_residual_model takes
   week_open_spread_home and predicts margin + that line -- the well-posed
   formulation. And it barely uses it: 1.6% of total gain, and feeding week 5
   the CURRENT line instead of the frozen opener moved its output by 0.15 on a
   9-point line change (PHI@JAX 1.5 -> -7.5 shifted residual_pred -0.42 ->
   -0.27). Zero of 15 games changed side.

   That is not a bug. Cover surplus genuinely is near-independent of the line in
   the training sample, because the line and team quality move together -- an
   efficient market. The tree learned the truth about the sample.

2. "Compute the edge arithmetically at the live line."
   margin_disagreement = predicted_margin + line, so the edge at any other
   line L is margin_disagreement + (L - open_line). No retraining needed. It
   also makes things WORSE, and arithmetically must: when the market runs away
   from a side, a line-aware edge grows by exactly the amount it ran. Week 5
   BAL@ATL went from -6.29 at the opener to -13.79 at today's number -- the
   model claiming a 13.8-point overlay on a team whose quarterback is doubtful,
   because its margin forecast (BAL by 10.8) is built from games Lamar Jackson
   played. Line-awareness faithfully converts the market's superior information
   into a phantom edge of precisely that size.

   So the line-INVARIANT residual model is accidentally protective. It declines
   to inflate when the market knows something it does not.

WHAT THE RECORD SAYS, AND WHERE IT RUNS OUT
Win rate of the margin rule's side by how far it disagrees with the market
(277 graded games with a stored margin forecast):

    |disagreement|  0-3   n=120   49.6%
                    3-5   n= 70   57.4%
                    5-7   n= 41   65.9%  (p=0.060)
                    7-10  n= 31   51.6%
                    10+   n= 15   53.3%

It rises, peaks in the 5-7 band, then decays -- the shape you would expect if
very large disagreement is more often the model's ignorance than the market's
error, and an argument for capping rather than trusting the tail. But n=31 and
n=15, and splitting those by movement direction puts 8 games in a cell that
reverses the sign of the 811-pick finding. Not enough to act on. Revisit with
a full season.

CONCLUSION
The structural criticism is correct and the available fixes are worse than the
flaw. What is actionable is narrower: stop presenting a line-free number beside
a specific line as though it were the edge at that line. spread_model's gate is
already disabled (SPREAD_MIN_EDGE = 999, "no demonstrated edge"), so the Edges
tab is driven by a model the project does not trust AND is ill-posed, while the
validated residual model sits beside it.

Nothing in this file is wired into the pipeline. It is the measurement.

Usage:
    python scripts/line_awareness_check.py
"""

import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest

warnings.filterwarnings("ignore")

from config import MODELS_DIR
from score_week import fetch_all


def feature_audit():
    print("FEATURE AUDIT — does each spread model see a line?")
    for name in ("spread", "nfl_residual", "margin"):
        p = MODELS_DIR / f"{name}_model.joblib"
        if not p.exists():
            print(f"  {name:14} missing")
            continue
        m = joblib.load(p)
        f = m.get_booster().feature_names
        imp = m.get_booster().get_score(importance_type="gain")
        tot = sum(imp.values()) or 1
        line = [c for c in f if "open_spread" in c or "market_spread" in c]
        g = f" gain {imp.get(line[0], 0) / tot:.1%}" if line else ""
        print(f"  {name:14} {len(f):3d} features | line feature: "
              f"{line[0] if line else 'NONE':24}{g}")


def disagreement_buckets():
    mh = pd.DataFrame(fetch_all("movement_history"))
    gr = pd.DataFrame(fetch_all("game_results"))[
        ["season", "week", "home_team", "away_team", "home_margin"]]
    sp = (mh[mh.bet_type == "spread"]
          .merge(gr, on=["season", "week", "home_team", "away_team"], how="inner"))
    sp = sp[sp.home_margin.notna() & sp.margin_disagreement.notna()].copy()
    sp["side"] = np.where(sp.margin_disagreement > 0, "home", "away")
    sp["absD"] = sp.margin_disagreement.abs()

    def rec(d):
        hc = d.home_margin + d.open_line
        m = hc != 0
        w = int(((hc[m] > 0) == (d.side[m] == "home")).sum())
        n = int(m.sum())
        return (f"{w:3d}-{n - w:<3d} {w / n:5.1%} (p={binomtest(w, n).pvalue:.3f})"
                if n else "n=0")

    print(f"\nDISAGREEMENT BUCKETS — {len(sp)} graded games")
    print("  does a bigger disagreement with the market win more?")
    for lo, hi in [(0, 3), (3, 5), (5, 7), (7, 10), (10, 99)]:
        d = sp[(sp.absD >= lo) & (sp.absD < hi)]
        print(f"    |disagreement| {lo:2d}-{hi:<3d} n={len(d):3d}  {rec(d)}")
    print("  rises, peaks at 5-7, decays past it. Tail cells are n=31 and n=15:")
    print("  suggestive of capping, nowhere near enough to act on.")


def reprice_live():
    """What the frozen picks look like re-priced at today's number."""
    cl = pd.DataFrame(fetch_all("clv_tracking"))
    d = cl[(cl.bet_type == "spread") & cl.qualifies
           & cl.margin_disagreement.notna() & cl.actual_movement.notna()].copy()
    if d.empty:
        return
    d["edge_now"] = d.margin_disagreement + d.actual_movement
    d["grew"] = d.edge_now.abs() - d.margin_disagreement.abs()
    worse = d[d.grew > 2]
    print(f"\nRE-PRICED AT THE LIVE LINE — {len(d)} qualifying picks")
    print(f"  {len(worse)} would have their apparent edge GROW by 2+ points, "
          f"purely because the market moved away from them:")
    for r in worse.sort_values("grew", ascending=False).head(6).itertuples():
        side = r.home_team if r.predicted_side == "home" else r.away_team
        print(f"    wk{r.week} {r.away_team}@{r.home_team:4} {side:4} "
              f"open {r.open_line:+5.1f} -> {r.closing_line:+5.1f}  "
              f"edge {r.margin_disagreement:+6.2f} -> {r.edge_now:+6.2f}")
    print("  This is why the arithmetic fix is rejected: it reads the market's")
    print("  information as our edge.")


if __name__ == "__main__":
    feature_audit()
    disagreement_buckets()
    reprice_live()
