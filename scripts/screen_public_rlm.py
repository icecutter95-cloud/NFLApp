"""
Reverse line movement, without buying public betting data.

The idea (borrowed from a college-basketball model): you cannot backtest
bet%/money% splits because no free historical archive exists -- but you may not
need them. Public betting behaviour is famously predictable. The public backs
favourites, backs overs, and backs popular franchises. If that lean can be
approximated from things we already have, RLM becomes testable on six seasons.

Why this is not a repeat of what we already rejected
----------------------------------------------------
The movement model predicts WHERE a line goes. RLM asks WHY. The same predicted
drift means different things depending on who is pushing it:

  line drifts toward the side the public is hammering -> probably just money
                                                          weight, and it often
                                                          overshoots
  line drifts AGAINST the public                       -> someone informed is
                                                          pushing against the
                                                          grain

Our model cannot currently tell those apart, so this is a conditioning variable
rather than a duplicate signal.

Honesty about the proxy
-----------------------
There is no ground truth for public% in this dataset, so the lean CANNOT be
trained -- it is hand-specified from well-documented public biases. That is a
real weakness and it is why the proxy is kept deliberately crude: three obvious
effects, no tuning. A hand-tuned proxy fitted until the answer looked good would
be indistinguishable from selecting on the test set.

Train 2020-2022, select 2023-2024, holdout 2025.

Usage:
    python screen_public_rlm.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES, TOTAL_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN, SELECT, HOLDOUT

# National draw / fan-base size. Hand-specified, NOT fitted -- these are the
# franchises that reliably attract casual money. Tier 2 = 1, tier 1 = 2.
POPULAR = {
    "DAL": 2, "KC": 2, "SF": 2, "PHI": 2, "GB": 2, "PIT": 2, "NE": 2, "BUF": 2,
    "BAL": 1, "DET": 1, "MIA": 1, "NYG": 1, "NYJ": 1, "CHI": 1, "SEA": 1,
    "MIN": 1, "CIN": 1, "LV": 1, "DEN": 1, "CLE": 1,
}


def public_lean(d: pd.DataFrame) -> pd.DataFrame:
    """Approximate which side casual money sits on. Positive = leans HOME."""
    d = d.copy()
    o = d["week_open_spread_home"]

    # 1. The public backs favourites, and more so as the favourite grows.
    #    Standard convention: negative spread = home favoured.
    fav = np.clip(-o / 7.0, -1.5, 1.5)

    # 2. The public backs popular franchises regardless of price.
    pop = (d["home_team"].map(POPULAR).fillna(0)
           - d["away_team"].map(POPULAR).fillna(0)) / 2.0

    d["public_home"] = fav + pop
    # The public backs overs. There is no "side" to vary per game here, so the
    # totals lean is a constant tilt and only its interaction with movement
    # can carry information.
    d["public_over"] = 1.0
    return d


def grade_spread(s, bet_home):
    surplus = s["home_margin"].values + s["week_open_spread_home"].values
    live = np.abs(surplus) > 1e-9
    won = np.where(bet_home, surplus > 0, surplus < 0)[live]
    w, l = int(won.sum()), int((~won).sum())
    n = max(w + l, 1)
    clv = np.where(bet_home, -s["week_spread_movement"], s["week_spread_movement"])
    return {"n": len(s), "w": w, "l": l, "wr": w / n * 100,
            "roi": (w * (100 / 110) - l) / n * 100, "clv": float(np.mean(clv))}


def grade_total(s, over):
    pts, line = s["total_points"].values, s["week_open_total"].values
    live = pts != line
    won = np.where(over, pts > line, pts < line)[live]
    w, l = int(won.sum()), int((~won).sum())
    n = max(w + l, 1)
    clv = np.where(over, s["week_total_movement"], -s["week_total_movement"])
    return {"n": len(s), "w": w, "l": l, "wr": w / n * 100,
            "roi": (w * (100 / 110) - l) / n * 100, "clv": float(np.mean(clv))}


def row(tag, r):
    print(f"    {tag:<34} n={r['n']:>3}  {r['w']:>3}-{r['l']:<3} {r['wr']:>5.1f}%  "
          f"ROI {r['roi']:>+6.1f}%  CLV {r['clv']:>+5.2f}")


def main():
    df = public_lean(load_joined())
    df["total_points"] = df["home_score"] + df["away_score"]

    # Sanity: does the proxy behave like public money? Public-backed sides
    # should get BET UP, i.e. the line should drift toward them on average.
    c = np.corrcoef(df.public_home, -df.week_spread_movement)[0, 1]
    print(f"proxy sanity: corr(public_home, line drifting toward home) = {c:+.3f}")
    print("  (positive means the side the proxy calls 'public' does get bet up)")

    tr = df[df.season.isin(TRAIN)]
    sf = [c for c in SPREAD_FEATURES if c in df.columns] + ["week_open_spread_home"]
    mv = train_model(tr[sf], tr["week_spread_movement"], tr, "p_move")
    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    mf = [c for c in SPREAD_FEATURES if c in full.columns]
    mfit = full[full.season.isin(TRAIN)].dropna(subset=["home_margin"])
    mm = train_model(mfit[mf], mfit["home_margin"], mfit, "p_margin")

    tdf = df.dropna(subset=["week_open_total", "week_total_movement", "total_points"])
    tf = [c for c in TOTAL_FEATURES if c in df.columns] + ["week_open_total"]
    tmv = train_model(tdf[tdf.season.isin(TRAIN)][tf],
                      tdf[tdf.season.isin(TRAIN)]["week_total_movement"],
                      tdf[tdf.season.isin(TRAIN)], "p_tmove")

    for tag, seasons in [("SELECT 2023-24", SELECT), ("HOLDOUT 2025", HOLDOUT)]:
        ev = df[df.season.isin(seasons)].copy()
        ev["move"] = mv.predict(ev[sf].fillna(0))
        ev["dis"] = mm.predict(ev[mf].fillna(0)) + ev["week_open_spread_home"].values
        ev["bet_home"] = ev["move"] < 0
        ev["qual"] = ((ev["dis"].abs() >= 3.0) & (ev["move"].abs() >= 0.5)
                      & ((ev["dis"] > 0) == (ev["move"] < 0)))

        # RLM: the line is predicted to move AGAINST the public lean, i.e. we
        # are siding with whoever is pushing against the crowd.
        ev["contrarian"] = ((ev.public_home > 0.25) & ~ev.bet_home) | \
                           ((ev.public_home < -0.25) & ev.bet_home)
        ev["with_public"] = ((ev.public_home > 0.25) & ev.bet_home) | \
                            ((ev.public_home < -0.25) & ~ev.bet_home)

        print(f"\n{tag}  SPREADS")
        q = ev[ev.qual]
        row("qualifying (what we ship)", grade_spread(q, q.bet_home.values))
        a = q[q.contrarian]
        b = q[q.with_public]
        if len(a) >= 15:
            row("  + against the public (RLM)", grade_spread(a, a.bet_home.values))
        if len(b) >= 15:
            row("  + with the public", grade_spread(b, b.bet_home.values))
        r2 = ev[ev.contrarian]
        row("contrarian alone (no filter)", grade_spread(r2, r2.bet_home.values))

        te = tdf[tdf.season.isin(seasons)].copy()
        te["move"] = tmv.predict(te[tf].fillna(0))
        te["over"] = te["move"] > 0
        tq = te[te["move"].abs() >= 1.25]
        print(f"{tag}  TOTALS")
        row("qualifying (what we ship)", grade_total(tq, tq.over.values))
        # Public backs overs, so a predicted move DOWN is the contrarian one.
        u = tq[~tq.over]
        o = tq[tq.over]
        if len(u) >= 15:
            row("  unders (against public)", grade_total(u, u.over.values))
        if len(o) >= 15:
            row("  overs (with public)", grade_total(o, o.over.values))


if __name__ == "__main__":
    main()
