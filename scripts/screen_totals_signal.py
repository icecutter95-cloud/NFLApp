"""
Does a SECOND signal exist for totals?

The gap this closes
-------------------
A qualifying spread needs two independent signals to agree: the movement model
expects drift, AND the margin model disagrees with the opener the same way.
Totals ship on movement alone, which is why they carry a lower-confidence
warning and a quarter-unit cap.

The reason given was that "the disagreement signal is worthless for totals,
49-51% at every threshold". But no script ever persisted that test, and the
margin model predicts a MARGIN -- by construction it says nothing about how many
total points get scored. Applying it to totals would be near-guaranteed to look
like a coin flip whether or not a real signal exists.

The correct analog is a TOTAL POINTS model measured against the opening total:

    spreads:  disagreement = predicted_margin + week_open_spread_home
    totals:   disagreement = predicted_total  - week_open_total

Direction agreement differs too. For spreads, rating home above the opener means
wanting the line to move toward home, i.e. movement < 0. For totals, projecting
more points than the opener means wanting the number to RISE, i.e. movement > 0.

Splits are strict: train 2020-2022, select 2023-2024, holdout 2025. A threshold
that looks good on select and dies on holdout is noise -- that is exactly how
the 1.5 totals bar was caught inverting (63.0% -> 47.4%).

Usage:
    python screen_totals_signal.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, TOTAL_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN, SELECT, HOLDOUT

MOVE_BAR = 1.25   # the shipped totals threshold


def grade(sel: pd.DataFrame, over: np.ndarray) -> dict:
    """Win rate and ROI taking the OPENING total, at -110."""
    pts = sel["total_points"].values
    line = sel["week_open_total"].values
    live = pts != line
    won = np.where(over, pts > line, pts < line)[live]
    w, l = int(won.sum()), int((~won).sum())
    n = max(w + l, 1)
    # over wants the number to rise, under wants it to fall
    clv = np.where(over, sel["week_total_movement"], -sel["week_total_movement"])
    return {"n": len(sel), "w": w, "l": l,
            "wr": w / n * 100, "roi": (w * (100 / 110) - l) / n * 100,
            "clv": float(np.mean(clv))}


def row(tag, r):
    print(f"    {tag:<30} n={r['n']:>3}  {r['w']:>3}-{r['l']:<3} "
          f"{r['wr']:>5.1f}%  ROI {r['roi']:>+6.1f}%  CLV {r['clv']:>+5.2f}")


def main():
    df = load_joined()
    df["total_points"] = df["home_score"] + df["away_score"]
    df = df.dropna(subset=["week_open_total", "week_total_movement", "total_points"])
    print(f"games with totals coverage: {len(df)}")

    tr = df[df.season.isin(TRAIN)]
    tfeats = [c for c in TOTAL_FEATURES if c in df.columns]

    # Movement model (the signal we already ship).
    mv_feats = tfeats + ["week_open_total"]
    mv = train_model(tr[mv_feats], tr["week_total_movement"], tr, "tot_move")

    # NEW: total points model -- the honest analog of the margin model.
    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    full["total_points"] = full["home_score"] + full["away_score"]
    ptf = [c for c in TOTAL_FEATURES if c in full.columns]
    pfit = full[full.season.isin(TRAIN)].dropna(subset=["total_points"])
    pm = train_model(pfit[ptf], pfit["total_points"], pfit, "tot_points")

    for tag, seasons in [("SELECT 2023-24", SELECT), ("HOLDOUT 2025", HOLDOUT)]:
        ev = df[df.season.isin(seasons)].copy()
        move = mv.predict(ev[mv_feats].fillna(0))
        dis = pm.predict(ev[ptf].fillna(0)) - ev["week_open_total"].values
        ev["move"], ev["dis"] = move, dis

        print(f"\n{tag}   (n={len(ev)})")
        print(f"  corr(disagreement, actual movement) = "
              f"{np.corrcoef(dis, ev['week_total_movement'])[0,1]:+.3f}")

        # Baseline: what we ship today.
        s = ev[np.abs(ev.move) >= MOVE_BAR]
        row(f"movement only >={MOVE_BAR}", grade(s, (s.move > 0).values))

        # Disagreement alone -- is it better than a coin flip at all?
        for db in [2.0, 3.0, 4.0]:
            s = ev[np.abs(ev.dis) >= db]
            if len(s) >= 25:
                row(f"disagreement only >={db}", grade(s, (s.dis > 0).values))

        # Both agree. Totals: model projects MORE points -> line should RISE.
        for db in [2.0, 3.0, 4.0]:
            for mb in [0.5, 1.0, 1.25]:
                s = ev[(np.abs(ev.dis) >= db) & (np.abs(ev.move) >= mb)
                       & ((ev.dis > 0) == (ev.move > 0))]
                if len(s) >= 20:
                    row(f"both agree d>={db} m>={mb}", grade(s, (s.dis > 0).values))


if __name__ == "__main__":
    main()
