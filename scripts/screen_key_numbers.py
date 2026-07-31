"""
Do KEY NUMBERS improve bet selection?

The movement model predicts how far a line will travel, and every point is
treated the same. Football scoring says otherwise: margins pile up on 3 and 7
because of how field goals and touchdowns combine, so moving a spread from -2.5
to -3.5 changes far more win probability than moving -6 to -7. If that is true,
the size of a predicted move is the wrong thing to rank on -- what matters is
whether the move CROSSES something.

Two parts:
  1. Derive the key numbers empirically from eight seasons of results, for both
     margins and totals, rather than repeating folklore. Totals are widely
     assumed to have weaker key numbers than spreads; this checks it.
  2. Test whether selecting for a PREDICTED crossing (knowable at bet time,
     since projected close = opener + predicted movement) beats the filter we
     ship today.

"Favourable crossing" means a key number sits between the number we hold and
the number the market closes at, with us on the better side of it. Expressed in
goodness units -- how many points our side effectively receives -- that is
simply: some key k with our_close <= k < our_open.

Splits stay strict: train 2020-2022, select 2023-2024, holdout 2025.

Usage:
    python screen_key_numbers.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES, TOTAL_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN, SELECT, HOLDOUT

SPREAD_KEYS = [3, 7]          # confirmed empirically in part 1
TOTAL_KEYS = [41, 44, 47, 51]


def distributions():
    """Which numbers actually occur? Establishes the keys instead of assuming."""
    d = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    d = d.dropna(subset=["home_margin"])
    d["total_points"] = d["home_score"] + d["away_score"]
    n = len(d)
    print(f"empirical distributions over {n} games ({d.season.min()}-{d.season.max()})\n")

    am = d["home_margin"].abs()
    print("  MARGIN   most common absolute margins")
    top = am.value_counts().head(8).sort_index()
    for v, c in top.items():
        print(f"    {int(v):>3} pts  {c / n * 100:>5.1f}%  {'#' * int(c / n * 300)}")

    print("\n  TOTAL    most common final totals")
    tp = d["total_points"].value_counts().head(8).sort_index()
    for v, c in tp.items():
        print(f"    {int(v):>3} pts  {c / n * 100:>5.1f}%  {'#' * int(c / n * 300)}")

    # A key number is only "key" if it beats its neighbours. Flatness means
    # there is nothing to cross.
    print("\n  spike test (share at N vs mean of N-1, N+1):")
    for label, series, keys in [("margin", am, SPREAD_KEYS + [10, 14]),
                                ("total", d["total_points"], TOTAL_KEYS)]:
        vc = series.value_counts()
        for k in keys:
            here = vc.get(k, 0)
            nb = (vc.get(k - 1, 0) + vc.get(k + 1, 0)) / 2
            ratio = here / nb if nb else float("nan")
            print(f"    {label:<7}{k:>3}: {here / n * 100:>4.1f}% vs {nb / n * 100:>4.1f}% "
                  f"-> {ratio:.2f}x {'KEY' if ratio >= 1.3 else ''}")


def crossings(our_open, our_close, keys_signed):
    """Count keys sitting between the number we hold and the close, in our favour."""
    out = np.zeros(len(our_open), dtype=int)
    for k in keys_signed:
        out += ((our_close <= k) & (k < our_open)).astype(int)
    return out


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
    print(f"    {tag:<32} n={r['n']:>3}  {r['w']:>3}-{r['l']:<3} {r['wr']:>5.1f}%  "
          f"ROI {r['roi']:>+6.1f}%  CLV {r['clv']:>+5.2f}")


def main():
    distributions()

    df = load_joined()
    df["total_points"] = df["home_score"] + df["away_score"]
    tr = df[df.season.isin(TRAIN)]

    sf = [c for c in SPREAD_FEATURES if c in df.columns] + ["week_open_spread_home"]
    mv = train_model(tr[sf], tr["week_spread_movement"], tr, "k_move")
    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    mf = [c for c in SPREAD_FEATURES if c in full.columns]
    mfit = full[full.season.isin(TRAIN)].dropna(subset=["home_margin"])
    mm = train_model(mfit[mf], mfit["home_margin"], mfit, "k_margin")

    tdf = df.dropna(subset=["week_open_total", "week_total_movement", "total_points"])
    tf = [c for c in TOTAL_FEATURES if c in df.columns] + ["week_open_total"]
    ttr = tdf[tdf.season.isin(TRAIN)]
    tmv = train_model(ttr[tf], ttr["week_total_movement"], ttr, "k_tmove")

    signed_spread = [k for key in SPREAD_KEYS for k in (key, -key)]

    for tag, seasons in [("SELECT 2023-24", SELECT), ("HOLDOUT 2025", HOLDOUT)]:
        ev = df[df.season.isin(seasons)].copy()
        ev["move"] = mv.predict(ev[sf].fillna(0))
        ev["dis"] = mm.predict(ev[mf].fillna(0)) + ev["week_open_spread_home"].values
        ev["bet_home"] = ev["move"] < 0
        ev["qual"] = ((ev["dis"].abs() >= 3.0) & (ev["move"].abs() >= 0.5)
                      & ((ev["dis"] > 0) == (ev["move"] < 0)))

        # In goodness units: points our side effectively receives.
        o = ev["week_open_spread_home"].values
        proj = o + ev["move"].values
        our_open = np.where(ev.bet_home, o, -o)
        our_proj = np.where(ev.bet_home, proj, -proj)
        ev["cross"] = crossings(our_open, our_proj, signed_spread)

        print(f"\n{tag}  SPREADS")
        q = ev[ev.qual]
        row("qualifying (what we ship)", grade_spread(q, q.bet_home.values))
        row("  + predicted key crossing", grade_spread(q[q.cross > 0], q[q.cross > 0].bet_home.values))
        row("  no crossing", grade_spread(q[q.cross == 0], q[q.cross == 0].bet_home.values))
        c = ev[ev.cross > 0]
        row("crossing alone (no filter)", grade_spread(c, c.bet_home.values))

        te = tdf[tdf.season.isin(seasons)].copy()
        te["move"] = tmv.predict(te[tf].fillna(0))
        te["over"] = te["move"] > 0
        to = te["week_open_total"].values
        tproj = to + te["move"].values
        # Over wants a LOWER number, under a higher one -- flip into goodness units.
        t_open = np.where(te.over, -to, to)
        t_proj = np.where(te.over, -tproj, tproj)
        tkeys = [k for key in TOTAL_KEYS for k in (key, -key)]
        te["cross"] = crossings(t_open, t_proj, tkeys)

        print(f"{tag}  TOTALS")
        tq = te[te["move"].abs() >= 1.25]
        row("qualifying (what we ship)", grade_total(tq, tq.over.values))
        row("  + predicted key crossing", grade_total(tq[tq.cross > 0], tq[tq.cross > 0].over.values))
        row("  no crossing", grade_total(tq[tq.cross == 0], tq[tq.cross == 0].over.values))


if __name__ == "__main__":
    main()
