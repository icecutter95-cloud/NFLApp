"""
The real CFB test: do team features beat the mean-reversion control?

model_line_movement.py's structure, re-run on college football:

  B  opener only            pure mean reversion -- the control
  C  opener + team form     the actual hypothesis

If C only matches B, the football features add nothing and any apparent edge is
regression to the mean off extreme openers. That is precisely why B exists.

Thresholds are DERIVED here, never inherited. The NFL bars (3.0 margin
disagreement, 0.5 predicted movement, 1.25 for totals) were fitted on NFL data
and carrying them over would dress an untested guess in a validated number's
clothing -- the same error refused for preseason.

Protocol matches the NFL exactly: train 2020-2022, choose on 2023-2024, and
report 2025 as a holdout that was not consulted while choosing. A threshold that
looks good on select and dies on holdout is noise, which is how the 1.5 totals
bar and the key-number filter were both caught.

Usage:
    python test_cfb_model.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR
from train_models import train_model
from build_cfb_dataset import FEATURE_COLS

TRAIN, SELECT, HOLDOUT = [2020, 2021, 2022], [2023, 2024], [2025]
OPEN, TGT = "week_open_spread_home", "week_spread_movement"


def grade(s: pd.DataFrame, bet_home: np.ndarray) -> dict:
    """Win rate and ROI taking the OPENER at -110, plus CLV."""
    surplus = s["home_margin"].values + s[OPEN].values
    live = np.abs(surplus) > 1e-9
    won = np.where(bet_home, surplus > 0, surplus < 0)[live]
    w, l = int(won.sum()), int((~won).sum())
    n = max(w + l, 1)
    clv = np.where(bet_home, -s[TGT].values, s[TGT].values)
    return {"n": len(s), "w": w, "l": l, "wr": w / n * 100,
            "roi": (w * (100 / 110) - l) / n * 100, "clv": float(np.mean(clv))}


def row(tag, r):
    print(f"    {tag:<30} n={r['n']:>4}  {r['w']:>3}-{r['l']:<3} {r['wr']:>5.1f}%  "
          f"ROI {r['roi']:>+6.1f}%  CLV {r['clv']:>+5.2f}")


def main():
    df = pd.read_parquet(DATA_DIR / "cfb_dataset.parquet")
    feats = [f"diff_{c}" for c in FEATURE_COLS] + ["neutral_site", "conference_game"]
    print(f"CFB dataset: {len(df)} games, {len(feats)} features")

    tr = df[df.season.isin(TRAIN)]
    b = train_model(tr[[OPEN]], tr[TGT], tr, "cfb_B")
    c = train_model(tr[feats + [OPEN]], tr[TGT], tr, "cfb_C")
    # Margin model for the second, independent signal.
    m = train_model(tr[feats], tr["home_margin"], tr, "cfb_margin")

    ev_sets = {}
    for tag, seasons in [("SELECT 2023-24", SELECT), ("HOLDOUT 2025", HOLDOUT)]:
        ev = df[df.season.isin(seasons)].copy()
        ev["pB"] = b.predict(ev[[OPEN]].fillna(0))
        ev["pC"] = c.predict(ev[feats + [OPEN]].fillna(0))
        ev["dis"] = m.predict(ev[feats].fillna(0)) + ev[OPEN].values
        ev_sets[tag] = ev

        print(f"\n{tag}   n={len(ev)}")
        for name, col in [("B opener only", "pB"), ("C opener + form", "pC")]:
            corr = np.corrcoef(ev[col], ev[TGT])[0, 1]
            moved = ev[TGT].abs() >= 0.5
            dacc = ((ev[col] < 0) == (ev[TGT] < 0))[moved].mean() * 100
            print(f"  {name:<18} corr {corr:+.3f}  dir {dacc:.1f}%", end="  ")
            row("", grade(ev, (ev[col] < 0).values))

    # Only sweep if C actually beat B; otherwise there is nothing to filter.
    print("\nTHRESHOLD SWEEP (chosen on SELECT, then read on HOLDOUT)")
    print(f"  {'disagree/move':<20}{'SELECT':>26}{'HOLDOUT':>26}")
    best = None
    for dbar in [3.0, 5.0, 7.0]:
        for mbar in [0.5, 1.0, 1.5]:
            out = []
            for tag in ("SELECT 2023-24", "HOLDOUT 2025"):
                ev = ev_sets[tag]
                q = ((ev["dis"].abs() >= dbar) & (ev["pC"].abs() >= mbar)
                     & ((ev["dis"] > 0) == (ev["pC"] < 0)))
                out.append(grade(ev[q], (ev[q]["pC"] < 0).values))
            s, h = out
            if s["n"] < 25:
                continue
            def cell(r):
                return f"n={r['n']:>4} {r['w']:>3}-{r['l']:<3} {r['wr']:>5.1f}% CLV {r['clv']:+.2f}"
            print(f"  d>={dbar} m>={mbar:<8}{cell(s):>28}{cell(h):>28}")
            if best is None or s["wr"] > best[0]:
                best = (s["wr"], dbar, mbar, s, h)

    if best:
        _, dbar, mbar, s, h = best
        print(f"\n  best on SELECT: d>={dbar} m>={mbar} -> {s['wr']:.1f}% "
              f"({s['w']}-{s['l']}), CLV {s['clv']:+.2f}")
        print(f"  same bar on HOLDOUT:            {h['wr']:.1f}% "
              f"({h['w']}-{h['l']}), CLV {h['clv']:+.2f}")
        print(f"  break-even at -110 is 52.38%")


if __name__ == "__main__":
    main()
