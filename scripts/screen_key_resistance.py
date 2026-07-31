"""
Are lines STICKY on key numbers, and can we use that?

This is a different question from screen_key_numbers.py. That asked whether our
FORECAST crossing a key improves selection (it does not). This asks about the
market's own behaviour: books are famously reluctant to move a spread off 3 or
7, preferring to shade the price -110 -> -115 -> -120 rather than go to 3.5. If
that reluctance is real, then a line parked on a key is harder to move, and a
movement model that ignores this is over-predicting movement for exactly those
games -- a systematic, correctable bias rather than a selection question.

Three parts:
  1. Is the resistance real? How much do lines on 3/7/10/14 actually move
     compared with lines sitting elsewhere?
  2. Does the model already know? If its errors are larger on key numbers, it
     does not, and the information is free.
  3. Does telling it help? Add distance-to-key features and test incremental
     value under the usual strict splits.

Train 2020-2022, select 2023-2024, holdout 2025.

Usage:
    python screen_key_resistance.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN, SELECT, HOLDOUT

# Confirmed empirically in screen_key_numbers.py: margins spike 2.9x on 3,
# 2.8x on 10 and 14, 1.6x on 7.
KEYS = [3, 7, 10, 14]


def add_key_features(d: pd.DataFrame) -> pd.DataFrame:
    a = d["week_open_spread_home"].abs()
    d = d.copy()
    d["abs_open"] = a
    d["dist_to_key"] = np.min([np.abs(a - k) for k in KEYS], axis=0)
    d["on_key"] = (d["dist_to_key"] < 0.01).astype(int)
    # A line at 3.5 or 2.5 is one tick from a key and has to cross it to move
    # inward; that is a different situation from sitting at 8.5.
    d["adjacent_key"] = ((d["dist_to_key"] > 0.01) & (d["dist_to_key"] <= 0.5)).astype(int)
    return d


def main():
    df = add_key_features(load_joined())
    print(f"games: {len(df)}")

    # ---- 1. Is the resistance real?
    print("\n1. HOW MUCH DO LINES MOVE, BY WHERE THEY OPEN")
    print(f"  {'opener sits':<20}{'n':>6}{'mean |move|':>13}{'moved at all':>14}{'moved 1+ pt':>13}")
    mv = df["week_spread_movement"].abs()
    for label, mask in [
        ("ON a key (3/7/10/14)", df.on_key == 1),
        ("within 0.5 of a key", df.adjacent_key == 1),
        ("1.0+ from any key", df.dist_to_key >= 1.0),
    ]:
        s = df[mask]
        if len(s) < 30:
            continue
        m = s["week_spread_movement"].abs()
        print(f"  {label:<20}{len(s):>6}{m.mean():>13.3f}"
              f"{(m > 0.01).mean()*100:>13.0f}%{(m >= 1.0).mean()*100:>12.0f}%")

    print("\n  by exact opening number (the ones with real volume):")
    for v in [2.5, 3.0, 3.5, 6.5, 7.0, 7.5, 9.5, 10.0]:
        s = df[np.isclose(df.abs_open, v)]
        if len(s) < 25:
            continue
        m = s["week_spread_movement"].abs()
        flag = "  <- KEY" if v in KEYS else ""
        print(f"    {v:>4}: n={len(s):>4}  mean |move| {m.mean():.3f}  "
              f"stayed put {(m <= 0.01).mean()*100:>3.0f}%{flag}")

    # ---- 2. Does the model already know?
    sf = [c for c in SPREAD_FEATURES if c in df.columns] + ["week_open_spread_home"]
    tr = df[df.season.isin(TRAIN)]
    base = train_model(tr[sf], tr["week_spread_movement"], tr, "kr_base")

    ev = df[df.season.isin(SELECT + HOLDOUT)].copy()
    ev["pred"] = base.predict(ev[sf].fillna(0))
    ev["err"] = ev["pred"] - ev["week_spread_movement"]

    print("\n2. CURRENT MODEL ERROR, BY DISTANCE TO A KEY")
    print(f"  {'opener sits':<20}{'n':>6}{'mean |pred|':>13}{'mean |actual|':>15}{'bias':>9}")
    for label, mask in [("ON a key", ev.on_key == 1),
                        ("within 0.5", ev.adjacent_key == 1),
                        ("1.0+ away", ev.dist_to_key >= 1.0)]:
        s = ev[mask]
        if len(s) < 30:
            continue
        print(f"  {label:<20}{len(s):>6}{s.pred.abs().mean():>13.3f}"
              f"{s.week_spread_movement.abs().mean():>15.3f}{s.err.mean():>+9.3f}")

    # ---- 3. Does telling it help?
    kf = sf + ["dist_to_key", "on_key", "adjacent_key"]
    keyed = train_model(tr[kf], tr["week_spread_movement"], tr, "kr_keyed")

    print("\n3. ADDING DISTANCE-TO-KEY AS FEATURES")
    print(f"  {'period':<12}{'variant':<10}{'corr':>8}{'MAE':>8}{'qual n':>8}{'win%':>8}{'CLV':>8}")
    for tag, seasons in [("select", SELECT), ("holdout", HOLDOUT)]:
        e = df[df.season.isin(seasons)].copy()
        full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
        mf = [c for c in SPREAD_FEATURES if c in full.columns]
        mfit = full[full.season.isin(TRAIN)].dropna(subset=["home_margin"])
        mm = train_model(mfit[mf], mfit["home_margin"], mfit, "kr_margin")
        dis = mm.predict(e[mf].fillna(0)) + e["week_open_spread_home"].values

        for name, model, feats in [("base", base, sf), ("+key", keyed, kf)]:
            p = model.predict(e[feats].fillna(0))
            corr = np.corrcoef(p, e["week_spread_movement"])[0, 1]
            mae = np.abs(p - e["week_spread_movement"]).mean()
            q = ((np.abs(dis) >= 3.0) & (np.abs(p) >= 0.5) & ((dis > 0) == (p < 0)))
            s = e[q]
            bh = p[q] < 0
            surplus = s["home_margin"].values + s["week_open_spread_home"].values
            live = np.abs(surplus) > 1e-9
            won = np.where(bh, surplus > 0, surplus < 0)[live]
            w, l = int(won.sum()), int((~won).sum())
            clv = np.where(bh, -s["week_spread_movement"], s["week_spread_movement"])
            print(f"  {tag:<12}{name:<10}{corr:>+8.3f}{mae:>8.3f}{len(s):>8}"
                  f"{w/max(w+l,1)*100:>7.1f}%{np.mean(clv):>+8.2f}")


if __name__ == "__main__":
    main()
