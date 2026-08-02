"""
GO / NO-GO for college football. Does the opener thesis travel?

The whole NFL app rests on one finding: we cannot beat the closing line at
predicting football (corr 0.374 against the market's 0.477), but the weekly
OPENER is soft, and predicting where the number travels earns CLV. If college
openers are already efficient, there is nothing to build here and no amount of
feature engineering will fix it.

This tests that with NO football data whatsoever -- no CFBD, no EPA, no team
metrics. It is the opener-only baseline (Model B from model_line_movement.py),
which asks a single question: given only the number on the board, is there
predictable drift?

Reading the result
------------------
  A  predict zero      no skill, the honest null
  B  opener only       pure mean reversion off the posted number

If B cannot beat A on CLV, the openers are efficient and the answer is NO GO.
If B does beat A, that is NOT proof of an edge -- it only means drift exists and
is partly predictable from the opener alone. Football features would then have
to add something on top, exactly as they did (modestly) for the NFL.

A caveat kept in view: B being profitable in-sample is unremarkable, since
extreme openers regress. The NFL version guarded against this by checking that
adding team features beat B. Here, B beating A is only a licence to continue.

Splits mirror the NFL protocol exactly: train 2020-2022, select 2023-2024,
holdout 2025.

Usage:
    python test_cfb_opener_thesis.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR
from train_models import train_model

TRAIN = [2020, 2021, 2022]
SELECT = [2023, 2024]
HOLDOUT = [2025]


def load() -> pd.DataFrame:
    frames = []
    for s in range(2020, 2026):
        p = DATA_DIR / f"cfb_open_close_{s}.parquet"
        if p.exists():
            d = pd.read_parquet(p)
            d["season"] = s
            frames.append(d)
    if not frames:
        raise SystemExit("No cfb_open_close_*.parquet — run fetch_cfb_lines.py first")
    df = pd.concat(frames, ignore_index=True)
    # Two snapshots minimum, or "movement" is just one reading against itself.
    return df[df.n_snapshots_week >= 2].dropna(
        subset=["week_open_spread_home", "week_spread_movement"])


def evaluate(name, pred, actual, opener) -> dict:
    corr = float(np.corrcoef(pred, actual)[0, 1]) if pred.std() > 1e-9 else float("nan")
    mae = float(np.abs(pred - actual).mean())
    moved = np.abs(actual) >= 0.5
    dir_acc = float(((pred < 0) == (actual < 0))[moved].mean()) if moved.any() else float("nan")
    # Bet the side the line is predicted to move toward, and hold the opener.
    #   home bet -> open - close ; away bet -> close - open
    clv = np.where(pred < 0, -actual, actual)
    return {"name": name, "corr": corr, "mae": mae, "dir": dir_acc,
            "clv": float(clv.mean()), "clv_pos": float((clv > 0).mean())}


def show(rows, title):
    print(f"\n{title}")
    print(f"  {'model':<22}{'corr':>8}{'MAE':>7}{'dir%':>8}{'CLV':>8}{'CLV+%':>8}")
    for r in rows:
        d = "  n/a" if np.isnan(r["dir"]) else f"{r['dir']*100:5.1f}%"
        print(f"  {r['name']:<22}{r['corr']:>+8.3f}{r['mae']:>7.2f}{d:>8}"
              f"{r['clv']:>+8.2f}{r['clv_pos']*100:>7.1f}%")


def main():
    df = load()
    print(f"CFB games with a usable open/close: {len(df)}")
    for s in sorted(df.season.unique()):
        n = int((df.season == s).sum())
        print(f"  {s}: {n}")

    OPEN, TGT = "week_open_spread_home", "week_spread_movement"
    print(f"\nmovement: mean |move| {df[TGT].abs().mean():.2f} pts, "
          f"moved at all {(df[TGT].abs() > 0.01).mean()*100:.0f}%, "
          f"moved 1+ pt {(df[TGT].abs() >= 1).mean()*100:.0f}%")

    tr = df[df.season.isin(TRAIN)]
    if len(tr) < 200:
        raise SystemExit(f"Only {len(tr)} training games — backfill incomplete")
    b = train_model(tr[[OPEN]], tr[TGT], tr, "cfb_opener_only")

    for tag, seasons in [("SELECT (2023-2024)", SELECT), ("HOLDOUT (2025)", HOLDOUT)]:
        ev = df[df.season.isin(seasons)]
        if len(ev) < 100:
            print(f"\n{tag}: only {len(ev)} games — skipping")
            continue
        actual, opener = ev[TGT].values, ev[OPEN].values
        rows = [evaluate("A predict zero", np.zeros(len(ev)), actual, opener),
                evaluate("B opener only", b.predict(ev[[OPEN]].fillna(0)), actual, opener)]
        show(rows, f"{tag}   n={len(ev)}")

    # Sample size is the structural reason CFB is worth doing at all: the NFL
    # holdout gives ~45 qualifying bets a year, whose win-rate CI spans
    # [43.3, 72.2] -- too wide to ever prove an edge.
    print(f"\nsample: {len(df)} CFB games across {df.season.nunique()} seasons "
          f"vs 1,612 NFL games across 6 — {len(df)/1612:.1f}x")


if __name__ == "__main__":
    main()
