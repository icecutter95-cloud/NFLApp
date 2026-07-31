"""
Public lean from PRICES instead of ticket counts.

Reverse line movement infers sharp-vs-public from bet%/money% splits, which we
cannot buy or backtest. But the same decomposition is available for free in the
book panel we already have.

Retail books (DraftKings, FanDuel, BetMGM, Caesars, BetRivers) take the public's
money and shade their numbers toward it -- that is their business model. Sharp,
low-margin shops (Pinnacle, BetOnline, LowVig) survive on accuracy and post
closer to the true price. So:

    retail_line - sharp_line  ~=  how far public money has bent the retail number

which is the same quantity RLM tries to recover from ticket percentages, in
points rather than percent. If the retail number is bent, it should snap back
toward the sharp number, and that is a directional, testable claim.

This is NOT a rerun of the dk_vs_consensus test in screen_multibook_signal.py.
That compared one book against the blended field, which mixes sharp and retail
together and cancels the very thing of interest. This splits the panel along the
line that theory says matters.

Train 2020-2022, select 2023-2024, holdout 2025.

Usage:
    python screen_sharp_vs_retail.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN, SELECT, HOLDOUT

SHARP = ["pinnacle", "betonlineag", "lowvig"]
RETAIL = ["draftkings", "fanduel", "betmgm", "williamhill_us", "betrivers"]
MIN_SHARP, MIN_RETAIL = 1, 2


def panel() -> pd.DataFrame:
    frames = [pd.read_parquet(DATA_DIR / f"multibook_{s}.parquet")
              for s in range(2020, 2026)
              if (DATA_DIR / f"multibook_{s}.parquet").exists()]
    d = pd.concat(frames, ignore_index=True).dropna(subset=["spread_home"])
    d["req"] = pd.to_datetime(d["requested_at"], utc=True)

    d["grp"] = np.where(d.book.isin(SHARP), "sharp",
                        np.where(d.book.isin(RETAIL), "retail", None))
    d = d[d.grp.notna()]

    g = d.groupby(["req", "home_team", "away_team", "grp"])["spread_home"]
    w = g.median().unstack("grp")
    n = g.count().unstack("grp")
    w = w.join(n, rsuffix="_n").reset_index()
    w = w[(w["sharp_n"] >= MIN_SHARP) & (w["retail_n"] >= MIN_RETAIL)]
    # >0 means retail hangs a number ABOVE the sharps, i.e. retail is shaded
    # toward the away side; the line should drift down toward the sharp price.
    w["retail_vs_sharp"] = w["retail"] - w["sharp"]
    return w


def grade(s, bet_home):
    surplus = s["home_margin"].values + s["week_open_spread_home"].values
    live = np.abs(surplus) > 1e-9
    won = np.where(bet_home, surplus > 0, surplus < 0)[live]
    w, l = int(won.sum()), int((~won).sum())
    n = max(w + l, 1)
    clv = np.where(bet_home, -s["week_spread_movement"], s["week_spread_movement"])
    return {"n": len(s), "w": w, "l": l, "wr": w / n * 100,
            "roi": (w * (100 / 110) - l) / n * 100, "clv": float(np.mean(clv))}


def row(tag, r):
    print(f"    {tag:<34} n={r['n']:>3}  {r['w']:>3}-{r['l']:<3} {r['wr']:>5.1f}%  "
          f"ROI {r['roi']:>+6.1f}%  CLV {r['clv']:>+5.2f}")


def main():
    df = load_joined()
    p = panel()

    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    oc["req"] = pd.to_datetime(oc["week_opened_at"], utc=True)
    ct = pd.to_datetime(oc["commence_time"], utc=True)
    oc["season"] = np.where(ct.dt.month >= 3, ct.dt.year, ct.dt.year - 1)
    k = oc[["season", "home_team", "away_team", "req"]].merge(
        p, on=["home_team", "away_team", "req"], how="inner")
    k = k.drop(columns=["req"]).drop_duplicates(subset=["season", "home_team", "away_team"])

    before = len(df)
    df = df.merge(k, on=["season", "home_team", "away_team"], how="left")
    assert len(df) == before, f"merge fanned out {before} -> {len(df)}"
    df = df[df.retail_vs_sharp.notna()].copy()
    print(f"games with both a sharp and 2+ retail quotes: {len(df)}")
    for s in sorted(df.season.unique()):
        print(f"  {s}: {int((df.season == s).sum())}")

    print(f"\ndisagreement: retail differs from sharp on "
          f"{(df.retail_vs_sharp.abs() > 0.01).mean()*100:.0f}% of games, "
          f"mean gap {df.retail_vs_sharp.abs().mean():.3f} pts")

    # Core claim: a shaded retail number snaps back toward the sharp price.
    c = np.corrcoef(df.retail_vs_sharp, df.week_spread_movement)[0, 1]
    print(f"corr(retail_vs_sharp, movement) = {c:+.3f}   (expect NEGATIVE)")
    for lo, hi in [(0.01, 0.5), (0.5, 1.0), (1.0, 99)]:
        s = df[(df.retail_vs_sharp.abs() >= lo) & (df.retail_vs_sharp.abs() < hi)]
        if len(s) < 25:
            continue
        toward = ((s.retail_vs_sharp > 0) == (s.week_spread_movement < 0))
        print(f"  gap {lo}-{hi}: n={len(s):>4}  line moved toward the sharp number "
              f"{toward.mean()*100:.1f}%")

    OPEN, TGT = "week_open_spread_home", "week_spread_movement"
    team = [c for c in SPREAD_FEATURES if c in df.columns]
    tr = df[df.season.isin(TRAIN)]
    bf = team + [OPEN]
    base = train_model(tr[bf], tr[TGT], tr, "sr_base")
    new = train_model(tr[bf + ["retail_vs_sharp"]], tr[TGT], tr, "sr_new")

    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    mf = [c for c in SPREAD_FEATURES if c in full.columns]
    mfit = full[full.season.isin(TRAIN)].dropna(subset=["home_margin"])
    mm = train_model(mfit[mf], mfit["home_margin"], mfit, "sr_margin")

    for tag, seasons in [("SELECT 2023-24", SELECT), ("HOLDOUT 2025", HOLDOUT)]:
        ev = df[df.season.isin(seasons)].copy()
        if len(ev) < 40:
            print(f"\n{tag}: only {len(ev)} games — too thin to read")
            continue
        ev["dis"] = mm.predict(ev[mf].fillna(0)) + ev[OPEN].values
        print(f"\n{tag}  (n={len(ev)})")
        for name, model, feats in [("base", base, bf),
                                   ("+sharp/retail", new, bf + ["retail_vs_sharp"])]:
            pr = model.predict(ev[feats].fillna(0))
            q = ((ev["dis"].abs() >= 3.0) & (np.abs(pr) >= 0.5)
                 & ((ev["dis"] > 0) == (pr < 0)))
            corr = np.corrcoef(pr, ev[TGT])[0, 1]
            r = grade(ev[q], (pr[q] < 0))
            print(f"    corr {corr:+.3f}", end="  ")
            row(name, r)


if __name__ == "__main__":
    main()
