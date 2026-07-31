"""
WHEN should the bet go down?

Our entire measured edge is closing line value, and the largest lever on CLV is
not which side you take -- it is what time you take it. That has never been
measured here. The logger freezes a pick the first time a game enters the weekly
window; whether acting on it immediately matters, or whether Saturday is just as
good, is an open question worth real money.

What this measures
------------------
Picks are made ONCE, at the weekly opener, exactly as the live logger does.
Then, holding that side fixed, CLV is recomputed as though the bet were placed
at each later snapshot. The result is a decay curve: how much value leaks away
per day of hesitation.

    home bet: CLV = line_at_bet - close     (you want to lay fewer points)
    away bet: CLV = close - line_at_bet     (you want to receive more)

Picks are out-of-sample throughout: models train on 2020-2022 and score
2023-2025 cold, same protocol as build_movement_history.py. Scoring a season
with a model that trained on it is the contamination that produced this
project's earlier fake results.

A caveat this cannot escape: the model is TRAINED on the weekly opener, so the
opener is its natural habitat. This answers "having decided at the opener, when
should I place?" -- not "would a model retrained on Saturday numbers do better?"

Usage:
    python screen_bet_timing.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN

EVAL = [2023, 2024, 2025]
HOLDOUT = [2025]
# Hours before kickoff. Snapshot density clusters around 11/7/6/4 days out and
# inside the final day, so buckets are drawn to match rather than evenly.
BUCKETS = [(150, 999, "6+ days"), (96, 150, "4-6 days"), (48, 96, "2-4 days"),
           (12, 48, "12-48 hrs"), (0, 12, "final 12 hrs")]


def picks() -> pd.DataFrame:
    """Out-of-sample side + qualification for every 2023-2025 game."""
    df = load_joined()
    tr = df[df.season.isin(TRAIN)]

    sf = [c for c in SPREAD_FEATURES if c in df.columns] + ["week_open_spread_home"]
    mv = train_model(tr[sf], tr["week_spread_movement"], tr, "t_move")

    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    mf = [c for c in SPREAD_FEATURES if c in full.columns]
    mfit = full[full.season.isin(TRAIN)].dropna(subset=["home_margin"])
    mm = train_model(mfit[mf], mfit["home_margin"], mfit, "t_margin")

    ev = df[df.season.isin(EVAL)].copy()
    ev["move"] = mv.predict(ev[sf].fillna(0))
    ev["dis"] = mm.predict(ev[mf].fillna(0)) + ev["week_open_spread_home"].values
    ev["bet_home"] = ev["move"] < 0
    ev["qualifies"] = ((ev["dis"].abs() >= 3.0) & (ev["move"].abs() >= 0.5)
                       & ((ev["dis"] > 0) == (ev["move"] < 0)))
    ev["kick_date"] = pd.to_datetime(ev["gameday"]).dt.date

    # The instant the pick becomes knowable. The model's input is the WEEKLY
    # opener, so any snapshot before week_opened_at is a number we could not
    # have acted on -- betting it would mean using a future input. Without this
    # the lookahead board (100+ days out) shows a fat +1.94 CLV that is pure
    # look-ahead bias.
    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    oc["week_opened_at"] = pd.to_datetime(oc["week_opened_at"], utc=True)
    oc["kick_date"] = pd.to_datetime(oc["commence_time"], utc=True).dt.date
    ev = ev.merge(oc[["home_team", "away_team", "kick_date", "week_opened_at"]],
                  on=["home_team", "away_team", "kick_date"], how="left")

    return ev[["season", "home_team", "away_team", "kick_date", "bet_home",
               "qualifies", "closing_spread_home", "week_open_spread_home",
               "week_opened_at"]]


def snapshots() -> pd.DataFrame:
    d = pd.read_parquet(DATA_DIR / "historical_lines_all.parquet")
    d = d.dropna(subset=["spread_home"])
    d["snap"] = pd.to_datetime(d["snapshot_at"], utc=True)
    d["kick"] = pd.to_datetime(d["commence_time"], utc=True)
    d["lead_h"] = (d["kick"] - d["snap"]).dt.total_seconds() / 3600
    d["kick_date"] = d["kick"].dt.date
    return d[d.lead_h > 0]


def main():
    p = picks()
    s = snapshots()
    m = s.merge(p, on=["home_team", "away_team", "kick_date"], how="inner")
    before = len(m)
    # Only snapshots from the moment the pick exists onward are bettable.
    m = m[m.week_opened_at.notna() & (m.snap >= m.week_opened_at)]
    print(f"picks {len(p)} | snapshots joined {before} "
          f"| bettable (at/after the weekly opener) {len(m)} "
          f"| games covered {m.groupby(['home_team','away_team','kick_date']).ngroups}")

    # CLV of placing at this snapshot instead of at the close.
    m["clv"] = np.where(m.bet_home,
                        m.spread_home - m.closing_spread_home,
                        m.closing_spread_home - m.spread_home)

    for label, sub in [("ALL PICKS", m), ("QUALIFYING ONLY", m[m.qualifies])]:
        print(f"\n{label}")
        print(f"  {'when you bet':<16}{'n':>6}{'mean CLV':>11}{'CLV>0':>8}"
              f"{'holdout CLV':>13}")
        for lo, hi, name in BUCKETS:
            b = sub[(sub.lead_h >= lo) & (sub.lead_h < hi)]
            if len(b) < 30:
                continue
            ho = b[b.season.isin(HOLDOUT)]
            hv = f"{ho.clv.mean():+.2f}" if len(ho) >= 20 else "--"
            print(f"  {name:<16}{len(b):>6}{b.clv.mean():>+11.2f}"
                  f"{(b.clv > 0).mean()*100:>7.0f}%{hv:>13}")

    # Same decay, framed as the thing you can actually control: how long you sit
    # on a pick after it appears.
    m["since_open_h"] = (m.snap - m.week_opened_at).dt.total_seconds() / 3600
    q = m[m.qualifies]
    print("\nQUALIFYING -- CLV by delay after the pick appears")
    print(f"  {'you waited':<16}{'n':>6}{'mean CLV':>11}{'CLV>0':>8}")
    for lo, hi, name in [(0, 6, "under 6 hrs"), (6, 24, "6-24 hrs"),
                         (24, 72, "1-3 days"), (72, 999, "3+ days")]:
        b = q[(q.since_open_h >= lo) & (q.since_open_h < hi)]
        if len(b) < 25:
            continue
        print(f"  {name:<16}{len(b):>6}{b.clv.mean():>+11.2f}{(b.clv > 0).mean()*100:>7.0f}%")

    # What does waiting actually cost? Compare the earliest and latest snapshot
    # available for the SAME game, so the comparison is paired.
    q = m[m.qualifies].sort_values("lead_h")
    g = q.groupby(["home_team", "away_team", "kick_date"])
    pair = pd.DataFrame({"early": g.clv.last(), "late": g.clv.first(),
                         "early_h": g.lead_h.last(), "late_h": g.lead_h.first()})
    pair = pair[(pair.early_h - pair.late_h) >= 24]
    if len(pair) >= 20:
        d = pair.early - pair.late
        t = d.mean() / (d.std() / np.sqrt(len(d)))
        print(f"\nPAIRED, same game (n={len(pair)}): betting early beats betting late "
              f"by {d.mean():+.2f} pts  (t={t:.1f})")
        print(f"  early avg {pair.early_h.mean():.0f}h out -> CLV {pair.early.mean():+.2f}")
        print(f"  late  avg {pair.late_h.mean():.0f}h out -> CLV {pair.late.mean():+.2f}")


if __name__ == "__main__":
    main()
