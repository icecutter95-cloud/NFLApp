"""
Out-of-sample track record for the line-movement model, one row per line.

Why this exists
---------------
The live CLV log (line_predictions) starts the day it was switched on, so it
carries no history. To answer "how have qualifying lines actually done?" we need
results, and the only honest way to produce them retroactively is to rebuild the
predictions the model WOULD have made without ever seeing the outcome.

So this trains on 2020-2022 and only 2020-2022, then predicts 2023, 2024 and
2025 cold. The production artifacts are deliberately NOT used: those are fit on
every season, and scoring 2024 with a model that trained on 2024 is exactly the
contamination that produced this project's earlier fake results.

Periods are labelled because they are not equally trustworthy:
  select   2023-2024  -- the thresholds were chosen looking at these
  holdout  2025       -- never looked at while choosing anything

Numbers from the holdout are the ones to believe. A filter that looks great on
select and mediocre on holdout is a filter that was fit to noise.

The bet is always taken AT THE OPENER, which is what the live logger freezes.

Usage:
    python build_movement_history.py            # rebuild and upload
    python build_movement_history.py --dry-run  # print, write nothing
"""

import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, SPREAD_FEATURES, TOTAL_FEATURES
from train_models import train_model
from model_line_movement import load_joined, TRAIN
from log_clv_predictions import TOTALS_MOVE_THRESHOLD
from score_week import supabase

SPREAD_DISAGREE_MIN = 3.0
SPREAD_MOVE_MIN = 0.5
EVAL_SEASONS = [2023, 2024, 2025]
HOLDOUT = [2025]


def period(season: int) -> str:
    return "holdout" if season in HOLDOUT else "select"


def build() -> pd.DataFrame:
    df = load_joined()
    df["total_points"] = df["home_score"] + df["away_score"]

    tr = df[df.season.isin(TRAIN)]
    print(f"training on {sorted(tr.season.unique())} ({len(tr)} games)")

    sfeats = [c for c in SPREAD_FEATURES if c in df.columns] + ["week_open_spread_home"]
    smodel = train_model(tr[sfeats], tr["week_spread_movement"], tr, "hist_movement")

    tfit = tr.dropna(subset=["week_total_movement", "week_open_total"])
    tfeats = [c for c in TOTAL_FEATURES if c in df.columns] + ["week_open_total"]
    tmodel = train_model(tfit[tfeats], tfit["week_total_movement"], tfit, "hist_total_movement")

    # Margin model for the disagreement signal. Trained on the full feature
    # dataset (not just games with line coverage) but still capped at 2022.
    full = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    mfeats = [c for c in SPREAD_FEATURES if c in full.columns]
    mfit = full[full.season.isin(TRAIN)].dropna(subset=["home_margin"])
    mmodel = train_model(mfit[mfeats], mfit["home_margin"], mfit, "hist_margin")

    ev = df[df.season.isin(EVAL_SEASONS)].copy()
    print(f"scoring   {sorted(ev.season.unique())} ({len(ev)} games)")

    mv = smodel.predict(ev[sfeats].fillna(0))
    disagree = mmodel.predict(ev[mfeats].fillna(0)) + ev["week_open_spread_home"].values
    tmv = tmodel.predict(ev[tfeats].fillna(0))

    rows = []
    for i, (_, g) in enumerate(ev.iterrows()):
        base = {
            "season": int(g["season"]), "week": int(g["week"]),
            "home_team": g["home_team"], "away_team": g["away_team"],
            "game_date": str(g["gameday"])[:10], "period": period(int(g["season"])),
        }

        # --- Spread. Bet the side the line is predicted to move toward.
        m, d = float(mv[i]), float(disagree[i])
        open_s = float(g["week_open_spread_home"])
        close_s = float(g["closing_spread_home"])
        home = m < 0
        # surplus > 0 means home covered the number we took.
        surplus = float(g["home_margin"]) + open_s
        rows.append({
            **base, "bet_type": "spread",
            "open_line": open_s, "closing_line": close_s,
            "predicted_movement": round(m, 3),
            "margin_disagreement": round(d, 3),
            "predicted_side": "home" if home else "away",
            "taken_line": open_s if home else -open_s,
            "actual_movement": round(close_s - open_s, 3),
            # home bet gains when the close moves toward home (more negative)
            "clv_points": round(-(close_s - open_s) if home else (close_s - open_s), 3),
            "result": ("push" if abs(surplus) < 1e-9
                       else "win" if (surplus > 0) == home else "loss"),
            "qualifies": bool(abs(d) >= SPREAD_DISAGREE_MIN
                              and abs(m) >= SPREAD_MOVE_MIN
                              and (d > 0) == (m < 0)),
        })

        # --- Total. Movement only; the disagreement signal is a coin flip here.
        if pd.isna(g["week_open_total"]) or pd.isna(g["closing_total"]):
            continue
        tm = float(tmv[i])
        open_t = float(g["week_open_total"])
        close_t = float(g["closing_total"])
        over = tm > 0
        pts = float(g["total_points"])
        rows.append({
            **base, "bet_type": "total",
            "open_line": open_t, "closing_line": close_t,
            "predicted_movement": round(tm, 3),
            "margin_disagreement": None,
            "predicted_side": "over" if over else "under",
            "taken_line": open_t,
            "actual_movement": round(close_t - open_t, 3),
            "clv_points": round((close_t - open_t) if over else -(close_t - open_t), 3),
            "result": ("push" if pts == open_t
                       else "win" if (pts > open_t) == over else "loss"),
            "qualifies": bool(abs(tm) >= TOTALS_MOVE_THRESHOLD),
        })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    """Print the same cuts the UI will show, as a cross-check on the page."""
    print(f"\n{'':<10}{'slice':<12}{'n':>5}{'W-L-P':>12}{'win%':>8}{'ROI':>8}{'CLV':>8}")
    for bt in ["spread", "total"]:
        for per in ["select", "holdout"]:
            for qual in [False, True]:
                s = df[(df.bet_type == bt) & (df.period == per)]
                if qual:
                    s = s[s.qualifies]
                if len(s) == 0:
                    continue
                w = int((s.result == "win").sum())
                l = int((s.result == "loss").sum())
                p = int((s.result == "push").sum())
                wr = w / max(w + l, 1) * 100
                roi = (w * (100 / 110) - l) / max(w + l, 1) * 100
                label = f"{per} {'qual' if qual else 'all'}"
                print(f"{bt:<10}{label:<12}{len(s):>5}{f'{w}-{l}-{p}':>12}"
                      f"{wr:>7.1f}%{roi:>+7.1f}%{s.clv_points.mean():>+8.2f}")


def main():
    dry = "--dry-run" in sys.argv
    df = build()
    summarize(df)

    if dry:
        print("\n--dry-run: nothing written")
        return

    supabase.table("movement_history").delete().neq("season", 0).execute()
    # NaN is not valid JSON -- postgrest rejects the whole batch on one of them.
    recs = [{k: (None if isinstance(v, float) and np.isnan(v) else v)
             for k, v in r.items()} for r in df.to_dict("records")]
    for i in range(0, len(recs), 500):
        supabase.table("movement_history").insert(recs[i:i + 500]).execute()
    print(f"\nwrote {len(recs)} rows to movement_history")


if __name__ == "__main__":
    main()
