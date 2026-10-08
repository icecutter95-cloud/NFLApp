"""
Re-predict frozen weeks with the metrics the model should have had.

WHY
team_metrics held no 2026 rows until 2026-10-08: compute_metrics defaulted to
ALL_HISTORICAL_SEASONS, which stops at 2025, and the Update Metrics workflow
had no schedule. So every 2026 prediction was built from end-of-2025 form.

Checked against each freeze timestamp, that cost less than it first appeared:

    pick week  frozen   2026 weeks complete  correct metrics   V1 used
    1          Aug 6    none                 pure 2025 prior   same
    2          Sep 14   none                 pure 2025 prior   same
    3          Sep 14   none                 pure 2025 prior   same
    4          Sep 21   1                    2026 wk2 (n=1)    stale
    5          Sep 28   1-2                  2026 wk3 (n=2)    stale
    6          Oct 5    1-3                  2026 wk4 (n=3)    stale

Weeks 1-3 were frozen before any 2026 week had finished, so pure-2025 prior
was the correct answer and v2 reproduces v1 exactly. Only weeks 4-6 move.

WHAT POINT-IN-TIME MEANS HERE
A week is "complete" only when its last kickoff is more than 3.5 hours before
the freeze -- a Monday night game that had not kicked off cannot inform a pick
frozen that afternoon. The metrics row for week N is built from games strictly
before N, so the legitimate row is (last complete week + 1). Using today's
week-5 row for a Sept 28 freeze would import games played Sept 30 to Oct 5 and
make the corrected model look better than it earned.

WHAT CANNOT BE RECONSTRUCTED, AND IS NOT FAKED
  * Injuries. injury_flags is delete-and-insert with no history, so there is no
    record of who was listed out on Sept 28. v2 leaves the ten inj_* features
    at zero, exactly as v1 had them. v2's weeks 4-6 are metrics-corrected only;
    weeks 7 onward, logged live, will have real injury data. That seam is real
    and is why this script prints it on every run.
  * Weather. One row per game, refreshed weekly, so it holds the latest
    forecast and not the one from the freeze. For every week rebuilt here the
    batch covering those games landed after the freeze, so weather is passed
    empty -- as v1 effectively had it. Including it flipped eight totals in
    weeks 1-3 while leaving the spreads identical, which is how the leak was
    found.

WHAT IS HONEST ABOUT THIS, AND WHAT IS NOT
Honest: the opener is a real frozen number, verifiable in line_history; the
models are byte-identical, not retrained or reselected; the features are
restricted to what existed at the freeze.

Not a track record: nobody could have bet these, because the app showed v1 at
the time. It is the best estimate of what the model would have said, and it is
labelled movement_v2 so it can never be silently added to realised results.

The trap to avoid: do NOT use v2's weeks 4-6 to choose qualifying thresholds
and then grade those thresholds on the same games. Fixing an input bug and
re-running is sound; selecting a rule on the data you then report is not.

Usage:
    python rebuild_predictions_v2.py --dry-run      # compare, write nothing
    python rebuild_predictions_v2.py                # write movement_v2 rows
    python rebuild_predictions_v2.py --weeks 5,6
"""

import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import MODELS_DIR, CURRENT_SEASON
from score_week import (supabase, fetch_all, fetch_current_schedule,
                        fetch_team_metrics, build_feature_matrix,
                        assert_feature_parity)

VERSION = "movement_v2"
PRIOR_VERSION = "movement_v1"

# Hours after the last kickoff of a week before it counts as complete.
WEEK_SETTLES_HOURS = 3.5

# Same thresholds clv_tracking applies, kept here only for the printed report.
RESIDUAL_MIN = 1.5
TOTAL_MOVE_MIN = 1.25


def complete_weeks(frozen_at, kickoffs) -> list:
    """2026 weeks whose last game had finished before `frozen_at`."""
    done = []
    for wk, last in kickoffs.items():
        if last + pd.Timedelta(hours=WEEK_SETTLES_HOURS) < frozen_at:
            done.append(int(wk))
    return sorted(done)


def main():
    dry = "--dry-run" in sys.argv
    want = None
    if "--weeks" in sys.argv:
        want = {int(x) for x in sys.argv[sys.argv.index("--weeks") + 1].split(",")}

    v1 = pd.DataFrame([r for r in fetch_all("line_predictions")
                       if r["model_version"] == PRIOR_VERSION
                       and r["season"] == CURRENT_SEASON])
    if v1.empty:
        print(f"no {PRIOR_VERSION} rows for {CURRENT_SEASON}")
        return
    v1["frozen"] = pd.to_datetime(v1.predicted_at, utc=True, format="ISO8601")
    v1["kick"] = pd.to_datetime(v1.commence_time, utc=True, format="ISO8601")
    kickoffs = v1.groupby("week").kick.max().to_dict()

    existing = {(r["game_id"], r["bet_type"]) for r in fetch_all("line_predictions")
                if r["model_version"] == VERSION}

    model = joblib.load(MODELS_DIR / "movement_model.joblib")
    feat_order = joblib.load(MODELS_DIR / "movement_features.joblib")
    res_model = joblib.load(MODELS_DIR / "nfl_residual_model.joblib")
    res_feats = joblib.load(MODELS_DIR / "nfl_residual_features.joblib")
    margin_model = joblib.load(MODELS_DIR / "margin_model.joblib")
    margin_feats = joblib.load(MODELS_DIR / "margin_features.joblib")
    tm = MODELS_DIR / "total_movement_model.joblib"
    total_model = joblib.load(tm) if tm.exists() else None
    total_feats = (joblib.load(MODELS_DIR / "total_movement_features.joblib")
                   if tm.exists() else None)

    now_iso = pd.Timestamp.now(tz="UTC").isoformat()
    rows, report = [], []

    for week in sorted(v1.week.unique()):
        if want and week not in want:
            continue
        wk = v1[v1.week == week]
        frozen = wk.frozen.min()
        done = complete_weeks(frozen, {k: v for k, v in kickoffs.items() if k < week})
        pit_week = (max(done) + 1) if done else None

        games = fetch_current_schedule(CURRENT_SEASON, week)
        if games.empty:
            continue
        pairs = list(zip(games["home_team"], games["away_team"]))

        # The frozen opener, used both as the model's market feature and as the
        # bet line -- never today's number.
        opener = {}
        for r in wk.itertuples():
            opener.setdefault((r.home_team, r.away_team), {})[r.bet_type] = r.open_line
        lines = {}
        for h, a in pairs:
            o = opener.get((h, a), {})
            if "spread" not in o:
                continue
            lines[(h, a)] = {"spread_home": float(o["spread"]),
                             "total": float(o["total"]) if "total" in o else None}
        games = games[[(h, a) in lines for h, a in pairs]].reset_index(drop=True)
        if games.empty:
            print(f"week {week}: no frozen openers to rebuild against")
            continue
        pairs = list(zip(games["home_team"], games["away_team"]))

        if pit_week is None:
            metrics = fetch_team_metrics(CURRENT_SEASON, 1)   # resolves to prior
            basis = "pure 2025 prior (nothing complete at freeze)"
        else:
            metrics = fetch_team_metrics(CURRENT_SEASON, pit_week)
            basis = f"2026 wk{pit_week} metrics (weeks {done} complete)"

        # Weather deliberately empty, like injuries. The weather table holds
        # ONE row per game, refreshed in weekly batches (2026: Sep 9, 16, 23,
        # 30, Oct 7), so it carries the latest forecast rather than the one
        # that existed at a freeze ten days before kickoff -- for every week
        # here the batch covering those games landed AFTER the freeze. Passing
        # it in imported post-freeze information, and it showed: weeks 1-3
        # spreads reproduced v1 exactly while eight TOTALS flipped, totals
        # being the weather-sensitive market. Dropping it makes weeks 1-3
        # reproduce v1 exactly in both markets, which is the check that the
        # only thing v2 changes is the metrics.
        feats = build_feature_matrix(games, metrics, lines, {}, {})
        feats["week_open_spread_home"] = feats["dk_spread"].astype(float)
        feats["week_open_total"] = feats["dk_total"].astype(float)

        assert_feature_parity(feats, feat_order, "movement")
        assert_feature_parity(feats, res_feats, "residual")
        assert_feature_parity(feats, margin_feats, "margin")
        mv = model.predict(feats[feat_order].fillna(0))
        resid = res_model.predict(feats[res_feats].fillna(0))
        pm = margin_model.predict(feats[margin_feats].fillna(0))
        disagree = pm + feats["week_open_spread_home"].values
        tv = (total_model.predict(feats[total_feats].fillna(0))
              if total_model is not None else None)

        print(f"\n--- week {week} | frozen {frozen:%Y-%m-%d %H:%M} | {basis}")
        for i, g in feats.iterrows():
            key = (g["home_team"], g["away_team"])
            for bet_type in ("spread", "total"):
                src = wk[(wk.home_team == key[0]) & (wk.away_team == key[1]) &
                         (wk.bet_type == bet_type)]
                if src.empty:
                    continue
                s = src.iloc[0]
                if bet_type == "spread":
                    movement = float(mv[i])
                    r = float(resid[i])
                    side = ("home" if r > 0 else "away") if abs(r) > 0 else None
                    q_new = abs(r) >= RESIDUAL_MIN
                    dis = float(disagree[i])
                else:
                    if tv is None:
                        continue
                    movement = float(tv[i])
                    r = None
                    side = "over" if movement > 0 else "under"
                    q_new = abs(movement) >= TOTAL_MOVE_MIN
                    dis = None
                rows.append({
                    "game_id": s.game_id, "bet_type": bet_type,
                    "season": int(s.season), "week": int(s.week),
                    "home_team": s.home_team, "away_team": s.away_team,
                    "commence_time": s.commence_time,
                    "predicted_at": now_iso,          # when it was COMPUTED
                    "open_spread_home": float(s.open_spread_home)
                    if pd.notna(s.open_spread_home) else None,
                    "open_line": float(s.open_line),
                    "predicted_movement": round(movement, 3),
                    "residual_pred": None if r is None else round(r, 3),
                    "margin_disagreement": None if dis is None else round(dis, 3),
                    "predicted_side": side,
                    "taken_line": float(s.open_line) if side in ("home", "over")
                    else -float(s.open_line) if side == "away"
                    else float(s.open_line),
                    "model_version": VERSION,
                })
                q_old = (abs(s.residual_pred) >= RESIDUAL_MIN
                         if bet_type == "spread" and pd.notna(s.residual_pred)
                         else abs(s.predicted_movement) >= TOTAL_MOVE_MIN
                         if pd.notna(s.predicted_movement) else False)
                if side != s.predicted_side or q_old != q_new:
                    report.append(
                        f"  wk{week} {s.away_team}@{s.home_team} {bet_type:6} "
                        f"line {float(s.open_line):+6.1f}  "
                        f"v1 {str(s.predicted_side):5}{'*' if q_old else ' '} -> "
                        f"v2 {str(side):5}{'*' if q_new else ' '}")

    print(f"\n{len(rows)} v2 rows built. Picks or flags that CHANGED "
          f"({len(report)}):")
    for line in report:
        print(line)
    if not report:
        print("  none — v2 reproduces v1 exactly")
    print("\n* = qualifies. Injuries are zero in every v2 row above: "
          "injury_flags has no history to reconstruct.")

    new = [r for r in rows if (r["game_id"], r["bet_type"]) not in existing]
    print(f"{len(new)} not yet written ({len(rows) - len(new)} already present)")
    if dry:
        print("--dry-run: nothing written")
        return
    for i in range(0, len(new), 500):
        supabase.table("line_predictions").insert(new[i:i + 500]).execute()
    print(f"wrote {len(new)} {VERSION} rows")


if __name__ == "__main__":
    main()
