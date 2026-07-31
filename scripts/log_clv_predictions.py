"""
Log a line-movement prediction for EVERY upcoming game, so closing line value
is captured automatically and completely.

Why every game, not just bets
-----------------------------
CLV is the fastest honest read on whether the movement model works -- it
resolves in weeks, where win rate needs a full season of coin flips. But it is
only trustworthy if it is measured on the complete slate. Logging only games we
liked would bias the sample toward whatever the model happened to feel strongly
about, which is exactly the kind of selection that has produced false positives
in this project before.

How it works
------------
- One row per game, written the FIRST time we see it inside the weekly window
  (UNIQUE(game_id) in the DB enforces this). That freezes the number that was
  actually available when the board posted.
- CLV itself is NOT stored. The clv_tracking view derives it by joining the
  frozen prediction to the current closing line, so it stays correct as the
  line firms up and needs no backfill job.

Only games kicking off within WEEKLY_WINDOW_DAYS are logged, matching the
"weekly opener" definition the model was trained and validated against -- the
lookahead lines posted months out are a different, much noisier animal.

Usage:
    python log_clv_predictions.py            # current season, auto week
    python log_clv_predictions.py 2026 1      # explicit
    python log_clv_predictions.py --dry-run   # show, write nothing
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import MODELS_DIR, CURRENT_SEASON
from score_week import (supabase, fetch_current_schedule, current_week_number,
                        fetch_team_metrics, fetch_latest_lines, fetch_weather,
                        fetch_injury_aggregates, build_feature_matrix,
                        assert_feature_parity)

WEEKLY_WINDOW_DAYS = 10


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    season = int(args[0]) if args else CURRENT_SEASON
    week = int(args[1]) if len(args) > 1 else current_week_number(season)

    model_path = MODELS_DIR / "movement_model.joblib"
    if not model_path.exists():
        print("No movement_model.joblib — run: python model_line_movement.py --save")
        return
    model = joblib.load(model_path)
    feat_order = joblib.load(MODELS_DIR / "movement_features.joblib")

    print(f"CLV logging: season={season} week={week}")
    games = fetch_current_schedule(season, week)
    if games.empty:
        print("  no regular-season games this week")
        return
    pairs = list(zip(games["home_team"], games["away_team"]))

    metrics = fetch_team_metrics(season, week)
    lines = fetch_latest_lines(pairs)
    weather = fetch_weather(pairs)
    injuries = fetch_injury_aggregates()
    feats = build_feature_matrix(games, metrics, lines, weather, injuries)

    # The movement model's extra input: the number currently on the board.
    # dk_spread already uses the standard convention (negative = home favored),
    # matching week_open_spread_home in training.
    feats["week_open_spread_home"] = feats["dk_spread"].astype(float)

    assert_feature_parity(feats, feat_order, "movement")
    preds = model.predict(feats[feat_order].fillna(0))

    # Which games already have a frozen prediction? Never overwrite: the whole
    # point is to hold the number from when the board first posted.
    existing = supabase.table("line_predictions").select("game_id") \
        .eq("season", season).eq("week", week).execute()
    already = {r["game_id"] for r in (existing.data or [])}

    # game_id must match line_open_close (The Odds API event id) so the view can
    # join; fall back to the team pair if a line row is missing.
    rows, skipped = [], 0
    for (_, g), pred in zip(feats.iterrows(), preds):
        line = lines.get((g["home_team"], g["away_team"]), {})
        gid = line.get("game_id") or f"{season}_{week}_{g['away_team']}_{g['home_team']}"
        if gid in already:
            skipped += 1
            continue
        opener = float(g["dk_spread"])
        side = "home" if pred < 0 else "away"
        rows.append({
            "game_id": gid,
            "season": int(g["season"]),
            "week": int(g["week"]),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "commence_time": g["game_time"],
            "open_spread_home": opener,
            "predicted_movement": round(float(pred), 3),
            "predicted_side": side,
            "taken_line": opener if side == "home" else -opener,
        })

    print(f"  {len(games)} games | {skipped} already logged | {len(rows)} new")
    for r in rows:
        print(f"    {r['away_team']:>3} @ {r['home_team']:<3}  open {r['open_spread_home']:+5.1f}  "
              f"pred {r['predicted_movement']:+6.2f} -> take {r['predicted_side']:<4} "
              f"at {r['taken_line']:+5.1f}")

    if dry:
        print("  --dry-run: nothing written")
        return
    if rows:
        supabase.table("line_predictions").upsert(rows, on_conflict="game_id").execute()
        print(f"  wrote {len(rows)} predictions")

    # Show CLV so far for this season.
    tr = supabase.table("clv_tracking").select("*").eq("season", season).execute()
    df = pd.DataFrame(tr.data or [])
    if not df.empty and df["clv_points"].notna().any():
        c = df["clv_points"].dropna()
        d = df["direction_correct"].dropna()
        print(f"\n  CLV so far ({len(c)} closed games): mean {c.mean():+.2f} pts, "
              f"positive on {(c > 0).mean()*100:.0f}%"
              + (f", direction right {d.mean()*100:.0f}%" if len(d) else ""))
    else:
        print("\n  CLV pending — no closing lines yet for these games")


if __name__ == "__main__":
    main()
