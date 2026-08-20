"""
Run preseason games through the movement/margin models. For fun, not for edge.

Read this before trusting a number it prints
--------------------------------------------
The models take opponent-adjusted EPA and success rates computed from STARTERS
in regular-season games. In preseason those players are wearing baseball caps on
the sideline. The features describe a team that is not on the field, so the
projections are a category error, not a slightly noisier version of the real
thing.

Nothing measured in this project transfers here. The 57.8% holdout win rate and
+1.55 CLV were both established on regular-season games only. There is no
preseason validation and none is possible -- we have no historical preseason
line data to backtest against.

It is kept strictly apart from the real pipeline:
  * predictions land in preseason_predictions, never line_predictions, so the
    Model page's track record cannot be polluted
  * lines come from preseason_lines, never line_open_close
  * no qualifying flag is written, because the thresholds were fitted on
    regular-season behaviour and mean nothing here

Also note 2026 metrics do not exist yet, so every team is carried at its end of
2025 strength -- last year's starters, projected onto this year's backups.

Usage:
    python log_preseason_predictions.py
    python log_preseason_predictions.py --dry-run
"""

import sys
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import MODELS_DIR, CURRENT_SEASON
from score_week import (supabase, fetch_team_metrics, fetch_weather,
                        fetch_injury_aggregates, build_feature_matrix,
                        assert_feature_parity)

# nflverse carries no preseason games at all -- REG and playoffs only -- so the
# schedule has to be reconstructed from the odds board itself.
DIVISIONS = {
    "AFCE": ["BUF", "MIA", "NE", "NYJ"], "AFCN": ["BAL", "CIN", "CLE", "PIT"],
    "AFCS": ["HOU", "IND", "JAX", "TEN"], "AFCW": ["DEN", "KC", "LV", "LAC"],
    "NFCE": ["DAL", "NYG", "PHI", "WAS"], "NFCN": ["CHI", "DET", "GB", "MIN"],
    "NFCS": ["ATL", "CAR", "NO", "TB"],   "NFCW": ["ARI", "LA", "SF", "SEA"],
}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}


def load_games(season: int) -> tuple:
    """One row per preseason game, plus the consensus line, from preseason_lines."""
    res = supabase.table("preseason_lines").select("*").execute()
    df = pd.DataFrame(res.data or [])
    if df.empty:
        return pd.DataFrame(), {}

    df["ct"] = pd.to_datetime(df["commence_time"], utc=True)
    g = df.groupby(["game_id", "home_team", "away_team", "ct"]).agg(
        spread_home=("spread_home", "median"),
        total=("total", "median"),
        n_books=("book", "nunique")).reset_index().sort_values("ct")

    # Preseason "weeks" are just the order the games fall in.
    g["week"] = g["ct"].dt.isocalendar().week
    g["week"] = g["week"].rank(method="dense").astype(int).clip(1, 4)
    g["season"] = season
    g["gameday"] = g["ct"].dt.strftime("%Y-%m-%d")
    g["gametime"] = g["ct"].dt.strftime("%H:%M")
    g["div_game"] = [int(TEAM_DIV.get(h) == TEAM_DIV.get(a) and TEAM_DIV.get(h) is not None)
                     for h, a in zip(g.home_team, g.away_team)]
    # No preseason rest data exists; the league default keeps the feature sane.
    g["home_rest"] = 7.0
    g["away_rest"] = 7.0

    lines = {(r.home_team, r.away_team): {"spread_home": r.spread_home, "total": r.total}
             for r in g.itertuples()}
    return g, lines


def main():
    dry = "--dry-run" in sys.argv
    season = CURRENT_SEASON

    mv_path = MODELS_DIR / "movement_model.joblib"
    if not mv_path.exists():
        print("No movement_model.joblib — run: python model_line_movement.py --save")
        return
    mv = joblib.load(mv_path)
    mv_f = joblib.load(MODELS_DIR / "movement_features.joblib")
    mm = joblib.load(MODELS_DIR / "margin_model.joblib")
    mm_f = joblib.load(MODELS_DIR / "margin_features.joblib")
    tm = joblib.load(MODELS_DIR / "total_movement_model.joblib")
    tm_f = joblib.load(MODELS_DIR / "total_movement_features.joblib")

    games, lines = load_games(season)
    if games.empty:
        print("No preseason lines stored — run fetch_preseason_lines.py first")
        return
    print(f"preseason games on the board: {len(games)}")

    pairs = list(zip(games["home_team"], games["away_team"]))
    metrics = fetch_team_metrics(season, 1)
    weather = fetch_weather(pairs)
    injuries = fetch_injury_aggregates()
    feats = build_feature_matrix(games, metrics, lines, weather, injuries)

    feats["week_open_spread_home"] = feats["dk_spread"].astype(float)
    feats["week_open_total"] = feats["dk_total"].astype(float)

    assert_feature_parity(feats, mv_f, "movement")
    move = mv.predict(feats[mv_f].fillna(0))
    assert_feature_parity(feats, mm_f, "margin")
    margin = mm.predict(feats[mm_f].fillna(0))
    assert_feature_parity(feats, tm_f, "total_movement")
    tmove = tm.predict(feats[tm_f].fillna(0))

    disagree = margin + feats["week_open_spread_home"].values

    # Stamped explicitly rather than left to the column default. This table is
    # upserted on (game_id, bet_type), and a DB default only fires on INSERT --
    # so re-running refreshed every number while leaving predicted_at frozen at
    # the first run forever. Nothing here freezes a price the way
    # line_predictions deliberately does, so "last recomputed" is the honest
    # meaning of this column.
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()

    rows = []
    for i, (_, g) in enumerate(feats.iterrows()):
        gid = games.iloc[i]["game_id"]
        opener = float(g["dk_spread"])
        rows.append({
            "game_id": gid, "bet_type": "spread", "season": season,
            "home_team": g["home_team"], "away_team": g["away_team"],
            "commence_time": g["game_time"], "open_line": opener,
            "predicted_movement": round(float(move[i]), 3),
            "projected_margin": round(float(margin[i]), 2),
            "margin_disagreement": round(float(disagree[i]), 3),
            "predicted_side": "home" if move[i] < 0 else "away",
            "predicted_at": now_iso,
        })
        rows.append({
            "game_id": gid, "bet_type": "total", "season": season,
            "home_team": g["home_team"], "away_team": g["away_team"],
            "commence_time": g["game_time"], "open_line": float(g["dk_total"]),
            "predicted_movement": round(float(tmove[i]), 3),
            "projected_margin": None,
            "margin_disagreement": None,
            "predicted_side": "over" if tmove[i] > 0 else "under",
            "predicted_at": now_iso,
        })

    # The side comes from the MOVEMENT model. The margin model is shown next to
    # it as context, and the two point opposite ways often enough that printing
    # the margin alone made the pick look inverted. Name the team rather than
    # 'home'/'away', and say plainly when the two models split.
    splits = 0
    for r in rows:
        if r["bet_type"] == "spread":
            team = r["home_team"] if r["predicted_side"] == "home" else r["away_team"]
            dis = r["margin_disagreement"]
            dis_team = r["home_team"] if dis > 0 else r["away_team"]
            split = dis_team != team
            splits += split
            note = (f"  << SPLIT: margin model likes {dis_team}" if split
                    else f"  (margin agrees: {dis_team})")
            print(f"  [SPR] {r['away_team']:>3} @ {r['home_team']:<3}  "
                  f"line {r['open_line']:+6.1f}  move {r['predicted_movement']:+6.2f}"
                  f"  -> {team:<3}  margin {r['projected_margin']:+6.2f}{note}")
        else:
            print(f"  [TOT] {r['away_team']:>3} @ {r['home_team']:<3}  "
                  f"line {r['open_line']:+6.1f}  move {r['predicted_movement']:+6.2f}"
                  f"  -> {r['predicted_side']}")
    n_spr = sum(1 for r in rows if r["bet_type"] == "spread")
    if n_spr:
        print(f"\n  movement and margin models split on {splits} of {n_spr} spreads")

    if dry:
        print("  --dry-run: nothing written")
        return
    supabase.table("preseason_predictions").upsert(rows, on_conflict="game_id,bet_type").execute()
    print(f"  wrote {len(rows)} preseason projections (NOT part of the tracked record)")


if __name__ == "__main__":
    main()
