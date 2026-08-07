"""
Score upcoming college games and freeze a prediction per (game, bet_type).

Mirrors log_clv_predictions.py but writes to the cfb_ tables and, importantly,
flags NOTHING as qualifying. CFB spreads sit at p = 0.077 on label permutation
and CFB totals at p = 0.192, so neither has evidence behind a recommendation.
The models' opinions get shown; the user decides.

Week 1 feature construction
---------------------------
The season has not started, so there is no in-season form. Every rolling
feature is zero, exactly as the training data has it for a team's first game,
and `games_played` is zero so the model knows how much evidence backs that.
What carries the prediction is the preseason block, which is what the CFB
ablation showed matters most anyway:

    SP+ and FPI from the PRIOR season (2025)
    returning production, published preseason
    recruiting talent
    Elo, carried over from the end of last season

Two honest compromises:
  * 2026 talent composites are not published yet, so 2025 is used for both
    talent and talent_prev. The composite is a multi-year rolling average, so
    it moves slowly, but it is a substitution and worth knowing about.
  * rest_days defaults to 7 in week 1 because there is no previous game.

Usage:
    python log_cfb_predictions.py
    python log_cfb_predictions.py --dry-run
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from config import DATA_DIR, MODELS_DIR
from cfb_teams import cfbd_to_key
from score_week import supabase
from build_cfb_dataset import (FEATURE_COLS, PRESEASON_COLS, INSEASON_COLS,
                               haversine)

load_dotenv(DATA_DIR.parent / ".env")
API = "https://api.collegefootballdata.com"
HDRS = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
SEASON = 2026
PRIOR = SEASON - 1


def get(path, **params):
    r = requests.get(f"{API}{path}", headers=HDRS, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def preseason_block() -> pd.DataFrame:
    """Team-level strength priors, all knowable before week 1."""
    rows = {}
    for r in get("/ratings/sp", year=PRIOR):
        k = cfbd_to_key(r.get("team"))
        if k:
            off, dfn = r.get("offense") or {}, r.get("defense") or {}
            rows.setdefault(k, {}).update({
                "sp_prev": r.get("rating"), "sp_off_prev": off.get("rating"),
                "sp_def_prev": dfn.get("rating")})
    for r in get("/ratings/fpi", year=PRIOR):
        k = cfbd_to_key(r.get("team"))
        if k:
            eff = r.get("efficiencies") or {}
            rows.setdefault(k, {}).update({
                "fpi_prev": r.get("fpi"), "fpi_off_prev": eff.get("offense"),
                "fpi_def_prev": eff.get("defense"),
                "fpi_st_prev": eff.get("specialTeams")})
    for r in get("/player/returning", year=SEASON):
        k = cfbd_to_key(r.get("team"))
        if k:
            rows.setdefault(k, {})["returning_ppa"] = r.get("percentPPA")
    # 2026 talent is unpublished; the prior year stands in for both columns.
    tal = get("/talent", year=PRIOR)
    for r in tal:
        k = cfbd_to_key(r.get("team"))
        if k:
            rows.setdefault(k, {}).update({"talent": r.get("talent"),
                                           "talent_prev": r.get("talent")})
    # Elo entering the season: each team's final rating from last year.
    elo = {}
    for w in range(16, 0, -1):
        try:
            for r in get("/ratings/elo", year=PRIOR, week=w):
                k = cfbd_to_key(r.get("team"))
                if k and k not in elo and r.get("elo") is not None:
                    elo[k] = float(r["elo"])
        except Exception:
            continue
        if len(elo) > 100:
            break
    for k, v in elo.items():
        rows.setdefault(k, {})["elo"] = v

    d = pd.DataFrame([{"team": t, **v} for t, v in rows.items()])
    for c in PRESEASON_COLS + INSEASON_COLS:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            d[c] = d[c].fillna(d[c].median())
        else:
            d[c] = 0.0
    return d


def venue_block() -> tuple:
    v = pd.DataFrame([{"venue_id": x.get("id"), "lat": x.get("latitude"),
                       "lon": x.get("longitude"), "elevation": x.get("elevation"),
                       "is_dome": int(bool(x.get("dome")))} for x in get("/venues")])
    for c in ("lat", "lon", "elevation"):
        v[c] = pd.to_numeric(v[c], errors="coerce")
    g = pd.DataFrame([{"home_raw": x.get("homeTeam"), "away_raw": x.get("awayTeam"),
                       "venue_id": x.get("venueId"),
                       "neutral_site": int(bool(x.get("neutralSite"))),
                       "conference_game": int(bool(x.get("conferenceGame"))),
                       "week": x.get("week")}
                      for x in get("/games", year=SEASON, seasonType="regular")])
    g["home_team"] = g.home_raw.map(cfbd_to_key)
    g["away_team"] = g.away_raw.map(cfbd_to_key)
    return g.dropna(subset=["home_team", "away_team"]), v


def main():
    dry = "--dry-run" in sys.argv

    res = supabase.table("cfb_line_history").select("*").execute()
    lines = pd.DataFrame(res.data or [])
    if lines.empty:
        print("No rows in cfb_line_history — run fetch_cfb_odds.py first")
        return
    lines["ct"] = pd.to_datetime(lines["commence_time"], utc=True)
    games = (lines.sort_values("recorded_at")
             .groupby(["home_team", "away_team"], as_index=False).first())
    print(f"CFB: {len(games)} games with a line")

    pre = preseason_block()
    sched, ven = venue_block()
    games = games.merge(sched.drop(columns=["home_raw", "away_raw"]),
                        on=["home_team", "away_team"], how="left")
    games = games.merge(ven, on="venue_id", how="left")

    # Travel and altitude, from each team's usual home venue.
    homes = sched.dropna(subset=["venue_id"]).groupby("home_team")["venue_id"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
    base = ven.set_index("venue_id")[["lat", "lon", "elevation"]]
    a_lat = games.away_team.map(homes).map(base["lat"])
    a_lon = games.away_team.map(homes).map(base["lon"])
    games["travel_miles"] = [haversine(la, lo, gl, gn) for la, lo, gl, gn
                             in zip(a_lat, a_lon, games["lat"], games["lon"])]
    games["elev_change"] = games["elevation"] - games.away_team.map(homes).map(base["elevation"])
    for c in ("travel_miles", "elev_change"):
        games[c] = pd.to_numeric(games[c], errors="coerce").fillna(0.0)
    for c in ("is_dome", "neutral_site", "conference_game"):
        games[c] = pd.to_numeric(games[c], errors="coerce").fillna(0).astype(int)

    # Feature matrix. Rolling form is zero in week 1, matching how the training
    # data represents a team's first game.
    X = pd.DataFrame(index=games.index)
    for side in ("home", "away"):
        p = pre.rename(columns={"team": f"{side}_team"})
        games = games.merge(p, on=f"{side}_team", how="left", suffixes=("", f"_{side}"))
    for c in PRESEASON_COLS + INSEASON_COLS:
        h = games[c] if c in games else 0.0
        a = games[f"{c}_away"] if f"{c}_away" in games else 0.0
        X[f"diff_{c}"] = pd.to_numeric(h, errors="coerce").fillna(0) - \
                         pd.to_numeric(a, errors="coerce").fillna(0)
    for c in FEATURE_COLS:
        X[f"diff_{c}"] = 0.0
    X["diff_games_played"] = 0.0
    X["diff_rest_days"] = 0.0
    for c in ("neutral_site", "conference_game", "travel_miles",
              "elev_change", "is_dome"):
        X[c] = games[c].astype(float)

    # Are the rolling-form features actually carrying information? Right now the
    # block above hardcodes every one of them to 0.0, so this is False for every
    # game in every week. Detecting it rather than hardcoding a week number means
    # totals switch back on by themselves the moment in-season form is wired in,
    # and cannot be re-enabled by the calendar alone.
    form_cols = [f"diff_{c}" for c in FEATURE_COLS if f"diff_{c}" in X.columns]
    form_populated = bool(X[form_cols].abs().to_numpy().sum() > 0) if form_cols else False

    rows = []
    withheld = 0
    for kind, model_name, line_col, market in [
            ("spread", "cfb_movement", "spread_home", "spread"),
            ("total", "cfb_total_residual", "total", "total")]:
        mp = MODELS_DIR / f"{model_name}_model.joblib"
        if not mp.exists():
            print(f"  missing {mp.name} — run build_production_models.py")
            continue
        model = joblib.load(mp)
        feats = joblib.load(MODELS_DIR / f"{model_name}_features.joblib")
        Xa = X.copy()
        Xa["week_open_spread_home"] = pd.to_numeric(games["spread_home"], errors="coerce").fillna(0)
        Xa["week_open_total"] = pd.to_numeric(games["total"], errors="coerce").fillna(0)
        missing = [c for c in feats if c not in Xa.columns]
        assert not missing, f"{market}: missing features {missing[:5]}"
        pred = model.predict(Xa[feats].fillna(0))

        if market == "spread":
            mm = joblib.load(MODELS_DIR / "cfb_margin_model.joblib")
            mf = joblib.load(MODELS_DIR / "cfb_margin_features.joblib")
            margin = mm.predict(Xa[mf].fillna(0))
        for i, g in games.iterrows():
            line = pd.to_numeric(g[line_col], errors="coerce")
            if pd.isna(line):
                continue
            if market == "spread":
                # The tested CFB spread rule is DIRECTION AGREEMENT between the
                # two models (see build_production_models.py) -- that is the
                # configuration the 54.7% walk-forward figure was measured on.
                # This previously took the movement model's sign alone, which is
                # a different and untested rule: the two disagree on roughly a
                # third of the board, and the movement model's median |pred| here
                # is 0.22 points, so on its own it was mostly signing noise.
                # Rows are still written for every game so the panel can show
                # what both models think; games without agreement simply carry no
                # side rather than a coin flip dressed as a pick.
                dis = float(margin[i] + line)
                mv_side = "home" if pred[i] < 0 else "away"
                dis_side = "home" if dis > 0 else "away"
                side = mv_side if mv_side == dis_side else None
                rows.append({
                    "game_id": g["game_id"], "bet_type": "spread", "season": SEASON,
                    "week": int(g["week"]) if pd.notna(g.get("week")) else None,
                    "home_team": g["home_team"], "away_team": g["away_team"],
                    "commence_time": g["commence_time"], "open_line": float(line),
                    "predicted_movement": round(float(pred[i]), 3),
                    "projected_value": round(float(margin[i]), 2),
                    "margin_disagreement": round(float(margin[i] + line), 3),
                    "predicted_side": side,
                    "taken_line": None if side is None else
                                  (float(line) if side == "home" else -float(line))})
            else:
                # Withheld until teams have real form. Rolling features are zero
                # in week 1 (see above), and a total is a LEVEL rather than a
                # difference, so those zeros do not cancel the way they do for
                # spreads -- they drag every prediction down. Measured on week-1
                # training games the model calls over 12% of the time against a
                # true rate of 38%, and on this live board it called 68 of 73
                # games under. That is a broken output, not a weak edge, so it
                # is suppressed at the source rather than deleted after the fact.
                # Deleting rows by hand did not hold: the 3-hourly cron simply
                # rewrote them.
                if not form_populated:
                    withheld += 1
                    continue
                side = "over" if pred[i] > 0 else "under"
                rows.append({
                    "game_id": g["game_id"], "bet_type": "total", "season": SEASON,
                    "week": int(g["week"]) if pd.notna(g.get("week")) else None,
                    "home_team": g["home_team"], "away_team": g["away_team"],
                    "commence_time": g["commence_time"], "open_line": float(line),
                    "predicted_movement": None,
                    "projected_value": round(float(line + pred[i]), 2),
                    "margin_disagreement": None,
                    "predicted_side": side, "taken_line": float(line)})

    ns = sum(1 for r in rows if r["bet_type"] == "spread")
    print(f"  {len(rows)} predictions ({ns} spread, {len(rows)-ns} total)")
    agree = sum(1 for r in rows
                if r["bet_type"] == "spread" and r["predicted_side"] is not None)
    if ns:
        print(f"  {agree}/{ns} spreads have a side (movement and margin models "
              f"agree); {ns-agree} shown without a pick")
    if withheld:
        print(f"  {withheld} totals withheld (rolling form all zero — "
              f"a level-valued prediction cannot be trusted without it)")
    # The side is decided by predicted_movement (which way the line is expected
    # to go), not by projected_value (the margin model's view of the game). Those
    # are two different models and they disagree often, so print the one that
    # actually drives the pick -- showing only the margin made every line look
    # like the sign was inverted.
    for r in rows[:10]:
        mv = r["predicted_movement"]
        drv = f"move {mv:+6.2f}" if mv is not None else "resid    n/a"
        print(f"    [{r['bet_type'][:3].upper()}] {r['away_team']:>16} @ "
              f"{r['home_team']:<16} line {r['open_line']:+7.1f}  {drv}  "
              f"margin {r['projected_value']:+7.2f}  -> "
              f"{r['predicted_side'] or 'no pick (models split)'}")

    if dry:
        print("  --dry-run: nothing written")
        return
    if rows:
        supabase.table("cfb_predictions").upsert(
            rows, on_conflict="game_id,bet_type").execute()
        print(f"  wrote {len(rows)} CFB predictions (nothing flagged as qualifying)")


if __name__ == "__main__":
    main()
