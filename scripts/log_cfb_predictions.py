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

Every run also appends to cfb_prediction_log, which is append-only. This table
upserts on (game_id, bet_type) and re-stamps predicted_at, so it holds only the
CURRENT prediction and cannot evidence what was on screen before a kickoff --
see the migration cfb_prediction_log_append_only for why that matters and what
the log records instead.

Usage:
    python log_cfb_predictions.py
    python log_cfb_predictions.py --dry-run
"""

import hashlib
import os
import subprocess
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
from score_week import supabase, fetch_all
from build_cfb_dataset import (FEATURE_COLS, PRESEASON_COLS, INSEASON_COLS,
                               haversine)

load_dotenv(DATA_DIR.parent / ".env")
API = "https://api.collegefootballdata.com"
HDRS = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
SEASON = 2026
PRIOR = SEASON - 1


# Every CFBD endpoint below returns data that is fixed for the season: final
# prior-season SP+/FPI/talent/Elo, the venue list, and the schedule. This job
# runs every 3 hours, so it was refetching all seven of them ~56 times a day for
# values that never changed, and on 2026-08-11 that exhausted the monthly call
# quota and took the logger down entirely.
#
# TTL controls when a refresh is ATTEMPTED. It is not an expiry: if CFBD is
# unreachable or out of quota, a stale row is always served instead of raising.
# Month-old SP+ is identical to today's SP+, and a run that produces predictions
# beats a run that dies on a 429.
CACHE_TTL_DAYS = {
    "/ratings/sp": 365,        # final prior-season ratings, never change
    "/ratings/fpi": 365,
    "/ratings/elo": 365,
    "/talent": 365,
    "/venues": 365,
    "/player/returning": 30,   # preseason figure, occasionally revised
    "/games": 7,               # kick times and venues do shift
}
_cache_stats = {"hit": 0, "miss": 0, "stale": 0}


def _cache_key(path, params):
    return path + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))


def _cache_read(key):
    try:
        r = (supabase.table("cfb_api_cache").select("payload, fetched_at")
             .eq("cache_key", key).limit(1).execute())
        return (r.data or [None])[0]
    except Exception:
        return None


def get(path, **params):
    key = _cache_key(path, params)
    row = _cache_read(key)
    fresh = False
    if row:
        age = (pd.Timestamp.now(tz="UTC")
               - pd.to_datetime(row["fetched_at"], utc=True)).days
        fresh = age <= CACHE_TTL_DAYS.get(path, 7)
    if row and fresh:
        _cache_stats["hit"] += 1
        return row["payload"]

    try:
        r = requests.get(f"{API}{path}", headers=HDRS, params=params, timeout=120)
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        # Serve stale rather than fail. This is the path that keeps the job
        # alive through a quota outage.
        if row is not None:
            _cache_stats["stale"] += 1
            print(f"  CFBD {path} failed ({type(e).__name__}) — using cached copy")
            return row["payload"]
        raise

    _cache_stats["miss"] += 1
    try:
        supabase.table("cfb_api_cache").upsert(
            {"cache_key": key, "payload": payload,
             "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
            on_conflict="cache_key").execute()
    except Exception as e:
        print(f"  warning: could not cache {path} ({type(e).__name__})")
    return payload


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


def model_hash() -> str:
    """Digest of the model files actually loaded, so a retrain is visible.

    A prediction is only reproducible against the weights that produced it.
    cfb_predictions cannot show that those weights changed; a hash in the
    append-only log can.
    """
    h = hashlib.sha256()
    for name in sorted(("cfb_movement", "cfb_margin", "cfb_total_residual")):
        f = MODELS_DIR / f"{name}_model.joblib"
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def code_version() -> str:
    """Short git sha, or 'dirty' when the tree has uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             cwd=str(DATA_DIR.parent)).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "scripts"],
                               capture_output=True, text=True, timeout=10,
                               cwd=str(DATA_DIR.parent)).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


# Fields that make one logged observation different from another. A run that
# changes none of them writes nothing: the 3-hourly cron would otherwise append
# ~1,600 identical rows a day and bury the changes that matter.
LOG_KEYS = ("open_line", "predicted_movement", "projected_value",
            "margin_disagreement", "predicted_side", "line_now")


def _num(v):
    """Plain Python float, or None. numpy scalars are not JSON serialisable."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def write_prediction_log(rows, model_h, code_v):
    """Append what the model said, plus the live line and splits beside it.

    Returns the number of rows written. Never updates or deletes -- the point of
    the table is that history here cannot be restated.
    """
    if not rows:
        return 0
    now = pd.Timestamp.now(tz="UTC")

    # The live line, and the most recent splits capture, per game.
    hist = pd.DataFrame(fetch_all("cfb_line_history"))
    live = {}
    if not hist.empty:
        hist = hist.sort_values("recorded_at")
        # The odds refresh does not stop at kickoff, so the newest snapshot for a
        # played game is a LIVE number -- TULSA opened +13.5 and the last row
        # says -13.5, recorded 90 minutes into the game. Take the last snapshot
        # before kickoff, the same rule cfb_open_close uses for its close.
        kick = pd.to_datetime(hist["commence_time"], utc=True, errors="coerce")
        rec_at = pd.to_datetime(hist["recorded_at"], utc=True, errors="coerce")
        hist = hist.assign(_pre=(rec_at < kick) | kick.isna())
        for gid, g in hist.groupby("game_id"):
            pre = g[g._pre]
            last = (pre if len(pre) else g).iloc[-1]
            # spread_home is signed relative to whichever side the feed called
            # home in THAT snapshot, and neutral-site games flip mid-week. Store
            # the orientation so it can be matched to the prediction's.
            live[gid] = (last.get("spread_home"), last.get("total"),
                         last.get("recorded_at"), last.get("home_team"))
    splits = {}
    sp = pd.DataFrame(fetch_all("cfb_public_splits"))
    if not sp.empty:
        sp = sp.sort_values("captured_at")
        for (gid, bt), g in sp.groupby(["game_id", "bet_type"]):
            splits[(gid, bt)] = g.iloc[-1]

    # The latest logged row per game, to log changes only.
    prev = {}
    for r in fetch_all("cfb_prediction_log"):
        k = (r["game_id"], r["bet_type"])
        if k not in prev or r["logged_at"] > prev[k]["logged_at"]:
            prev[k] = r

    out = []
    for r in rows:
        gid, bt = r["game_id"], r["bet_type"]
        sh, tot, at, live_home = live.get(gid, (None, None, None, None))
        if sh is not None and live_home and live_home != r["home_team"]:
            sh = -float(sh)          # feed flipped home/away since the opener
        line_now = sh if bt == "spread" else tot
        s = splits.get((gid, bt))
        kick = pd.to_datetime(r["commence_time"], utc=True, errors="coerce")
        rec = {
            "game_id": gid, "bet_type": bt,
            "season": None if r["season"] is None else int(r["season"]),
            "week": None if r["week"] is None else int(r["week"]),
            "home_team": r["home_team"],
            "away_team": r["away_team"], "commence_time": r["commence_time"],
            "hours_to_kickoff": (None if pd.isna(kick) else
                                 round((kick - now).total_seconds() / 3600, 2)),
            "open_line": _num(r["open_line"]),
            "predicted_movement": _num(r["predicted_movement"]),
            "projected_value": _num(r["projected_value"]),
            "margin_disagreement": _num(r["margin_disagreement"]),
            "predicted_side": r["predicted_side"],
            "taken_line": _num(r["taken_line"]),
            "line_now": _num(line_now),
            "line_now_at": at,
            "splits_captured_at": None if s is None else s["captured_at"],
            "home_bets_pct": None if s is None else _num(s["home_bets_pct"]),
            "away_bets_pct": None if s is None else _num(s["away_bets_pct"]),
            "home_money_pct": None if s is None else _num(s["home_money_pct"]),
            "away_money_pct": None if s is None else _num(s["away_money_pct"]),
            "model_hash": model_h, "code_version": code_v,
        }
        p = prev.get((gid, bt))
        if p is not None:
            same = all((p.get(k) is None and rec[k] is None) or
                       (p.get(k) is not None and rec[k] is not None and
                        abs(float(p[k]) - float(rec[k])) < 1e-6)
                       if k != "predicted_side" else p.get(k) == rec[k]
                       for k in LOG_KEYS)
            if same and p.get("model_hash") == model_h:
                continue
            rec["note"] = "changed"
        elif (rec["hours_to_kickoff"] or 0) < 0:
            # An entry created after the game started proves only that the
            # prediction is reproducible, NOT that it was on screen in time.
            # Say so in the row rather than letting it read as a live pick.
            rec["note"] = "backfill after kickoff"
        else:
            rec["note"] = "first observation"
        out.append(rec)

    for i in range(0, len(out), 500):
        supabase.table("cfb_prediction_log").insert(out[i:i + 500]).execute()
    return len(out)


def main():
    dry = "--dry-run" in sys.argv

    lines = pd.DataFrame(fetch_all("cfb_line_history"))
    if lines.empty:
        print("No rows in cfb_line_history — run fetch_cfb_odds.py first")
        return
    lines["ct"] = pd.to_datetime(lines["commence_time"], utc=True)
    games = (lines.sort_values("recorded_at")
             .groupby(["home_team", "away_team"], as_index=False).first())

    # One row per GAME, not per orientation. A neutral-site game can arrive with
    # its home and away sides swapped partway through -- OKLAHOMA @ TEXAS did
    # exactly that, 33 snapshots one way then one the other -- which makes two
    # groups above sharing a single game_id, and cfb_predictions is keyed on
    # (game_id, bet_type). The upsert then fails outright with "ON CONFLICT DO
    # UPDATE command cannot affect row a second time". Keep the orientation the
    # feed used most, which is the same rule cfb_open_close applies.
    counts = (lines.groupby(["game_id", "home_team", "away_team"])
              .size().rename("n").reset_index())
    games = games.merge(counts, on=["game_id", "home_team", "away_team"], how="left")
    before = len(games)
    games = (games.sort_values("n", ascending=False)
             .drop_duplicates("game_id", keep="first")
             .drop(columns="n").reset_index(drop=True))
    if before != len(games):
        print(f"  collapsed {before - len(games)} flipped-orientation duplicate(s)")
    print(f"CFB: {len(games)} games with a line")

    # A quota outage with nothing cached is an external condition, not a code
    # fault, and it self-heals when the allowance resets. Exit cleanly so the
    # 3-hourly cron does not raise an alert every run for days on end -- but say
    # so loudly, because the visible symptom is that predictions stop updating
    # while the odds keep flowing.
    try:
        pre = preseason_block()
        sched, ven = venue_block()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print("\n  !! CFBD monthly call quota exhausted and nothing cached.")
            print("     Odds and line history are unaffected; CFB predictions")
            print("     will not refresh until the allowance resets.")
            print("     Existing logged predictions are frozen and still valid.\n")
            return
        raise
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

    # Stamped explicitly, not left to the column default. cfb_predictions upserts
    # on (game_id, bet_type) and a DB default only fires on INSERT, so every
    # re-run refreshed the numbers while predicted_at stayed pinned to the first
    # run forever -- the tab could not report how fresh it was, and the Refresh
    # button has nothing to poll for completion. Same bug preseason had.
    now_iso = pd.Timestamp.now(tz="UTC").isoformat()

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
                    "predicted_at": now_iso,
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
                    "predicted_side": side, "taken_line": float(line),
                    "predicted_at": now_iso})

    print(f"  CFBD calls: {_cache_stats['miss']} live, "
          f"{_cache_stats['hit']} cached, {_cache_stats['stale']} served stale")

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
        # The upsert above overwrote the previous state. This does not.
        mh, cv = model_hash(), code_version()
        n = write_prediction_log(rows, mh, cv)
        print(f"  prediction log: +{n} row(s) (models {mh}, code {cv})"
              if n else f"  prediction log: no change (models {mh})")


if __name__ == "__main__":
    main()
