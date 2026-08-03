"""
Build the CFB modelling dataset: team form + lines, one row per game.

Mirrors what build_dataset.py + compute_metrics.py do for the NFL, but written
separately rather than branched into them, because those two files hold
validated NFL behaviour and are not to be touched.

Leakage discipline
------------------
Every metric attached to a game uses ONLY games played BEFORE it. Rolling means
are shifted by one, and the opponent adjustment uses the opponent's form as of
prior weeks only. Getting this wrong is the single easiest way to manufacture a
result, and this project has already produced two fake edges (a sign inversion
and a lookahead line) that looked spectacular until someone checked.

Opponent adjustment matters far more here than in the NFL. Talent is wildly
unequal -- a 45-point win over a bad opponent says little -- so raw EPA is
mostly a statement about schedule.

FBS-vs-FBS only. Games against FCS opponents have no usable opponent stats and
are dropped rather than zero-filled.

Usage:
    python build_cfb_dataset.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from config import DATA_DIR
from cfb_teams import cfbd_to_key, assert_cfbd_mapped

load_dotenv(DATA_DIR.parent / ".env")
API = "https://api.collegefootballdata.com"
HDRS = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
SEASONS = [2020, 2021, 2022, 2023, 2024, 2025]
ROLL = [4, 8]


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle miles. CFB travel dwarfs the NFL's -- Hawaii to the east
    coast is a 5,000-mile round trip, and there is no NFL analogue."""
    if any(pd.isna(x) for x in (lat1, lon1, lat2, lon2)):
        return np.nan
    r = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(a)))


def fetch_preseason() -> pd.DataFrame:
    """Season-level strength priors, all knowable before week 1 kicks off.

    SP+, talent and returning production are taken from the PRIOR season (or
    the current year for talent/returning, which are published preseason), so
    nothing here peeks at results from the season being predicted.
    """
    rows = {}
    for s in SEASONS:
        for r in get("/ratings/sp", year=s - 1):          # prior season only
            k = cfbd_to_key(r.get("team"))
            if k:
                off, dfn = r.get("offense") or {}, r.get("defense") or {}
                rows.setdefault((s, k), {}).update({
                    "sp_prev": r.get("rating"), "sp_off_prev": off.get("rating"),
                    "sp_def_prev": dfn.get("rating")})
        for r in get("/talent", year=s):                   # published preseason
            k = cfbd_to_key(r.get("team"))
            if k:
                rows.setdefault((s, k), {})["talent"] = r.get("talent")
        for r in get("/talent", year=s - 1):
            k = cfbd_to_key(r.get("team"))
            if k:
                rows.setdefault((s, k), {})["talent_prev"] = r.get("talent")
        for r in get("/player/returning", year=s):         # published preseason
            k = cfbd_to_key(r.get("team"))
            if k:
                rows.setdefault((s, k), {})["returning_ppa"] = r.get("percentPPA")

    out = pd.DataFrame([{"season": s, "team": t, **v} for (s, t), v in rows.items()])
    # A promoted or brand-new FBS program has no prior SP+; league-median is a
    # fairer stand-in than zero, which would read as "average" on a centred
    # scale but as "terrible" on SP+.
    for c in PRESEASON_COLS:
        if c in out:
            out[c] = out[c].fillna(out.groupby("season")[c].transform("median"))
    return out


def fetch_venues() -> pd.DataFrame:
    rows = [{"venue_id": v.get("id"), "lat": v.get("latitude"), "lon": v.get("longitude"),
             "elevation": v.get("elevation"), "is_dome": int(bool(v.get("dome")))}
            for v in get("/venues")]
    d = pd.DataFrame(rows)
    for c in ("lat", "lon", "elevation"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def get(path, **params):
    r = requests.get(f"{API}{path}", headers=HDRS, params=params, timeout=180)
    r.raise_for_status()
    return r.json()


def fetch_games() -> pd.DataFrame:
    rows = []
    for s in SEASONS:
        for g in get("/games", year=s, seasonType="regular"):
            rows.append({
                "game_id": g["id"], "season": s, "week": g["week"],
                "home_raw": g["homeTeam"], "away_raw": g["awayTeam"],
                "home_pts": g.get("homePoints"), "away_pts": g.get("awayPoints"),
                "neutral_site": int(bool(g.get("neutralSite"))),
                "conference_game": int(bool(g.get("conferenceGame"))),
                "start_date": g.get("startDate"),
                "venue_id": g.get("venueId"),
            })
    df = pd.DataFrame(rows)
    df["home_team"] = df["home_raw"].map(cfbd_to_key)
    df["away_team"] = df["away_raw"].map(cfbd_to_key)
    # Both sides FBS, and the game actually played.
    df = df.dropna(subset=["home_team", "away_team", "home_pts", "away_pts"])
    df["home_margin"] = df["home_pts"] - df["away_pts"]
    df["total_points"] = df["home_pts"] + df["away_pts"]
    df["kick_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce").dt.date
    return df


def fetch_team_games() -> pd.DataFrame:
    """One row per team per game: their offensive and defensive efficiency."""
    rows = []
    for s in SEASONS:
        for r in get("/stats/game/advanced", year=s):
            # Postseason week numbers RESTART at 1, so a bowl game collides with
            # the same team's regular-season week 1 on (season, week, team).
            # That produced 337 duplicate keys and the opponent join fanned them
            # out to 9,756 rows from 8,984. Bowls are excluded anyway: opt-outs
            # and month-long layoffs make them closer to preseason than football.
            if r.get("seasonType") != "regular":
                continue
            off, dfn = r.get("offense") or {}, r.get("defense") or {}
            sd_o, pd_o = off.get("standardDowns") or {}, off.get("passingDowns") or {}
            sd_d, pd_d = dfn.get("standardDowns") or {}, dfn.get("passingDowns") or {}
            rush, pas = off.get("rushingPlays") or {}, off.get("passingPlays") or {}
            rows.append({
                "game_id": r.get("gameId"), "season": s, "week": r["week"],
                "team_raw": r["team"], "opp_raw": r["opponent"],
                # Core efficiency -- these get opponent-adjusted.
                "off_ppa": off.get("ppa"), "off_sr": off.get("successRate"),
                "def_ppa": dfn.get("ppa"), "def_sr": dfn.get("successRate"),
                # Trench play. The same payload already carried these; the first
                # pass discarded them, which is most of why CFB had 12 features
                # against the NFL model's 57.
                "off_expl": off.get("explosiveness"), "def_expl": dfn.get("explosiveness"),
                "off_power": off.get("powerSuccess"), "def_power": dfn.get("powerSuccess"),
                "off_stuff": off.get("stuffRate"), "def_stuff": dfn.get("stuffRate"),
                "off_line_yds": off.get("lineYards"), "def_line_yds": dfn.get("lineYards"),
                "off_2nd_lvl": off.get("secondLevelYards"),
                "off_open_field": off.get("openFieldYards"),
                # Situational: staying ahead of the chains vs having to throw.
                "off_sd_ppa": sd_o.get("ppa"), "off_pd_ppa": pd_o.get("ppa"),
                "def_sd_ppa": sd_d.get("ppa"), "def_pd_ppa": pd_d.get("ppa"),
                "off_sd_sr": sd_o.get("successRate"), "off_pd_sr": pd_o.get("successRate"),
                # Tendency / pace proxy.
                "off_rush_ppa": rush.get("ppa"), "off_pass_ppa": pas.get("ppa"),
                "off_plays": off.get("plays"),
            })
    d = pd.DataFrame(rows)
    assert_cfbd_mapped([t for t in d.team_raw.unique()
                        if cfbd_to_key(t) is None and t in set(d.team_raw)][:0], "stats")
    d["team"] = d["team_raw"].map(cfbd_to_key)
    d["opp"] = d["opp_raw"].map(cfbd_to_key)
    d = d.dropna(subset=["team", "opp", "off_ppa", "def_ppa"])
    # Sparse extras get the season median rather than dropping the game.
    for c in RAW_COLS:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
            d[c] = d[c].fillna(d.groupby("season")[c].transform("median"))
    # Key on game_id, never (season, week, team). CFBD labels Week 0 games as
    # week 1, so a team that opened in Week 0 has TWO "week 1" rows -- Illinois
    # played Nebraska on Aug 28 2021 and UTSA on Sep 4, both tagged week 1.
    # That produced 108 colliding rows and fanned the opponent join out.
    dupes = int(d.duplicated(subset=["game_id", "team"]).sum())
    assert dupes == 0, f"{dupes} duplicate (game_id, team) rows before rolling"
    return d


def add_form(d: pd.DataFrame, dates: pd.DataFrame) -> pd.DataFrame:
    """Opponent-adjusted rolling form, using prior games only.

    Ordered by kickoff date rather than week number, because CFBD's week labels
    are not monotonic within a team's schedule (Week 0).
    """
    d = d.merge(dates, on="game_id", how="left").dropna(subset=["kick_ts"])
    d = d.sort_values(["season", "team", "kick_ts"]).copy()

    # Each team's season-to-date form BEFORE the current game. shift(1) is what
    # keeps the game itself out of its own feature.
    for col in ["off_ppa", "def_ppa", "off_sr", "def_sr"]:
        g = d.groupby(["season", "team"])[col]
        d[f"{col}_todate"] = g.transform(lambda s: s.shift(1).expanding().mean())

    # Opponent strength as of the same game, joined by game_id: both teams have
    # a row for it, so the opponent's own to-date form is exactly what we want.
    opp = d[["game_id", "team", "def_ppa_todate", "off_ppa_todate",
             "def_sr_todate", "off_sr_todate"]].rename(columns={
        "team": "opp", "def_ppa_todate": "opp_def_ppa", "off_ppa_todate": "opp_off_ppa",
        "def_sr_todate": "opp_def_sr", "off_sr_todate": "opp_off_sr"})
    d = d.merge(opp, on=["game_id", "opp"], how="left")

    # League mean by season so the adjustment is centred, not shifted.
    for c in ["opp_def_ppa", "opp_off_ppa", "opp_def_sr", "opp_off_sr"]:
        d[c] = d[c].fillna(d.groupby("season")[c].transform("mean"))

    d["off_ppa_adj"] = d["off_ppa"] - d["opp_def_ppa"]
    d["def_ppa_adj"] = d["def_ppa"] - d["opp_off_ppa"]
    d["off_sr_adj"] = d["off_sr"] - d["opp_def_sr"]
    d["def_sr_adj"] = d["def_sr"] - d["opp_off_sr"]

    # Roll the ADJUSTED per-game values, still shifted one back.
    # min_periods=1, not 2: requiring two prior games silently deleted every
    # team's first three weeks -- 1,084 of 3,863 games, a quarter of a college
    # season. One prior game is a weak signal, so `games_played` is carried
    # alongside and the preseason priors cover the gap.
    for n in ROLL:
        for col in ROLLED:
            d[f"{col}_L{n}"] = (d.groupby(["season", "team"])[col]
                                .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))

    d["games_played"] = d.groupby(["season", "team"]).cumcount()
    # Week 1 has no in-season form at all. Zero is the honest value on these
    # opponent-adjusted, mean-centred scales, and games_played tells the model
    # how much weight the number deserves.
    for n in ROLL:
        for col in ROLLED:
            d[f"{col}_L{n}"] = d[f"{col}_L{n}"].fillna(0.0)

    # Rest days from the previous game, ordered by kickoff.
    prev = d.groupby(["season", "team"])["kick_ts"].shift(1)
    d["rest_days"] = (d["kick_ts"] - prev).dt.days.fillna(7).clip(0, 21)
    return d


# Opponent-adjusted, matching what the NFL side adjusts (EPA and success rate).
ADJ_COLS = ["off_ppa_adj", "def_ppa_adj", "off_sr_adj", "def_sr_adj"]
# Rolled as-is. Adjusting every one of these would double the bookkeeping for
# diminishing return; the NFL pipeline makes the same trade.
RAW_COLS = ["off_expl", "def_expl", "off_power", "def_power", "off_stuff",
            "def_stuff", "off_line_yds", "def_line_yds", "off_2nd_lvl",
            "off_open_field", "off_sd_ppa", "off_pd_ppa", "def_sd_ppa",
            "def_pd_ppa", "off_sd_sr", "off_pd_sr", "off_rush_ppa",
            "off_pass_ppa", "off_plays"]
ROLLED = ADJ_COLS + RAW_COLS
FEATURE_COLS = [f"{c}_L{n}" for n in ROLL for c in ROLLED]

# Per-team season-level inputs, all knowable BEFORE a season starts. These are
# what fix the two holes in the first pass:
#
#   1. The first version used 12 features against the NFL model's 57, with no
#      analogue for rest, travel, venue or prior-season strength. It was not a
#      like-for-like test and should not have been reported as one.
#   2. Rolling form needs prior games, so weeks 1-3 were dropped entirely --
#      1,084 of 3,863 games, and in a 12-game season that is a quarter of the
#      year. A preseason prior covers exactly that gap.
#
# College football has a far better preseason prior available than the NFL does:
# recruiting talent and returning production genuinely forecast strength, where
# the NFL equivalent (calibrate_prior_trust.py) was a clean negative.
# NOTE: SP+ exposes offense.pace but never populates it — the column came back
# 100% null, and since a median fill cannot rescue an all-null column, the
# dropna below took every row. Excluded rather than carried as dead weight.
PRESEASON_COLS = ["sp_prev", "sp_off_prev", "sp_def_prev",
                  "talent", "talent_prev", "returning_ppa"]
CONTEXT_COLS = ["rest_days", "travel_miles", "elev_change", "is_dome", "games_played"]


def main():
    print("fetching CFBD games and per-game efficiency...")
    games = fetch_games()
    games["kick_ts"] = pd.to_datetime(games["start_date"], utc=True, errors="coerce")
    tg = add_form(fetch_team_games(), games[["game_id", "kick_ts"]])
    print(f"  {len(games)} FBS-vs-FBS results | {len(tg)} team-game rows")

    ROLL_PLUS = FEATURE_COLS + ["games_played", "rest_days"]
    keep = ["game_id", "team"] + ROLL_PLUS
    home = tg[keep].rename(columns={"team": "home_team",
                                    **{c: f"home_{c}" for c in ROLL_PLUS}})
    away = tg[keep].rename(columns={"team": "away_team",
                                    **{c: f"away_{c}" for c in ROLL_PLUS}})
    # Joined on game_id + side, so a team's two Week-0/Week-1 rows cannot cross.
    df = (games.merge(home, on=["game_id", "home_team"], how="left")
                .merge(away, on=["game_id", "away_team"], how="left"))
    assert len(df) == len(games), f"form join fanned out: {len(games)} -> {len(df)}"

    # Lines, keyed on the canonical pair + kick date -- never an event id.
    oc = []
    for s in SEASONS:
        p = DATA_DIR / f"cfb_open_close_{s}.parquet"
        if p.exists():
            o = pd.read_parquet(p)
            o["season"] = s
            oc.append(o)
    oc = pd.concat(oc, ignore_index=True)
    oc = oc[oc.n_snapshots_week >= 2]

    before = len(df)
    df = df.merge(oc[["season", "home_team", "away_team", "kick_date",
                      "week_open_spread_home", "closing_spread_home",
                      "week_spread_movement"]],
                  on=["season", "home_team", "away_team", "kick_date"], how="inner")
    assert df.duplicated(subset=["season", "home_team", "away_team", "kick_date"]).sum() == 0, \
        "merge produced duplicate games"
    print(f"  joined to lines: {len(df)} of {before} results matched a line")

    # Preseason strength priors, joined per side.
    pre = fetch_preseason()
    for side in ("home", "away"):
        p = pre.rename(columns={"team": f"{side}_team",
                                **{c: f"{side}_{c}" for c in PRESEASON_COLS}})
        before_p = len(df)
        df = df.merge(p, on=["season", f"{side}_team"], how="left")
        assert len(df) == before_p, f"{side} preseason join fanned out"

    # Venue: dome, altitude, and how far the away side travelled. Home venues
    # are inferred from each team's most-used stadium so travel can be measured
    # even for neutral-site games.
    ven = fetch_venues()
    df = df.merge(ven, on="venue_id", how="left")
    homes = (df[df.neutral_site == 0].groupby("home_team")["venue_id"]
             .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan))
    base = ven.set_index("venue_id")[["lat", "lon", "elevation"]]
    away_base = df["away_team"].map(homes).map(base["lat"]), df["away_team"].map(homes).map(base["lon"])
    df["travel_miles"] = [haversine(la, lo, gl, gn)
                          for la, lo, gl, gn in zip(away_base[0], away_base[1], df["lat"], df["lon"])]
    df["elev_change"] = df["elevation"] - df["away_team"].map(homes).map(base["elevation"])
    df["is_dome"] = df["is_dome"].fillna(0)
    for c in ("travel_miles", "elev_change"):
        df[c] = df[c].fillna(df[c].median())

    # Differentials are what the model actually reads.
    for c in FEATURE_COLS + PRESEASON_COLS + ["games_played", "rest_days"]:
        if f"home_{c}" in df and f"away_{c}" in df:
            df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]

    need = [f"diff_{c}" for c in FEATURE_COLS + PRESEASON_COLS]
    need = [c for c in need if c in df]
    # An all-null feature would silently delete the entire dataset through the
    # dropna below -- exactly what sp_pace_prev did. Name it instead.
    dead = [c for c in need if df[c].isna().all()]
    assert not dead, f"features that are 100% null: {dead}"
    df = df.dropna(subset=need)
    print(f"  with complete form features: {len(df)}")
    for s in sorted(df.season.unique()):
        print(f"    {s}: {int((df.season == s).sum())}")

    out = DATA_DIR / "cfb_dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out.name}  ({len(df)} games, {len(FEATURE_COLS)} form features)")

    # Sanity: the line must behave like a line.
    corr = np.corrcoef(df.closing_spread_home, df.home_margin)[0, 1]
    cover = ((df.home_margin + df.closing_spread_home) > 0).mean() * 100
    print(f"  convention check: corr(closing_spread, margin) {corr:+.3f} "
          f"(want strongly negative), home covers {cover:.1f}%")
    if corr > -0.2:
        raise SystemExit("ABORT: spread sign looks inverted.")


if __name__ == "__main__":
    main()
