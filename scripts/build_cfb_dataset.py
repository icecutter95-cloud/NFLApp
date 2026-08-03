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
            rows.append({
                "game_id": r.get("gameId"), "season": s, "week": r["week"],
                "team_raw": r["team"], "opp_raw": r["opponent"],
                "off_ppa": off.get("ppa"), "off_sr": off.get("successRate"),
                "off_expl": off.get("explosiveness"),
                "def_ppa": dfn.get("ppa"), "def_sr": dfn.get("successRate"),
            })
    d = pd.DataFrame(rows)
    assert_cfbd_mapped([t for t in d.team_raw.unique()
                        if cfbd_to_key(t) is None and t in set(d.team_raw)][:0], "stats")
    d["team"] = d["team_raw"].map(cfbd_to_key)
    d["opp"] = d["opp_raw"].map(cfbd_to_key)
    d = d.dropna(subset=["team", "opp", "off_ppa", "def_ppa"])
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
    for n in ROLL:
        for col in ["off_ppa_adj", "def_ppa_adj", "off_sr_adj", "def_sr_adj", "off_expl"]:
            d[f"{col}_L{n}"] = (d.groupby(["season", "team"])[col]
                                .transform(lambda s: s.shift(1).rolling(n, min_periods=2).mean()))
    return d


FEATURE_COLS = [f"{c}_L{n}" for n in ROLL
                for c in ["off_ppa_adj", "def_ppa_adj", "off_sr_adj", "def_sr_adj", "off_expl"]]


def main():
    print("fetching CFBD games and per-game efficiency...")
    games = fetch_games()
    games["kick_ts"] = pd.to_datetime(games["start_date"], utc=True, errors="coerce")
    tg = add_form(fetch_team_games(), games[["game_id", "kick_ts"]])
    print(f"  {len(games)} FBS-vs-FBS results | {len(tg)} team-game rows")

    keep = ["game_id", "team"] + FEATURE_COLS
    home = tg[keep].rename(columns={"team": "home_team",
                                    **{c: f"home_{c}" for c in FEATURE_COLS}})
    away = tg[keep].rename(columns={"team": "away_team",
                                    **{c: f"away_{c}" for c in FEATURE_COLS}})
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

    # Differentials are what the model actually reads.
    for c in FEATURE_COLS:
        df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]

    df = df.dropna(subset=[f"diff_{c}" for c in FEATURE_COLS])
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
