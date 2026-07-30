"""
Phase 1 — Step 3: Merge team metrics + schedule lines into a unified training dataset.

Primary line source: nfl_data_py schedules (spread_line, total_line columns from nflverse).
Fallback: PFR scraped lines (pfr_lines_all.parquet).

Output: data/historical_dataset.parquet
  - One row per game
  - All team efficiency features for home + away (with _home / _away suffix)
  - Target columns: home_margin, combined_score
  - Closing lines: closing_spread_home, closing_total

Usage:
    python build_dataset.py
"""

import numpy as np
import pandas as pd
import nfl_data_py as nfl

from config import DATA_DIR, ALL_HISTORICAL_SEASONS, HFA_OVERRIDES, HFA_DEFAULT, DOME_TEAMS


# ---------------------------------------------------------------------------
# Schedule loading
# ---------------------------------------------------------------------------

def load_schedules(seasons: list) -> pd.DataFrame:
    print("Loading schedules via nfl_data_py...")
    sched = nfl.import_schedules(seasons)

    keep = [
        "season", "week", "game_id", "home_team", "away_team", "game_type",
        "gameday", "gametime", "home_score", "away_score",
        "spread_line", "total_line", "div_game", "roof",
    ]
    cols = [c for c in keep if c in sched.columns]
    sched = sched[cols].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"], errors="coerce")
    print(f"  {len(sched)} games loaded (seasons {min(seasons)}–{max(seasons)})")
    return sched


# ---------------------------------------------------------------------------
# Rest days / bye week features
# ---------------------------------------------------------------------------

def add_rest_features(sched: pd.DataFrame) -> pd.DataFrame:
    """Compute rest_days and had_bye for home and away teams."""
    home = sched[["season", "week", "home_team", "gameday"]].rename(
        columns={"home_team": "team", "gameday": "game_date"})
    away = sched[["season", "week", "away_team", "gameday"]].rename(
        columns={"away_team": "team", "gameday": "game_date"})

    all_games = (pd.concat([home, away])
                 .sort_values(["season", "team", "week"])
                 .reset_index(drop=True))
    all_games["prev_date"] = all_games.groupby(["season", "team"])["game_date"].shift(1)
    all_games["rest_days"] = (all_games["game_date"] - all_games["prev_date"]).dt.days.fillna(7)
    all_games["had_bye"] = (all_games["rest_days"] >= 13).astype(int)

    home_rest = (all_games.rename(columns={"team": "home_team", "rest_days": "rest_days_home",
                                            "had_bye": "had_bye_home"})
                 [["season", "week", "home_team", "rest_days_home", "had_bye_home"]])
    away_rest = (all_games.rename(columns={"team": "away_team", "rest_days": "rest_days_away",
                                            "had_bye": "had_bye_away"})
                 [["season", "week", "away_team", "rest_days_away", "had_bye_away"]])

    sched = (sched
             .merge(home_rest, on=["season", "week", "home_team"], how="left")
             .merge(away_rest, on=["season", "week", "away_team"], how="left"))
    sched["rest_diff"] = sched["rest_days_home"] - sched["rest_days_away"]
    sched["is_short_week_home"] = (sched["rest_days_home"] <= 5).astype(int)
    sched["is_short_week_away"] = (sched["rest_days_away"] <= 5).astype(int)
    return sched


# ---------------------------------------------------------------------------
# Stadium / game context features
# ---------------------------------------------------------------------------

def add_game_context(sched: pd.DataFrame) -> pd.DataFrame:
    sched["home_field_advantage"] = sched["home_team"].map(
        lambda t: HFA_OVERRIDES.get(t, HFA_DEFAULT))
    sched["is_dome"] = sched["home_team"].isin(DOME_TEAMS).astype(int)
    sched["is_divisional"] = sched.get("div_game", pd.Series(0, index=sched.index)).fillna(0).astype(int)
    sched["week_number"] = sched["week"].astype(int)
    sched["is_playoffs"] = (sched["game_type"] != "REG").astype(int) if "game_type" in sched.columns else 0

    # Neutral-site HFA reduction (London, Mexico City, Super Bowl)
    if "game_type" in sched.columns:
        neutral = sched["game_type"].isin(["SB", "CON"])
        sched.loc[neutral, "home_field_advantage"] -= 0.5

    return sched


# ---------------------------------------------------------------------------
# Team metrics merge
# ---------------------------------------------------------------------------

def merge_team_metrics(sched: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in metrics.columns if c not in ("team", "season", "week")]

    home_m = (metrics.rename(columns={c: f"{c}_home" for c in metric_cols})
              .rename(columns={"team": "home_team"}))
    away_m = (metrics.rename(columns={c: f"{c}_away" for c in metric_cols})
              .rename(columns={"team": "away_team"}))

    df = (sched
          .merge(home_m, on=["season", "week", "home_team"], how="left")
          .merge(away_m, on=["season", "week", "away_team"], how="left"))
    return df


# ---------------------------------------------------------------------------
# Lines (primary = nfl_data_py schedule, fallback = PFR)
# ---------------------------------------------------------------------------

def add_closing_lines(df: pd.DataFrame) -> pd.DataFrame:
    pfr_path = DATA_DIR / "pfr_lines_all.parquet"
    if pfr_path.exists():
        pfr = pd.read_parquet(pfr_path)[["season", "week", "home_team", "away_team",
                                          "pfr_spread_home", "pfr_total"]]
        df = df.merge(pfr, on=["season", "week", "home_team", "away_team"], how="left")
    else:
        print("  WARNING: pfr_lines_all.parquet not found — run scrape_pfr_lines.py first")
        df["pfr_spread_home"] = np.nan
        df["pfr_total"] = np.nan

    # Primary source from nfl_data_py schedule, fallback to PFR
    raw_spread = df["spread_line"].fillna(df["pfr_spread_home"]) if "spread_line" in df.columns else df["pfr_spread_home"]
    df["closing_total"] = df["total_line"].fillna(df["pfr_total"]) if "total_line" in df.columns else df["pfr_total"]

    # ---- SIGN NORMALISATION (critical) ----------------------------------
    # nflverse `spread_line` uses POSITIVE = home favored, which is the
    # OPPOSITE of the sportsbook standard. The Odds API (our live source,
    # line_history.spread_home -> score_week.py's dk_spread) uses the
    # standard convention: NEGATIVE = home favored. Normalise to the
    # standard here so training data and live scoring agree, and so
    # add_targets()'s `home_margin + closing_spread_home` is correct.
    #
    # Getting this backwards silently corrupts the model TARGET rather than
    # a feature, which is why it survived the earlier circular-dependency
    # cleanup: with the wrong sign the target becomes
    #     wrong = home_margin + spread_line = correct + 2*spread_line
    # so the model learns mostly to reconstruct the market's own line, and
    # evaluating sign(pred)==sign(actual) with the same 2*spread_line term
    # on both sides inflates the apparent win rate.
    #
    # Worked example (MIA home vs NE, 2019 wk2, MIA lost 0-43):
    #   spread_line = -18  ->  NE favored by 18
    #   correct surplus = -43 - (-18) = -25  (missed the number by 25)
    #   wrong   surplus = -43 + (-18) = -61  (impossible: they only lost by 43)
    df["closing_spread_home"] = -raw_spread
    # Keep the raw source value for provenance / debugging.
    df["nflverse_spread_line_raw"] = raw_spread
    return df


# ---------------------------------------------------------------------------
# Travel / body-clock features
# ---------------------------------------------------------------------------

# Standard-time UTC offset of each team's home market. Used to measure how many
# time zones the visiting team crossed -- a well-documented effect (a Pacific
# team playing an early Eastern kickoff is on its body clock's early morning)
# that our previous rest_diff / is_short_week flags did not capture at all.
_TEAM_TZ = {
    **{t: -5 for t in ["BUF","MIA","NE","NYJ","NYG","BAL","CIN","CLE","PIT",
                       "JAX","IND","DET","WAS","PHI","CAR","TB","ATL"]},
    **{t: -6 for t in ["CHI","GB","MIN","NO","DAL","HOU","KC","TEN"]},
    **{t: -7 for t in ["DEN","ARI"]},
    **{t: -8 for t in ["SEA","SF","LA","LAC","LV"]},
}


def _haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def add_travel_features(df: pd.DataFrame) -> pd.DataFrame:
    """Distance flown by the visiting team and time zones crossed."""
    try:
        from fetch_historical_weather import STADIUM_COORDS, get_stadium_coords
    except Exception:
        print("  WARNING: stadium coords unavailable — skipping travel features")
        df["away_travel_miles"] = 0.0
        df["tz_delta"] = 0.0
        df["abs_tz_delta"] = 0.0
        return df

    miles, tzd = [], []
    for _, r in df.iterrows():
        home, away, season = r["home_team"], r["away_team"], int(r["season"])
        hc = get_stadium_coords(home, season)
        ac = get_stadium_coords(away, season)
        if hc and ac:
            miles.append(_haversine_miles(ac[0], ac[1], hc[0], hc[1]))
        else:
            miles.append(0.0)
        # positive = visitor moved east (loses hours off the body clock)
        tzd.append(float(_TEAM_TZ.get(home, -5) - _TEAM_TZ.get(away, -5)))

    df["away_travel_miles"] = miles
    df["tz_delta"] = tzd
    df["abs_tz_delta"] = np.abs(tzd)
    print(f"  Travel: mean visitor distance {np.mean(miles):.0f} mi, "
          f"{int((np.abs(tzd) >= 2).sum())} games crossing 2+ time zones")
    return df


# ---------------------------------------------------------------------------
# Injury features
# ---------------------------------------------------------------------------

_OFF_POS = {"QB", "RB", "FB", "WR", "TE", "T", "G", "C", "OL", "OT", "OG"}
_DEF_POS = {"DE", "DT", "NT", "LB", "ILB", "OLB", "MLB", "CB", "S", "FS", "SS", "DB", "DL"}
_UNAVAILABLE = {"Out", "Doubtful"}


def _load_injuries(seasons: list) -> pd.DataFrame:
    """Weekly injury reports, cached locally.

    NOT leakage: the injury report for week W is published before week W's
    games, so it is legitimate pre-game information -- unlike the rolling
    performance metrics, which must use only weeks strictly before W.
    """
    cache = DATA_DIR / "injuries_all.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    try:
        inj = nfl.import_injuries(seasons)
        inj.to_parquet(cache, index=False)
        print(f"  Cached {len(inj)} injury rows -> {cache.name}")
        return inj
    except Exception as exc:
        print(f"  WARNING: could not load injuries: {exc}")
        return pd.DataFrame()


def _injury_team_week(seasons: list) -> pd.DataFrame:
    inj = _load_injuries(seasons)
    if inj.empty:
        return pd.DataFrame()

    inj = inj[inj.get("game_type", "REG") == "REG"].copy()
    inj["status"] = inj["report_status"].fillna("")
    inj["pos"] = inj["position"].fillna("")

    unavailable = inj["status"].isin(_UNAVAILABLE)
    inj["is_out"] = unavailable.astype(int)
    inj["is_q"] = (inj["status"] == "Questionable").astype(int)
    inj["qb_out"] = (unavailable & (inj["pos"] == "QB")).astype(int)
    inj["off_out"] = (unavailable & inj["pos"].isin(_OFF_POS)).astype(int)
    inj["def_out"] = (unavailable & inj["pos"].isin(_DEF_POS)).astype(int)

    agg = (inj.groupby(["season", "week", "team"], as_index=False)
           .agg(inj_qb_out=("qb_out", "max"),
                inj_out_off=("off_out", "sum"),
                inj_out_def=("def_out", "sum"),
                inj_out_total=("is_out", "sum"),
                inj_questionable=("is_q", "sum")))
    return agg


def add_injury_features(df: pd.DataFrame, seasons: list) -> pd.DataFrame:
    agg = _injury_team_week(seasons)
    cols = ["inj_qb_out", "inj_out_off", "inj_out_def", "inj_out_total", "inj_questionable"]
    if agg.empty:
        for side in ("home", "away"):
            for c in cols:
                df[f"{c}_{side}"] = 0.0
        return df

    for side, key in (("home", "home_team"), ("away", "away_team")):
        renamed = agg.rename(columns={"team": key, **{c: f"{c}_{side}" for c in cols}})
        df = df.merge(renamed, on=["season", "week", key], how="left")

    for side in ("home", "away"):
        for c in cols:
            df[f"{c}_{side}"] = df[f"{c}_{side}"].fillna(0.0)

    qb_games = int((df["inj_qb_out_home"] + df["inj_qb_out_away"]).gt(0).sum())
    print(f"  Injuries: {qb_games} games with a QB listed out/doubtful; "
          f"mean players out {df['inj_out_total_home'].mean():.1f}/team")
    return df


# ---------------------------------------------------------------------------
# Weather features
# ---------------------------------------------------------------------------

def add_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Merge historical weather into the game dataframe.

    1. Try to load data/historical_weather.parquet and join on game_id.
       Only the weather measurement columns are merged — is_dome is already
       set by add_game_context() and is not duplicated here.
    2. For dome games (home team in DOME_TEAMS), overwrite weather with dome
       defaults (72 °F, 0 wind, 0 precip) regardless of API data.
    3. Fill remaining NaN (outdoor games without weather data) with
       conservative outdoor defaults so features are never NaN.

    Historical note: LA (Rams; nfl_data_py uses "LA" not "LAR") and LAC both
    played outdoor 2018-2019 (LA Coliseum / StubHub Center), then moved into
    SoFi Stadium (dome) in 2020. Apply dome defaults only from 2020 onward
    for these two teams.
    """
    from config import DOME_TEAMS

    all_dome = DOME_TEAMS  # config now includes full dome set

    # Teams that became domes mid-period: dome defaults apply only from season X onward
    _DOME_FROM_SEASON = {"LA": 2020, "LAC": 2020}

    weather_path = DATA_DIR / "historical_weather.parquet"
    if weather_path.exists():
        wx = pd.read_parquet(weather_path)[
            # Exclude is_dome — it's already in df from add_game_context
            ["game_id", "temp_fahrenheit", "wind_speed_mph", "precipitation_prob"]
        ]
        df = df.merge(wx, on="game_id", how="left")
        n_with_data = wx["temp_fahrenheit"].notna().sum()
        print(f"  Merged weather: {n_with_data}/{len(wx)} games have data")
    else:
        print("  WARNING: historical_weather.parquet not found — run fetch_historical_weather.py first")
        print("           Weather features will be NaN (totals model will be unreliable)")
        for col in ["temp_fahrenheit", "wind_speed_mph", "precipitation_prob"]:
            df[col] = np.nan

    # Ensure is_dome is set for the full dome set (config may be narrower than reality)
    dome_mask = df["home_team"].isin(all_dome)
    # Exclude teams that weren't dome yet in early seasons
    for team, from_season in _DOME_FROM_SEASON.items():
        dome_mask = dome_mask & ~((df["home_team"] == team) & (df["season"] < from_season))
    df.loc[dome_mask, "is_dome"] = 1

    df["is_dome"] = df["is_dome"].fillna(0).astype(int)

    # Dome teams always get controlled-environment defaults
    dome_mask = df["is_dome"] == 1
    df.loc[dome_mask, "temp_fahrenheit"]    = 72.0
    df.loc[dome_mask, "wind_speed_mph"]     = 0.0
    df.loc[dome_mask, "precipitation_prob"] = 0.0

    # For outdoor games still missing weather, fill with reasonable outdoor defaults
    outdoor_mask = df["is_dome"] == 0
    df.loc[outdoor_mask & df["temp_fahrenheit"].isna(),    "temp_fahrenheit"]    = 55.0
    df.loc[outdoor_mask & df["wind_speed_mph"].isna(),     "wind_speed_mph"]     = 8.0
    df.loc[outdoor_mask & df["precipitation_prob"].isna(), "precipitation_prob"] = 0.1

    wx_coverage = df["temp_fahrenheit"].notna().sum()
    print(f"  Weather coverage after fill: {wx_coverage}/{len(df)} ({wx_coverage/len(df)*100:.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Sanity guards
# ---------------------------------------------------------------------------

def assert_line_conventions(df: pd.DataFrame) -> None:
    """Fail loudly if the spread sign convention is inverted.

    A correctly-signed home spread (negative = home favored) must correlate
    NEGATIVELY with realised home margin: the bigger the home favorite, the
    more negative the number, the larger the expected margin. And because a
    closing line is a well-calibrated market price, cover outcomes must land
    near 50/50.

    Both checks failed before the sign fix (corr was +0.451, covers 56/42),
    and the bad sign corrupted the TARGET rather than a feature -- invisible
    to feature-level review. Guard it here so it can never regress silently.
    """
    d = df.dropna(subset=["closing_spread_home", "home_margin"])
    if len(d) < 100:
        return

    corr = d["closing_spread_home"].corr(d["home_margin"])
    if corr > -0.2:
        raise AssertionError(
            f"closing_spread_home vs home_margin corr = {corr:+.3f}; expected strongly "
            f"NEGATIVE. The spread sign convention looks inverted -- see the sign "
            f"normalisation note in add_closing_lines()."
        )

    surplus = d["home_margin"] + d["closing_spread_home"]
    home_cover_pct = (surplus > 0.01).mean() * 100
    if not (44.0 <= home_cover_pct <= 56.0):
        raise AssertionError(
            f"home cover rate = {home_cover_pct:.1f}%; expected ~50% against a "
            f"calibrated closing line. Check the spread sign / line source."
        )

    print(f"  Line sanity OK: corr(spread, margin)={corr:+.3f}, "
          f"home covers {home_cover_pct:.1f}%, mean surplus {surplus.mean():+.2f}")


# ---------------------------------------------------------------------------
# Target variables
# ---------------------------------------------------------------------------

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["home_margin"] = df["home_score"] - df["away_score"]
    df["combined_score"] = df["home_score"] + df["away_score"]

    # Model targets (what we train on):
    #   home_cover_surplus > 0 → home covered the spread
    #   home_cover_surplus < 0 → away covered the spread
    #   ou_surplus > 0 → over hit
    #   ou_surplus < 0 → under hit
    df["home_cover_surplus"] = df["home_margin"] + df["closing_spread_home"]
    df["ou_surplus"] = df["combined_score"] - df["closing_total"]

    # Human-readable labels (for reference)
    df["spread_result"] = np.where(
        df["home_cover_surplus"] > 0.01, "home_covered",
        np.where(df["home_cover_surplus"] < -0.01, "away_covered", "push")
    )
    df["total_result"] = np.where(
        df["ou_surplus"] > 0.01, "over",
        np.where(df["ou_surplus"] < -0.01, "under", "push")
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load schedules
    sched = load_schedules(ALL_HISTORICAL_SEASONS)
    sched = add_rest_features(sched)
    sched = add_game_context(sched)

    # 2. Load team metrics (must run compute_metrics.py first)
    metrics_path = DATA_DIR / "team_metrics_all.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError("Run compute_metrics.py first to generate team_metrics_all.parquet")
    print("Loading team metrics...")
    metrics = pd.read_parquet(metrics_path)

    # 3. Merge
    print("Merging schedules + metrics...")
    df = merge_team_metrics(sched, metrics)

    # 4. Lines
    print("Adding closing lines...")
    df = add_closing_lines(df)

    # 5. Keep market lines as reference columns (NOT in features — see config.py)
    df["market_spread_home"] = df["closing_spread_home"]
    df["market_total"]       = df["closing_total"]

    # 5b. Weather features
    print("Adding weather features...")
    df = add_weather(df)

    # 5c. Travel + injury features (Tier 2)
    print("Adding travel features...")
    df = add_travel_features(df)
    print("Adding injury features...")
    df = add_injury_features(df, ALL_HISTORICAL_SEASONS)

    # 6. Targets
    df = add_targets(df)

    # Runs after add_targets because it needs home_margin.
    print("Checking line sign conventions...")
    assert_line_conventions(df)

    # 7. Drop rows without scores or lines (can't train/evaluate on them)
    before = len(df)
    df = df.dropna(subset=["home_score", "closing_spread_home"])
    print(f"Dropped {before - len(df)} rows with missing scores or lines ({len(df)} remain)")

    # 7. Regular season only for training (keep playoffs separate)
    regular = df[df["is_playoffs"] == 0].copy()
    print(f"Regular season games: {len(regular)}")

    # 8. Save
    out = DATA_DIR / "historical_dataset.parquet"
    df.to_parquet(out, index=False)
    print(f"\nFull dataset ({len(df)} games) -> {out}")

    out_reg = DATA_DIR / "historical_dataset_regular.parquet"
    regular.to_parquet(out_reg, index=False)
    print(f"Regular season ({len(regular)} games) -> {out_reg}")

    # Quick coverage summary
    print("\nLine coverage by season:")
    cov = (df.groupby("season")["closing_spread_home"]
           .apply(lambda x: f"{x.notna().sum()}/{len(x)}")
           .reset_index()
           .rename(columns={"closing_spread_home": "spread_coverage"}))
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
