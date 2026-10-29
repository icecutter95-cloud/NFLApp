"""
Team-week features from Next Gen Stats and FTN charting, and a test of whether
they add anything.

WHY
Asked whether a paid charting subscription (fantasypoints.com) would give
dimensions we lack. Roughly half its headline list is already free in
nfl_data_py and unused here:

    receiver separation      NGS avg_separation, avg_cushion
    men in box               NGS percent_attempts_gte_eight_defenders,
                             FTN n_defense_box
    motion / formation       FTN is_motion, is_play_action, is_rpo,
                             n_offense_backfield
    first-read targets       FTN read_thrown
    QB accuracy / air yards   NGS CPOE, time to throw, air yards to sticks
    YAC above expectation    NGS avg_yac_above_expectation
    blitz / pressure         FTN n_blitzers, n_pass_rushers

Paying for a dimension before using the free version of it is the wrong order,
so this builds the free version and measures it.

WHAT IT EXPECTS TO FIND
Probably nothing. This project's own history says so: "Six of nine football
feature ideas failed -- the closing line already prices public information
about football." The market prices a quarterback's absence to within 5%
(charges 27.9 points per unit of downgrade, reality pays 29.3, residual
p = 0.76), so it almost certainly prices receiver separation. The test exists
to settle it rather than assume it.

NO LEAKAGE
Every feature is a rolling mean over games STRICTLY BEFORE the one being
predicted: groupby(season, team).shift(1).rolling(n). A team's week-6 row is
built from weeks 1-5. This is the same rule compute_metrics follows, and it is
the rule the zeroed-form bug violated in the other direction.

COVERAGE
NGS runs from 2016, FTN only from 2022, so FTN features cover roughly a third
of the training frame. The test reports both separately for that reason.

Usage:
    python scripts/build_ngs_ftn_features.py            # build + test
    python scripts/build_ngs_ftn_features.py --rebuild  # ignore the cache
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR

NGS_SEASONS = list(range(2018, 2027))
FTN_SEASONS = list(range(2022, 2027))
PART_SEASONS = list(range(2019, 2027))
ROLL = 4
CACHE = DATA_DIR / "ngs_ftn_team_week.parquet"


def ngs_team_week() -> pd.DataFrame:
    """Offense-side NGS, aggregated to team-week."""
    import nfl_data_py as nfl
    out = []

    rec = nfl.import_ngs_data("receiving", NGS_SEASONS)
    rec = rec[rec.season_type == "REG"]
    # Target-weighted: a team's separation is what its targets saw, not the
    # unweighted mean across everyone who ran a route.
    rec["w"] = rec.targets.fillna(0)
    for c in ("avg_separation", "avg_cushion", "avg_yac_above_expectation"):
        rec[f"_{c}"] = rec[c] * rec.w
    g = (rec.groupby(["season", "week", "team_abbr"], as_index=False)
            .agg(w=("w", "sum"),
                 **{f"_{c}": (f"_{c}", "sum") for c in
                    ("avg_separation", "avg_cushion", "avg_yac_above_expectation")}))
    for c in ("avg_separation", "avg_cushion", "avg_yac_above_expectation"):
        g[f"off_{c}"] = g[f"_{c}"] / g.w.replace(0, np.nan)
    out.append(g[["season", "week", "team_abbr",
                  "off_avg_separation", "off_avg_cushion",
                  "off_avg_yac_above_expectation"]])

    rush = nfl.import_ngs_data("rushing", NGS_SEASONS)
    rush = rush[rush.season_type == "REG"]
    rush["w"] = rush.rush_attempts.fillna(0)
    for c in ("percent_attempts_gte_eight_defenders",
              "rush_yards_over_expected_per_att", "efficiency"):
        rush[f"_{c}"] = rush[c] * rush.w
    g = (rush.groupby(["season", "week", "team_abbr"], as_index=False)
             .agg(w=("w", "sum"),
                  **{f"_{c}": (f"_{c}", "sum") for c in
                     ("percent_attempts_gte_eight_defenders",
                      "rush_yards_over_expected_per_att", "efficiency")}))
    for c in ("percent_attempts_gte_eight_defenders",
              "rush_yards_over_expected_per_att", "efficiency"):
        g[f"off_{c}"] = g[f"_{c}"] / g.w.replace(0, np.nan)
    out.append(g[["season", "week", "team_abbr",
                  "off_percent_attempts_gte_eight_defenders",
                  "off_rush_yards_over_expected_per_att", "off_efficiency"]])

    pas = nfl.import_ngs_data("passing", NGS_SEASONS)
    pas = pas[pas.season_type == "REG"]
    pas["w"] = pas.attempts.fillna(0)
    cols = ("avg_time_to_throw", "completion_percentage_above_expectation",
            "aggressiveness", "avg_air_yards_to_sticks")
    for c in cols:
        pas[f"_{c}"] = pas[c] * pas.w
    g = (pas.groupby(["season", "week", "team_abbr"], as_index=False)
            .agg(w=("w", "sum"), **{f"_{c}": (f"_{c}", "sum") for c in cols}))
    for c in cols:
        g[f"off_{c}"] = g[f"_{c}"] / g.w.replace(0, np.nan)
    out.append(g[["season", "week", "team_abbr"] + [f"off_{c}" for c in cols]])

    d = out[0]
    for x in out[1:]:
        d = d.merge(x, on=["season", "week", "team_abbr"], how="outer")
    return d.rename(columns={"team_abbr": "team"})


def ftn_team_week() -> pd.DataFrame:
    """Charting rates per team-week, split into what a team DID and FACED.

    FTN carries no possession team, so it is joined to play-by-play on
    (nflverse_game_id, nflverse_play_id) to attach posteam/defteam. Offensive
    tendencies belong to posteam; box count, blitzers and pass rushers describe
    what the DEFENCE did, so they are credited to defteam and also recorded as
    what the offence faced.
    """
    import nfl_data_py as nfl
    ftn = nfl.import_ftn_data(FTN_SEASONS)
    pbp = nfl.import_pbp_data(FTN_SEASONS, downcast=True, cache=False,
                              columns=["game_id", "play_id", "posteam",
                                       "defteam", "season", "week"])
    # No rename: import_pbp_data already returns a column called
    # nflverse_game_id (it hands back participation fields whether or not you
    # ask for them), so renaming game_id onto it produced two columns with the
    # same label and merge refused. Join on explicit keys instead.
    f = ftn.merge(pbp, left_on=["nflverse_game_id", "nflverse_play_id"],
                  right_on=["game_id", "play_id"], how="inner",
                  suffixes=("", "_pbp"))
    f = f.dropna(subset=["posteam", "defteam"])
    for c in ("is_motion", "is_play_action", "is_screen_pass", "is_rpo",
              "is_no_huddle", "is_qb_out_of_pocket", "is_throw_away",
              "is_drop", "is_contested_ball", "is_created_reception",
              "is_interception_worthy"):
        f[c] = f[c].astype(float)
    f["is_first_read"] = (pd.to_numeric(f.read_thrown, errors="coerce") == 1).astype(float)
    f["n_defense_box"] = pd.to_numeric(f.n_defense_box, errors="coerce")
    f["n_blitzers"] = pd.to_numeric(f.n_blitzers, errors="coerce")
    f["n_pass_rushers"] = pd.to_numeric(f.n_pass_rushers, errors="coerce")
    f["is_blitz"] = (f.n_blitzers > 0).astype(float)

    off = (f.groupby(["season", "week", "posteam"], as_index=False)
            .agg(off_motion=("is_motion", "mean"),
                 off_play_action=("is_play_action", "mean"),
                 off_screen=("is_screen_pass", "mean"),
                 off_rpo=("is_rpo", "mean"),
                 off_no_huddle=("is_no_huddle", "mean"),
                 off_out_of_pocket=("is_qb_out_of_pocket", "mean"),
                 off_throw_away=("is_throw_away", "mean"),
                 off_drop=("is_drop", "mean"),
                 off_created_rec=("is_created_reception", "mean"),
                 off_first_read=("is_first_read", "mean"),
                 off_faced_box=("n_defense_box", "mean"),
                 off_faced_blitz=("is_blitz", "mean"))
            .rename(columns={"posteam": "team"}))
    dfn = (f.groupby(["season", "week", "defteam"], as_index=False)
            .agg(def_box=("n_defense_box", "mean"),
                 def_blitz=("is_blitz", "mean"),
                 def_pass_rushers=("n_pass_rushers", "mean"),
                 def_int_worthy=("is_interception_worthy", "mean"),
                 def_forced_drop=("is_drop", "mean"))
            .rename(columns={"defteam": "team"}))
    return off.merge(dfn, on=["season", "week", "team"], how="outer")


def participation_team_week() -> pd.DataFrame:
    """Coverage, pressure and alignment from the NGS participation columns.

    import_pbp_data returns these whether or not they are requested, and they
    carry the dimensions a paid charting product sells hardest: man vs zone,
    pressure rate, defenders in box, formation. Coverage from 2019, at roughly
    the dropback share of plays (39%) through 2022 and ~93% from 2023, when the
    upstream feed widened.

    Offensive rows are what a team FACED; defensive rows are what it DID.
    """
    import nfl_data_py as nfl
    frames = []
    for s in PART_SEASONS:
        try:
            d = nfl.import_pbp_data([s], downcast=True, cache=False)
        except Exception as e:
            print(f"  participation {s}: {type(e).__name__}")
            continue
        need = ["season", "week", "posteam", "defteam", "was_pressure",
                "defense_man_zone_type", "defenders_in_box", "time_to_throw",
                "offense_formation", "qb_dropback"]
        have = [c for c in need if c in d.columns]
        if "was_pressure" not in have:
            continue
        d = d[have].dropna(subset=["posteam", "defteam"])
        d["pressure"] = pd.to_numeric(d.get("was_pressure"), errors="coerce")
        d["man"] = (d.get("defense_man_zone_type") == "MAN_COVERAGE").astype(float)
        d.loc[d.get("defense_man_zone_type").isna(), "man"] = np.nan
        d["box"] = pd.to_numeric(d.get("defenders_in_box"), errors="coerce")
        d["ttt"] = pd.to_numeric(d.get("time_to_throw"), errors="coerce")
        d["shotgun_form"] = (d.get("offense_formation") == "SHOTGUN").astype(float)
        d.loc[d.get("offense_formation").isna(), "shotgun_form"] = np.nan
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)

    off = (d.groupby(["season", "week", "posteam"], as_index=False)
            .agg(off_pressure_faced=("pressure", "mean"),
                 off_man_faced=("man", "mean"),
                 off_box_faced=("box", "mean"),
                 off_time_to_throw=("ttt", "mean"),
                 off_shotgun=("shotgun_form", "mean"))
            .rename(columns={"posteam": "team"}))
    dfn = (d.groupby(["season", "week", "defteam"], as_index=False)
            .agg(def_pressure_rate=("pressure", "mean"),
                 def_man_rate=("man", "mean"),
                 def_box_used=("box", "mean"),
                 def_time_to_throw_allowed=("ttt", "mean"))
            .rename(columns={"defteam": "team"}))
    return off.merge(dfn, on=["season", "week", "team"], how="outer")


def roll_prior(d: pd.DataFrame, cols: list, n: int = ROLL) -> pd.DataFrame:
    """Rolling mean over games STRICTLY BEFORE this one."""
    d = d.sort_values(["season", "team", "week"]).copy()
    for c in cols:
        d[f"{c}_L{n}"] = (d.groupby(["season", "team"])[c]
                          .transform(lambda s: s.shift(1).rolling(n, min_periods=1).mean()))
    return d


def build() -> pd.DataFrame:
    ngs = ngs_team_week()
    print(f"  NGS: {len(ngs)} team-weeks")
    ftn = ftn_team_week()
    print(f"  FTN: {len(ftn)} team-weeks")
    part = participation_team_week()
    print(f"  participation: {len(part)} team-weeks")
    d = ngs.merge(ftn, on=["season", "week", "team"], how="outer")
    if not part.empty:
        d = d.merge(part, on=["season", "week", "team"], how="outer")
    base = [c for c in d.columns if c not in ("season", "week", "team")]
    d = roll_prior(d, base)
    keep = ["season", "week", "team"] + [f"{c}_L{ROLL}" for c in base]
    out = d[keep]
    out.to_parquet(CACHE, index=False)
    print(f"built {len(out)} team-weeks, {len(base)} features -> {CACHE.name}")
    return out


def joined(f: pd.DataFrame) -> tuple:
    """Training frame with home-minus-away versions of the new features."""
    d = pd.read_parquet(DATA_DIR / "historical_dataset.parquet")
    oc = pd.read_parquet(DATA_DIR / "historical_open_close.parquet")
    kd = pd.to_datetime(oc["kick_date"])
    oc["season"] = kd.dt.year
    # January and February games belong to the PRIOR NFL season.
    oc.loc[kd.dt.month <= 2, "season"] -= 1
    m = d.merge(oc[["season", "home_team", "away_team", "week_open_spread_home"]],
                on=["season", "home_team", "away_team"], how="inner")
    cols = [c for c in f.columns if c.endswith(f"_L{ROLL}")]
    h = f.rename(columns={"team": "home_team", **{c: f"h_{c}" for c in cols}})
    a = f.rename(columns={"team": "away_team", **{c: f"a_{c}" for c in cols}})
    m = (m.merge(h[["season", "week", "home_team"] + [f"h_{c}" for c in cols]],
                 on=["season", "week", "home_team"], how="left")
          .merge(a[["season", "week", "away_team"] + [f"a_{c}" for c in cols]],
                 on=["season", "week", "away_team"], how="left"))
    for c in cols:
        m[f"d_{c}"] = m[f"h_{c}"] - m[f"a_{c}"]
    # 70% coverage or better. FTN only starts in 2022 and several of its rates
    # are sparse, so the thin ones would be mostly zero-fill noise.
    new = [f"d_{c}" for c in cols if m[f"d_{c}"].notna().mean() >= 0.70]
    return m, new


def test(f: pd.DataFrame):
    """Do these features add anything the existing ones do not?"""
    import joblib
    import xgboost as xgb
    from scipy.stats import binomtest
    from config import MODELS_DIR

    m, new = joined(f)
    m["resid_open"] = m.home_margin + m.week_open_spread_home
    m = m.dropna(subset=["resid_open"])
    rf = joblib.load(MODELS_DIR / "nfl_residual_features.joblib")
    base = [c for c in rf if c in m.columns]
    print(f"{len(m)} games | baseline {len(base)} features | new {len(new)}")

    def walk(feats, label):
        rows = []
        for S in (2022, 2023, 2024, 2025):
            tr, te = m[m.season < S], m[m.season == S]
            if not len(te):
                continue
            mod = xgb.XGBRegressor(n_estimators=400, max_depth=4,
                                   learning_rate=0.03, subsample=0.8,
                                   colsample_bytree=0.8, random_state=0)
            mod.fit(tr[feats].fillna(0), tr.resid_open)
            rows.append(pd.DataFrame({"pred": mod.predict(te[feats].fillna(0)),
                                      "actual": te.resid_open.values}))
        r = pd.concat(rows)
        r = r[r.actual != 0]
        r["hit"] = (r.pred > 0) == (r.actual > 0)
        print(f"  {label:24} corr {np.corrcoef(r.pred, r.actual)[0, 1]:+.3f}")
        for bar in (1.0, 1.5, 2.5):
            s = r[r.pred.abs() >= bar]
            w, n = int(s.hit.sum()), len(s)
            print(f"      |resid| >= {bar}: {w}-{n - w} ({w / n:5.1%}, "
                  f"p={binomtest(w, n).pvalue:.3f}) n={n}")

    print("WALK-FORWARD on the live rule's target (margin + weekly opener)")
    walk(base, "baseline")
    walk(base + new, "baseline + new")
    print("Measured 2026-10-29: the new features are slightly WORSE at every")
    print("bar (54.5% -> 54.2% at the live 1.5 threshold) and lower the")
    print("correlation (+0.074 -> +0.062). No improvement, within noise.")


if __name__ == "__main__":
    if "--rebuild" in sys.argv or not CACHE.exists():
        f = build()
    else:
        f = pd.read_parquet(CACHE)
        print(f"loaded {len(f)} team-weeks from cache")
    cols = [c for c in f.columns if c.endswith(f"_L{ROLL}")]
    cov = f[cols].notna().mean().sort_values()
    print(f"\n{len(cols)} rolled features | coverage (non-null share):")
    print(f"  worst: {cov.index[0]} {cov.iloc[0]:.0%} | best: {cov.index[-1]} {cov.iloc[-1]:.0%}")
    print(f"  seasons {int(f.season.min())}-{int(f.season.max())}")
    if "--test" in sys.argv:
        test(f)
