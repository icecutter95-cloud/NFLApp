"""
Current-season rolling form for college teams, so the models stop being told
every game is week 1.

THE BUG THIS FIXES
log_cfb_predictions hardcoded all 27 rolling form features to 0.0 at serve
time. For DIFFERENCE features zero sounds neutral, so the under bias on totals
was blamed on "a total is a level, not a difference". That was wrong. Measured
on the training frame:

    rows where all form is zero (272, mostly weeks 1-2)   residual -3.26, 60.7% under
    rows with real form (3,327)                           residual +0.20, 50.8% under
    week 1 residual -2.93 | week 2 -1.81 | week 3 -0.13 | week 4 +1.97

All-zero form is not a neutral input -- in training it IS a week-1 game, and
week-1 college games go under. The model with form zeroed predicts a residual
of -1.91 at a total of 38 and -3.70 at 65, which is exactly what it should
predict for the input it was handed. Nothing was biased; we were lying to it.

Both markets were affected. Spreads are differences so the zeros hurt less,
but they were just as uninformed -- their only varying inputs were the
preseason priors, the venue and the line.

WHY THIS IS CHEAP
The form comes from CFBD /stats/game/advanced?year=, which returns every game
in a season in ONE call, plus one /games call for kickoff times. Two requests
against a monthly quota of ~1,000, cached in cfb_api_cache. The belief that
in-season form was quota-prohibitive -- the reason this sat unfixed since
August -- was simply untested.

HOW PARITY IS KEPT
It does NOT reimplement the rolling windows. fetch_team_games, fetch_games and
add_form are imported from build_cfb_dataset, the same functions that built the
training set, so the serve-time numbers are computed by the training code. A
second implementation would drift, and a drifted form feature is worse than a
zeroed one because it looks plausible.

Form "as of now" is obtained by appending ONE synthetic future row per team.
add_form shifts every rolling window back by one game, so a row dated after a
team's last played game receives the mean of their last 4 and last 8 ACTUAL
games -- precisely the quantity a prediction needs. The synthetic row's own
(absent) stats are never read, because a row never feeds its own features.

Usage:
    python scripts/fetch_cfb_form.py              # write cfb_team_form
    python scripts/fetch_cfb_form.py --dry-run
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import CURRENT_SEASON as SEASON
from build_cfb_dataset import (fetch_team_games, fetch_games,
                               fetch_drive_stats, add_form, FEATURE_COLS)
from score_week import supabase

SYNTH_PREFIX = "SYNTH_"


def build() -> pd.DataFrame:
    tg = fetch_team_games([SEASON])
    if tg.empty:
        print(f"no advanced stats for {SEASON} yet")
        return pd.DataFrame()

    # add_form rolls ADJ_COLS + RAW_COLS, and RAW_COLS ends with the drive
    # columns (start field position, scoring rate, drives) that come from
    # /drives rather than /stats/game/advanced. main() merges them in before
    # calling add_form; omitting that merge here raised
    # "Column not found: off_start_fp". Same order, same asserts.
    dr = fetch_drive_stats([SEASON])
    before = len(tg)
    tg = tg.merge(dr, on=["game_id", "team"], how="left")
    assert len(tg) == before, "drive-stat join fanned out"

    games = fetch_games([SEASON])
    dates = games[["game_id", "start_date"]].copy()
    dates["kick_ts"] = pd.to_datetime(dates.start_date, utc=True, errors="coerce")
    dates = dates[["game_id", "kick_ts"]]

    played = tg.merge(dates, on="game_id", how="inner").dropna(subset=["kick_ts"])
    print(f"{SEASON}: {len(played)} team-games with advanced stats, "
          f"{played.team.nunique()} teams, weeks "
          f"{int(played.week.min())}-{int(played.week.max())}")

    # One synthetic row per team, dated after everything played. add_form's
    # shift(1) means this row's features are the mean of the team's last N real
    # games, which is the form entering their next game.
    last = played.kick_ts.max() + pd.Timedelta(days=7)
    synth = pd.DataFrame({
        "game_id": [f"{SYNTH_PREFIX}{t}" for t in sorted(played.team.unique())],
        "season": SEASON,
        "week": int(played.week.max()) + 1,
        "team": sorted(played.team.unique()),
        "opp": "SYNTH_OPP",
    })
    d = pd.concat([played.drop(columns=["kick_ts"]), synth], ignore_index=True)
    dates_all = pd.concat([
        dates,
        pd.DataFrame({"game_id": synth.game_id, "kick_ts": last}),
    ], ignore_index=True)

    rolled = add_form(d, dates_all)
    out = rolled[rolled.game_id.astype(str).str.startswith(SYNTH_PREFIX)].copy()

    cols = [c for c in FEATURE_COLS if c in out.columns]
    missing = [c for c in FEATURE_COLS if c not in out.columns]
    if missing:
        print(f"  warning: {len(missing)} feature(s) absent from the rolled "
              f"frame, will serve as 0: {missing[:4]}")
    keep = ["season", "team", "games_played"] + cols
    out = out[keep]
    out["week"] = int(played.week.max()) + 1
    return out


def main():
    dry = "--dry-run" in sys.argv
    f = build()
    if f.empty:
        return

    cols = [c for c in FEATURE_COLS if c in f.columns]
    nz = (f[cols].abs().to_numpy() > 1e-9).mean()
    print(f"\n{len(f)} teams, {len(cols)} form columns, "
          f"{nz:.0%} of values non-zero")
    print("  sanity — a few teams' opponent-adjusted offensive PPA (L4):")
    s = f.sort_values("off_ppa_adj_L4", ascending=False)
    for _, r in pd.concat([s.head(3), s.tail(3)]).iterrows():
        print(f"    {r.team:18} {r.off_ppa_adj_L4:+.3f}  "
              f"(def {r.def_ppa_adj_L4:+.3f}, {int(r.games_played)} games)")

    if dry:
        print("\n--dry-run: nothing written")
        return

    # Column names go in LOWERCASE. Postgres folds unquoted identifiers, so
    # off_ppa_adj_L4 in the DDL became off_ppa_adj_l4 in the table, and writing
    # the mixed-case name returns PGRST204 "could not find the column". The
    # alternative -- quoting every identifier forever -- is worse. The reader in
    # log_cfb_predictions maps back to FEATURE_COLS by lowercasing, so the
    # mapping lives in exactly two places, both commented.
    rows = []
    for _, r in f.iterrows():
        rec = {"season": int(r.season), "week": int(r.week), "team": r.team,
               "games_played": int(r.games_played)}
        for c in cols:
            v = r[c]
            rec[c.lower()] = None if pd.isna(v) else float(v)
        rows.append(rec)
    for i in range(0, len(rows), 200):
        supabase.table("cfb_team_form").upsert(
            rows[i:i + 200], on_conflict="season,team").execute()
    print(f"\nwrote form for {len(rows)} teams")


if __name__ == "__main__":
    main()
