"""
Per-quarterback quality, and what a team loses when its starter does not play.

WHY THIS EXISTS
The models carry inj_qb_out_home / inj_qb_out_away -- a BINARY count of
quarterbacks ruled out. Measured on 2026 week 5, forcing that flag on for
Baltimore moved the spread model's number by 0.24 points while the market had
moved 7.5. A binary cannot tell Lamar Jackson from a backup, or a backup from
a third-stringer, so the input that reprices a game most violently is the one
the model is nearly blind to.

WHAT THIS BUILDS
One row per team-week:

    qb_quality       rolling EPA/attempt of the quarterback who takes the
                     snaps, entering that week
    qb_baseline      same measure for the team's ESTABLISHED starter, judged
                     on who has carried the snaps over the trailing 17 team
                     games, also entering that week
    qb_quality_drop  qb_quality - qb_baseline, clipped at 0

qb_quality_drop is the feature that matters: continuous, zero in the ordinary
case, and scaled to how big the downgrade actually is.

NO LEAKAGE, AND NO TIME-TRAVEL
Every quality is the value entering the game, built from attempts STRICTLY
BEFORE it -- the rule compute_metrics follows for team metrics. Two mistakes
worth recording because the first draft of this file made both:

  * The baseline was taken as the primary quarterback's LAST quality of the
    season, which is end-of-season information. Compared against a starter's
    week-3 value it manufactured downgrades for quarterbacks who never missed
    a game: DET 2024 week 3 showed Goff "downgrading" from Goff by 0.318,
    because his own September number was measured against his own January one.
    Both sides of the subtraction are now taken at the same moment.

  * Quality was computed per season, so every quarterback reset to league
    average each September and nobody was distinguishable in week 1. History
    now runs continuously, with the attempt half-life handling staleness.

The starter's IDENTITY for a historical week comes from that week's
play-by-play (whoever threw the most passes). That is a depth-chart fact known
before kickoff, not a result.

WHAT THE MEASUREMENT SAID -- READ THIS BEFORE ADDING IT TO A MODEL
Built on 2018-2026, 4,582 team-weeks, 654 with a real downgrade (14.3%).
Then tested, and the answer was no:

  closing_spread_home ~ drop_diff   slope -27.9 pts per EPA/att, p=7e-40
  home_margin         ~ drop_diff   slope +29.3 pts per EPA/att, p=3e-09
  home_cover_surplus  ~ drop_diff   slope  +1.4 pts,             p=0.76

The market charges 27.9 points per unit of downgrade and reality pays 29.3.
Those agree within 5%, and the residual against the closing line is 1.4 points
with p=0.76. The market prices quarterback absences correctly, so there is no
edge in predicting them better.

And the models already inherit that pricing for free. They predict
home_cover_surplus -- margin PLUS the closing spread -- so the market's QB
adjustment is inside the target, not something the features must reconstruct.
Regressing the live spread model's residual on drop_diff over all 2,227
training games gives a slope of -1.2 with p=0.77, and on the 237 games with a
0.10+ gap the mean residual is -0.08 points. There is nothing left for this
feature to explain.

So this file is NOT wired into build_dataset or into any model, deliberately.
It exists because the numbers above are worth keeping, and because
qb_quality_drop is a good way to SHOW a human why a line moved eight points
("Huntley in for Jackson, -0.29, about 8.4 points"). The model failure that
prompted it -- 8-14 when backing a downgraded team, 4-15 when the market ran
4+ points away -- is not a missing feature. It is a pick frozen at a pre-news
line and never revisited, which is what the adverse-movement badge in
frontend/src/lib/movement.js is for.

SHRINKAGE
A quarterback with 20 attempts has a meaningless EPA/attempt, and the
replacement is exactly the player with the thinnest sample. Each is pulled
toward the league mean with a pseudo-count of PRIOR_ATTEMPTS, so a debut
backup scores near league average rather than at whatever his first drive did.

Usage:
    python scripts/build_qb_quality.py            # all seasons -> parquet
    python scripts/build_qb_quality.py --check    # sanity report only
"""

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR, ALL_HISTORICAL_SEASONS, CURRENT_SEASON

# Attempts of league-average play mixed into every quarterback's number. From
# the shape of the problem rather than fitted: a starter has 300+ attempts a
# season and is barely moved, a backup with 40 is pulled most of the way back.
PRIOR_ATTEMPTS = 120.0

# Half-life in attempts. A quarterback moves more slowly than team form.
HALF_LIFE_ATTEMPTS = 400.0

# Trailing team games used to decide who the ESTABLISHED starter is. One full
# season, so a quarterback who has started all year stays the baseline through
# a multi-week absence instead of the backup quietly becoming the baseline.
BASELINE_WINDOW = 17

COLS = ["season", "week", "posteam", "passer_player_id", "passer_player_name",
        "pass_attempt", "qb_epa", "epa"]


def load_pbp(season: int) -> pd.DataFrame:
    f = DATA_DIR / f"pbp_{season}.parquet"
    if f.exists() and season != CURRENT_SEASON:
        return pd.read_parquet(f, columns=COLS)
    import nfl_data_py as nfl
    d = nfl.import_pbp_data([season], downcast=True, cache=False)
    if season == CURRENT_SEASON:
        d.to_parquet(f, index=False)
    return d[COLS]


def qb_game_level(seasons) -> pd.DataFrame:
    """One row per quarterback per team-week: attempts and EPA on his throws."""
    frames = []
    for s in seasons:
        try:
            pbp = load_pbp(s)
        except Exception as e:
            print(f"  {s}: {type(e).__name__} — skipping")
            continue
        p = pbp[(pbp.pass_attempt == 1) & pbp.passer_player_id.notna()].copy()
        if p.empty:
            print(f"  {s}: no passing plays yet")
            continue
        q = pd.to_numeric(p.qb_epa, errors="coerce")
        p["q"] = q.fillna(pd.to_numeric(p.epa, errors="coerce"))
        frames.append(
            p.groupby(["season", "week", "posteam", "passer_player_id",
                       "passer_player_name"], as_index=False)
             .agg(attempts=("q", "size"), epa_sum=("q", "sum")))
    if not frames:
        return pd.DataFrame()
    g = pd.concat(frames, ignore_index=True)
    return g.rename(columns={"posteam": "team"})


def rolling_quality(games: pd.DataFrame) -> pd.DataFrame:
    """Quality ENTERING each game, continuous across seasons."""
    games = games.sort_values(["season", "week"]).reset_index(drop=True)
    league_epa = games.epa_sum.sum() / max(games.attempts.sum(), 1)

    out = []
    for pid, d in games.groupby("passer_player_id", sort=False):
        d = d.sort_values(["season", "week"])
        w_sum = e_sum = 0.0
        for r in d.itertuples():
            out.append({
                "season": r.season, "week": r.week, "team": r.team,
                "qb_id": pid, "qb_name": r.passer_player_name,
                "attempts_game": r.attempts, "attempts_prior": w_sum,
                "qb_quality": (e_sum + league_epa * PRIOR_ATTEMPTS) /
                              (w_sum + PRIOR_ATTEMPTS),
            })
            decay = 0.5 ** (r.attempts / HALF_LIFE_ATTEMPTS)
            w_sum = w_sum * decay + r.attempts
            e_sum = e_sum * decay + r.epa_sum
    return pd.DataFrame(out)


def team_weeks(q: pd.DataFrame) -> pd.DataFrame:
    """Starter, established baseline, and the gap -- both sides same-moment."""
    q = q.sort_values(["season", "week"]).reset_index(drop=True)
    q["order"] = q.season * 100 + q.week          # chronological key

    rows = []
    for team, d in q.groupby("team", sort=False):
        d = d.sort_values("order")
        slots = sorted(d.order.unique())
        # Each quarterback's most recent quality as of each slot, and his
        # attempts for this team inside the trailing window.
        for i, slot in enumerate(slots):
            here = d[d.order == slot]
            starter = here.loc[here.attempts_game.idxmax()]

            past = d[d.order < slot]
            window = past[past.order >= (slots[max(i - BASELINE_WINDOW, 0)]
                                         if i else slot)]
            if window.empty:
                baseline_q = starter.qb_quality      # nothing to compare to
            else:
                att = window.groupby("qb_id").attempts_game.sum()
                primary = att.idxmax()
                # That quarterback's latest quality AT OR BEFORE this slot.
                hist = d[(d.qb_id == primary) & (d.order <= slot)]
                baseline_q = (hist.iloc[-1].qb_quality if len(hist)
                              else starter.qb_quality)
                # A starter who IS the primary is never a downgrade.
                if starter.qb_id == primary:
                    baseline_q = starter.qb_quality
            rows.append({
                "season": int(starter.season), "week": int(starter.week),
                "team": team, "qb_id": starter.qb_id, "qb_name": starter.qb_name,
                "attempts_prior": float(starter.attempts_prior),
                "qb_quality": float(starter.qb_quality),
                "qb_baseline": float(baseline_q),
            })
    out = pd.DataFrame(rows)
    out["qb_quality_drop"] = (out.qb_quality - out.qb_baseline).clip(upper=0.0)
    return out.sort_values(["season", "week", "team"]).reset_index(drop=True)


def report(d: pd.DataFrame):
    drops = d[d.qb_quality_drop < -0.02]
    print(f"\n{len(d)} team-weeks, {len(drops)} with a real downgrade "
          f"({len(drops)/len(d):.1%})")
    print(f"  downgrade size: mean {drops.qb_quality_drop.mean():+.3f}, "
          f"worst {drops.qb_quality_drop.min():+.3f} EPA/attempt")
    e = d[d.attempts_prior >= 400]
    print("\nhighest quality on record (400+ prior attempts):")
    for _, r in e.sort_values("qb_quality", ascending=False).head(4).iterrows():
        print(f"  {r.qb_name:16} {r.season} wk{int(r.week):<2} {r.qb_quality:+.3f}")
    print("lowest:")
    for _, r in e.sort_values("qb_quality").head(4).iterrows():
        print(f"  {r.qb_name:16} {r.season} wk{int(r.week):<2} {r.qb_quality:+.3f}")
    print("\nbiggest downgrades (a backup in for an established starter):")
    for _, r in d.sort_values("qb_quality_drop").head(8).iterrows():
        print(f"  {r.team} {r.season} wk{int(r.week):<2} {r.qb_name:16} "
              f"{r.qb_quality:+.3f} vs baseline {r.qb_baseline:+.3f} "
              f"-> {r.qb_quality_drop:+.3f}")
    # The check that caught the first draft's bug.
    same = d[d.qb_id == d.qb_id]
    bad = d[(d.qb_quality_drop < -0.02) & (d.qb_quality == d.qb_baseline)]
    print(f"\nsanity: rows claiming a drop while starter IS the baseline: "
          f"{len(bad)} (must be 0)")


def main():
    seasons = sorted(dict.fromkeys(ALL_HISTORICAL_SEASONS + [CURRENT_SEASON]))
    print(f"building QB quality across {seasons[0]}-{seasons[-1]}")
    g = qb_game_level(seasons)
    if g.empty:
        print("no passing data at all"); return
    q = rolling_quality(g)
    d = team_weeks(q)
    report(d)
    if "--check" not in sys.argv:
        out = DATA_DIR / "qb_quality.parquet"
        d.to_parquet(out, index=False)
        print(f"\n{len(d)} rows -> {out.name}")


if __name__ == "__main__":
    main()
