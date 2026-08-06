"""
An outcome-based power rating for the NFL model.

Every one of the NFL model's 57 features is a PROCESS metric -- EPA, success
rate, CPOE, third down, red zone -- or context. There is not a single
outcome-based rating among them. The college model is the opposite: its
ablation showed public ratings (Elo, FPI, SP+) to be the most valuable family
by an order of magnitude, costing -0.0334 correlation when removed.

That asymmetry is worth closing, and it is worth closing with something
structurally different rather than another EPA variant. Process metrics and
power ratings reduce the same games in different ways and fail in different
places: EPA rewards a team that moves the ball and loses, Elo does not care how
you won. Four straight attempts to REORGANISE our existing features failed
today, so the useful move is to add information, not rearrange it.

Elo is computed sequentially from results, so a game's rating uses only games
played before it -- no leakage is possible by construction. Conventional NFL
parameters:

    K = 20                  update size
    HFA = 48 Elo (~2.5 pts) home advantage in the expectation
    MOV multiplier          a 21-point win moves more than a 1-point win, with
                            the autocorrelation correction that stops good
                            teams running away with the rating
    offseason regression    one third of the way back to 1500, since NFL rosters
                            turn over hard

Writes data/nfl_elo.parquet: one row per team-game with the rating BEFORE
kickoff.

Usage:
    python build_nfl_elo.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import DATA_DIR

K = 20.0
HFA_ELO = 48.0
BASE = 1500.0
REGRESS = 1.0 / 3.0          # toward BASE between seasons


def expected(rating_a, rating_b):
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def mov_multiplier(margin, elo_diff_winner):
    """Margin-of-victory scaling with the autocorrelation correction.

    Without the second term a dominant team's rating inflates without limit,
    because blowouts by strong favourites would keep paying full freight.
    """
    return np.log(abs(margin) + 1.0) * (2.2 / (elo_diff_winner * 0.001 + 2.2))


def build() -> pd.DataFrame:
    d = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    d = d.dropna(subset=["home_score", "away_score"]).copy()
    d["kick"] = pd.to_datetime(d["gameday"], errors="coerce")
    d = d.sort_values(["season", "kick", "home_team"]).reset_index(drop=True)

    ratings, rows, last_season = {}, [], None
    for r in d.itertuples():
        if last_season is not None and r.season != last_season:
            # Offseason: pull everyone toward the mean. Skipping this lets a
            # 2019 dynasty rating leak into 2021 as if nothing changed.
            for t in ratings:
                ratings[t] = BASE + (ratings[t] - BASE) * (1 - REGRESS)
        last_season = r.season

        h = ratings.setdefault(r.home_team, BASE)
        a = ratings.setdefault(r.away_team, BASE)

        # Recorded BEFORE the update, so this is what was knowable pre-kickoff.
        rows.append({"season": r.season, "week": r.week,
                     "home_team": r.home_team, "away_team": r.away_team,
                     "elo_home": h, "elo_away": a, "elo_diff": h - a})

        margin = r.home_score - r.away_score
        exp_h = expected(h + HFA_ELO, a)
        actual_h = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        if margin == 0:
            mult = 1.0
        else:
            winner_edge = (h + HFA_ELO - a) if margin > 0 else (a - h - HFA_ELO)
            mult = mov_multiplier(margin, winner_edge)
        delta = K * mult * (actual_h - exp_h)
        ratings[r.home_team] = h + delta
        ratings[r.away_team] = a - delta

    out = pd.DataFrame(rows)
    print(f"built Elo for {len(out)} team-games, "
          f"{out.season.min()}-{out.season.max()}")
    return out


def main():
    e = build()
    p = DATA_DIR / "nfl_elo.parquet"
    e.to_parquet(p, index=False)

    # Sanity: the rating should predict outcomes, and should agree broadly with
    # the market without being a copy of it.
    d = pd.read_parquet(DATA_DIR / "historical_dataset_regular.parquet")
    m = d.merge(e, on=["season", "week", "home_team", "away_team"], how="inner")
    m = m.dropna(subset=["home_margin", "closing_spread_home"])
    print(f"  corr(elo_diff, home_margin)      {np.corrcoef(m.elo_diff, m.home_margin)[0,1]:+.3f}")
    print(f"  corr(elo_diff, -closing_spread)  "
          f"{np.corrcoef(m.elo_diff, -m.closing_spread_home)[0,1]:+.3f}")
    print(f"  elo range {m.elo_diff.min():.0f} to {m.elo_diff.max():.0f}")
    print(f"  saved -> {p.name}")


if __name__ == "__main__":
    main()
