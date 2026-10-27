"""
Load final scores into game_results so logged predictions can be graded.

Why this exists
---------------
CLV resolves the week a line closes, but win/loss needs the game to be played
and the score to be somewhere the database can see it. Nothing was writing that
table, so the live season could show CLV and nothing else.

Sign convention
---------------
nflverse `spread_line` is POSITIVE when the home team is favored, which is the
OPPOSITE of how sportsbooks quote it and the opposite of every other spread in
this project. Getting that backwards silently inverted the model's target once
already and cost weeks, so the flip happens here, once, explicitly:

    closing_spread_home = -spread_line

`assert_result_conventions` then refuses to write if the stored numbers do not
behave like real spreads. Its primary test is the SIGN CORRELATION, which must
be strongly negative; the cover-rate test is secondary and scales with the
sample, after a fixed 44-56% band blocked grading on 2026-10-27 over a 41.6%
rate whose binomial p was 0.111.

Usage:
    python fetch_game_results.py              # current season
    python fetch_game_results.py 2025         # a specific season
    python fetch_game_results.py 2024 2025    # several
    python fetch_game_results.py --dry-run
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest
import nfl_data_py as nfl

warnings.filterwarnings("ignore")

from config import CURRENT_SEASON
from score_week import supabase


def assert_result_conventions(df: pd.DataFrame):
    """Refuse to write scores whose spreads point the wrong way.

    A correctly-signed home spread is NEGATIVE when home is favored, so it must
    be negatively correlated with the home margin, and home teams must cover
    somewhere near half the time. This is the same guard build_dataset.py runs.
    """
    d = df.dropna(subset=["closing_spread_home", "home_margin"])
    if len(d) < 30:
        print(f"  (only {len(d)} graded games — skipping convention check)")
        return

    corr = float(np.corrcoef(d["closing_spread_home"], d["home_margin"])[0, 1])
    if corr > -0.2:
        raise SystemExit(
            f"ABORT: corr(closing_spread_home, home_margin) = {corr:+.3f}. "
            "A correctly-signed spread is strongly NEGATIVE here. The sign is flipped."
        )

    # The cover rate is a SECONDARY check and its threshold now scales with the
    # sample, because a fixed 44-56% band is only +/-1.2 standard deviations at
    # a half-season and fires on clean data roughly one year in five.
    #
    # It fired on 2026-10-27 at 41.6% (42 of 101), binomial p = 0.111 against
    # 50% -- ordinary variance, not a fault. The abort happens before any
    # insert, so week 7 went ungraded: fourteen games including the Cowboys
    # at Eagles Monday nighter, held back by a guard that was right in spirit
    # and wrong in calibration.
    #
    # A genuinely flipped sign does not produce a mildly low rate. Adding a
    # home favourite's spread with the wrong sign makes it "cover" whenever it
    # loses by less than the number, pushing the rate far toward 100% AND
    # flipping the correlation above positive -- which the check above already
    # catches, and which is the real detector. So this one only needs to catch
    # the extreme, and a binomial test does that without punishing a season in
    # which home teams genuinely underperform.
    surplus = d["home_margin"] + d["closing_spread_home"]
    live = surplus[surplus.abs() > 1e-9]
    covers = int((live > 0).sum())
    n = int(len(live))
    rate = covers / n * 100 if n else float("nan")
    p = binomtest(covers, n).pvalue if n else 1.0
    if p < 0.001:
        raise SystemExit(
            f"ABORT: home teams cover {covers}/{n} = {rate:.1f}% "
            f"(binomial p = {p:.2g} against 50%). That is far beyond variance; "
            "something is wrong with the lines or the scores."
        )
    note = "" if p >= 0.05 else f" -- unusual but within variance (p = {p:.3f})"
    print(f"  convention check OK (corr {corr:+.3f}, home covers "
          f"{rate:.1f}% of {n}{note})")


def build(seasons: list) -> pd.DataFrame:
    sched = nfl.import_schedules(seasons)
    played = sched.dropna(subset=["home_score", "away_score"]).copy()
    print(f"  {len(played)} completed of {len(sched)} scheduled")
    if played.empty:
        return played

    played["home_margin"] = played["home_score"] - played["away_score"]
    played["total_points"] = played["home_score"] + played["away_score"]
    # See module docstring: nflverse quotes this backwards from the book.
    played["closing_spread_home"] = -played["spread_line"]
    played["closing_total"] = played["total_line"]

    surplus = played["home_margin"] + played["closing_spread_home"]
    played["spread_result"] = np.where(
        played["closing_spread_home"].isna(), None,
        np.where(surplus > 0, "home_covered",
                 np.where(surplus < 0, "away_covered", "push")))
    played["total_result"] = np.where(
        played["closing_total"].isna(), None,
        np.where(played["total_points"] > played["closing_total"], "over",
                 np.where(played["total_points"] < played["closing_total"], "under", "push")))

    out = played[[
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "home_score", "away_score", "home_margin", "total_points",
        "closing_spread_home", "closing_total", "spread_result", "total_result",
    ]].rename(columns={"gameday": "game_date"})

    for c in ["home_score", "away_score", "home_margin", "total_points", "season", "week"]:
        out[c] = out[c].astype(int)
    out["game_date"] = out["game_date"].astype(str).str[:10]
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    seasons = [int(a) for a in args] or [CURRENT_SEASON]

    print(f"Fetching results for {seasons}")
    df = build(seasons)
    if df.empty:
        print("  nothing to write — no completed games yet")
        return

    assert_result_conventions(df)

    by_season = df.groupby("season").size().to_dict()
    print(f"  {dict(by_season)}")
    print(f"  home covered {int((df.spread_result == 'home_covered').sum())}, "
          f"over {int((df.total_result == 'over').sum())}")

    if dry:
        print("  --dry-run: nothing written")
        return

    recs = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v)
             for k, v in r.items()} for r in df.to_dict("records")]
    for i in range(0, len(recs), 500):
        supabase.table("game_results").upsert(recs[i:i + 500], on_conflict="game_id").execute()
    print(f"  wrote {len(recs)} rows to game_results")

    # Report what that unlocked: predictions that now have a graded outcome.
    for s in seasons:
        tr = supabase.table("clv_tracking").select("bet_type, result") \
            .eq("season", s).execute()
        rows = [r for r in (tr.data or []) if r["result"]]
        if not rows:
            continue
        for bt in ("spread", "total"):
            g = [r for r in rows if r["bet_type"] == bt]
            if not g:
                continue
            w = sum(r["result"] == "win" for r in g)
            l = sum(r["result"] == "loss" for r in g)
            p = sum(r["result"] == "push" for r in g)
            print(f"  {s} {bt}: {w}-{l}-{p} graded")


if __name__ == "__main__":
    main()
