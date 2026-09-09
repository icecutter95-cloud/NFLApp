"""
Do public splits predict where the line goes NEXT?

This is the question worth asking of the splits data, and it is a much better
question than "do these buckets win". Line movement resolves in hours, every
game contributes whether or not it was bet, and the sample grows every capture
instead of every settled wager. Win rate needs hundreds of bets against a
measured noise floor of +/-3.3pp; this needs weeks.

The hypothesis, stated so it can fail
-------------------------------------
Money moves lines. If the money is on our side, the book should shift TOWARD our
side, our number gets worse, and the right response is to bet early. If the money
is against us, the number should drift our way and waiting is better.

    our_money_pct high  ->  line moves toward us  ->  clv NEGATIVE
    our_money_pct low   ->  line moves away       ->  clv POSITIVE

So the prediction is a NEGATIVE correlation between the money share on our side
and subsequent CLV. A positive correlation falsifies it; a zero says splits carry
no timing information.

What makes this test honest
---------------------------
Only movement STRICTLY AFTER the capture counts. A contemporaneous comparison --
today's splits against movement already banked -- is contaminated, because both
reflect the same action. Run on 2026-09-06 that contaminated version gave +0.06
to +0.12 across 63 rows: no relationship, and the wrong sign if any. This script
exists so that number can be replaced with one that means something.

Each capture is paired with the last line snapshot at or before it, and with the
latest snapshot after it. Captures with no later snapshot are skipped rather than
counted as zero movement.

Usage:
    python test_splits_movement.py              # both leagues
    python test_splits_movement.py --league nfl
    python test_splits_movement.py --min-hours 6
"""

import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from score_week import supabase

# join: how a split row finds its line history.
#
#   cfb  by game_id -- cfb_predictions takes its ids straight from
#        cfb_line_history, so they are the same ids.
#   nfl  by TEAM PAIR -- line_predictions falls back to synthetic ids
#        ("2026_1_NE_SEA") because Odds API event ids are unstable, while
#        line_history keeps the event id. Measured: 16 prediction ids, 272
#        history ids, ZERO overlap. Joining on game_id here silently produced no
#        pairs at all rather than an error.
LEAGUES = {
    "nfl": {"splits": "nfl_public_splits", "lines": "line_history",
            "signal": "nfl_signal", "join": "teams"},
    "cfb": {"splits": "cfb_public_splits", "lines": "cfb_line_history",
            "signal": "cfb_signal", "join": "game_id"},
}
MIN_MOVE = 0.5          # half a point is the smallest tick worth scoring


def pull(table, cols="*"):
    out, page = [], 1000
    for i in range(0, 100000, page):
        r = supabase.table(table).select(cols).range(i, i + page - 1).execute()
        d = r.data or []
        out.extend(d)
        if len(d) < page:
            break
    return pd.DataFrame(out)


def analyse(league, min_hours):
    cfg = LEAGUES[league]
    splits = pull(cfg["splits"])
    if splits.empty:
        print(f"\n{league.upper()}: no captures yet")
        return None
    lines = pull(cfg["lines"],
                 "game_id, home_team, away_team, recorded_at, spread_home, total")
    sig = pull(cfg["signal"],
               "game_id, bet_type, predicted_side, home_team, away_team, "
               "commence_time")
    if lines.empty:
        print(f"\n{league.upper()}: no line history")
        return None

    # ISO8601 explicitly: captures written before the timestamps were floored
    # carry microseconds and later ones do not, so letting pandas infer a single
    # format from the first row fails on the rest.
    splits["captured_at"] = pd.to_datetime(splits["captured_at"],
                                           format="ISO8601", utc=True)
    lines["recorded_at"] = pd.to_datetime(lines["recorded_at"],
                                          format="ISO8601", utc=True)
    meta = {(r.game_id, r.bet_type): r for r in sig.itertuples()} if not sig.empty else {}

    # One line per (game, timestamp): the median across books, so a single
    # outlier book cannot masquerade as movement.
    key = ["home_team", "away_team"] if cfg["join"] == "teams" else ["game_id"]
    lines = (lines.groupby(key + ["recorded_at"], as_index=False)
             .agg(spread_home=("spread_home", "median"), total=("total", "median")))

    rows = []
    for s in splits.itertuples():
        m = meta.get((s.game_id, s.bet_type))
        if m is None or not m.predicted_side:
            continue
        side = m.predicted_side
        col = "spread_home" if s.bet_type == "spread" else "total"
        if cfg["join"] == "teams":
            # A team pair repeats across seasons, so without a date guard a
            # 2025 snapshot could pair with a 2026 capture. Two weeks either
            # side of kickoff is wider than any line history for one game and
            # far narrower than a season.
            kick = pd.to_datetime(m.commence_time, format="ISO8601", utc=True)
            g = lines[(lines.home_team == m.home_team)
                      & (lines.away_team == m.away_team)
                      & (lines.recorded_at >= kick - pd.Timedelta(days=14))
                      & (lines.recorded_at <= kick + pd.Timedelta(days=1))]
        else:
            g = lines[lines.game_id == s.game_id]
        g = g.sort_values("recorded_at")
        if g.empty:
            continue
        before = g[g.recorded_at <= s.captured_at]
        after = g[g.recorded_at > s.captured_at + pd.Timedelta(hours=min_hours)]
        if before.empty or after.empty:
            continue
        at = before.iloc[-1][col]
        later = after.iloc[-1][col]
        if pd.isna(at) or pd.isna(later):
            continue
        move = float(later) - float(at)
        # CLV convention, identical to the views: home and under profit when the
        # number falls; away and over when it rises.
        clv = -move if side in ("home", "under") else move
        our_money = s.home_money_pct if side in ("home", "over") else s.away_money_pct
        our_bets = s.home_bets_pct if side in ("home", "over") else s.away_bets_pct
        if our_money is None or our_bets is None:
            continue
        rows.append({"bet_type": s.bet_type, "our_money": our_money,
                     "our_bets": our_bets, "gap": our_money - our_bets,
                     "subsequent_move": move, "subsequent_clv": clv,
                     "hours": (after.iloc[-1].recorded_at - s.captured_at)
                              .total_seconds() / 3600})
    d = pd.DataFrame(rows)
    print(f"\n{league.upper()}: {len(splits)} captures -> {len(d)} usable pairs "
          f"(needs a snapshot >{min_hours}h after the capture)")
    if len(d) < 20:
        print("  too few to say anything. Let the scheduled task run for a week or two.")
        return d if len(d) else None

    moved = d[d.subsequent_clv.abs() >= MIN_MOVE]
    print(f"  {len(moved)} of those actually moved half a point or more")
    print(f"  median gap between capture and later line: {d.hours.median():.1f}h")
    print()
    print("  HYPOTHESIS: money on our side -> line moves toward us -> NEGATIVE corr")
    for label, x in (("money share on our side", d.our_money),
                     ("ticket share on our side", d.our_bets),
                     ("handle gap (money - tickets)", d.gap)):
        r = np.corrcoef(x, d.subsequent_clv)[0, 1]
        verdict = ("supports it" if r <= -0.15 else
                   "contradicts it" if r >= 0.15 else "nothing")
        print(f"    corr({label:28}, subsequent CLV) = {r:+.3f}   {verdict}")

    print()
    print("  by money share on our side:")
    bins = [(0, 40, "under 40%"), (40, 60, "40-60%"), (60, 101, "over 60%")]
    for lo, hi, name in bins:
        b = d[(d.our_money >= lo) & (d.our_money < hi)]
        if len(b) < 5:
            continue
        print(f"    {name:10} n={len(b):>4}  mean subsequent CLV {b.subsequent_clv.mean():+.2f}"
              f"   moved to us {100*(b.subsequent_clv > 0).mean():.0f}% of the time")
    return d


def main():
    min_hours = 0
    if "--min-hours" in sys.argv:
        min_hours = float(sys.argv[sys.argv.index("--min-hours") + 1])
    only = None
    if "--league" in sys.argv:
        only = sys.argv[sys.argv.index("--league") + 1]

    frames = []
    for lg in LEAGUES:
        if only and lg != only:
            continue
        d = analyse(lg, min_hours)
        if d is not None and len(d):
            frames.append(d)

    if len(frames) > 1:
        d = pd.concat(frames, ignore_index=True)
        if len(d) >= 20:
            r = np.corrcoef(d.gap, d.subsequent_clv)[0, 1]
            print(f"\nPOOLED: n={len(d)}  corr(handle gap, subsequent CLV) = {r:+.3f}")
            print("  Pooling two sports assumes the effect is the same in both,")
            print("  which is itself an assumption worth checking separately.")


if __name__ == "__main__":
    main()
