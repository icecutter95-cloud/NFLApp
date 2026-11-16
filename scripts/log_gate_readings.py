"""
Stamp the two-gate reading for upcoming NFL picks, before kickoff.

WHY THIS EXISTS
The gates are used to SIZE, not to select: 1u on every qualifying pick at the
opener, plus a 0.5u top-up on the ones where both gates fire, read about two
days out. See bet_time_gate.py for why that ordering wins and
nfl_threshold_sweep.py for why the gate is not trusted to select.

Acting on the gate means having read it at a real moment before kickoff. That
reading is not currently recorded anywhere -- it is reconstructible from
line_history and nfl_public_splits, but reconstructing it is error-prone: the
first attempt lost 20 of 78 games to a UTC date join and nearly used a
commence_time that drifts by hours. So the decision gets written down when it
is made, rather than re-derived later with hindsight about which horizon to
pick.

WHAT IT WRITES
One row per (game_id, bet_type, read_at) into nfl_gate_readings, every run,
for every QUALIFYING pick whose kickoff is still ahead. A dense series rather
than a single stamp, which is what answers "when did the gate first fire" and
costs a few hundred rows a week.

The durable facts are move_toward_us and our_tix. rule1/rule2/gate_fires are
generated in the database from those plus the bars stored in the same row, so
changing the constants here can never restate a past decision.

A NULL rule2 means the splits had not been captured yet at read_at. That is
not the same as the gate not firing, and the two must not be collapsed.

WHICH LINE THE MOVEMENT IS MEASURED AGAINST
DraftKings, from book_lines -- the basis open_line already uses, and the one
bet_time_gate.py measured the horizons on. nfl_public_splits also carries
line_at_capture, which is the only line perfectly synchronised with the ticket
percentages, but it is Action Network's consensus and carries a basis against
DraftKings: measured 2026-11-12 over 42 upcoming markets, mean 0.15 pts on
spreads and 0.25 on totals, max 1.00, with 5 markets a full point apart. That
is enough to push a market across a 1.5-point bar on source disagreement
alone, so it is stored for audit and not used for the gate.

NOTHING READS FROM THIS YET
Purely additive. No existing table, view, workflow or app path is touched --
clv_tracking is unchanged, and this only ever INSERTs into its own table.

Usage:
    python scripts/log_gate_readings.py
    python scripts/log_gate_readings.py --dry-run
    python scripts/log_gate_readings.py --within 240   # hours before kickoff
"""

import sys
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from score_week import supabase, fetch_all, ts

MOVE_BAR = 1.5      # rule 1: points the line must come toward our side
TICKET_BAR = 50.0   # rule 2: max % of tickets on our side
BOOK = "draftkings"  # the book bet_time_gate.py measured the horizons on
WITHIN_HOURS = 336  # log from two weeks out; the pick freezes ~320h before
ALARM_HOURS = 36    # only imminent games with no reading are worth shouting


def toward_us(side, open_line, line):
    """Points the line has come TOWARD our side. THE canonical definition.

    home/under gain when the number falls, away/over when it rises. Mirrors
    clv_tracking's clv_points CASE and build_movement_history.py:108 exactly,
    so a reading here is directly comparable to the stored clv_points. Imported
    by bet_time_gate.py rather than restated there -- this sign convention has
    bitten this project before and there should be one copy of it.
    """
    delta = np.asarray(line, dtype=float) - np.asarray(open_line, dtype=float)
    rising = pd.Series(side).isin(["away", "over"]).values
    return np.where(rising, delta, -delta)


def upcoming_picks(within_hours):
    """Qualifying NFL picks whose kickoff is still ahead.

    Qualifiers only: that is the bet set, and a reading for a pick we would
    never place is noise. The non-qualifying half stays available through
    clv_tracking if it is ever wanted.
    """
    d = pd.DataFrame(fetch_all("clv_tracking"))
    d = d[d.bet_type.isin(["spread", "total"])].copy()
    d["kick"] = ts(d.commence_time)
    now = pd.Timestamp.now(tz="UTC")
    return d[(d.qualifies == True)                      # noqa: E712
             & d.predicted_side.notna() & d.open_line.notna()
             & (d.kick > now)
             & (d.kick < now + pd.Timedelta(hours=within_hours))]


def current_lines(picks):
    """The market line right now, matched on the pair within a day.

    book_lines is the CURRENT state -- one row per (game_id, book), refreshed
    by the hourly odds cron -- so it is 1.6k rows instead of line_history's
    57k, and no "latest snapshot" ordering is needed.

    Matched on (home_team, away_team) with a day of slack, not on an exact
    kickoff date: a Sunday 8:20pm ET kickoff is Monday in UTC and the tables
    do not agree on which day that is. An exact join silently drops night
    games. clv_tracking's own definition uses the same +/-1 day tolerance.
    """
    bl = pd.DataFrame(fetch_all(
        "book_lines",
        "game_id, home_team, away_team, commence_time, spread_home, total, "
        "updated_at", book=BOOK))
    if bl.empty:
        return picks.assign(line_at_read=np.nan)
    bl["bl_kick"] = ts(bl.commence_time)

    key = ["home_team", "away_team"]
    m = (picks[key + ["game_id", "bet_type", "kick"]]
         .merge(bl[key + ["bl_kick", "spread_home", "total"]],
                on=key, how="inner"))
    m = m[(m.bl_kick - m.kick).abs() <= pd.Timedelta(days=1)]
    m["line_at_read"] = np.where(m.bet_type == "spread", m.spread_home, m.total)
    m = (m.sort_values("bl_kick")
         .groupby(["game_id", "bet_type"]).tail(1)
         [["game_id", "bet_type", "line_at_read"]])
    return picks.merge(m, on=["game_id", "bet_type"], how="left")


def current_splits(picks):
    """The most recent pre-now ticket/money split per market."""
    sp = pd.DataFrame(fetch_all("nfl_public_splits"))
    if sp.empty:
        return picks.assign(our_tix=np.nan, our_money=np.nan,
                            splits_captured_at=None)
    sp["c"] = ts(sp.captured_at)
    sp = sp[sp.c <= pd.Timestamp.now(tz="UTC")]
    last = (sp.sort_values("c").groupby(["game_id", "bet_type"]).tail(1)
            [["game_id", "bet_type", "c", "home_bets_pct", "away_bets_pct",
              "home_money_pct", "away_money_pct", "line_at_capture"]])
    d = picks.merge(last, on=["game_id", "bet_type"], how="left")
    # home column = over, per action_splits.outcomes(). So over reads the home
    # slot and under the away slot; the same mapping season_record.py and
    # nfl_threshold_sweep.py use. Getting this backwards inverts every total.
    home = d.predicted_side.isin(["home", "over"])
    d["our_tix"] = np.where(home, d.home_bets_pct, d.away_bets_pct)
    d["our_money"] = np.where(home, d.home_money_pct, d.away_money_pct)
    d["splits_captured_at"] = d.c
    return d


def opener_age(picks):
    """How stale the line open_line was measured against.

    Read from the nfl_pick_opener_age view, which does the line_history join
    in the database -- the alternative is transferring 57k snapshot rows every
    two hours to compute one number per pick.

    This is the single most important qualifier on a reading. Measured
    2026-11-12: when rule 1 fires on a FRESH opener the record is 9-10, and on
    a stale one 8-3, with average CLV -0.02 against +1.57. A big
    move_toward_us on a 60-day-old opener is a lookahead correction that
    already happened, not an edge still on the table.
    """
    a = pd.DataFrame(fetch_all("nfl_pick_opener_age",
                               "game_id, bet_type, opener_age_days"))
    if a.empty:
        return picks.assign(opener_age_days=np.nan)
    return picks.merge(a, on=["game_id", "bet_type"], how="left")


def main():
    dry = "--dry-run" in sys.argv
    within = WITHIN_HOURS
    if "--within" in sys.argv:
        within = float(sys.argv[sys.argv.index("--within") + 1])

    # Floored to the minute, matching nfl_public_splits.captured_at, so two
    # runs inside the same minute collapse onto one row instead of recording
    # the same observation twice.
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    picks = upcoming_picks(within)
    print(f"{len(picks)} qualifying NFL market(s) kicking within {within:.0f}h")
    if picks.empty:
        print("  nothing upcoming; no readings to take")
        return

    d = opener_age(current_splits(current_lines(picks)))
    d["move_toward_us"] = toward_us(d.predicted_side, d.open_line,
                                    d.line_at_read)
    d["hours"] = (d.kick - pd.Timestamp(now)).dt.total_seconds() / 3600

    rows = []
    for r in d.itertuples():
        if not np.isfinite(r.move_toward_us):
            continue  # no line to read; recorded as absent rather than 0
        rows.append({
            "game_id": r.game_id, "bet_type": r.bet_type,
            "read_at": now.isoformat(),
            "commence_time": r.kick.isoformat(),
            "hours_to_kick": round(float(r.hours), 2),
            "predicted_side": r.predicted_side,
            "open_line": float(r.open_line),
            "line_at_read": float(r.line_at_read),
            "move_toward_us": round(float(r.move_toward_us), 3),
            "our_tix": None if pd.isna(r.our_tix) else float(r.our_tix),
            "our_money": None if pd.isna(r.our_money) else float(r.our_money),
            "splits_captured_at": (None if pd.isna(r.splits_captured_at)
                                   else r.splits_captured_at.isoformat()),
            # Audit only -- see the column comment. Movement is DraftKings.
            "line_at_splits": (None if pd.isna(r.line_at_capture)
                               else float(r.line_at_capture)),
            "opener_age_days": (None if pd.isna(r.opener_age_days)
                                else round(float(r.opener_age_days), 1)),
            "move_bar": MOVE_BAR, "ticket_bar": TICKET_BAR,
        })

    got_line = len(rows)
    got_tix = sum(1 for x in rows if x["our_tix"] is not None)
    fires = [x for x in rows if x["our_tix"] is not None
             and x["move_toward_us"] >= MOVE_BAR
             and x["our_tix"] < TICKET_BAR]
    print(f"  {got_line}/{len(d)} have a {BOOK} line, "
          f"{got_tix}/{got_line} also have splits")
    print(f"  gate fires on {len(fires)} market(s) right now")
    for x in sorted(fires, key=lambda x: x["hours_to_kick"]):
        print(f"    FIRES  {x['game_id']:<18} {x['bet_type']:<7}"
              f" {x['predicted_side']:<5} move {x['move_toward_us']:+5.2f}"
              f"  tix {x['our_tix']:4.1f}%  T-{x['hours_to_kick']:.0f}h")

    # Only imminent gaps are worth alarming on. A missing reading two weeks
    # out means the market is not posted yet; a missing one inside 36 hours
    # means a bet would be placed blind.
    blind = d[(d.hours < ALARM_HOURS)
              & (~np.isfinite(d.move_toward_us) | d.our_tix.isna())]
    if not blind.empty:
        print(f"  WARNING: {len(blind)} market(s) kicking within "
              f"{ALARM_HOURS}h have no complete reading: "
              f"{', '.join(blind.game_id.head(6))}")

    if dry:
        print("  --dry-run: nothing written")
        return
    if not rows:
        print("  no readable markets; nothing written")
        return

    for i in range(0, len(rows), 500):
        supabase.table("nfl_gate_readings").upsert(
            rows[i:i + 500],
            on_conflict="game_id,bet_type,read_at").execute()
    print(f"  wrote {len(rows)} reading(s) at {now.isoformat()}")


if __name__ == "__main__":
    main()
