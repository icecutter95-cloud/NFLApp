"""
Put finished CFB predictions back to what they said BEFORE kickoff.

WHY
cfb_predictions upserts on (game_id, bet_type), so the 3-hourly job rewrote
every finished game with the current day's inputs. Harmless while the inputs
were static; outright look-ahead once in-season form was wired in on
2026-10-20, because a week-3 game was then being predicted from form that
includes its own result and every game since.

Measured before the guard went in: 381 finished games re-stamped after kickoff,
158 of them with a CHANGED SIDE. Not a timestamp artefact --
cfb_prediction_log writes only when a value differs, verified when it was built
(an identical rerun writes zero rows). JAMES_MADISON @ OLD_DOMINION, played
2026-09-26, read home/-2.71 before kickoff and away/-10.78 after.

Any record graded off the contaminated table is the model marking its own
homework.

WHAT THIS DOES
For every finished game-market, finds the last cfb_prediction_log row written
BEFORE kickoff and writes those values back, including the original
predicted_at. log_cfb_predictions now refuses to touch finished games, so the
restore sticks.

WHAT IT CANNOT DO
cfb_prediction_log began on 2026-09-25. A game that kicked off before then has
no pre-kickoff entry, so its current row cannot be trusted and cannot be
repaired -- it is reported, not silently left to look fine. Those games should
be excluded from any record until they can be re-derived from line history.

Usage:
    python scripts/restore_cfb_prekickoff.py --dry-run
    python scripts/restore_cfb_prekickoff.py
"""

import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

from score_week import supabase, fetch_all

FIELDS = ["predicted_side", "predicted_movement", "projected_value",
          "margin_disagreement", "open_line", "taken_line"]


def main():
    dry = "--dry-run" in sys.argv
    now = pd.Timestamp.now(tz="UTC")

    preds = pd.DataFrame(fetch_all("cfb_predictions"))
    log = pd.DataFrame(fetch_all("cfb_prediction_log"))
    if preds.empty or log.empty:
        print("nothing to work with")
        return
    for d in (preds, log):
        d["kick"] = pd.to_datetime(d.commence_time, utc=True, format="ISO8601")
    log["la"] = pd.to_datetime(log.logged_at, utc=True, format="ISO8601")

    finished = preds[preds.kick < now]
    print(f"{len(finished)} finished game-markets in cfb_predictions")

    pre = log[log.la < log.kick].sort_values("la")
    latest_pre = {(r.game_id, r.bet_type): r
                  for r in pre.itertuples()}

    restore, unfixable, already = [], [], 0
    for r in finished.itertuples():
        key = (r.game_id, r.bet_type)
        src = latest_pre.get(key)
        if src is None:
            unfixable.append(r)
            continue
        same = all(
            (pd.isna(getattr(src, f)) and pd.isna(getattr(r, f)))
            or (not pd.isna(getattr(src, f)) and not pd.isna(getattr(r, f))
                and (getattr(src, f) == getattr(r, f)
                     if f == "predicted_side"
                     else abs(float(getattr(src, f)) - float(getattr(r, f))) < 1e-6))
            for f in FIELDS)
        if same:
            already += 1
            continue
        rec = {"game_id": r.game_id, "bet_type": r.bet_type,
               "season": int(r.season), "week": int(r.week) if pd.notna(r.week) else None,
               "home_team": r.home_team, "away_team": r.away_team,
               "commence_time": r.commence_time,
               "predicted_at": src.logged_at}
        for f in FIELDS:
            v = getattr(src, f)
            rec[f] = None if pd.isna(v) else (v if f == "predicted_side" else float(v))
        restore.append(rec)

    print(f"  {already} already match their pre-kickoff value")
    print(f"  {len(restore)} to restore")
    print(f"  {len(unfixable)} have NO pre-kickoff log entry and cannot be "
          f"repaired (the log began 2026-09-25)")
    if unfixable:
        wk = sorted({int(r.week) for r in unfixable if pd.notna(r.week)})
        print(f"    affected weeks: {wk}")
        print(f"    exclude these from any graded record until they are "
              f"re-derived from line history")

    flips = sum(1 for rec in restore
                if rec["predicted_side"] != finished.set_index(
                    ["game_id", "bet_type"]).loc[
                        (rec["game_id"], rec["bet_type"]), "predicted_side"])
    print(f"  of the restores, {flips} put the SIDE back")

    if dry:
        print("\n--dry-run: nothing written")
        return
    for i in range(0, len(restore), 200):
        supabase.table("cfb_predictions").upsert(
            restore[i:i + 200], on_conflict="game_id,bet_type").execute()
    print(f"\nrestored {len(restore)} rows to their pre-kickoff values")


if __name__ == "__main__":
    main()
