"""
Has the college prediction history been restated?

cfb_predictions is an upsert table: it holds only what the model says NOW.
cfb_prediction_log is append-only and records every change, with the hash of
the model files that produced it. This script reads the log and answers the
questions the upsert table cannot:

  1. Did any prediction change AFTER its kickoff? A model is a function of the
     opener and static team ratings, so the answer must be no. Anything here is
     either a retrain applied to settled games or a bug.
  2. Which rows are evidence of a pre-kickoff call, and which are backfill? A
     row first written after the game started proves the number is
     reproducible, not that it was on screen in time.
  3. Have the model weights changed, and when?

Exit code 1 if anything in (1) is found, so it can be wired to a workflow.

Usage:
    python scripts/check_cfb_prediction_log.py
    python scripts/check_cfb_prediction_log.py --changes    # list every change
"""

import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from score_week import fetch_all  # noqa: E402


def main():
    l = pd.DataFrame(fetch_all("cfb_prediction_log"))
    if l.empty:
        print("cfb_prediction_log is empty — run log_cfb_predictions.py")
        return 0
    l["logged"] = pd.to_datetime(l.logged_at, utc=True, format="ISO8601")
    l["kick"] = pd.to_datetime(l.commence_time, utc=True, format="ISO8601")
    games = l.groupby(["game_id", "bet_type"]).ngroups
    print(f"{len(l)} log rows over {games} game-markets, "
          f"{l.logged.min():%Y-%m-%d} to {l.logged.max():%Y-%m-%d}")
    for note, n in l.note.value_counts().items():
        print(f"  {n:4d}  {note}")

    print("\nmodel weights seen:")
    for h, g in l.groupby("model_hash"):
        print(f"  {h}  {len(g):4d} rows, first {g.logged.min():%Y-%m-%d %H:%M}, "
              f"last {g.logged.max():%Y-%m-%d %H:%M}")
    if l.model_hash.nunique() > 1:
        print("  !! more than one model hash: predictions logged under the older")
        print("     hash were made by different weights. Any record that mixes")
        print("     them is not a single model's track record.")

    # A change logged after kickoff means settled history moved.
    after = l[(l.note == "changed") & (l.logged > l.kick)]
    print(f"\npredictions changed after kickoff: {len(after)}")
    if len(after):
        cols = ["home_team", "away_team", "bet_type", "logged_at",
                "open_line", "predicted_side", "model_hash"]
        print(after[cols].to_string(index=False))
        print("\n  !! settled games were restated. Compare the model hashes above")
        print("     and treat any record spanning the change as two records.")

    ev = l[l.note == "first observation"]
    bf = l[l.note == "backfill after kickoff"]
    print(f"\nevidenced pre-kickoff: {len(ev)} game-markets"
          f"  |  reproducible but not evidenced: {len(bf)}")
    if len(bf):
        print("  (the log began 2026-09-25; everything played before that is"
              " backfill by construction, not a data fault)")

    if "--changes" in sys.argv:
        ch = l[l.note == "changed"].sort_values("logged")
        print(f"\n{len(ch)} changes:")
        for _, r in ch.iterrows():
            print(f"  {r.logged:%m-%d %H:%M}  {r.away_team} @ {r.home_team} "
                  f"{r.bet_type}: open {r.open_line:+.1f} now {r.line_now:+.1f} "
                  f"-> {r.predicted_side or 'no pick'}")
    return 1 if len(after) else 0


if __name__ == "__main__":
    sys.exit(main())
