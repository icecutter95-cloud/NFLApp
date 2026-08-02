"""
Prove the NFL side is untouched. Run before and after any CFB work.

Why this exists
---------------
College football is being added alongside the NFL app, and the NFL half is the
part with a validated track record behind it. This session alone turned up a
playoff join that silently duplicated rows, a lookahead bias worth a fake +1.94
CLV, a decimal/American odds mismatch, and a team-pair collision that joined 32
predictions to zero books. Every one of those was found by checking rather than
assuming.

So CFB work is additive only -- new cfb_ tables, new scripts, no edits to NFL
tables, views, models or components -- and this script is the tripwire that
proves it, instead of asking anyone to take it on faith.

Usage:
    python check_nfl_invariants.py --save     # snapshot the current good state
    python check_nfl_invariants.py            # compare against the snapshot
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from score_week import supabase

SNAPSHOT = Path(__file__).parent / "nfl_invariants.json"

# Row counts that must not move, and the views the UI depends on. A CFB change
# that alters any of these has reached into the NFL pipeline.
TABLES = [
    "line_predictions", "clv_tracking", "movement_history", "game_results",
    "line_history", "book_lines", "best_book_lines", "line_open_close",
    "preseason_predictions", "preseason_lines",
]


def counts() -> dict:
    out = {}
    for t in TABLES:
        try:
            r = supabase.table(t).select("*", count="exact", head=True).execute()
            out[t] = r.count
        except Exception as exc:
            out[t] = f"ERROR: {str(exc)[:80]}"
    return out


def track_record() -> dict:
    """The published holdout figures. These are the numbers on the Model page."""
    rows = []
    for frm in range(0, 4000, 1000):
        r = supabase.table("movement_history").select(
            "bet_type, period, qualifies, result").range(frm, frm + 999).execute()
        if not r.data:
            break
        rows += r.data

    out = {}
    for bt in ("spread", "total"):
        for per in ("select", "holdout"):
            s = [x for x in rows if x["bet_type"] == bt and x["period"] == per and x["qualifies"]]
            w = sum(x["result"] == "win" for x in s)
            l = sum(x["result"] == "loss" for x in s)
            out[f"{bt}_{per}_qualifying"] = f"{w}-{l}"
    return out


def main():
    state = {"counts": counts(), "track_record": track_record()}

    if "--save" in sys.argv:
        SNAPSHOT.write_text(json.dumps(state, indent=2))
        print(f"saved baseline -> {SNAPSHOT.name}")
        for k, v in state["counts"].items():
            print(f"  {k:<26}{v}")
        for k, v in state["track_record"].items():
            print(f"  {k:<26}{v}")
        return

    if not SNAPSHOT.exists():
        print("No baseline. Run: python check_nfl_invariants.py --save")
        return

    old = json.loads(SNAPSHOT.read_text())
    drift = []
    for section in ("counts", "track_record"):
        for k, v in old[section].items():
            now = state[section].get(k)
            if now != v:
                drift.append(f"  {section}.{k}: {v} -> {now}")

    # A new key is fine (a table added); a changed value is not.
    if drift:
        print("NFL STATE CHANGED -- CFB work has reached into the NFL pipeline:")
        for d in drift:
            print(d)
        raise SystemExit(1)

    print(f"NFL side unchanged ({len(old['counts'])} tables, "
          f"{len(old['track_record'])} track-record figures verified)")
    for k, v in old["track_record"].items():
        print(f"  {k:<26}{v}")


if __name__ == "__main__":
    main()
