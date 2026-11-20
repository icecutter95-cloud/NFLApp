# What the schedules in this directory actually do

Read this before trusting a `cron:` line here, and before raising one.

**GitHub delivers roughly a third of the scheduled runs these workflows ask
for, and raising the cron frequency does not increase delivery.** Measured
2026-11-19 over the 14 days from 2026-11-05. Every dropped run is invisible:
GitHub does not fail them, it declines to start them, so there is nothing in
the Actions tab to see and no notification. The repo is public, so this is not
a billing limit -- Actions minutes are free and unlimited here -- and there are
no `concurrency` blocks. It is GitHub's documented best-effort scheduling for
`schedule` triggers.

| workflow | cron | expected / 14d | actual | delivered |
|---|---|---|---|---|
| `collect-splits` | every 2h | 168 | 48 | 29% |
| `log-cfb` | every 3h | 112 | 43 | 38% |
| `log-clv` | every 3h | 112 | 44 | 39% |
| `capture-td-props` | every 2h | ~170 | 51 | 30% |

## Asking more often delivers less, not more

In October the two **2-hourly** workflows (`collect-splits` 3.45/day,
`capture-td-props` 3.68/day) got *fewer* absolute runs than the two **3-hourly**
ones (`log-cfb` 4.0/day, `log-clv` 3.87/day). The cron interval barely matters.

## Why: the repo is woken in waves, not per workflow

`log-clv` (cron `:30`) and `log-cfb` (cron `:45`) fire **6-17 minutes apart,
every single time** -- about their configured 15-minute offset. GitHub wakes
this repo's scheduler roughly three times a day and then releases one run of
each due workflow in a batch.

```
11-18T01:05  log-clv          11-19T02:41  log-clv
11-18T01:17  log-cfb          11-19T02:47  log-cfb
11-18T03:54  collect-splits   11-19T03:51  collect-splits
11-18T11:59  log-clv          11-19T11:56  log-clv
11-18T12:12  log-cfb          11-19T12:06  log-cfb
```

**This is why consolidating these workflows into one would not raise their
frequency.** Three waves a day times one workflow is the same ~3 runs/day as
three waves times four workflows. Per-workflow delivery was somewhat better
when the repo had fewer workflows (Sept, 2 workflows: 4.6/day each; Oct, 4
workflows: 3.8/day; Nov: 3.2/day), so merging might reach ~5/day -- a modest
gain, not the 12/day a 2-hourly cron implies. Consolidation is worth doing for
ordering and compute, not for cadence.

## pg_cron is the only scheduler here that keeps its promises

`refresh-odds` (pg_cron, hourly, the only writer of `line_history`) delivered
304 of 336 possible snapshot-hours over the same 14 days: **90%**. pg_cron lives
in the database and nothing throttles it. See `supabase/cron_jobs.sql`.

The catch is that pg_cron runs SQL, and Supabase Edge Functions are
Deno/TypeScript. There is no Python. So moving a job here to Supabase means
rewriting it in SQL or TS, and for anything that imports `cfb_teams.py` that
means a **second copy of the team crosswalk** -- which is the exact shape of the
UL Monroe hyphen bug (Action sends `UL-Monroe Warhawks`, the crosswalk had
`UL Monroe Warhawks`, `to_key()` returned `None`, and the game was silently
counted non-FBS). `log_cfb_predictions.py` imports `cfb_teams` and cannot leave
Python, so that copy can never be deleted. Two crosswalks would have to be
maintained forever, and the one that falls behind fails silently by design.

The exception is `scripts/log_gate_readings.py`: it reads only Postgres
(`clv_tracking`, `book_lines`, `nfl_public_splits`, `nfl_pick_opener_age`) and
writes only Postgres (`nfl_gate_readings`). No external API, no model, no
Python-only library. It could be a pg_cron SQL job with no edge function at
all, and its primary input (`book_lines`) is already written hourly by pg_cron
-- an hourly-fresh table currently being read 2-3 times a day. Not done, and
not to be done without asking, because it would loosen the deliberate ordering
in `collect-splits.yml` (gate readings run *after* NFL splits so rule 2 reads
the splits just captured).

## What the throttling actually costs

Measured, so that nobody pays to fix a problem that is not there:

* **NFL odds: nothing.** The DraftKings spread changes in **2.7%** of
  consecutive hourly snapshots (134 of 4,969). Hourly already oversamples the
  thing it watches by ~37x.
* **The closing number: almost nothing.** Across 86 graded games the spread
  moves **0.093 points on average over the final ~3 hours**, with 7 of 86
  moving half a point or more. The NFL close is captured 0.75h before kickoff
  (median 1.0, worst 3.3).
* **CFB lines: the one real degradation, and it is small.** Snapshot gaps
  average **8.3h, max 12h, and not one gap in 14 days honoured the configured
  3h**. The CFB close lands 3.28h pre-kickoff (median 3.03, worst 7.5) against
  the NFL's 0.75h -- worth ~0.09 points by the number above.
* **Gate readings: 2-3 per day, not the 12 the 2-hourly cron implies.** This
  degrades an audit trail that nothing reads yet; it resolves "when did the gate
  first fire" to ~8h instead of ~2h, on a question answered once per bet at a
  6-7 day horizon.

## Deliberate duplication is the only lever that works

Because delivery is per-wave and not per-cron, **the way to capture something
more often is to run it in a second workflow, not to ask for it more often.**

`scripts/fetch_cfb_splits.py` runs in `collect-splits.yml` (as "College
splits") and `log-cfb.yml` (as "Fetch public betting splits"), and also lands
at a fixed 04:00 and 16:00 UTC from a path outside GitHub:
`scripts/collect_splits.bat`, run by a Windows scheduled task named "Betting
splits" at 09:00 and 21:00 local. Confirmed 2026-11-20: state Ready, last
result 0. Three paths in all.

The two kinds of path are equally thorough. Over the 7 days to 2026-11-20 the
GitHub waves averaged **103.3 rows per capture** across 45 captures and the
scheduled task **99.7** across 14 -- about 6.4 captures a day from CI and 2
from the task. An earlier version of this file claimed the task's captures
were far larger (345-365 against ~107); that was an artifact of grouping by
clock time, since 04:00 and 16:00 recur daily and accumulate while each
throttled GitHub timestamp appears once.

The redundancy started as an accident and is now load-bearing; merging the
workflows would quietly drop it.

`scripts/fetch_cfb_odds.py` had exactly one path until 2026-11-20 and now runs
in both workflows too, for a measured reason. Over the 4 days to 2026-11-20:

| | distinct captures / 4d | per day |
|---|---|---|
| CFB splits | 34 | ~8.5 |
| CFB lines | 11 | ~2.75 |

The lines refreshed a third as often as the tickets beside them. On 2026-11-20
the `WYOMING`/`NEW_MEXICO` spread in `cfb_line_history` still read 7.5 from a
pull four days earlier, while the 16:00 splits capture carried 9.0 -- the
difference between reading rule 1 as +3.0 (passes easily) and +1.5 (exactly at
the bar). Adding the pull to `collect-splits.yml` roughly doubles line captures
and, more usefully, puts the line and the tickets in the same run seconds
apart. Cost: 2 Odds API credits per extra call, ~180/month.

`cfb_line_history` is append-only -- surrogate primary key, `recorded_at`
defaults to `now()` -- so the second pull can never overwrite an opener or
conflict with the first. Note it is `continue-on-error` in `collect-splits.yml`
and deliberately not in `log-cfb.yml`: there a failed odds pull should stop the
run, because the predictions after it would be meaningless, whereas here the
NFL gate readings come afterwards and must not be skipped for it.

### Do not compare a line from one source against a line from another

The splits carry `line_at_capture`, which is Action Network's consensus, not
DraftKings. Measured 2026-11-12 over 42 upcoming markets: mean 0.15 pts on
spreads, 0.25 on totals, max 1.00, with 5 markets a full point apart -- enough
to cross a 1.5-point bar on source disagreement alone. `log_gate_readings.py`
therefore computes movement from DraftKings and stores `line_at_splits` for
audit only. A stale DraftKings line is not a licence to substitute Action's;
the fix is a fresher DraftKings pull, which is what the step above is for.

## Open question, not yet measured

`log-cfb.yml` cites the +1.55-CLV-in-six-hours finding
(`scripts/screen_bet_timing.py`) as its reason for running often. With real
gaps up to 12h, a CFB board can sit unlogged long enough for the frozen
"opener" to have already absorbed that move. An attempt to measure this on
2026-11-19 was confounded: `cfb_line_history` sees the whole board weeks before
a game enters the prediction window, so first-snapshot-to-freeze came out at a
meaningless 262h median. Unresolved, and the same shape as the stale-opener
problem that overturned the NFL gated record (see `scripts/bet_time_gate.py`).

## Verify

```bash
gh run list --workflow=log-clv.yml --limit 200 --json createdAt,event \
  --jq '[.[] | select(.event=="schedule")] | length'
```
