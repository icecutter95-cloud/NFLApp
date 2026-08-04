# Frozen production state — 2026 season

Tag: `prod-2026-frozen`

This is the exact state the 2026 NFL season will be bet from. It is frozen so
that the season produces **one clean prospective test** rather than another
round of "we adjusted and it improved". Research continues on separate
branches; nothing below changes until the season ends.

---

## What is frozen

### The betting rule (NFL spreads)

```
qualify when:  |margin_disagreement| >= 3.0
               AND (disagreement > 0) == (predicted_movement < 0)
bet:           the side the line is predicted to move toward
at:            the opening line, best available price across the book panel
```

There is deliberately **no movement-magnitude bar**. It was removed on
2026-08-03: the walk-forward selector rejected it in every recent fold, removing
it raised the record from 36-29 (55.4%) to 120-86 (58.3%) with volume and rate
rising together, and the rule without it passes label permutation where the
tighter rule does not.

Defined in three places that must agree:
- `scripts/log_clv_predictions.py` — `SPREAD_DISAGREE_MIN`, the loop's `q`
- `clv_tracking` view in Supabase — the `qualifies` CASE
- `scripts/build_movement_history.py` — `SPREAD_DISAGREE_MIN`, `SPREAD_MOVE_MIN`

### NFL totals

`|predicted_movement| >= 1.25`, single signal. **Tracked, not recommended.**
It fails label permutation (5/25 noise runs reach it, p = 0.231) and its noise
floor (52.6%) sits above break-even. Visible in the app with CLV tracked; the
qualifying flag should be treated as informational only.

### College football

**Not in the app.** Research only. CFB spreads sit at p = 0.077; CFB totals do
not exist.

### Models

Artifacts in Supabase Storage, all uploaded 2026-07-31T20:28 UTC:

| file | features |
|---|---|
| `movement_model.joblib` | 58 (SPREAD_FEATURES + week_open_spread_home) |
| `margin_model.joblib` | 57 (SPREAD_FEATURES) |
| `total_movement_model.joblib` | 62 (TOTAL_FEATURES + week_open_total) |

Trained on all seasons with line coverage (2020-2025). XGBoost: depth 3,
lr 0.03, reg_alpha 1.0, reg_lambda 5.0, recency-weighted 0.75^seasons_back.

### Expected performance

Plan on **54–55.5%**, not the historical 58.3%. Two independent routes give
that range: the worst leave-one-season-out case is 55.4%, and shrinkage for
selection optimism lands in the same place. Treating 58.3% or the +11.2% ROI as
the forward expectation would be a mistake.

Stake **0.20–0.35 units flat**. No Kelly, no in-season threshold changes, no
rule adaptation.

---

## What the season is actually testing

Not whether it starts 8-3 or 3-8. Over a full season:

- positive CLV against a consensus close (not just DraftKings)
- realistic execution — were the quoted prices actually available
- calibration holding
- a return consistent with the shrunk 54–55.5% estimate

---

## How to restore this exact state

```bash
git checkout prod-2026-frozen
python scripts/download_models.py          # pulls the 2026-07-31 artifacts
python scripts/check_nfl_invariants.py     # confirms the DB matches
```

The `clv_tracking` view is DB state, not code. If it has been changed, the
frozen definition is in the migration `clv_tracking_drop_movement_bar`.

Invariant baseline at freeze (`scripts/nfl_invariants.json`):

```
line_predictions 32   clv_tracking 32   movement_history 1640
game_results 285      book_lines 679    best_book_lines 272
spread select 109-63  spread holdout 49-40
total  select  71-49  total  holdout 30-23
```

---

## Rules for research from here

1. Research happens on branches, never on `main` before the season ends.
2. `check_nfl_invariants.py` must pass before and after any experiment.
3. Nothing gets promoted into production mid-season. If something looks good in
   October, it goes live in 2027.
4. A promotion candidate must clear: genuine nested walk-forward, max-statistic
   permutation against every alternative tested in its research family, minimum
   sample and season breadth, robustness to nearby specifications, and
   prospective shadow performance.

The failure mode this exists to prevent: taking a result that has finally
survived scrutiny and tuning it until it looks better.
