// Has the number moved for us or against us?
//
// Shared by the NFL and college panels deliberately. This is a sign convention,
// and sign conventions duplicated across files are how this project has
// repeatedly ended up with two subtly different answers to the same question.
//
// clv_points already encodes the answer, and encodes it correctly for all four
// bet types, which is why the movement is NOT re-derived here:
//
//     home / under   clv = open - close   (profits when the number falls)
//     away / over    clv = close - open   (profits when the number rises)
//
// So positive CLV means we hold a better number than the market does now --
// the line moved toward our side. Reading the sign of actual_movement instead
// would be wrong half the time, because which direction is good depends
// entirely on which side was taken.

// A move smaller than a full point is not worth colouring: markets tick in
// half-points, so a single tick is noise rather than the market disagreeing.
export const MOVE_HIGHLIGHT_MIN = 1.0

export function moveTone(row) {
  // No side taken (the college models split) means there is no "us" for the
  // number to have moved toward.
  if (row.predicted_side == null) return null
  if (row.actual_movement == null) return null
  if (Math.abs(row.actual_movement) < MOVE_HIGHLIGHT_MIN) return null
  if (row.clv_points == null) return null
  if (row.clv_points > 0) return 'good'
  if (row.clv_points < 0) return 'bad'
  return null
}

export function moveClass(row, neutral = 'text-gray-400') {
  const t = moveTone(row)
  return t === 'good' ? 'text-green-400 font-medium'
       : t === 'bad' ? 'text-red-400 font-medium'
       : neutral
}

export function moveTitle(row) {
  const t = moveTone(row)
  if (!t) return undefined
  // "Moved away from under" reads badly; totals want an article.
  const side = row.predicted_side === 'home' ? row.home_team
             : row.predicted_side === 'away' ? row.away_team
             : `the ${row.predicted_side}`
  return t === 'good'
    ? `Moved toward ${side} — the number we hold is better than the market's now`
    : `Moved away from ${side} — the market has a better number than the one we hold`
}

// ---------------------------------------------------------------------------
// Adverse movement
// ---------------------------------------------------------------------------
//
// A frozen pick is graded at the number it was taken at, which is the honest
// way to measure CLV but hides a question the bettor actually faces: the
// market has run several points away from this side, so is the pick still
// worth making at today's number?
//
// Measured on all 811 graded spread picks in movement_history (2023-2025),
// bucketed by how far the line moved before kickoff. Record AT THE OPENING
// NUMBER, so this is not a CLV artefact -- it is whether the pick won:
//
//     ran away 6+ pts      1-5    16.7%
//     ran away 4-6         3-10   23.1%
//     ran away 2-4        23-29   44.2%
//     ran away 0-2        94-86   52.2%
//     flat                81-67   54.7%
//     came to us 0-2     158-121  56.6%
//     came to us 2-4      56-47   54.4%
//     came to us 4+       22-8    73.3%
//
// Combined: ran away 4+ is 4-15 (21%, p=0.019); everything else is 411-329
// (55.5%, p=0.003). Eight buckets, monotonic, symmetric at both ends. Totals
// show the same shape -- ran away 2+ is 26-50 (34.2%, p=0.008).
//
// The model's entire measured edge lives in games the market did NOT move
// against. Partly that is self-fulfilling: a six-point move happens on news,
// and news is the input the pipeline is weakest on. That is the mechanism, not
// a confound, which is exactly why it is worth a badge.
//
// DISPLAY ONLY. This changes no pick, no tier and no qualifying flag.
export const ADVERSE_WARN = 3.0
export const ADVERSE_SEVERE = 6.0

export function adverse(row) {
  if (row.predicted_side == null) return null
  if (row.clv_points == null) return null
  const against = -row.clv_points
  if (against < ADVERSE_WARN) return null
  const severe = against >= ADVERSE_SEVERE
  const side = row.predicted_side === 'home' ? row.home_team
             : row.predicted_side === 'away' ? row.away_team
             : row.predicted_side
  return {
    level: severe ? 'severe' : 'warn',
    label: `−${against.toFixed(1)}`,
    pts: against,
    title:
      `The market has moved ${against.toFixed(1)} points away from ${side} ` +
      `since this pick was frozen.\n\n` +
      `Historically (811 graded spread picks, 2023-2025) picks the market ` +
      `ran ${severe ? '6+' : '4+'} points away from went ` +
      `${severe ? '1-5 (17%)' : '4-15 (21%, p=0.02)'} at the number they were ` +
      `taken at, against 55.5% for every other pick.\n\n` +
      `A move this size is usually news the model has not priced. ` +
      `Display only — this does not change the pick or the tier.`,
  }
}

export function adverseClass(a) {
  return a?.level === 'severe'
    ? 'text-red-300 bg-red-950/60 border-red-800'
    : 'text-orange-300 bg-orange-950/50 border-orange-800'
}
