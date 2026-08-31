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
