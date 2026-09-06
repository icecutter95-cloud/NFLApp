// How public money lines up with the side we took.
//
// Shared by the NFL and college panels so the two cannot drift, for the same
// reason the movement colouring is shared: it is a convention, not a format.
//
// The tone ordering says how the money sits relative to our side. It is NOT a
// claim that any bucket wins more often -- there is no evidence of that yet,
// which is the whole reason the captures are being stored.
export const SIGNAL_TONE = {
  'reverse line movement':           'text-green-400',
  'sharp agreement':                 'text-green-400',
  'money with us, line still':       'text-green-400/80',
  'public side, number likely gone': 'text-amber-300',
  'money against us, line still':    'text-amber-300',
  'public trap':                     'text-red-400',
  'against the money':               'text-red-400',
  'neutral':                         'text-gray-400',
  'no move yet':                     'text-gray-500',
}

// A one-line plain reading, for a tooltip or the expanded row.
export function signalText(sig) {
  if (!sig || !sig.signal) return null
  const side = sig.predicted_side
  const team = side === 'home' ? sig.home_team
             : side === 'away' ? sig.away_team
             : `the ${side}`
  const gap = sig.handle_gap
  const gapText = gap == null ? ''
    : gap <= -10 ? ` The money is ${Math.abs(gap)} points lighter on ${team} than the tickets are — bigger bets are on the other side.`
    : gap >= 10  ? ` The money is ${gap} points heavier on ${team} than the tickets are — fewer, larger bets back it.`
    : ''
  return `${team} has ${sig.our_bets_pct}% of tickets and ${sig.our_money_pct}% of money.${gapText}`
}
