import EdgeRow from './EdgeRow'

export default function EdgeList({ projections, loading, onBetLogged, filters }) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
        Loading projections...
      </div>
    )
  }

  if (projections.length === 0) {
    // Distinguish "filtered everything out" from "no data at all" -- with both
    // models gated off, "Bets only" legitimately matches nothing, and that
    // shouldn't look like a broken pipeline.
    const filteredOut = filters?.recommendedOnly
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2">
        {filteredOut ? (
          <>
            <span>No recommended bets this week</span>
            <span className="text-xs text-gray-700 text-center max-w-md">
              Neither model currently clears its edge threshold, so nothing is
              flagged as a bet. Turn off “Bets only” to see the full slate and lines.
            </span>
          </>
        ) : (
          <>
            <span>No projections found</span>
            <span className="text-xs text-gray-700">Run score_week.py to generate this week's projections</span>
          </>
        )}
      </div>
    )
  }

  return (
    <div className="divide-y divide-gray-900">
      {/* Column headers */}
      <div className="hidden md:grid grid-cols-[2fr_1fr_80px_1fr_80px_80px_80px_80px_100px_120px] gap-2 px-4 py-2 text-xs text-gray-600 uppercase tracking-wider">
        <span>Game</span>
        <span>Time</span>
        <span>Type</span>
        <span>Side</span>
        <span>Model</span>
        <span>DK</span>
        <span>Edge</span>
        <span>EV%</span>
        <span>Tier</span>
        <span>Signals</span>
      </div>

      {projections.map(proj => (
        <EdgeRow key={`${proj.game_id}-${proj.bet_type}`} projection={proj} onBetLogged={onBetLogged} />
      ))}
    </div>
  )
}
