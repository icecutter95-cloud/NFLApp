import { BarChart2 } from 'lucide-react'

export default function Header({ season, week, weeks, bets, onWeekChange, onSeasonChange, onTogglePerformance, showPerformance }) {
  const closed = bets.filter(b => b.result !== 'pending')
  const wins = closed.filter(b => b.result === 'win').length
  const losses = closed.filter(b => b.result === 'loss').length
  const units = closed.reduce((acc, b) => {
    if (b.result === 'win') return acc + (b.units ?? 1) * (100 / 110)
    if (b.result === 'loss') return acc - (b.units ?? 1)
    return acc
  }, 0)

  return (
    <header className="sticky top-0 z-50 bg-gray-950 border-b border-gray-800 px-4 py-2 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3">
        <span className="text-green-400 font-bold text-lg tracking-tight">NFL EDGE</span>
        <span className="text-gray-600 text-xs">|</span>

        <select
          value={season}
          onChange={e => onSeasonChange(Number(e.target.value))}
          className="bg-gray-900 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1"
        >
          {[2025, 2024, 2023, 2022].map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>

        <select
          value={week ?? ''}
          onChange={e => onWeekChange(e.target.value ? Number(e.target.value) : null)}
          className="bg-gray-900 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1"
        >
          <option value="">All Weeks</option>
          {weeks.map(w => (
            <option key={w} value={w}>Week {w}</option>
          ))}
        </select>
      </div>

      <div className="flex items-center gap-4">
        {closed.length > 0 && (
          <div className="text-xs text-gray-400 flex gap-3">
            <span className="text-white">{wins}-{losses}</span>
            <span className={units >= 0 ? 'text-green-400' : 'text-red-400'}>
              {units >= 0 ? '+' : ''}{units.toFixed(1)}u
            </span>
          </div>
        )}
        <button
          onClick={onTogglePerformance}
          className={`flex items-center gap-1 text-xs px-3 py-1 rounded border transition-colors ${
            showPerformance
              ? 'border-green-500 text-green-400 bg-green-950'
              : 'border-gray-700 text-gray-400 hover:border-gray-500'
          }`}
        >
          <BarChart2 size={12} />
          Performance
        </button>
      </div>
    </header>
  )
}
