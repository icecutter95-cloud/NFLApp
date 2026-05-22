import { Zap, RefreshCw } from 'lucide-react'

export default function FilterBar({ filters, onChange }) {
  function set(key, val) {
    onChange(prev => ({ ...prev, [key]: val }))
  }

  return (
    <div className="sticky top-[49px] z-40 bg-gray-950 border-b border-gray-800 px-4 py-2 flex flex-wrap items-center gap-3">

      {/* Sort */}
      <div className="flex items-center gap-1">
        <span className="text-gray-500 text-xs">Sort</span>
        <select
          value={filters.sortBy}
          onChange={e => set('sortBy', e.target.value)}
          className="bg-gray-900 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1"
        >
          <option value="ev_pct">EV%</option>
          <option value="edge">Edge pts</option>
          <option value="game_time">Game time</option>
          <option value="confidence">Confidence</option>
        </select>
      </div>

      {/* Bet type */}
      <div className="flex rounded overflow-hidden border border-gray-700">
        {['all', 'spread', 'total'].map(t => (
          <button
            key={t}
            onClick={() => set('betType', t)}
            className={`px-3 py-1 text-xs transition-colors ${
              filters.betType === t
                ? 'bg-gray-700 text-white'
                : 'bg-gray-900 text-gray-400 hover:bg-gray-800'
            }`}
          >
            {t === 'all' ? 'All' : t === 'spread' ? 'Spreads' : 'Totals'}
          </button>
        ))}
      </div>

      {/* Confidence */}
      <div className="flex rounded overflow-hidden border border-gray-700">
        {['all', 'A', 'B', 'C', 'watch'].map(t => (
          <button
            key={t}
            onClick={() => set('confidence', t)}
            className={`px-3 py-1 text-xs transition-colors ${
              filters.confidence === t
                ? 'bg-gray-700 text-white'
                : 'bg-gray-900 text-gray-400 hover:bg-gray-800'
            }`}
          >
            {t === 'all' ? 'All' : t === 'A' ? '🔥A' : t === 'B' ? '⭐B' : t === 'C' ? '📊C' : '👀'}
          </button>
        ))}
      </div>

      {/* Min edge slider */}
      <div className="flex items-center gap-2">
        <span className="text-gray-500 text-xs">Min edge</span>
        <input
          type="range" min={0} max={5} step={0.5}
          value={filters.minEdge}
          onChange={e => set('minEdge', Number(e.target.value))}
          className="w-20 accent-green-500"
        />
        <span className="text-gray-300 text-xs w-8">{filters.minEdge}+</span>
      </div>

      {/* Signal toggles */}
      <button
        onClick={() => set('steamOnly', !filters.steamOnly)}
        className={`flex items-center gap-1 px-3 py-1 text-xs rounded border transition-colors ${
          filters.steamOnly
            ? 'border-yellow-500 text-yellow-400 bg-yellow-950'
            : 'border-gray-700 text-gray-500 hover:border-gray-500'
        }`}
      >
        <Zap size={11} /> Steam
      </button>

      <button
        onClick={() => set('rlmOnly', !filters.rlmOnly)}
        className={`flex items-center gap-1 px-3 py-1 text-xs rounded border transition-colors ${
          filters.rlmOnly
            ? 'border-blue-500 text-blue-400 bg-blue-950'
            : 'border-gray-700 text-gray-500 hover:border-gray-500'
        }`}
      >
        <RefreshCw size={11} /> RLM
      </button>
    </div>
  )
}
