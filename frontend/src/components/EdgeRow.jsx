import { useState } from 'react'
import { ChevronDown, ChevronUp, Zap, RefreshCw, AlertTriangle } from 'lucide-react'
import GameDetailPanel from './GameDetailPanel'

const TIER_CONFIG = {
  A: { label: '🔥 A', className: 'text-orange-400 bg-orange-950 border-orange-800' },
  B: { label: '⭐ B', className: 'text-yellow-400 bg-yellow-950 border-yellow-800' },
  C: { label: '📊 C', className: 'text-blue-400 bg-blue-950 border-blue-800' },
  watch: { label: '👀', className: 'text-gray-400 bg-gray-900 border-gray-700' },
}

function formatTime(ts) {
  if (!ts) return '—'
  try {
    return new Date(ts).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
  } catch { return '—' }
}

function formatLine(val, betType, side) {
  if (val == null) return '—'
  if (betType === 'total') return val.toFixed(1)
  return val >= 0 ? `+${val.toFixed(1)}` : val.toFixed(1)
}

export default function EdgeRow({ projection: p, onBetLogged }) {
  const [expanded, setExpanded] = useState(false)
  const tier = TIER_CONFIG[p.confidence_tier] ?? TIER_CONFIG.watch

  const edgePositive = (p.edge_points ?? 0) > 0
  const evPositive = (p.ev_pct ?? 0) > 0

  const sideLabel = p.bet_type === 'total'
    ? p.side?.toUpperCase()
    : p.side === p.home_team
      ? `${p.home_team} (H)`
      : `${p.away_team} (A)`

  return (
    <div className={`border-b border-gray-900 ${p.conflict_flag ? 'bg-red-950/20' : ''}`}>
      {/* Main row */}
      <div
        className="grid grid-cols-[2fr_1fr_80px_1fr_80px_80px_80px_80px_100px_1fr] gap-2 px-4 py-3 items-center cursor-pointer hover:bg-gray-900/50 transition-colors text-sm"
        onClick={() => setExpanded(e => !e)}
      >
        {/* Game */}
        <div className="min-w-0">
          <span className="text-gray-100 font-medium">{p.away_team} @ {p.home_team}</span>
          <span className="text-gray-600 text-xs ml-2">Wk {p.week}</span>
        </div>

        {/* Time */}
        <div className="text-gray-400 text-xs">{formatTime(p.game_time)}</div>

        {/* Type badge */}
        <div>
          <span className={`text-xs px-2 py-0.5 rounded font-medium ${
            p.bet_type === 'spread'
              ? 'bg-purple-950 text-purple-300 border border-purple-800'
              : 'bg-cyan-950 text-cyan-300 border border-cyan-800'
          }`}>
            {p.bet_type === 'spread' ? 'SPREAD' : 'TOTAL'}
          </span>
        </div>

        {/* Side */}
        <div className="text-gray-200 font-medium text-xs">{sideLabel}</div>

        {/* Model line */}
        <div className="text-gray-300 text-xs tabular-nums">
          {formatLine(p.model_line, p.bet_type, p.side)}
        </div>

        {/* DK line */}
        <div className="text-gray-500 text-xs tabular-nums">
          {formatLine(p.dk_line, p.bet_type, p.side)}
        </div>

        {/* Edge */}
        <div className={`text-xs tabular-nums font-medium ${edgePositive ? 'text-green-400' : 'text-red-400'}`}>
          {edgePositive ? '+' : ''}{(p.edge_points ?? 0).toFixed(1)}
        </div>

        {/* EV% */}
        <div className={`text-xs tabular-nums font-bold ${evPositive ? 'text-green-400' : 'text-gray-500'}`}>
          {evPositive ? '+' : ''}{((p.ev_pct ?? 0) * 100).toFixed(1)}%
        </div>

        {/* Tier badge */}
        <div>
          <span className={`text-xs px-2 py-0.5 rounded border font-medium ${tier.className}`}>
            {tier.label}
          </span>
        </div>

        {/* Signals */}
        <div className="flex items-center gap-1">
          {p.steam_flag && (
            <span className="flex items-center gap-0.5 text-xs text-yellow-400 bg-yellow-950 border border-yellow-800 px-1.5 py-0.5 rounded">
              <Zap size={10} /> Steam
            </span>
          )}
          {p.rlm_flag && (
            <span className="flex items-center gap-0.5 text-xs text-blue-400 bg-blue-950 border border-blue-800 px-1.5 py-0.5 rounded">
              <RefreshCw size={10} /> RLM
            </span>
          )}
          {p.conflict_flag && (
            <span className="flex items-center gap-0.5 text-xs text-red-400 bg-red-950 border border-red-800 px-1.5 py-0.5 rounded">
              <AlertTriangle size={10} /> Conflict
            </span>
          )}
          <span className="ml-auto text-gray-700">
            {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </span>
        </div>
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <GameDetailPanel projection={p} onBetLogged={onBetLogged} />
      )}
    </div>
  )
}
