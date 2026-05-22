import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { supabase } from '../lib/supabase'

const TIERS = ['A', 'B', 'C', 'watch']

function calcStats(bets) {
  const closed = bets.filter(b => b.result !== 'pending')
  const wins = closed.filter(b => b.result === 'win').length
  const losses = closed.filter(b => b.result === 'loss').length
  const pushes = closed.filter(b => b.result === 'push').length
  const units = closed.reduce((acc, b) => {
    if (b.result === 'win') return acc + (b.units ?? 1) * (100 / 110)
    if (b.result === 'loss') return acc - (b.units ?? 1)
    return acc
  }, 0)
  const roi = (wins + losses) > 0 ? (units / (wins + losses)) * 100 : 0
  const avgClv = closed.filter(b => b.clv != null).reduce((acc, b, _, arr) => acc + b.clv / arr.length, 0)
  return { wins, losses, pushes, units, roi, avgClv, total: closed.length }
}

export default function PerformancePanel({ bets }) {
  const [updating, setUpdating] = useState(null)

  const stats = calcStats(bets)
  const pending = bets.filter(b => b.result === 'pending')

  // Cumulative units chart
  const sorted = [...bets].filter(b => b.result !== 'pending').sort((a, b) => new Date(a.logged_at) - new Date(b.logged_at))
  let cumUnits = 0
  const chartData = sorted.map(b => {
    if (b.result === 'win') cumUnits += (b.units ?? 1) * (100 / 110)
    if (b.result === 'loss') cumUnits -= (b.units ?? 1)
    return { week: `Wk${b.week}`, units: parseFloat(cumUnits.toFixed(2)) }
  })

  async function updateResult(betId, result) {
    setUpdating(betId)
    await supabase.from('bets').update({ result }).eq('id', betId)
    setUpdating(null)
    window.location.reload()
  }

  return (
    <div className="p-4 space-y-6 max-w-4xl mx-auto">

      {/* Overall record */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: 'Record', value: `${stats.wins}-${stats.losses}${stats.pushes ? `-${stats.pushes}` : ''}` },
          { label: 'Units', value: `${stats.units >= 0 ? '+' : ''}${stats.units.toFixed(1)}u`, color: stats.units >= 0 ? 'text-green-400' : 'text-red-400' },
          { label: 'ROI', value: `${stats.roi >= 0 ? '+' : ''}${stats.roi.toFixed(1)}%`, color: stats.roi >= 0 ? 'text-green-400' : 'text-red-400' },
          { label: 'Avg CLV', value: stats.avgClv ? `${stats.avgClv >= 0 ? '+' : ''}${stats.avgClv.toFixed(2)}` : '—' },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-gray-900 rounded border border-gray-800 p-3">
            <div className="text-xs text-gray-500 mb-1">{label}</div>
            <div className={`text-xl font-bold tabular-nums ${color ?? 'text-gray-100'}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Cumulative units chart */}
      {chartData.length > 1 && (
        <div className="bg-gray-900 rounded border border-gray-800 p-4">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Cumulative Units</h3>
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={chartData}>
              <XAxis dataKey="week" tick={{ fontSize: 9, fill: '#6b7280' }} />
              <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} width={35} />
              <Tooltip contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }} />
              <Line type="monotone" dataKey="units" stroke="#22c55e" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* By confidence tier */}
      <div className="bg-gray-900 rounded border border-gray-800 p-4">
        <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">By Confidence Tier</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-600 border-b border-gray-800">
              <th className="text-left pb-2">Tier</th>
              <th className="text-right pb-2">W-L</th>
              <th className="text-right pb-2">Units</th>
              <th className="text-right pb-2">ROI</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {TIERS.map(tier => {
              const tierBets = bets.filter(b => b.confidence_tier === tier)
              const s = calcStats(tierBets)
              const label = { A: '🔥 A', B: '⭐ B', C: '📊 C', watch: '👀 Watch' }[tier]
              return (
                <tr key={tier}>
                  <td className="py-2 text-gray-300">{label}</td>
                  <td className="py-2 text-right text-gray-300">{s.wins}-{s.losses}</td>
                  <td className={`py-2 text-right tabular-nums ${s.units >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {s.units >= 0 ? '+' : ''}{s.units.toFixed(1)}
                  </td>
                  <td className={`py-2 text-right tabular-nums ${s.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {s.roi >= 0 ? '+' : ''}{s.roi.toFixed(1)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* By bet type */}
      <div className="bg-gray-900 rounded border border-gray-800 p-4">
        <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">By Bet Type</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-600 border-b border-gray-800">
              <th className="text-left pb-2">Type</th>
              <th className="text-right pb-2">W-L</th>
              <th className="text-right pb-2">ROI</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {['spread', 'total'].map(type => {
              const s = calcStats(bets.filter(b => b.bet_type === type))
              return (
                <tr key={type}>
                  <td className="py-2 text-gray-300 capitalize">{type}s</td>
                  <td className="py-2 text-right text-gray-300">{s.wins}-{s.losses}</td>
                  <td className={`py-2 text-right tabular-nums ${s.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {s.roi >= 0 ? '+' : ''}{s.roi.toFixed(1)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Pending bets — update result */}
      {pending.length > 0 && (
        <div className="bg-gray-900 rounded border border-gray-800 p-4">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Pending Results ({pending.length})</h3>
          <div className="space-y-2">
            {pending.map(b => (
              <div key={b.id} className="flex items-center justify-between text-xs">
                <span className="text-gray-300">{b.away_team} @ {b.home_team} — {b.bet_type} {b.side}</span>
                <div className="flex gap-1">
                  {['win', 'loss', 'push'].map(r => (
                    <button
                      key={r}
                      disabled={updating === b.id}
                      onClick={() => updateResult(b.id, r)}
                      className={`px-2 py-0.5 rounded border text-xs transition-colors ${
                        r === 'win' ? 'border-green-800 text-green-400 hover:bg-green-950'
                        : r === 'loss' ? 'border-red-800 text-red-400 hover:bg-red-950'
                        : 'border-gray-700 text-gray-400 hover:bg-gray-800'
                      }`}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
