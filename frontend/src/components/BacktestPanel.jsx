import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { supabase } from '../lib/supabase'

export default function BacktestPanel() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [betType, setBetType] = useState('spread')
  const [threshold, setThreshold] = useState(1.5)

  useEffect(() => {
    supabase
      .from('backtest_results')
      .select('*')
      .order('season')
      .order('week')
      .then(({ data: rows }) => {
        setData(rows ?? [])
        setLoading(false)
      })
  }, [])

  const filtered = useMemo(() =>
    data.filter(d => d.bet_type === betType && d.edge_points >= threshold),
    [data, betType, threshold]
  )

  const overall = useMemo(() => {
    const wins   = filtered.filter(d => d.result === 'win').length
    const losses = filtered.filter(d => d.result === 'loss').length
    const pushes = filtered.filter(d => d.result === 'push').length
    const units  = filtered.reduce((acc, d) => acc + (d.units ?? 0), 0)
    const roi    = (wins + losses) > 0 ? (units / (wins + losses)) * 100 : 0
    const winPct = (wins + losses) > 0 ? (wins / (wins + losses)) * 100 : 0
    return { wins, losses, pushes, units, roi, winPct, bets: filtered.length }
  }, [filtered])

  const bySeasonStats = useMemo(() => {
    const seasons = [...new Set(filtered.map(d => d.season))].sort()
    return seasons.map(season => {
      const rows = filtered.filter(d => d.season === season)
      const wins   = rows.filter(d => d.result === 'win').length
      const losses = rows.filter(d => d.result === 'loss').length
      const pushes = rows.filter(d => d.result === 'push').length
      const units  = rows.reduce((acc, d) => acc + (d.units ?? 0), 0)
      const roi    = (wins + losses) > 0 ? (units / (wins + losses)) * 100 : 0
      return { season, wins, losses, pushes, units, roi, bets: rows.length }
    })
  }, [filtered])

  const chartData = useMemo(() => {
    let cum = 0
    return filtered.map((d, i) => {
      cum += d.units ?? 0
      return { i: i + 1, units: parseFloat(cum.toFixed(2)) }
    })
  }, [filtered])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-gray-600 text-sm">
        Loading backtest data...
      </div>
    )
  }

  if (data.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-3">
        <span>No backtest data yet</span>
        <div className="text-xs text-gray-700 text-center max-w-sm">
          <p>1. Make sure models are trained and data covers 2023–2025</p>
          <p className="mt-1">2. Run <code className="text-gray-500 bg-gray-900 px-1 rounded">python scripts/backtest.py</code> from your terminal</p>
        </div>
      </div>
    )
  }

  const finalUnits = chartData[chartData.length - 1]?.units ?? 0

  return (
    <div className="p-4 space-y-6 max-w-5xl mx-auto">

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex rounded overflow-hidden border border-gray-700">
          {['spread', 'total'].map(t => (
            <button
              key={t}
              onClick={() => setBetType(t)}
              className={`px-4 py-1.5 text-xs transition-colors ${
                betType === t ? 'bg-gray-700 text-white' : 'bg-gray-900 text-gray-400 hover:bg-gray-800'
              }`}
            >
              {t === 'spread' ? 'Spreads' : 'Totals'}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-gray-500 text-xs">Min edge</span>
          <div className="flex rounded overflow-hidden border border-gray-700">
            {[1.0, 1.5, 2.0, 2.5, 3.0].map(t => (
              <button
                key={t}
                onClick={() => setThreshold(t)}
                className={`px-3 py-1 text-xs transition-colors ${
                  threshold === t ? 'bg-gray-700 text-white' : 'bg-gray-900 text-gray-400 hover:bg-gray-800'
                }`}
              >
                {t}+
              </button>
            ))}
          </div>
        </div>

        <span className="text-xs text-gray-600 ml-auto">{overall.bets} bets · out-of-sample (2023–2025)</span>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {[
          { label: 'Record', value: `${overall.wins}-${overall.losses}${overall.pushes ? `-${overall.pushes}` : ''}` },
          { label: 'Units', value: `${overall.units >= 0 ? '+' : ''}${overall.units.toFixed(1)}u`, color: overall.units >= 0 ? 'text-green-400' : 'text-red-400' },
          { label: 'ROI', value: `${overall.roi >= 0 ? '+' : ''}${overall.roi.toFixed(1)}%`, color: overall.roi >= 0 ? 'text-green-400' : 'text-red-400' },
          { label: 'Win Rate', value: `${overall.winPct.toFixed(1)}%` },
          { label: 'Total Bets', value: overall.bets },
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
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">
            Cumulative Units · {betType}s · edge ≥ {threshold}
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData}>
              <XAxis dataKey="i" tick={{ fontSize: 9, fill: '#6b7280' }} label={{ value: 'Bet #', position: 'insideBottomRight', offset: -5, fontSize: 9, fill: '#4b5563' }} />
              <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} width={40} />
              <ReferenceLine y={0} stroke="#374151" strokeDasharray="4 2" />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
                formatter={(v) => [`${v >= 0 ? '+' : ''}${v}u`, 'Cumulative']}
                labelFormatter={(i) => `Bet #${i}`}
              />
              <Line
                type="monotone"
                dataKey="units"
                stroke={finalUnits >= 0 ? '#22c55e' : '#ef4444'}
                dot={false}
                strokeWidth={2}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* By season */}
      <div className="bg-gray-900 rounded border border-gray-800 p-4">
        <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Season Breakdown</h3>
        {bySeasonStats.length === 0 ? (
          <div className="text-xs text-gray-600 py-4 text-center">No bets at this threshold</div>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-gray-600 border-b border-gray-800">
                <th className="text-left pb-2">Season</th>
                <th className="text-right pb-2">W-L-P</th>
                <th className="text-right pb-2">Bets</th>
                <th className="text-right pb-2">Units</th>
                <th className="text-right pb-2">ROI</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {bySeasonStats.map(s => (
                <tr key={s.season}>
                  <td className="py-2 font-medium text-gray-200">{s.season}</td>
                  <td className="py-2 text-right text-gray-300">{s.wins}-{s.losses}-{s.pushes}</td>
                  <td className="py-2 text-right text-gray-500">{s.bets}</td>
                  <td className={`py-2 text-right tabular-nums ${s.units >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {s.units >= 0 ? '+' : ''}{s.units.toFixed(1)}u
                  </td>
                  <td className={`py-2 text-right tabular-nums ${s.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {s.roi >= 0 ? '+' : ''}{s.roi.toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

    </div>
  )
}
