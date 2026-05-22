import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { supabase } from '../lib/supabase'
import BetLogForm from './BetLogForm'

export default function GameDetailPanel({ projection: p, onBetLogged }) {
  const [lineHistory, setLineHistory] = useState([])
  const [publicBetting, setPublicBetting] = useState(null)
  const [weather, setWeather] = useState(null)
  const [showBetForm, setShowBetForm] = useState(false)

  useEffect(() => {
    fetchDetail()
  }, [p.game_id])

  async function fetchDetail() {
    const [lh, pb, wx] = await Promise.all([
      supabase.from('line_history').select('*').eq('game_id', p.game_id).order('recorded_at'),
      supabase.from('public_betting').select('*').eq('game_id', p.game_id).order('recorded_at', { ascending: false }).limit(1),
      supabase.from('weather').select('*').eq('game_id', p.game_id).single(),
    ])
    setLineHistory(lh.data ?? [])
    setPublicBetting(pb.data?.[0] ?? null)
    setWeather(wx.data ?? null)
  }

  const chartData = lineHistory.map(h => ({
    time: new Date(h.recorded_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: 'numeric' }),
    spread: h.spread_home,
    total: h.total,
  }))

  const spreadOpen = lineHistory.find(h => h.is_opening)?.spread_home
  const spreadCurrent = lineHistory[lineHistory.length - 1]?.spread_home
  const spreadMove = spreadOpen != null && spreadCurrent != null ? spreadCurrent - spreadOpen : null

  return (
    <div className="bg-gray-900/60 border-t border-gray-800 px-6 py-4 grid grid-cols-1 lg:grid-cols-3 gap-6">

      {/* Column 1 — Model inputs + edge driver */}
      <div className="space-y-3">
        <h4 className="text-xs text-gray-500 uppercase tracking-wider">Model Inputs</h4>
        <table className="w-full text-xs">
          <thead>
            <tr className="text-gray-600">
              <th className="text-left pb-1">Metric</th>
              <th className="text-right pb-1">{p.home_team}</th>
              <th className="text-right pb-1">{p.away_team}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {[
              { label: 'EPA/play (off)', home: p.home_epa_off, away: p.away_epa_off },
              { label: 'EPA/play (def)', home: p.home_epa_def, away: p.away_epa_def },
              { label: 'CPOE', home: p.home_cpoe, away: p.away_cpoe },
            ].map(({ label, home, away }) => (
              <tr key={label}>
                <td className="py-1 text-gray-500">{label}</td>
                <td className="py-1 text-right text-gray-300">{home != null ? home.toFixed(2) : '—'}</td>
                <td className="py-1 text-right text-gray-300">{away != null ? away.toFixed(2) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <div className="text-xs text-gray-500 pt-2 border-t border-gray-800">
          <span className="text-gray-400 font-medium">Model: </span>
          {p.bet_type === 'spread' ? `Projected margin: ${p.model_line > 0 ? '+' : ''}${p.model_line}` : `Projected total: ${p.model_line}`}
          {p.weather_adj !== 0 && <span className="text-blue-400 ml-1">(weather adj: {p.weather_adj > 0 ? '+' : ''}{p.weather_adj})</span>}
        </div>
      </div>

      {/* Column 2 — Line movement chart */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h4 className="text-xs text-gray-500 uppercase tracking-wider">Line Movement</h4>
          {spreadMove != null && (
            <span className={`text-xs ${Math.abs(spreadMove) >= 1 ? 'text-yellow-400' : 'text-gray-500'}`}>
              {spreadMove >= 0 ? '+' : ''}{spreadMove.toFixed(1)} pts
            </span>
          )}
        </div>

        {chartData.length > 1 ? (
          <ResponsiveContainer width="100%" height={120}>
            <LineChart data={chartData}>
              <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#6b7280' }} hide />
              <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} width={30} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
                labelStyle={{ color: '#9ca3af' }}
              />
              {p.bet_type === 'spread'
                ? <Line type="monotone" dataKey="spread" stroke="#22c55e" dot={false} strokeWidth={2} />
                : <Line type="monotone" dataKey="total" stroke="#22c55e" dot={false} strokeWidth={2} />
              }
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[120px] flex items-center justify-center text-gray-700 text-xs">
            {chartData.length === 1 ? 'Opening line only — no movement yet' : 'No line history'}
          </div>
        )}

        {/* Public betting */}
        {publicBetting && (
          <div className="space-y-1 pt-2 border-t border-gray-800">
            <h4 className="text-xs text-gray-500 uppercase tracking-wider">Public Betting</h4>
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Bets on {p.home_team}</span>
              <span className="text-gray-300">{publicBetting.bet_pct_home?.toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-1.5">
              <div
                className="bg-green-500 h-1.5 rounded-full"
                style={{ width: `${publicBetting.bet_pct_home ?? 50}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-gray-600">
              <span>Money: {publicBetting.money_pct_home?.toFixed(0)}%</span>
              <span className="text-gray-700">{publicBetting.source}</span>
            </div>
          </div>
        )}
      </div>

      {/* Column 3 — Weather + bet log */}
      <div className="space-y-3">
        {weather && !p.is_dome && (
          <div className="space-y-1">
            <h4 className="text-xs text-gray-500 uppercase tracking-wider">Weather</h4>
            <div className="grid grid-cols-2 gap-1 text-xs">
              <div className="text-gray-500">Wind</div>
              <div className={`text-right ${weather.wind_speed_mph >= 15 ? 'text-yellow-400' : 'text-gray-300'}`}>
                {weather.wind_speed_mph} mph {weather.wind_direction}
              </div>
              <div className="text-gray-500">Temp</div>
              <div className={`text-right ${weather.temp_fahrenheit <= 32 ? 'text-blue-400' : 'text-gray-300'}`}>
                {weather.temp_fahrenheit}°F
              </div>
              <div className="text-gray-500">Precip</div>
              <div className={`text-right ${weather.precipitation_prob >= 0.5 ? 'text-blue-400' : 'text-gray-300'}`}>
                {((weather.precipitation_prob ?? 0) * 100).toFixed(0)}%
              </div>
            </div>
            {p.weather_adj !== 0 && (
              <div className="text-xs text-blue-400 mt-1">Total adj: {p.weather_adj > 0 ? '+' : ''}{p.weather_adj} pts</div>
            )}
          </div>
        )}
        {p.is_dome && (
          <div className="text-xs text-gray-600">🏟️ Dome — no weather impact</div>
        )}

        {/* Bet log button */}
        <div className="pt-2 border-t border-gray-800">
          {showBetForm ? (
            <BetLogForm
              projection={p}
              onLogged={() => { setShowBetForm(false); onBetLogged() }}
              onCancel={() => setShowBetForm(false)}
            />
          ) : (
            <button
              onClick={() => setShowBetForm(true)}
              className="w-full text-xs py-2 rounded border border-green-800 text-green-400 hover:bg-green-950 transition-colors"
            >
              + Log This Bet
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
