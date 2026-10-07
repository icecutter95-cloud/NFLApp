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
    // line_history.game_id is The Odds API's own opaque event ID, not the
    // nfl_data_py game_id used everywhere else — join on the team pair instead
    // (refresh-odds stores home_team/away_team specifically for this).
    // Public splits live in nfl_public_splits, keyed on the LOGGER's synthetic
    // game_id (2026_1_NO_DET) while a projection carries the nfl_data_py one
    // (2026_01_NO_DET). The zero-padded week means querying by p.game_id
    // matches nothing, which is half of why the old public_betting block was
    // always blank -- the other half being that its writer called an endpoint
    // Action Network retired. Resolve the id through the board, then read the
    // split for THIS market.
    const [lh, board, wx] = await Promise.all([
      supabase.from('line_history').select('*').eq('home_team', p.home_team).eq('away_team', p.away_team).order('recorded_at'),
      supabase.from('line_predictions').select('game_id, commence_time').eq('home_team', p.home_team).eq('away_team', p.away_team).limit(1),
      supabase.from('weather').select('*').eq('game_id', p.game_id).single(),
    ])
    setLineHistory(lh.data ?? [])
    setWeather(wx.data ?? null)
    const gid = board.data?.[0]?.game_id
    const kick = board.data?.[0]?.commence_time
    if (gid) {
      // Last capture BEFORE kickoff. The collector keeps running through the
      // game, so the newest row for a finished game is a post-game snapshot --
      // a Week 1 game was showing a split captured six days after it ended.
      let q = supabase.from('nfl_public_splits').select('*')
        .eq('game_id', gid).eq('bet_type', p.bet_type)
      if (kick) q = q.lt('captured_at', kick)
      const { data } = await q.order('captured_at', { ascending: false }).limit(1)
      setPublicBetting(data?.[0] ?? null)
    } else {
      setPublicBetting(null)
    }
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
        {publicBetting && (() => {
          // The parser stores the OVER as the home column of a total row, so
          // the label has to follow the market or a 73% over reads as 73% on
          // the home team.
          const label = p.bet_type === 'total' ? 'Over' : p.home_team
          const bets = publicBetting.home_bets_pct
          const money = publicBetting.home_money_pct
          const gap = bets != null && money != null ? money - bets : null
          return (
            <div className="space-y-1 pt-2 border-t border-gray-800">
              <h4 className="text-xs text-gray-500 uppercase tracking-wider">Public Betting</h4>
              <div className="flex justify-between text-xs">
                <span className="text-gray-500">Tickets on {label}</span>
                <span className="text-gray-300">{bets?.toFixed(0)}%</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-1.5">
                <div className="bg-green-500 h-1.5 rounded-full"
                     style={{ width: `${bets ?? 50}%` }} />
              </div>
              <div className="flex justify-between text-xs text-gray-600">
                <span>Money: {money?.toFixed(0)}%</span>
                {gap != null && Math.abs(gap) >= 10 && (
                  <span className={gap > 0 ? 'text-blue-400' : 'text-orange-400'}>
                    {gap > 0 ? '+' : ''}{gap.toFixed(0)} handle gap
                  </span>
                )}
              </div>
              {p.rlm_flag && (
                <div className="text-xs text-blue-400 pt-0.5">
                  Reverse line movement — money pushing back toward{' '}
                  {p.rlm_sharp_side === 'home' ? p.home_team
                    : p.rlm_sharp_side === 'away' ? p.away_team
                    : p.rlm_sharp_side}
                </div>
              )}
              <div className="text-xs text-gray-700">
                {publicBetting.source} · {new Date(publicBetting.captured_at)
                  .toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric' })}
                {' '}· display only, does not affect the tier
              </div>
            </div>
          )
        })()}
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
