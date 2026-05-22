import { useState } from 'react'
import { supabase } from '../lib/supabase'

export default function BetLogForm({ projection: p, onLogged, onCancel }) {
  const [units, setUnits] = useState(1)
  const [odds, setOdds] = useState(-110)
  const [notes, setNotes] = useState('')
  const [saving, setSaving] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setSaving(true)

    const { error } = await supabase.from('bets').insert({
      game_id: p.game_id,
      game_date: p.game_date,
      week: p.week,
      season: p.season,
      home_team: p.home_team,
      away_team: p.away_team,
      bet_type: p.bet_type,
      side: p.side,
      model_line: p.model_line,
      dk_line: p.dk_line,
      edge_points: p.edge_points,
      ev_pct: p.ev_pct,
      confidence_tier: p.confidence_tier,
      steam_flag: p.steam_flag,
      rlm_flag: p.rlm_flag,
      odds: Number(odds),
      units: Number(units),
      result: 'pending',
      notes: notes || null,
    })

    setSaving(false)
    if (!error) onLogged()
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-2">
      <div className="text-xs text-gray-400 font-medium">Log bet: {p.bet_type} {p.side}</div>

      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="text-xs text-gray-600 block mb-0.5">Units</label>
          <input
            type="number" min={0.5} max={5} step={0.5}
            value={units}
            onChange={e => setUnits(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1"
          />
        </div>
        <div>
          <label className="text-xs text-gray-600 block mb-0.5">Odds</label>
          <input
            type="number"
            value={odds}
            onChange={e => setOdds(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1"
          />
        </div>
      </div>

      <div>
        <label className="text-xs text-gray-600 block mb-0.5">Notes</label>
        <input
          type="text"
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Optional..."
          className="w-full bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1"
        />
      </div>

      <div className="flex gap-2 pt-1">
        <button
          type="submit"
          disabled={saving}
          className="flex-1 text-xs py-1.5 rounded bg-green-800 hover:bg-green-700 text-green-100 transition-colors disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Log Bet'}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="flex-1 text-xs py-1.5 rounded border border-gray-700 text-gray-400 hover:bg-gray-800 transition-colors"
        >
          Cancel
        </button>
      </div>
    </form>
  )
}
