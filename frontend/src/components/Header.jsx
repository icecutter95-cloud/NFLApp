import { useState } from 'react'
import { BarChart2, FlaskConical, RefreshCw, Zap, Play } from 'lucide-react'
import { supabase } from '../lib/supabase'

export default function Header({
  season, week, weeks, bets,
  onSeasonChange, onWeekChange,
  view, onViewChange,
}) {
  const [oddsStatus, setOddsStatus]   = useState(null)  // null | 'loading' | 'ok' | 'error'
  const [projStatus, setProjStatus]   = useState(null)
  const [oddsMsg, setOddsMsg]         = useState('')
  const [projMsg, setProjMsg]         = useState('')

  const closed  = bets.filter(b => b.result !== 'pending')
  const wins    = closed.filter(b => b.result === 'win').length
  const losses  = closed.filter(b => b.result === 'loss').length
  const units   = closed.reduce((acc, b) => {
    if (b.result === 'win')  return acc + (b.units ?? 1) * (100 / 110)
    if (b.result === 'loss') return acc - (b.units ?? 1)
    return acc
  }, 0)

  async function refreshOdds() {
    setOddsStatus('loading')
    setOddsMsg('')
    try {
      const { data, error } = await supabase.functions.invoke('refresh-odds')
      if (error) throw error
      setOddsStatus('ok')
      setOddsMsg(`${data?.count ?? '?'} lines`)
      setTimeout(() => setOddsStatus(null), 4000)
    } catch (err) {
      setOddsStatus('error')
      setOddsMsg('Failed')
    }
  }

  async function runProjections() {
    setProjStatus('loading')
    setProjMsg('')
    try {
      const { data, error } = await supabase.functions.invoke('trigger-pipeline', {
        body: { season, week },
      })
      if (error) throw error
      setProjStatus('ok')
      setProjMsg('Running (~2 min)')
      setTimeout(() => setProjStatus(null), 8000)
    } catch (err) {
      setProjStatus('error')
      setProjMsg('Failed — check secrets')
    }
  }

  const navBtn = (id, icon, label) => (
    <button
      key={id}
      onClick={() => onViewChange(id)}
      className={`flex items-center gap-1 text-xs px-3 py-1 rounded border transition-colors ${
        view === id
          ? 'border-green-500 text-green-400 bg-green-950'
          : 'border-gray-700 text-gray-400 hover:border-gray-500'
      }`}
    >
      {icon}
      {label}
    </button>
  )

  const pipelineBtn = ({ onClick, status, msg, idleIcon, idleLabel, loadingLabel, okClass, okLabel }) => (
    <button
      onClick={onClick}
      disabled={status === 'loading'}
      className={`flex items-center gap-1 text-xs px-3 py-1 rounded border transition-colors disabled:opacity-50 ${
        status === 'ok'    ? `border-green-700 text-green-400 bg-green-950 ${okClass}` :
        status === 'error' ? 'border-red-700 text-red-400 bg-red-950' :
                             'border-gray-700 text-gray-500 hover:border-gray-400'
      }`}
    >
      {status === 'loading' ? <RefreshCw size={11} className="animate-spin" /> : idleIcon}
      {status === 'loading' ? loadingLabel :
       status === 'ok'      ? (msg || okLabel) :
       status === 'error'   ? (msg || 'Error') :
       idleLabel}
    </button>
  )

  return (
    <header className="sticky top-0 z-50 bg-gray-950 border-b border-gray-800 px-4 py-2 flex flex-wrap items-center justify-between gap-3">

      {/* Left — brand + season/week selectors */}
      <div className="flex items-center gap-3">
        <span className="text-green-400 font-bold text-lg tracking-tight">NFL EDGE</span>
        <span className="text-gray-700 text-xs">|</span>

        <select
          value={season}
          onChange={e => onSeasonChange(Number(e.target.value))}
          className="bg-gray-900 border border-gray-700 text-gray-300 text-xs rounded px-2 py-1"
        >
          {[2026, 2025, 2024, 2023, 2022].map(s => (
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

      {/* Center — W-L + units (if any closed bets) */}
      {closed.length > 0 && (
        <div className="text-xs text-gray-400 flex gap-3">
          <span className="text-white font-medium">{wins}-{losses}</span>
          <span className={units >= 0 ? 'text-green-400' : 'text-red-400'}>
            {units >= 0 ? '+' : ''}{units.toFixed(1)}u
          </span>
        </div>
      )}

      {/* Right — pipeline buttons + nav tabs */}
      <div className="flex items-center gap-2 flex-wrap">

        {/* Pipeline controls */}
        {pipelineBtn({
          onClick: refreshOdds,
          status: oddsStatus,
          msg: oddsMsg,
          idleIcon: <Zap size={11} />,
          idleLabel: 'Refresh Odds',
          loadingLabel: 'Fetching…',
          okLabel: 'Updated',
        })}

        {pipelineBtn({
          onClick: runProjections,
          status: projStatus,
          msg: projMsg,
          idleIcon: <Play size={11} />,
          idleLabel: 'Run Projections',
          loadingLabel: 'Queuing…',
          okLabel: 'Queued',
        })}

        <span className="text-gray-800 text-xs">|</span>

        {/* View nav */}
        {navBtn('edges',       null,                        'Edges')}
        {navBtn('performance', <BarChart2 size={12} />,    'Performance')}
        {navBtn('backtest',    <FlaskConical size={12} />, 'Backtest')}
      </div>

    </header>
  )
}
