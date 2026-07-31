import { useState, useEffect, useMemo } from 'react'
import { TrendingUp, TrendingDown, Minus, Info, Target } from 'lucide-react'
import { supabase } from '../lib/supabase'

// This view tracks the LINE MOVEMENT model, which is a different question from
// the Edges tab. Edges asks "who covers?" — we measured that against closing
// lines and found no edge (52.5% vs a 52.38% break-even), which is why every
// row there reads "no bet". This asks "where is the number going?", which did
// survive an out-of-sample holdout.
//
// CLV (closing line value) is the metric because it resolves in weeks rather
// than a season: if you consistently hold a better number than the close, and
// the close is efficient, you are getting +EV prices even without predicting
// games better than the market.

function fmtLine(v) {
  if (v == null) return '—'
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)
}

function Stat({ label, value, sub, color }) {
  return (
    <div className="bg-gray-900 rounded border border-gray-800 p-3">
      <div className="text-xs text-gray-500 mb-1">{label}</div>
      <div className={`text-xl font-bold tabular-nums ${color ?? 'text-gray-100'}`}>{value}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  )
}

export default function ClvPanel({ season }) {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [qualifyingOnly, setQualifyingOnly] = useState(false)

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      const PAGE = 1000
      let all = []
      for (let from = 0; ; from += PAGE) {
        const { data, error } = await supabase
          .from('clv_tracking')
          .select('*')
          .eq('season', season)
          .order('commence_time')
          .range(from, from + PAGE - 1)
        if (error || !data || data.length === 0) break
        all = all.concat(data)
        if (data.length < PAGE) break
      }
      if (!cancelled) { setRows(all); setLoading(false) }
    }
    load()
    return () => { cancelled = true }
  }, [season])

  const stats = useMemo(() => {
    const closed = rows.filter(r => r.clv_points != null)
    const moved = rows.filter(r => r.direction_correct != null)
    const clv = closed.map(r => r.clv_points)
    const mean = clv.length ? clv.reduce((a, b) => a + b, 0) / clv.length : null
    return {
      qualifying: rows.filter(r => r.qualifies).length,
      tracked: rows.length,
      resolved: closed.length,
      meanClv: mean,
      pctPositive: clv.length ? clv.filter(v => v > 0).length / clv.length * 100 : null,
      dirAcc: moved.length ? moved.filter(r => r.direction_correct).length / moved.length * 100 : null,
      dirN: moved.length,
    }
  }, [rows])

  if (loading) {
    return <div className="flex items-center justify-center h-48 text-gray-600 text-sm">Loading CLV tracking…</div>
  }

  if (rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2">
        <span>No predictions logged yet for {season}</span>
        <span className="text-xs text-gray-700">
          Runs daily at 12:30 UTC, or: <code className="bg-gray-900 px-1 rounded">python scripts/log_clv_predictions.py</code>
        </span>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-5 max-w-6xl mx-auto">

      <div className="flex items-start gap-2 text-xs text-gray-500 bg-gray-900/40 border border-gray-800 rounded p-3">
        <Info size={14} className="mt-0.5 shrink-0 text-gray-600" />
        <span>
          Tracks where the model thinks each line will <span className="text-gray-300">move</span>, not who covers.
          A prediction is frozen the first time a game appears, and <span className="text-gray-300">CLV</span> measures
          how much better that number is than the eventual close. Positive CLV means you hold a better price than
          the market's final answer — the fastest honest read on whether the model works.
          <span className="text-gray-600"> Paper tracking; nothing here is a bet recommendation.</span>
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat label="Games tracked" value={stats.tracked} sub={`${stats.resolved} with a close`} />
        <Stat
          label="Mean CLV"
          value={stats.meanClv == null ? '—' : `${stats.meanClv >= 0 ? '+' : ''}${stats.meanClv.toFixed(2)}`}
          sub="points vs close"
          color={stats.meanClv == null ? undefined : stats.meanClv > 0 ? 'text-green-400' : stats.meanClv < 0 ? 'text-red-400' : 'text-gray-300'}
        />
        <Stat
          label="CLV positive"
          value={stats.pctPositive == null ? '—' : `${stats.pctPositive.toFixed(0)}%`}
          sub="beat the close"
        />
        <Stat
          label="Direction right"
          value={stats.dirAcc == null ? '—' : `${stats.dirAcc.toFixed(0)}%`}
          sub={stats.dirN ? `of ${stats.dirN} that moved` : 'none moved yet'}
          color={stats.dirAcc == null ? undefined : stats.dirAcc > 50 ? 'text-green-400' : 'text-red-400'}
        />
        <Stat label="Qualifying" value={stats.qualifying} sub="both signals agree" color="text-green-400" />
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={() => setQualifyingOnly(v => !v)}
          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded border transition-colors ${
            qualifyingOnly
              ? 'border-green-500 text-green-400 bg-green-950'
              : 'border-gray-700 text-gray-500 hover:border-gray-500'
          }`}
        >
          <Target size={12} /> Qualifying only
        </button>
        <span className="text-xs text-gray-600">
          Qualifying = margin model disagrees with the opener by 3+ pts AND the movement
          model expects 0.5+ pts of drift the same way. Held 61.5% / 60.0% across the two
          test periods at ~2.5 games a week.
        </span>
      </div>

      <div className="bg-gray-900 rounded border border-gray-800 overflow-hidden">
        <div className="hidden md:grid grid-cols-[1.6fr_70px_80px_90px_80px_90px_80px_70px] gap-2 px-4 py-2 text-xs text-gray-600 uppercase tracking-wider border-b border-gray-800">
          <span>Game</span>
          <span className="text-right">Opened</span>
          <span className="text-right">We project</span>
          <span className="text-right">Take</span>
          <span className="text-right">Current</span>
          <span className="text-right">Moved</span>
          <span className="text-right">CLV</span>
          <span className="text-right">Dir</span>
        </div>

        <div className="divide-y divide-gray-800/50">
          {(qualifyingOnly ? rows.filter(r => r.qualifies) : rows).map(r => {
            const pending = r.clv_points == null
            const clv = r.clv_points ?? 0
            const side = r.predicted_side === 'home' ? r.home_team : r.away_team
            return (
              <div key={r.game_id}
                   className={`grid grid-cols-[1.6fr_70px_80px_90px_80px_90px_80px_70px] gap-2 px-4 py-2.5 text-sm items-center hover:bg-gray-800/30 ${
                     r.qualifies ? 'bg-green-950/20 border-l-2 border-l-green-600' : ''
                   }`}>
                <div className="min-w-0">
                  {r.qualifies && <Target size={11} className="inline mr-1.5 text-green-500" />}
                  <span className="text-gray-100">{r.away_team} @ {r.home_team}</span>
                  <span className="text-gray-600 text-xs ml-2">Wk {r.week}</span>
                </div>
                <div className="text-right text-gray-400 text-xs tabular-nums">{fmtLine(r.open_spread_home)}</div>
                <div className="text-right text-gray-300 text-xs tabular-nums">{fmtLine(r.projected_close)}</div>
                <div className="text-right text-xs">
                  <span className={r.qualifies ? 'text-green-300 font-semibold' : 'text-gray-200 font-medium'}>{side}</span>
                  <span className="text-gray-500 ml-1 tabular-nums">{fmtLine(r.taken_line)}</span>
                </div>
                <div className="text-right text-gray-400 text-xs tabular-nums">{fmtLine(r.closing_spread_home)}</div>
                <div className="text-right text-xs tabular-nums text-gray-400">
                  {r.actual_movement == null ? '—' : fmtLine(r.actual_movement)}
                </div>
                <div className={`text-right text-xs tabular-nums font-medium ${
                  pending ? 'text-gray-600' : clv > 0 ? 'text-green-400' : clv < 0 ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {pending ? 'pending' : `${clv >= 0 ? '+' : ''}${clv.toFixed(1)}`}
                </div>
                <div className="text-right">
                  {r.direction_correct == null
                    ? <Minus size={13} className="inline text-gray-700" />
                    : r.direction_correct
                      ? <TrendingUp size={13} className="inline text-green-400" />
                      : <TrendingDown size={13} className="inline text-red-400" />}
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <p className="text-xs text-gray-600">
        “Moved” and “CLV” stay blank until the line actually moves off its opener — with the season still weeks out,
        every game currently sits at its opening number, so zero movement and zero CLV is the correct reading.
      </p>
    </div>
  )
}
