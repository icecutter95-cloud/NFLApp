import { useState, useEffect, useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'
import { Info, Target, AlertTriangle } from 'lucide-react'
import { supabase } from '../lib/supabase'

// Performance of the LINE MOVEMENT model at the line level — every line it has
// an opinion on, and separately the ones that clear the qualifying bar.
//
// The numbers come from movement_history, which is rebuilt by
// scripts/build_movement_history.py with models trained on 2020-2022 ONLY and
// scored cold on 2023-2025. The production models are not used there: they are
// fit on every season, and grading 2024 with a model that trained on 2024 is
// the contamination that produced this project's earlier fake results.
//
// The select/holdout split matters more than any single number. Thresholds were
// chosen while looking at 2023-24, so those figures are flattered by selection.
// 2025 was never looked at. When the two disagree, believe 2025.

const BREAK_EVEN = 52.38   // win rate needed at -110

function pct(x) { return x == null ? '—' : `${x.toFixed(1)}%` }
function signed(x, d = 1) { return x == null ? '—' : `${x >= 0 ? '+' : ''}${x.toFixed(d)}` }

// -110 juice: a win returns 100/110 units, a loss costs 1, a push is nothing.
function tally(rows) {
  const w = rows.filter(r => r.result === 'win').length
  const l = rows.filter(r => r.result === 'loss').length
  const p = rows.filter(r => r.result === 'push').length
  const decided = w + l
  const units = w * (100 / 110) - l
  const clv = rows.filter(r => r.clv_points != null).map(r => r.clv_points)
  return {
    n: rows.length, w, l, p, units,
    winPct: decided ? (w / decided) * 100 : null,
    roi: decided ? (units / decided) * 100 : null,
    clv: clv.length ? clv.reduce((a, b) => a + b, 0) / clv.length : null,
    clvPos: clv.length ? (clv.filter(v => v > 0).length / clv.length) * 100 : null,
  }
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

function edgeColor(winPct) {
  if (winPct == null) return 'text-gray-300'
  return winPct >= BREAK_EVEN ? 'text-green-400' : 'text-red-400'
}

function SliceTable({ rows, betType }) {
  const cuts = [
    { key: 'select-all',   label: 'All lines',  period: 'select',  qual: false },
    { key: 'select-qual',  label: 'Qualifying', period: 'select',  qual: true },
    { key: 'holdout-all',  label: 'All lines',  period: 'holdout', qual: false },
    { key: 'holdout-qual', label: 'Qualifying', period: 'holdout', qual: true },
  ]
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="text-gray-600 border-b border-gray-800">
          <th className="text-left pb-2 font-normal">Period</th>
          <th className="text-left pb-2 font-normal">Slice</th>
          <th className="text-right pb-2 font-normal">N</th>
          <th className="text-right pb-2 font-normal">W-L-P</th>
          <th className="text-right pb-2 font-normal">Win%</th>
          <th className="text-right pb-2 font-normal">ROI</th>
          <th className="text-right pb-2 font-normal">Units</th>
          <th className="text-right pb-2 font-normal">CLV</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-gray-800/50">
        {cuts.map(c => {
          const s = tally(rows.filter(r =>
            r.period === c.period && (!c.qual || r.qualifies) &&
            (betType === 'all' || r.bet_type === betType)))
          if (!s.n) return null
          const hold = c.period === 'holdout'
          return (
            <tr key={c.key} className={hold ? 'bg-gray-800/20' : ''}>
              <td className="py-2">
                <span className={hold ? 'text-gray-200' : 'text-gray-500'}>
                  {hold ? '2025 holdout' : '2023-24 select'}
                </span>
              </td>
              <td className="py-2">
                {c.qual
                  ? <span className="text-green-400 flex items-center gap-1"><Target size={10} /> Qualifying</span>
                  : <span className="text-gray-500">All lines</span>}
              </td>
              <td className="py-2 text-right text-gray-400 tabular-nums">{s.n}</td>
              <td className="py-2 text-right text-gray-300 tabular-nums">{s.w}-{s.l}{s.p ? `-${s.p}` : ''}</td>
              <td className={`py-2 text-right tabular-nums font-medium ${edgeColor(s.winPct)}`}>{pct(s.winPct)}</td>
              <td className={`py-2 text-right tabular-nums ${s.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {signed(s.roi)}%
              </td>
              <td className={`py-2 text-right tabular-nums ${s.units >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {signed(s.units)}u
              </td>
              <td className={`py-2 text-right tabular-nums ${s.clv >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {signed(s.clv, 2)}
              </td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

export default function ModelPerformancePanel({ season }) {
  const [hist, setHist] = useState([])
  const [live, setLive] = useState([])
  const [loading, setLoading] = useState(true)
  const [betType, setBetType] = useState('all')

  useEffect(() => {
    let cancelled = false
    async function pageAll(table, apply) {
      const PAGE = 1000
      let all = []
      for (let from = 0; ; from += PAGE) {
        let q = supabase.from(table).select('*').range(from, from + PAGE - 1)
        q = apply(q)
        const { data, error } = await q
        if (error || !data || data.length === 0) break
        all = all.concat(data)
        if (data.length < PAGE) break
      }
      return all
    }
    async function load() {
      setLoading(true)
      const [h, l] = await Promise.all([
        pageAll('movement_history', q => q.order('game_date')),
        pageAll('clv_tracking', q => q.eq('season', season).order('commence_time')),
      ])
      if (!cancelled) { setHist(h); setLive(l); setLoading(false) }
    }
    load()
    return () => { cancelled = true }
  }, [season])

  const inType = useMemo(
    () => betType === 'all' ? hist : hist.filter(r => r.bet_type === betType),
    [hist, betType],
  )

  // Cumulative units on qualifying lines only, in date order — the equity curve
  // you would actually have ridden. The holdout boundary is marked because the
  // curve before it is partly a product of hindsight.
  const chart = useMemo(() => {
    const q = inType.filter(r => r.qualifies).sort((a, b) => (a.game_date ?? '').localeCompare(b.game_date ?? ''))
    let u = 0, holdoutAt = null
    const pts = q.map((r, i) => {
      if (r.result === 'win') u += 100 / 110
      else if (r.result === 'loss') u -= 1
      if (holdoutAt == null && r.period === 'holdout') holdoutAt = i
      return { i, date: r.game_date, units: parseFloat(u.toFixed(2)) }
    })
    return { pts, holdoutAt }
  }, [inType])

  const holdoutQual = useMemo(
    () => tally(inType.filter(r => r.period === 'holdout' && r.qualifies)),
    [inType],
  )

  const liveStats = useMemo(() => {
    const rows = betType === 'all' ? live : live.filter(r => r.bet_type === betType)
    const closed = rows.filter(r => r.clv_points != null)
    const clv = closed.map(r => r.clv_points)
    return {
      tracked: rows.length,
      qualifying: rows.filter(r => r.qualifies).length,
      meanClv: clv.length ? clv.reduce((a, b) => a + b, 0) / clv.length : null,
    }
  }, [live, betType])

  if (loading) {
    return <div className="flex items-center justify-center h-48 text-gray-600 text-sm">Loading model performance…</div>
  }

  if (hist.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2">
        <span>No track record built yet</span>
        <span className="text-xs text-gray-700">
          Run: <code className="bg-gray-900 px-1 rounded">python scripts/build_movement_history.py</code>
        </span>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-5 max-w-5xl mx-auto">

      <div className="flex items-start gap-2 text-xs text-gray-500 bg-gray-900/40 border border-gray-800 rounded p-3">
        <Info size={14} className="mt-0.5 shrink-0 text-gray-600" />
        <span>
          How the movement model's lines have actually graded, taking every bet{' '}
          <span className="text-gray-300">at the opener</span> and paying −110. Built by training on
          2020–2022 and scoring 2023–2025 cold, so no game was graded by a model that had seen it.
          Thresholds were chosen while looking at <span className="text-gray-300">2023–24</span>;
          <span className="text-gray-300"> 2025</span> was never looked at.
          When the two rows disagree, the 2025 row is the honest one.
        </span>
      </div>

      <div className="flex items-center gap-1 border-b border-gray-800">
        {[['all', 'All'], ['spread', 'Spreads'], ['total', 'Totals']].map(([k, label]) => (
          <button
            key={k}
            onClick={() => setBetType(k)}
            className={`px-3 py-1.5 text-xs border-b-2 -mb-px transition-colors ${
              betType === k ? 'border-gray-300 text-gray-100' : 'border-transparent text-gray-600 hover:text-gray-400'
            }`}
          >
            {label}{' '}
            <span className="text-gray-600 tabular-nums">
              {k === 'all' ? hist.length : hist.filter(r => r.bet_type === k).length}
            </span>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat
          label="Holdout qualifying"
          value={pct(holdoutQual.winPct)}
          sub={`${holdoutQual.w}-${holdoutQual.l}${holdoutQual.p ? `-${holdoutQual.p}` : ''} in 2025`}
          color={edgeColor(holdoutQual.winPct)}
        />
        <Stat
          label="Holdout ROI"
          value={`${signed(holdoutQual.roi)}%`}
          sub={`${signed(holdoutQual.units)}u at −110`}
          color={holdoutQual.roi >= 0 ? 'text-green-400' : 'text-red-400'}
        />
        <Stat
          label="Break-even"
          value={`${BREAK_EVEN}%`}
          sub="what −110 demands"
        />
        <Stat
          label="Holdout CLV"
          value={signed(holdoutQual.clv, 2)}
          sub={holdoutQual.clvPos == null ? 'points vs close' : `${holdoutQual.clvPos.toFixed(0)}% beat the close`}
          color={holdoutQual.clv >= 0 ? 'text-green-400' : 'text-red-400'}
        />
      </div>

      <div className="bg-gray-900 rounded border border-gray-800 p-4">
        <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">
          All lines vs qualifying lines
        </h3>
        <SliceTable rows={hist} betType={betType} />
        <p className="text-xs text-gray-600 mt-3 leading-relaxed">
          The gap between the two slices is the whole argument for the filter. If “qualifying” does not
          clearly beat “all lines” in the 2025 row, the filter is not doing work worth the bet.
        </p>
      </div>

      {chart.pts.length > 1 && (
        <div className="bg-gray-900 rounded border border-gray-800 p-4">
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">
            Qualifying lines — cumulative units at −110
          </h3>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chart.pts}>
              <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#6b7280' }} minTickGap={40} />
              <YAxis tick={{ fontSize: 9, fill: '#6b7280' }} width={38} />
              <Tooltip
                contentStyle={{ background: '#111827', border: '1px solid #374151', fontSize: 11 }}
                formatter={v => [`${v >= 0 ? '+' : ''}${v}u`, 'Cumulative']}
              />
              <ReferenceLine y={0} stroke="#374151" />
              {chart.holdoutAt != null && (
                <ReferenceLine
                  x={chart.pts[chart.holdoutAt]?.date}
                  stroke="#f59e0b"
                  strokeDasharray="3 3"
                  label={{ value: 'holdout begins', fill: '#f59e0b', fontSize: 9, position: 'insideTopLeft' }}
                />
              )}
              <Line type="monotone" dataKey="units" stroke="#22c55e" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-gray-600 mt-2">
            Everything left of the dashed line is partly hindsight — the thresholds were picked while
            looking at it. Only the segment to its right is a clean out-of-sample result.
          </p>
        </div>
      )}

      {betType !== 'spread' && (
        <div className="flex items-start gap-2 text-xs bg-amber-950/30 border border-amber-900/60 rounded p-3">
          <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-500" />
          <span className="text-amber-200/70">
            Totals carry a caveat the raw win rate hides: they qualify on the movement model
            <span className="text-amber-200"> alone</span>, where a spread needs two independent signals to
            agree. The 1.25-point bar held in both periods, but the next bar up (1.5) inverted out of
            sample — 63.0% → 47.4%. Size them at a quarter unit at most.
          </span>
        </div>
      )}

      <div className="bg-gray-900 rounded border border-gray-800 p-4">
        <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">
          Live — {season} season
        </h3>
        {liveStats.tracked === 0 ? (
          <p className="text-xs text-gray-600">No lines logged yet for {season}.</p>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-3">
              <Stat label="Lines logged" value={liveStats.tracked} sub="frozen at the opener" />
              <Stat label="Qualifying" value={liveStats.qualifying} sub="clear the bar" color="text-green-400" />
              <Stat
                label="Mean CLV"
                value={signed(liveStats.meanClv, 2)}
                sub="points vs close"
                color={liveStats.meanClv > 0 ? 'text-green-400' : liveStats.meanClv < 0 ? 'text-red-400' : undefined}
              />
            </div>
            <p className="text-xs text-gray-600 mt-3">
              Win/loss for the live season is not graded yet — no final scores are loaded for {season}.
              CLV resolves first and is the earlier honest signal; the record above is what to expect meanwhile.
            </p>
          </>
        )}
      </div>
    </div>
  )
}
