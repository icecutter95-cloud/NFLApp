import { useState, useEffect, useMemo } from 'react'
import { AlertTriangle, Info, ChevronRight, ChevronDown, RefreshCw } from 'lucide-react'
import { supabase } from '../lib/supabase'
import { moveClass, moveTitle } from '../lib/movement'

// College football. Deliberately has NO qualifying flag anywhere.
//
// CFB spreads sit at p = 0.077 on label permutation and CFB totals at p = 0.192
// — neither clears the bar the NFL spread model does (p <= 0.038). So this page
// shows what the models think and lets the user decide, rather than dressing an
// unvalidated signal in the same green flag the NFL page uses.
//
// Totals are absent on purpose: the model called 68 of 73 games UNDER, and
// checking against training data showed it predicts over on 12% of week-1 games
// where the truth is 38%. That is a bias, not a signal, so totals return once
// teams have games played and the rolling features actually exist. The logger
// enforces this by inspecting whether the rolling features are populated, so
// they come back on their own rather than on a remembered date.
//
// Spreads only carry a side when the movement and margin models agree, which is
// the configuration the 54.7% walk-forward number was measured on. Roughly a
// third of the board splits; those rows still show both models' numbers but
// deliberately show no pick.

const fmtLine = v => v == null ? '—' : v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)

function fmtDate(s) {
  if (!s) return ''
  const d = new Date(s)
  const wd = d.toLocaleDateString(undefined, { weekday: 'short' })
  const md = d.toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' })
  const hm = d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })
    .replace(' ', '').replace('AM', 'a').replace('PM', 'p')
  return `${wd} ${md} ${hm}`
}

// Long form, for the expanded row where there is room for it.
function fmtDateFull(s) {
  if (!s) return ''
  return new Date(s).toLocaleString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  })
}

// The side, stated in full, for the expanded row.
//
// predicted_side is only populated when the movement and margin models agree --
// that is the tested rule -- so a null needs explaining rather than hiding. Both
// underlying signals are on the row, so the disagreement can be shown directly
// instead of just asserting there was one.
function sideDetail(r) {
  const mvTeam = r.predicted_movement == null ? null
               : r.predicted_movement < 0 ? r.home_team : r.away_team
  const disTeam = r.margin_disagreement == null ? null
                : r.margin_disagreement > 0 ? r.home_team : r.away_team
  if (r.predicted_side) {
    const team = r.predicted_side === 'home' ? r.home_team : r.away_team
    // taken_line is already mirrored for the away side by the logger.
    const num = r.taken_line != null ? r.taken_line
              : (r.predicted_side === 'home' ? r.open_line : -r.open_line)
    return { agree: true, team, num, mvTeam, disTeam }
  }
  return { agree: false, team: null, num: null, mvTeam, disTeam }
}

function Field({ label, value, color }) {
  return (
    <div>
      <div className="text-[10px] text-gray-600 uppercase tracking-wider">{label}</div>
      <div className={`text-xs tabular-nums ${color ?? 'text-gray-200'}`}>{value}</div>
    </div>
  )
}

export default function CfbPanel({ season }) {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(null)
  // Defaults to kickoff, not disagreement. Sorting by disagreement leads with
  // 40-point spreads where the model has the least to go on -- the extremes of a
  // distribution whose median disagreement is only 3.5 points.
  const [sort, setSort] = useState('time')
  const [status, setStatus] = useState('upcoming')
  const [week, setWeek] = useState('all')
  const [refresh, setRefresh] = useState(null)
  const [refreshMsg, setRefreshMsg] = useState('')

  async function load() {
    const PAGE = 1000
    let all = []
    for (let from = 0; ; from += PAGE) {
      const { data, error } = await supabase
        .from('cfb_tracking').select('*').eq('season', season)
        .order('commence_time').range(from, from + PAGE - 1)
      if (error || !data || data.length === 0) break
      all = all.concat(data)
      if (data.length < PAGE) break
    }
    setRows(all)
  }

  // Dispatches log-cfb, then polls for completion.
  //
  // Completion is measured on cfb_line_history, not on predicted_at. The odds
  // pull runs every time, but log_cfb_predictions exits early when CFBD is out
  // of quota with nothing cached -- which is exactly the state it was in on
  // 2026-08-31 -- so predicted_at can legitimately not move on a run that
  // otherwise succeeded. Polling it meant the button never resolved on a job
  // that had actually finished.
  async function refreshBoard() {
    setRefresh('running'); setRefreshMsg('Starting…')
    const stamp = async () => {
      const { data } = await supabase.from('cfb_line_history')
        .select('recorded_at').order('recorded_at', { ascending: false }).limit(1)
      return data?.[0] ? +new Date(data[0].recorded_at) : 0
    }
    const beforeLines = await stamp()
    const beforePreds = rows.length
      ? Math.max(...rows.map(r => +new Date(r.predicted_at || 0))) : 0
    try {
      const { error } = await supabase.functions.invoke('trigger-pipeline', {
        body: { action: 'log-cfb' },
      })
      if (error) throw error
      setRefreshMsg('Running (~3 min)…')
      for (let i = 0; i < 45; i++) {
        await new Promise(r => setTimeout(r, 6000))
        if (await stamp() > beforeLines) {
          await load()
          const { data } = await supabase.from('cfb_tracking')
            .select('predicted_at').eq('season', season)
            .order('predicted_at', { ascending: false }).limit(1)
          const movedPreds = data?.[0] && +new Date(data[0].predicted_at) > beforePreds
          setRefresh('ok')
          setRefreshMsg(movedPreds ? 'Board updated'
                                   : 'Lines and results updated — projections unchanged')
          setTimeout(() => setRefresh(null), 8000)
          return
        }
      }
      setRefresh('error'); setRefreshMsg('Timed out — check the Actions tab')
    } catch (err) {
      setRefresh('error'); setRefreshMsg('Failed to start')
    }
  }

  // Record so far: the bet result, the CLV, and -- separately -- whether the
  // movement model got the DIRECTION right. Direction is the cleaner read on
  // that model, because CLV mixes being right about which way the number moves
  // with how far it happens to travel.
  const record = useMemo(() => {
    const graded = rows.filter(r => r.result)
    const clv = rows.filter(r => r.clv_points != null)
    const dir = rows.filter(r => r.direction_correct != null)
    if (!graded.length && !clv.length && !dir.length) return null
    const t = graded.reduce((a, r) => (a[r.result] = (a[r.result] || 0) + 1, a), {})
    const n = (t.win || 0) + (t.loss || 0)
    const dRight = dir.filter(r => r.direction_correct).length
    return {
      hasGraded: graded.length > 0,
      rec: `${t.win || 0}-${t.loss || 0}${t.push ? `-${t.push}` : ''}`,
      pct: n ? `${((t.win || 0) / n * 100).toFixed(0)}%` : '—',
      n,
      played: rows.filter(r => r.home_score != null).length,
      noSide: rows.filter(r => r.home_score != null && r.predicted_side == null).length,
      avgClv: clv.length
        ? (clv.reduce((a, r) => a + r.clv_points, 0) / clv.length).toFixed(2) : null,
      clvN: clv.length,
      dirRec: `${dRight}-${dir.length - dRight}`,
      dirPct: dir.length ? `${(dRight / dir.length * 100).toFixed(0)}%` : null,
      dirN: dir.length,
      flat: rows.filter(r => r.closing_line != null && r.direction_correct == null).length,
    }
  }, [rows])

  useEffect(() => {
    (async () => { setLoading(true); await load(); setLoading(false) })()
  }, [season])

  // Played vs still to come. A game counts as final once it has a score, so
  // rows never fall between the two buckets while grading catches up.
  //
  // Worth knowing: there is no "week 0" to filter on. CFBD labels the opening
  // Friday/Saturday games as week 1, so week 1 here runs Aug 29 to Sep 7 and
  // covers both the openers and the following weekend. Played-vs-upcoming is
  // what actually separates the games already behind us.
  const bucket = useMemo(() => ({
    upcoming: rows.filter(r => r.home_score == null),
    final: rows.filter(r => r.home_score != null),
  }), [rows])

  const pool = status === 'all' ? rows : bucket[status]

  const weeks = useMemo(() => {
    const seen = new Map()
    for (const r of pool) {
      const k = r.week ?? 'none'
      seen.set(k, (seen.get(k) || 0) + 1)
    }
    return [...seen.entries()].sort((a, b) =>
      a[0] === 'none' ? 1 : b[0] === 'none' ? -1 : a[0] - b[0])
  }, [pool])

  const visible = useMemo(() => {
    let s = week === 'all' ? pool
          : pool.filter(r => (r.week ?? 'none') === week)
    s = [...s]
    if (sort === 'disagree') {
      s.sort((a, b) => Math.abs(b.margin_disagreement ?? 0) - Math.abs(a.margin_disagreement ?? 0))
    } else if (status === 'final') {
      // looking back: most recent first
      s.sort((a, b) => new Date(b.commence_time) - new Date(a.commence_time))
    } else {
      s.sort((a, b) => new Date(a.commence_time) - new Date(b.commence_time))
    }
    return s
  }, [pool, week, sort, status])

  if (loading) {
    return <div className="flex items-center justify-center h-48 text-gray-600 text-sm">Loading college football…</div>
  }
  if (rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2">
        <span>No college predictions logged for {season}</span>
        <span className="text-xs text-gray-700">
          Run <code className="bg-gray-900 px-1 rounded">fetch_cfb_odds.py</code> then{' '}
          <code className="bg-gray-900 px-1 rounded">log_cfb_predictions.py</code>
        </span>
      </div>
    )
  }

  const GRID = 'grid-cols-[24px_1fr_98px_122px_58px_62px_88px_74px_146px_46px]'

  return (
    <div className="p-4 space-y-4 max-w-6xl mx-auto">

      <div className="flex items-start gap-2 text-xs bg-amber-950/30 border border-amber-900/60 rounded p-3">
        <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-500" />
        <div className="space-y-1.5 text-amber-200/70">
          <div className="text-amber-300 font-semibold uppercase tracking-wider text-[11px]">
            Nothing here is flagged as a bet — on purpose
          </div>
          <div>
            The college spread model reaches <span className="text-amber-200">p = 0.077</span> against a
            label-permutation null, where the NFL spread model reaches p ≤ 0.038. In plain terms: one run in
            thirteen of pure noise, put through this same pipeline, produces a result this good. That is not
            enough to recommend a side, so the model's opinion is shown and the decision is yours.
          </div>
          <div className="text-amber-200/50">
            Walk-forward record was 54.7% across 1,242 bets, with a clustered interval of [51.8, 57.7] that
            still contains the 52.38% break-even.
          </div>
        </div>
      </div>

      <div className="flex items-start gap-2 text-xs text-gray-500 bg-gray-900/40 border border-gray-800 rounded p-3">
        <Info size={14} className="mt-0.5 shrink-0 text-gray-600" />
        <span>
          <span className="text-gray-300">Model margin</span> is what the model expects the home side to win
          by; <span className="text-gray-300">vs line</span> is how far that sits from the opener. Positive
          means it likes the home team more than the market does.
          {' '}<span className="text-gray-400">Totals are not shown</span> — the model called 93% of week-1
          games under against a true rate of 38%, so it is withheld until in-season form exists.
          <span className="block mt-1.5">
            Typical disagreement is <span className="text-gray-300">3.5 points</span>; anything past 10 is
            almost always a lopsided spread where the model has least to go on. Sorting by disagreement
            surfaces those first, so treat the top of that list with more scepticism, not less.
          </span>
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {[['upcoming', `Upcoming ${bucket.upcoming.length}`],
          ['final', `Final ${bucket.final.length}`],
          ['all', `All ${rows.length}`]].map(([k, l]) => (
          <button key={k} onClick={() => { setStatus(k); setWeek('all') }}
            className={`px-2 py-1.5 text-xs rounded border transition-colors ${
              status === k ? 'border-gray-500 text-gray-200 bg-gray-800'
                           : 'border-gray-800 text-gray-600 hover:border-gray-600'}`}>
            {l}
          </button>
        ))}
        {weeks.length > 1 && (
          <>
            <span className="text-gray-800 px-1">|</span>
            <span className="text-xs text-gray-600">Week</span>
            <button onClick={() => setWeek('all')}
              className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                week === 'all' ? 'border-gray-500 text-gray-200 bg-gray-800'
                               : 'border-gray-800 text-gray-600 hover:border-gray-600'}`}>
              All
            </button>
            {weeks.map(([w, n]) => (
              <button key={String(w)} onClick={() => setWeek(w)}
                title={w === 'none' ? 'CFBD returned no week for this game' : `${n} games`}
                className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                  week === w ? 'border-gray-500 text-gray-200 bg-gray-800'
                             : 'border-gray-800 text-gray-600 hover:border-gray-600'}`}>
                {w === 'none' ? '—' : w}
                <span className="text-gray-700 ml-1">{n}</span>
              </button>
            ))}
          </>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs text-gray-600">Sort</span>
        {[['disagree', 'Disagreement'], ['time', 'Kickoff']].map(([k, l]) => (
          <button key={k} onClick={() => setSort(k)}
            className={`px-2 py-1.5 text-xs rounded border transition-colors ${
              sort === k ? 'border-gray-500 text-gray-200 bg-gray-800'
                         : 'border-gray-800 text-gray-600 hover:border-gray-600'}`}>
            {l}
          </button>
        ))}
        <span className="text-xs text-gray-600 ml-2">
          {visible.length} shown{visible.length !== rows.length && ` of ${rows.length}`}
        </span>
        <div className="ml-auto flex items-center gap-2 text-xs">
          {refreshMsg && (
            <span className={refresh === 'error' ? 'text-red-400'
                           : refresh === 'ok' ? 'text-green-400' : 'text-gray-500'}>
              {refreshMsg}
            </span>
          )}
          <button onClick={refreshBoard} disabled={refresh === 'running'}
                  title="Pull the current college board, grade finished games and rescore"
                  className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded border ${
                    refresh === 'running' ? 'border-gray-800 text-gray-600 cursor-not-allowed'
                                          : 'border-gray-700 text-gray-300 hover:border-gray-500'}`}>
            <RefreshCw size={11} className={refresh === 'running' ? 'animate-spin' : ''} />
            {refresh === 'running' ? 'Refreshing…' : 'Refresh board'}
          </button>
        </div>
      </div>

      {record && (
        <div className="bg-gray-900 rounded border border-gray-800 px-4 py-3 text-xs space-y-2">
          <div className="flex flex-wrap gap-x-6 gap-y-1">
            <span className="text-gray-500">Season to date:</span>
            {record.hasGraded && (
              <span className="text-gray-300">
                against the spread <span className="text-gray-100">{record.rec} ({record.pct})</span>
              </span>
            )}
            {record.dirPct && (
              <span className="text-gray-300">
                movement direction <span className="text-gray-100">{record.dirRec} ({record.dirPct})</span>
              </span>
            )}
            {record.avgClv != null && (
              <span className="text-gray-300">
                average CLV <span className="text-gray-100">{record.avgClv > 0 ? '+' : ''}{record.avgClv}</span>
                <span className="text-gray-600"> over {record.clvN}</span>
              </span>
            )}
          </div>
          <div className="text-gray-600 leading-relaxed">
            Direction is scored separately from CLV, and is the cleaner read on the movement model —
            CLV mixes being right about which way a number moves with how far it happens to travel.
            {record.flat > 0 && ` ${record.flat} games are excluded because the line has not moved half a point,
            which is the smallest tick the market trades in.`}
            {record.hasGraded
              ? ` ${record.played} final, ${record.noSide} with no side because the models split.`
              : ' No games have finished yet.'}
            {' '}Samples this small say nothing either way — the walk-forward estimate was 54.7% with an
            interval that still contains break-even.
          </div>
        </div>
      )}

      <div className="bg-gray-900 rounded border border-gray-800 overflow-hidden">
        <div className={`hidden md:grid ${GRID} gap-2 px-4 py-2 text-xs text-gray-600 uppercase tracking-wider border-b border-gray-800`}>
          <span />
          <span>Game</span>
          <span>Kickoff</span>
          <span className="text-right">Open → now</span>
          <span className="text-right">Moved</span>
          <span className="text-right">Proj.</span>
          <span className="text-right">Model margin</span>
          <span className="text-right">vs line</span>
          <span className="text-right">Leans</span>
          <span className="text-right">W/L</span>
        </div>

        <div className="divide-y divide-gray-800/50">
          {visible.length === 0 && (
            <div className="px-4 py-6 text-center text-xs text-gray-600">
              {status === 'final'
                ? 'No games have finished yet.'
                : 'Nothing matches this filter.'}
            </div>
          )}
          {visible.map(r => {
            const key = `${r.game_id}_${r.bet_type}`
            const isOpen = open === key
            // A null side means the movement and margin models point opposite
            // ways, which is the tested rule declining to pick. Ternary on
            // 'home' alone would silently render the AWAY team as the pick.
            const side = r.predicted_side == null ? null
                       : r.predicted_side === 'home' ? r.home_team : r.away_team
            const dis = r.margin_disagreement
            return (
              <div key={key}>
                <div role="button" tabIndex={0}
                     onClick={() => setOpen(isOpen ? null : key)}
                     onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(isOpen ? null : key) } }}
                     className={`grid ${GRID} gap-2 px-4 py-2.5 text-sm items-center cursor-pointer hover:bg-gray-800/30`}>
                  <div>{isOpen ? <ChevronDown size={12} className="text-gray-500" />
                               : <ChevronRight size={12} className="text-gray-600" />}</div>
                  <div className="text-gray-100 truncate text-xs"
                       title={`${r.away_team} @ ${r.home_team}`}>{r.away_team} @ {r.home_team}</div>
                  <div className="text-gray-500 text-xs truncate">{fmtDate(r.commence_time)}</div>
                  <div className="text-right text-xs tabular-nums whitespace-nowrap"
                       title={r.closing_line == null ? 'No current line captured yet'
                              : `Opened ${fmtLine(r.open_line)}, currently ${fmtLine(r.closing_line)}`}>
                    <span className="text-gray-600">{fmtLine(r.open_line)}</span>
                    <span className="text-gray-700 mx-0.5">→</span>
                    <span className="text-gray-200">
                      {r.closing_line == null ? '—' : fmtLine(r.closing_line)}
                    </span>
                  </div>
                  <div className={`text-right text-xs tabular-nums ${
                    !r.actual_movement ? 'text-gray-700' : moveClass(r, 'text-gray-200')
                  }`} title={moveTitle(r)}>
                    {r.actual_movement == null ? '—'
                     : r.actual_movement === 0 ? '0.0' : fmtLine(r.actual_movement)}
                  </div>
                  <div className="text-right text-xs tabular-nums text-gray-500">
                    {r.predicted_movement == null ? '—' : fmtLine(r.predicted_movement)}
                  </div>
                  <div className="text-right text-gray-300 text-xs tabular-nums">
                    {r.projected_value == null ? '—' : fmtLine(r.projected_value)}
                  </div>
                  <div className={`text-right text-xs tabular-nums ${
                    dis == null ? 'text-gray-600'
                    : Math.abs(dis) >= 3 ? 'text-gray-200 font-medium' : 'text-gray-500'
                  }`}>{dis == null ? '—' : fmtLine(dis)}</div>
                  <div className={`text-right text-xs truncate ${
                    side ? 'text-gray-300' : 'text-gray-600 italic'
                  }`} title={side ? `Model backs ${side}` : 'The movement and margin models disagree, so the rule declines to pick'}>
                    {side ?? 'models split'}
                  </div>
                  <div className="text-right text-xs">
                    {r.result ? (
                      <span className={`px-1 py-0.5 rounded text-[10px] uppercase ${
                        r.result === 'win' ? 'bg-green-950 text-green-400'
                        : r.result === 'loss' ? 'bg-red-950 text-red-400'
                        : 'bg-gray-800 text-gray-400'}`}>
                        {r.result === 'win' ? 'W' : r.result === 'loss' ? 'L' : 'P'}
                      </span>
                    ) : <span className="text-gray-700">—</span>}
                  </div>
                </div>

                {isOpen && (
                  <div className="px-4 py-3 bg-gray-950/60 border-t border-gray-800/70">
                    {(() => {
                      const d = sideDetail(r)
                      return (
                        <div className="mb-3 text-xs leading-relaxed bg-gray-900/50 border-l-2 border-gray-700 pl-3 py-2">
                          {d.agree ? (
                            <>
                              <span className="text-gray-500">Model's side: </span>
                              <span className="text-gray-100 font-medium">
                                {d.team} {fmtLine(d.num)}
                              </span>
                              <div className="text-gray-500 mt-1">
                                Both models point the same way — the movement model expects the number to
                                move toward {d.mvTeam}, and the margin model makes {d.disTeam} the side.
                                Shown as information only; nothing on this page is flagged as a bet.
                              </div>
                            </>
                          ) : (
                            <>
                              <span className="text-gray-500">Model's side: </span>
                              <span className="text-amber-300/90 font-medium">none — the models split</span>
                              <div className="text-gray-500 mt-1">
                                The movement model leans <span className="text-gray-300">{d.mvTeam ?? '—'}</span>,
                                the margin model leans <span className="text-gray-300">{d.disTeam ?? '—'}</span>.
                                The tested rule only takes a side when they agree, so it declines here.
                              </div>
                            </>
                          )}
                        </div>
                      )
                    })()}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3 gap-y-4">
                      <Field label="Kickoff" value={fmtDateFull(r.commence_time)} />
                      {r.home_score != null && (
                        <Field label="Final"
                               value={`${r.away_team} ${r.away_score} — ${r.home_team} ${r.home_score}`} />
                      )}
                      <Field label="Opening line" value={fmtLine(r.open_line)} />
                      <Field label="Current line"
                             value={r.closing_line == null ? 'not captured' : fmtLine(r.closing_line)}
                             color={r.closing_line == null ? 'text-gray-500' : 'text-gray-100'} />
                      <Field label="Moved so far"
                             value={r.actual_movement == null ? '—'
                                    : r.actual_movement === 0 ? 'no move yet' : fmtLine(r.actual_movement)}
                             color={r.actual_movement ? moveClass(r, 'text-gray-200') : 'text-gray-500'} />
                      <Field label="Predicted move" value={r.predicted_movement == null ? '—' : fmtLine(r.predicted_movement)} />
                      <Field label="Model margin" value={r.projected_value == null ? '—' : fmtLine(r.projected_value)} />
                      <Field label="vs the line" value={dis == null ? '—' : fmtLine(dis)} />
                      <Field label="Direction so far"
                             value={r.direction_correct == null ? 'no move yet'
                                    : r.direction_correct ? 'right' : 'wrong'}
                             color={r.direction_correct == null ? 'text-gray-500'
                                    : r.direction_correct ? 'text-green-400' : 'text-red-400'} />
                      <Field label="Snapshots" value={r.n_snapshots ?? '—'}
                             color="text-gray-500" />
                      <Field
                        label="CLV so far"
                        value={r.predicted_side == null ? 'no side taken'
                               : r.clv_points == null ? 'pending' : fmtLine(r.clv_points)}
                        color={r.clv_points == null ? 'text-gray-500'
                               : r.clv_points > 0 ? 'text-green-400'
                               : r.clv_points < 0 ? 'text-red-400' : 'text-gray-300'}
                      />
                    </div>
                    <p className="text-[11px] text-gray-600 mt-3">
                      Week 1 predictions rest entirely on preseason inputs — prior-season SP+ and FPI,
                      recruiting talent, returning production and carried-over Elo. There is no in-season form
                      yet, so treat these as the model's prior rather than a read on how teams are playing.
                    </p>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
