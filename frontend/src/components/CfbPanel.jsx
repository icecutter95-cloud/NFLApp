import { useState, useEffect, useMemo } from 'react'
import { AlertTriangle, Info, ChevronRight, ChevronDown } from 'lucide-react'
import { supabase } from '../lib/supabase'

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

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
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
      if (!cancelled) { setRows(all); setLoading(false) }
    }
    load()
    return () => { cancelled = true }
  }, [season])

  const visible = useMemo(() => {
    const s = [...rows]
    if (sort === 'disagree') {
      s.sort((a, b) => Math.abs(b.margin_disagreement ?? 0) - Math.abs(a.margin_disagreement ?? 0))
    } else {
      s.sort((a, b) => new Date(a.commence_time) - new Date(b.commence_time))
    }
    return s
  }, [rows, sort])

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

  const GRID = 'grid-cols-[24px_1fr_145px_72px_94px_84px_158px]'

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

      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-600">Sort</span>
        {[['disagree', 'Disagreement'], ['time', 'Kickoff']].map(([k, l]) => (
          <button key={k} onClick={() => setSort(k)}
            className={`px-2 py-1.5 text-xs rounded border transition-colors ${
              sort === k ? 'border-gray-500 text-gray-200 bg-gray-800'
                         : 'border-gray-800 text-gray-600 hover:border-gray-600'}`}>
            {l}
          </button>
        ))}
        <span className="text-xs text-gray-600 ml-2">{rows.length} games</span>
      </div>

      <div className="bg-gray-900 rounded border border-gray-800 overflow-hidden">
        <div className={`hidden md:grid ${GRID} gap-2 px-4 py-2 text-xs text-gray-600 uppercase tracking-wider border-b border-gray-800`}>
          <span />
          <span>Game</span>
          <span>Kickoff</span>
          <span className="text-right">Line</span>
          <span className="text-right">Model margin</span>
          <span className="text-right">vs line</span>
          <span className="text-right">Leans</span>
        </div>

        <div className="divide-y divide-gray-800/50">
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
                  <div className="text-right text-gray-400 text-xs tabular-nums">{fmtLine(r.open_line)}</div>
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
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                      <Field label="Opening line" value={fmtLine(r.open_line)} />
                      <Field label="Model margin" value={r.projected_value == null ? '—' : fmtLine(r.projected_value)} />
                      <Field label="vs the line" value={dis == null ? '—' : fmtLine(dis)} />
                      <Field label="Predicted move" value={r.predicted_movement == null ? '—' : fmtLine(r.predicted_movement)} />
                      <Field
                        label="CLV so far"
                        value={r.clv_points == null ? 'pending' : fmtLine(r.clv_points)}
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
