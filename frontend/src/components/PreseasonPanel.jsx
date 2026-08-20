import { useState, useEffect, useMemo } from 'react'
import { AlertTriangle, ChevronRight, ChevronDown, RefreshCw } from 'lucide-react'
import { supabase } from '../lib/supabase'

// Preseason board plus model projections — deliberately walled off from
// everything the rest of the app measures.
//
// The models read opponent-adjusted EPA and success rates computed from
// STARTERS in regular-season games. In preseason those players are on the
// sideline, so the features describe a team that is not playing. No figure
// established in this project (57.8% holdout, +1.55 CLV) applies here; all of
// it was measured on regular-season games, and no preseason backtest is
// possible because we hold no historical preseason lines.
//
// There is intentionally no "qualifies" flag. The spread thresholds and the
// 1.25 totals bar were fitted on regular-season behaviour; reusing them here
// would dress a guess in a validated number's clothing.

const BOOK_NAMES = {
  draftkings: 'DraftKings', fanduel: 'FanDuel', betmgm: 'BetMGM',
  williamhill_us: 'Caesars', betrivers: 'BetRivers', espnbet: 'ESPN Bet',
  betonlineag: 'BetOnline', lowvig: 'LowVig', bovada: 'Bovada',
}
const bookName = k => BOOK_NAMES[k] ?? k

const fmtLine = v => v == null ? '—' : v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)
const fmtNum = v => v == null ? '—' : v.toFixed(1)
const fmtPrice = v => v == null ? '' : v > 0 ? `+${v}` : `${v}`

// How much the model actually has to say about a game, for ranking only.
//
// Spreads score on the margin model's gap to the line, because that is the
// quantity in points -- the movement model's output is tenths and does not
// separate anything. But a game where the two models point opposite ways is a
// coin flip no matter how large that gap is, so splits are demoted below every
// game the models agree on rather than topping the list on a number that has an
// argument against it. Totals have only the one signal.
function convictionScore(r) {
  if (r.bet_type === 'total') return Math.abs(r.predicted_movement ?? 0)
  const gap = Math.abs(r.margin_disagreement ?? 0)
  return leanExplain(r)?.split ? gap - 1000 : gap
}

// Say which model drove the lean, and whether the other one agrees.
//
// The side comes from the MOVEMENT model, but the row also shows the margin
// model and its gap to the line. Those two point opposite ways on a couple of
// games a week, and when they do the panel read as though the lean was
// inverted: MIN @ NYG leaned MIN while displaying a +3.6 margin for NYG.
// Nothing is being changed about how the lean is chosen -- preseason has no
// validated rule to change it to -- only about saying out loud where it came
// from.
function leanExplain(r) {
  if (r.bet_type === 'total' || r.margin_disagreement == null) return null
  const lean = r.predicted_side === 'home' ? r.home_team : r.away_team
  const marginTeam = r.margin_disagreement > 0 ? r.home_team : r.away_team
  const split = marginTeam !== lean
  return {
    lean, marginTeam, split,
    move: Math.abs(r.predicted_movement ?? 0).toFixed(2),
    gap: Math.abs(r.margin_disagreement).toFixed(1),
  }
}

function fmtDate(s) {
  if (!s) return ''
  return new Date(s).toLocaleString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
  })
}

export default function PreseasonPanel() {
  const [rows, setRows] = useState([])
  const [quotes, setQuotes] = useState([])
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(null)
  const [sort, setSort] = useState('time')
  const [type, setType] = useState('all')
  const [refresh, setRefresh] = useState(null)   // null | 'running' | 'ok' | 'error'
  const [refreshMsg, setRefreshMsg] = useState('')

  async function load() {
    const [p, q] = await Promise.all([
      supabase.from('preseason_projections').select('*').order('commence_time'),
      supabase.from('preseason_lines').select('*'),
    ])
    setRows(p.data ?? []); setQuotes(q.data ?? [])
  }

  // Kicks off the GitHub workflow, then polls for the new rows. The job takes
  // a couple of minutes (install, model download, two API pulls), so the button
  // keeps reporting until rows actually change rather than claiming success the
  // moment the dispatch is accepted -- a 200 here only means GitHub queued it.
  async function refreshBoard() {
    setRefresh('running'); setRefreshMsg('Starting…')
    const before = rows.length ? Math.max(...rows.map(r => +new Date(r.predicted_at || 0))) : 0
    try {
      const { error } = await supabase.functions.invoke('trigger-pipeline', {
        body: { action: 'log-preseason' },
      })
      if (error) throw error
      setRefreshMsg('Running (~2 min)…')
      for (let i = 0; i < 40; i++) {
        await new Promise(r => setTimeout(r, 6000))
        const { data } = await supabase.from('preseason_projections')
          .select('predicted_at').order('predicted_at', { ascending: false }).limit(1)
        const latest = data?.[0] ? +new Date(data[0].predicted_at) : 0
        if (latest > before) {
          await load()
          setRefresh('ok'); setRefreshMsg('Board updated')
          setTimeout(() => setRefresh(null), 5000)
          return
        }
      }
      setRefresh('error'); setRefreshMsg('Timed out — check the Actions tab')
    } catch (err) {
      setRefresh('error'); setRefreshMsg('Failed to start')
    }
  }

  // Games that have already kicked off are dropped. Nothing here grades a
  // result, so a finished game is pure clutter -- and under "strongest lean" it
  // sorted into the middle of the list on last week's number, which reads as a
  // play that is still available.
  const upcoming = useMemo(() => {
    const now = Date.now()
    return rows.filter(r => new Date(r.commence_time).getTime() > now)
  }, [rows])

  const visible = useMemo(() => {
    const s = upcoming.filter(r => type === 'all' || r.bet_type === type)
    return sort === 'lean'
      ? [...s].sort((a, b) => convictionScore(b) - convictionScore(a))
      : [...s].sort((a, b) => new Date(a.commence_time) - new Date(b.commence_time))
  }, [upcoming, sort, type])

  useEffect(() => {
    (async () => { setLoading(true); await load(); setLoading(false) })()
  }, [])

  // Defined before the early returns on purpose: an empty or fully-kicked-off
  // board is precisely when this button is needed, and it used to live only in
  // the controls row that those returns skip past.
  const refreshButton = (
    <div className="flex items-center gap-2 text-xs">
      {refreshMsg && (
        <span className={
          refresh === 'error' ? 'text-red-400' :
          refresh === 'ok' ? 'text-green-400' : 'text-gray-500'
        }>{refreshMsg}</span>
      )}
      <button
        onClick={refreshBoard}
        disabled={refresh === 'running'}
        title="Pulls the current preseason board from the Odds API and reprojects it"
        className={`flex items-center gap-1.5 px-2.5 py-1 rounded border ${
          refresh === 'running'
            ? 'border-gray-800 text-gray-600 cursor-not-allowed'
            : 'border-gray-700 text-gray-300 hover:border-gray-500'
        }`}>
        <RefreshCw size={11} className={refresh === 'running' ? 'animate-spin' : ''} />
        {refresh === 'running' ? 'Refreshing…' : 'Refresh board'}
      </button>
    </div>
  )

  if (loading) {
    return <div className="flex items-center justify-center h-48 text-gray-600 text-sm">Loading preseason…</div>
  }

  if (rows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2">
        <span>No preseason projections yet</span>
        <span className="text-xs text-gray-700">Pull the current board to get started</span>
        {refreshButton}
      </div>
    )
  }

  if (upcoming.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2">
        <span>Every projected preseason game has already kicked off</span>
        <span className="text-xs text-gray-700">Pull the next slate to see fresh projections</span>
        {refreshButton}
      </div>
    )
  }

  const GRID = 'grid-cols-[24px_52px_1.3fr_1fr_80px_110px_90px_60px]'

  return (
    <div className="p-4 space-y-4 max-w-5xl mx-auto">

      <div className="flex items-start gap-2 text-xs bg-amber-950/30 border border-amber-900/60 rounded p-3">
        <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-500" />
        <div className="space-y-1.5 text-amber-200/70">
          <div className="text-amber-300 font-semibold uppercase tracking-wider text-[11px]">
            Entertainment only — none of the model's track record applies here
          </div>
          <div>
            The models read opponent-adjusted EPA and success rates from <span className="text-amber-200">starters
            in regular-season games</span>. In preseason those players are on the sideline in baseball caps, so
            the features describe a team that is not on the field. This is a category error, not a noisier version
            of the real thing.
          </div>
          <div>
            The 57.8% holdout and +1.55 CLV were both measured on regular-season games and do{' '}
            <span className="text-amber-200">not</span> carry over. No preseason backtest is possible — we hold no
            historical preseason lines to test against. Nothing here is flagged as qualifying, because those
            thresholds were fitted on regular-season behaviour.
          </div>
          <div className="text-amber-200/50">
            2026 team metrics do not exist yet either, so every team is carried at its end-of-2025 strength.
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        {[['all', `All ${upcoming.length}`],
          ['spread', `Spreads ${upcoming.filter(r => r.bet_type === 'spread').length}`],
          ['total', `Totals ${upcoming.filter(r => r.bet_type === 'total').length}`]].map(([k, label]) => (
          <button key={k} onClick={() => setType(k)}
                  className={`px-2.5 py-1 rounded border ${
                    type === k ? 'border-gray-500 text-gray-200 bg-gray-800'
                               : 'border-gray-800 text-gray-500 hover:text-gray-300'}`}>
            {label}
          </button>
        ))}
        <span className="text-gray-700 px-1">|</span>
        {[['time', 'Kickoff'], ['lean', 'Strongest lean']].map(([k, label]) => (
          <button key={k} onClick={() => setSort(k)}
                  className={`px-2.5 py-1 rounded border ${
                    sort === k ? 'border-gray-500 text-gray-200 bg-gray-800'
                               : 'border-gray-800 text-gray-500 hover:text-gray-300'}`}>
            {label}
          </button>
        ))}
        {sort === 'lean' && (
          <span className="text-gray-600">
            ranked by the margin model's gap to the line — split games sink to the bottom
          </span>
        )}

        <div className="ml-auto">{refreshButton}</div>
      </div>

      <div className="bg-gray-900 rounded border border-gray-800 overflow-hidden">
        <div className={`hidden md:grid ${GRID} gap-2 px-4 py-2 text-xs text-gray-600 uppercase tracking-wider border-b border-gray-800`}>
          <span />
          <span>Type</span>
          <span>Game</span>
          <span>Kickoff</span>
          <span className="text-right">Line</span>
          <span className="text-right">Model says (move)</span>
          <span className="text-right">Leans</span>
          <span className="text-right">Books</span>
        </div>

        <div className="divide-y divide-gray-800/50">
          {visible.map(r => {
            const key = `${r.game_id}_${r.bet_type}`
            const isTotal = r.bet_type === 'total'
            const fmt = isTotal ? fmtNum : fmtLine
            const isOpen = open === key
            const mine = quotes.filter(q => q.game_id === r.game_id)
              .sort((a, b) => (a.spread_home ?? 99) - (b.spread_home ?? 99))
            const side = isTotal
              ? (r.predicted_side === 'over' ? 'Over' : 'Under')
              : (r.predicted_side === 'home' ? r.home_team : r.away_team)
            const why = leanExplain(r)
            return (
              <div key={key}>
                <div role="button" tabIndex={0}
                     onClick={() => setOpen(isOpen ? null : key)}
                     onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(isOpen ? null : key) } }}
                     className={`grid ${GRID} gap-2 px-4 py-2.5 text-sm items-center cursor-pointer hover:bg-gray-800/30`}>
                  <div>
                    {isOpen ? <ChevronDown size={12} className="text-gray-500" />
                            : <ChevronRight size={12} className="text-gray-600" />}
                  </div>
                  <div>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider ${
                      isTotal ? 'bg-amber-950 text-amber-500' : 'bg-gray-800 text-gray-400'
                    }`}>{isTotal ? 'Tot' : 'Spr'}</span>
                  </div>
                  <div className="text-gray-100 truncate">{r.away_team} @ {r.home_team}</div>
                  <div className="text-gray-500 text-xs truncate">{fmtDate(r.commence_time)}</div>
                  <div className="text-right text-gray-400 text-xs tabular-nums">{fmt(r.open_line)}</div>
                  <div className="text-right text-xs tabular-nums">
                    <span className="text-gray-300">{fmt(r.projected_close)}</span>
                    <span className="text-gray-600 ml-1.5">({fmtLine(r.predicted_movement)})</span>
                  </div>
                  <div className="text-right text-xs font-medium truncate">
                    <span className="text-gray-200">{side}</span>
                    {why?.split && (
                      <span className="ml-1.5 text-[9px] uppercase tracking-wider text-amber-500/80"
                            title={`The movement model leans ${why.lean}; the margin model leans ${why.marginTeam}.`}>
                        split
                      </span>
                    )}
                  </div>
                  <div className="text-right text-gray-500 text-xs tabular-nums">{r.n_books ?? '—'}</div>
                </div>

                {isOpen && (
                  <div className="px-4 py-3 bg-gray-950/60 border-t border-gray-800/70 space-y-3">
                    {why && (
                      <div className="text-xs leading-relaxed bg-gray-900/50 border-l-2 border-gray-700 pl-3 py-2 text-gray-300">
                        <span className="text-gray-500">Why {why.lean}: </span>
                        the lean follows the <span className="text-gray-100">movement</span> model,
                        which expects the number to move {why.move} toward {why.lean}.
                        {why.split ? (
                          <> The margin model disagrees — it makes{' '}
                            <span className="text-amber-300/90">{why.marginTeam}</span> the side by{' '}
                            {why.gap} points, so treat this one as a coin flip.</>
                        ) : (
                          <> The margin model agrees, making {why.marginTeam} the side by {why.gap} points.</>
                        )}
                      </div>
                    )}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                      <div>
                        <div className="text-[10px] text-gray-600 uppercase tracking-wider">Consensus line</div>
                        <div className="text-gray-200 tabular-nums">{fmt(r.current_line)}</div>
                      </div>
                      <div>
                        <div className="text-[10px] text-gray-600 uppercase tracking-wider">Predicted move</div>
                        <div className="text-gray-200 tabular-nums">{fmtLine(r.predicted_movement)}</div>
                      </div>
                      {!isTotal && (
                        <>
                          <div title="Second opinion — does NOT decide the lean. Positive favours home.">
                            <div className="text-[10px] text-gray-600 uppercase tracking-wider">Model margin</div>
                            <div className="text-gray-400 tabular-nums">
                              {r.projected_margin == null ? '—' : fmtLine(r.projected_margin)}
                            </div>
                          </div>
                          <div title="The margin model's gap to the line. Second opinion — does NOT decide the lean.">
                            <div className="text-[10px] text-gray-600 uppercase tracking-wider">vs the line</div>
                            <div className="text-gray-400 tabular-nums">
                              {r.margin_disagreement == null ? '—' : fmtLine(r.margin_disagreement)}
                            </div>
                          </div>
                        </>
                      )}
                    </div>

                    <div>
                      <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">
                        Every book
                        {r.book_spread_span > 0 && (
                          <span className="ml-2 text-gray-600 normal-case tracking-normal">
                            books span {r.book_spread_span.toFixed(1)} pts
                            {r.book_spread_span >= 1.5 && ' — unusually wide, the market is guessing too'}
                          </span>
                        )}
                      </div>
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="text-gray-600 border-b border-gray-800">
                            <th className="text-left pb-1.5 font-normal">Book</th>
                            <th className="text-right pb-1.5 font-normal">Spread (home)</th>
                            <th className="text-right pb-1.5 font-normal">Total</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-800/40">
                          {mine.map(q => (
                            <tr key={q.book}>
                              <td className="py-1.5 text-gray-300">{bookName(q.book)}</td>
                              <td className="py-1.5 text-right tabular-nums text-gray-300">
                                {q.spread_home == null ? '—' : (
                                  <>{fmtLine(q.spread_home)}
                                    <span className="text-gray-600 ml-1">{fmtPrice(q.spread_home_price)}</span></>
                                )}
                              </td>
                              <td className="py-1.5 text-right tabular-nums text-gray-300">
                                {q.total == null ? '—' : (
                                  <>{q.total.toFixed(1)}
                                    <span className="text-gray-600 ml-1">{fmtPrice(q.over_price)}</span></>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      <p className="text-xs text-gray-600">
        The board fills in through August as books post the rest of the slate. Refresh with{' '}
        <code className="bg-gray-900 px-1 rounded">fetch_preseason_lines.py</code> then{' '}
        <code className="bg-gray-900 px-1 rounded">log_preseason_predictions.py</code>.
      </p>
    </div>
  )
}
