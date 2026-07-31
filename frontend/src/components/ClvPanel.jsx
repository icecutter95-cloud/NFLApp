import { useState, useEffect, useMemo } from 'react'
import { TrendingUp, TrendingDown, Minus, Info, Target, AlertTriangle, ChevronRight, ChevronDown } from 'lucide-react'
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
//
// Spreads and totals are both tracked, but they are NOT equally trustworthy and
// the UI says so loudly — see TOTALS_CAVEAT below.

const TABS = [
  { key: 'all',    label: 'All' },
  { key: 'spread', label: 'Spreads' },
  { key: 'total',  label: 'Totals' },
]

// Conviction sorts on the ABSOLUTE predicted move: a 2-point drop toward the
// home team and a 2-point drift toward the away team are equally strong reads,
// they just point opposite ways.
const SORTS = {
  time:     { label: 'Game time', fn: (a, b) => new Date(a.commence_time) - new Date(b.commence_time) },
  movement: { label: 'Proj. move', fn: (a, b) => Math.abs(b.predicted_movement ?? 0) - Math.abs(a.predicted_movement ?? 0) },
  clv:      { label: 'CLV', fn: (a, b) => (b.clv_points ?? -Infinity) - (a.clv_points ?? -Infinity) },
}

function fmtLine(v) {
  if (v == null) return '—'
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)
}

// Totals are quoted as a single number, so the sign prefix a spread needs is
// just noise here.
function fmtNum(v) {
  return v == null ? '—' : v.toFixed(1)
}

// American odds always carry an explicit sign.
function fmtPrice(v) {
  return v == null ? '' : v > 0 ? `+${v}` : `${v}`
}

const BOOK_NAMES = {
  draftkings: 'DraftKings', fanduel: 'FanDuel', betmgm: 'BetMGM',
  williamhill_us: 'Caesars', betrivers: 'BetRivers', espnbet: 'ESPN Bet',
  betonlineag: 'BetOnline', lowvig: 'LowVig', bovada: 'Bovada',
  pinnacle: 'Pinnacle',
}
const bookName = k => BOOK_NAMES[k] ?? k

// The best number available right now for the side we are on, pulled from
// best_book_lines. Shopping is worth ~+0.30 pts a bet — about 15% on top of
// the qualifying filter's +1.93 CLV — so the headline row shows where to bet,
// not just what DraftKings says.
function bestFor(row, books) {
  if (!books) return null
  const m = {
    home:  ['home_book', 'home_line', 'home_price'],
    away:  ['away_book', 'away_line', 'away_price'],
    over:  ['over_book', 'over_line', 'over_price'],
    under: ['under_book', 'under_line', 'under_price'],
  }[row.predicted_side]
  if (!m || books[m[0]] == null) return null
  return { book: books[m[0]], line: books[m[1]], price: books[m[2]] }
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

function TotalsCaveat() {
  return (
    <div className="flex items-start gap-2 text-xs bg-amber-950/30 border border-amber-900/60 rounded p-3">
      <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-500" />
      <div className="space-y-1.5 text-amber-200/70">
        <div className="text-amber-300 font-semibold uppercase tracking-wider text-[11px]">
          Totals are lower confidence than spreads — bet them smaller
        </div>
        <div>
          A qualifying <span className="text-amber-200">spread</span> needs two independent signals to agree.
          A qualifying <span className="text-amber-200">total</span> rests on <span className="text-amber-200">one</span>:
          the movement model alone. The margin-disagreement signal that carries half the weight on spreads is
          worthless on totals — 49–51% at every threshold tested, i.e. a coin flip.
        </div>
        <div>
          The 1.25-point bar is also fragile. It held in both test periods (59.2% select / 56.6% holdout),
          but the very next bar up, 1.5, <span className="text-amber-200">inverted</span> out of sample —
          63.0% on the selection years, 47.4% on the holdout. That is what an unstable signal looks like,
          and it is why there is no "more conviction = more confident" ladder here.
        </div>
        <div className="text-amber-200/50">
          Treat a qualifying total as a quarter unit at most, and never size it like a qualifying spread.
        </div>
      </div>
    </div>
  )
}

function Field({ label, value, color, title }) {
  return (
    <div title={title}>
      <div className="text-[10px] text-gray-600 uppercase tracking-wider">{label}</div>
      <div className={`text-xs tabular-nums ${color ?? 'text-gray-200'}`}>{value}</div>
    </div>
  )
}

// Everything we know about one line, opened up: how the model got here, where
// the number has travelled, and what every book is currently offering.
function RowDetail({ row, books, quotes, loading }) {
  const isTotal = row.bet_type === 'total'
  const fmt = isTotal ? fmtNum : fmtLine
  const side = row.predicted_side
  const best = bestFor(row, books)

  // Which column of the book table is the side we are actually on.
  const rank = q => {
    if (!isTotal) return side === 'home' ? q.spread_home : -q.spread_home
    return side === 'over' ? -q.total : q.total
  }
  const sorted = [...(quotes ?? [])].sort((a, b) => (rank(b) ?? -99) - (rank(a) ?? -99))
  const dk = (quotes ?? []).find(q => q.book === 'draftkings')
  const dkLine = dk ? (isTotal ? dk.total : (side === 'away' ? -dk.spread_home : dk.spread_home)) : null
  const dkPrice = dk ? ({ home: dk.spread_home_price, away: dk.spread_away_price,
                          over: dk.over_price, under: dk.under_price })[side] : null

  // Points first, but a better PRICE on the same number is real value too:
  // -105 pays more than -110 on an identical bet. In American odds the larger
  // number is always the better payout, on either side of zero.
  let shopping = null
  if (best && dkLine != null) {
    const pts = isTotal && side === 'over' ? dkLine - best.line : best.line - dkLine
    if (pts > 0) shopping = { good: true, text: `+${pts.toFixed(1)} pts` }
    else if (pts === 0 && best.price != null && dkPrice != null && best.price > dkPrice)
      shopping = { good: true, text: `same number, better price (${fmtPrice(best.price)} vs ${fmtPrice(dkPrice)})` }
    else shopping = { good: false, text: 'DraftKings is already the best' }
  }

  return (
    <div className="px-4 py-4 bg-gray-950/60 border-t border-gray-800/70 space-y-4">

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Field label="Opened" value={fmt(row.open_line)} />
        <Field label="Projected close" value={fmt(row.projected_close)}
               title="Where the movement model expects the number to settle" />
        <Field label="Current close" value={fmt(row.closing_line)} />
        <Field
          label="Moved so far"
          value={row.actual_movement == null ? 'no move yet' : fmtLine(row.actual_movement)}
          color={row.actual_movement ? 'text-gray-200' : 'text-gray-500'}
        />
        <Field
          label="CLV"
          value={row.clv_points == null ? 'pending' : `${row.clv_points >= 0 ? '+' : ''}${row.clv_points.toFixed(2)}`}
          color={row.clv_points == null ? 'text-gray-500'
                 : row.clv_points > 0 ? 'text-green-400' : row.clv_points < 0 ? 'text-red-400' : 'text-gray-300'}
        />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-1 border-t border-gray-800/50">
        <Field label="Predicted move" value={fmtLine(row.predicted_movement)} />
        {!isTotal && (
          <Field
            label="Margin disagreement"
            value={row.margin_disagreement == null ? '—' : fmtLine(row.margin_disagreement)}
            title="How far the margin model sits from what the opener implies. Positive favours home."
          />
        )}
        <Field label="Direction so far"
               value={row.direction_correct == null ? 'no move yet' : row.direction_correct ? 'right' : 'wrong'}
               color={row.direction_correct == null ? 'text-gray-500'
                      : row.direction_correct ? 'text-green-400' : 'text-red-400'} />
        <Field
          label="Qualifies"
          value={row.qualifies ? 'yes' : 'no'}
          color={row.qualifies ? 'text-green-400' : 'text-gray-500'}
          title={isTotal
            ? 'Totals need 1.25+ pts of predicted movement — one signal only'
            : 'Spreads need 3+ pts of margin disagreement AND 0.5+ pts of predicted drift the same way'}
        />
      </div>

      <div>
        <div className="flex items-baseline justify-between mb-2">
          <span className="text-[10px] text-gray-500 uppercase tracking-wider">
            Every book — {isTotal ? (side === 'over' ? 'best over is the lowest number' : 'best under is the highest number')
                                  : 'best is the most points'}
          </span>
          {shopping && (
            <span className="text-[10px] text-gray-500">
              shopping vs DraftKings:{' '}
              <span className={shopping.good ? 'text-green-400' : 'text-gray-500'}>
                {shopping.text}
              </span>
            </span>
          )}
        </div>

        {loading ? (
          <div className="text-xs text-gray-600 py-2">Loading books…</div>
        ) : sorted.length === 0 ? (
          <div className="text-xs text-gray-600 py-2">
            No book quotes stored for this game yet — they arrive with the next odds refresh.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs min-w-[420px]">
              <thead>
                <tr className="text-gray-600 border-b border-gray-800">
                  <th className="text-left pb-1.5 font-normal">Book</th>
                  <th className="text-right pb-1.5 font-normal">Spread (home)</th>
                  <th className="text-right pb-1.5 font-normal">Total</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/40">
                {sorted.map(q => {
                  const isBest = best && q.book === best.book
                  return (
                    <tr key={q.book} className={isBest ? 'bg-green-950/25' : ''}>
                      <td className="py-1.5">
                        <span className={isBest ? 'text-green-300 font-medium' : 'text-gray-300'}>
                          {bookName(q.book)}
                        </span>
                        {isBest && <span className="text-green-500 text-[10px] ml-1.5">← best</span>}
                      </td>
                      <td className="py-1.5 text-right tabular-nums text-gray-300">
                        {q.spread_home == null ? '—' : (
                          <>
                            {fmtLine(q.spread_home)}
                            <span className="text-gray-600 ml-1">
                              {fmtPrice(side === 'away' ? q.spread_away_price : q.spread_home_price)}
                            </span>
                          </>
                        )}
                      </td>
                      <td className="py-1.5 text-right tabular-nums text-gray-300">
                        {q.total == null ? '—' : (
                          <>
                            {fmtNum(q.total)}
                            <span className="text-gray-600 ml-1">
                              {fmtPrice(side === 'under' ? q.under_price : q.over_price)}
                            </span>
                          </>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default function ClvPanel({ season }) {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(true)
  const [qualifyingOnly, setQualifyingOnly] = useState(false)
  const [tab, setTab] = useState('all')
  const [sort, setSort] = useState('time')
  const [best, setBest] = useState({})        // "AWAY@HOME" -> best_book_lines row
  const [expanded, setExpanded] = useState(null)
  const [quotes, setQuotes] = useState({})    // "AWAY@HOME" -> every book's quote
  const [loadingQuotes, setLoadingQuotes] = useState(false)

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
      // Best available number per game. Keyed on the team pair because the
      // Odds API's event ids are not stable, so line_predictions falls back to
      // a synthetic id that would never match.
      const { data: bb } = await supabase.from('best_book_lines').select('*')
      const byPair = {}
      for (const b of bb ?? []) byPair[`${b.away_team}@${b.home_team}`] = b

      if (!cancelled) { setRows(all); setBest(byPair); setLoading(false) }
    }
    load()
    return () => { cancelled = true }
  }, [season])

  // Full book panel is fetched lazily — only the game being opened.
  async function toggle(key, row) {
    if (expanded === key) { setExpanded(null); return }
    setExpanded(key)
    const pair = `${row.away_team}@${row.home_team}`
    if (quotes[pair]) return
    setLoadingQuotes(true)
    const { data } = await supabase.from('book_lines').select('*')
      .eq('home_team', row.home_team).eq('away_team', row.away_team)
    setQuotes(q => ({ ...q, [pair]: data ?? [] }))
    setLoadingQuotes(false)
  }

  const inTab = useMemo(
    () => (tab === 'all' ? rows : rows.filter(r => r.bet_type === tab)),
    [rows, tab],
  )

  const stats = useMemo(() => {
    const closed = inTab.filter(r => r.clv_points != null)
    const moved = inTab.filter(r => r.direction_correct != null)
    const clv = closed.map(r => r.clv_points)
    const mean = clv.length ? clv.reduce((a, b) => a + b, 0) / clv.length : null
    return {
      qualifying: inTab.filter(r => r.qualifies).length,
      qualSpread: inTab.filter(r => r.qualifies && r.bet_type === 'spread').length,
      qualTotal: inTab.filter(r => r.qualifies && r.bet_type === 'total').length,
      tracked: inTab.length,
      resolved: closed.length,
      meanClv: mean,
      pctPositive: clv.length ? clv.filter(v => v > 0).length / clv.length * 100 : null,
      dirAcc: moved.length ? moved.filter(r => r.direction_correct).length / moved.length * 100 : null,
      dirN: moved.length,
    }
  }, [inTab])

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

  const visible = [...(qualifyingOnly ? inTab.filter(r => r.qualifies) : inTab)]
    .sort(SORTS[sort].fn)
  const showTotalsCaveat = tab !== 'spread'
  const GRID = 'grid-cols-[52px_1.4fr_70px_110px_100px_80px_90px_80px_60px_54px]'

  return (
    <div className="p-4 space-y-5 max-w-6xl mx-auto">

      <div className="flex items-start gap-2 text-xs text-gray-500 bg-gray-900/40 border border-gray-800 rounded p-3">
        <Info size={14} className="mt-0.5 shrink-0 text-gray-600" />
        <span>
          Tracks where the model thinks each line will <span className="text-gray-300">move</span>, not who covers.
          A prediction is frozen the first time a game appears, and <span className="text-gray-300">CLV</span> measures
          how much better that number is than the eventual close. Positive CLV means you hold a better price than
          the market's final answer — the fastest honest read on whether the model works.
          {' '}<span className="text-gray-400">Take</span> shows the best number and price on the board right
          now, across every book — worth about +0.30 pts a bet on its own. Click any row for the full panel.
        </span>
      </div>

      {showTotalsCaveat && <TotalsCaveat />}

      <div className="flex items-center gap-1 border-b border-gray-800">
        {TABS.map(t => {
          const n = t.key === 'all' ? rows.length : rows.filter(r => r.bet_type === t.key).length
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`px-3 py-1.5 text-xs border-b-2 -mb-px transition-colors ${
                tab === t.key
                  ? 'border-gray-300 text-gray-100'
                  : 'border-transparent text-gray-600 hover:text-gray-400'
              }`}
            >
              {t.label} <span className="text-gray-600 tabular-nums">{n}</span>
            </button>
          )
        })}
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat label="Lines tracked" value={stats.tracked} sub={`${stats.resolved} with a close`} />
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
        <Stat
          label="Qualifying"
          value={stats.qualifying}
          sub={tab === 'all' ? `${stats.qualSpread} spread · ${stats.qualTotal} total` : 'meets the bar'}
          color="text-green-400"
        />
      </div>

      <div className="flex items-start gap-3">
        <button
          onClick={() => setQualifyingOnly(v => !v)}
          className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded border shrink-0 transition-colors ${
            qualifyingOnly
              ? 'border-green-500 text-green-400 bg-green-950'
              : 'border-gray-700 text-gray-500 hover:border-gray-500'
          }`}
        >
          <Target size={12} /> Qualifying only
        </button>

        <div className="flex items-center gap-1 shrink-0">
          <span className="text-xs text-gray-600 mr-0.5">Sort</span>
          {Object.entries(SORTS).map(([key, s]) => (
            <button
              key={key}
              onClick={() => setSort(key)}
              className={`px-2 py-1.5 text-xs rounded border transition-colors ${
                sort === key
                  ? 'border-gray-500 text-gray-200 bg-gray-800'
                  : 'border-gray-800 text-gray-600 hover:border-gray-600'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>

        <span className="text-xs text-gray-600 leading-relaxed">
          <span className="text-gray-400">Spread</span> qualifies when the margin model disagrees with the opener
          by 3+ pts AND the movement model expects 0.5+ pts of drift the same way — 61.5% / 60.0% across the two
          test periods at ~2.5 games a week.{' '}
          <span className="text-amber-500/80">Total</span> qualifies on 1.25+ pts of predicted movement alone,
          which is a weaker and less stable bar — see the warning above.
        </span>
      </div>

      <div className="bg-gray-900 rounded border border-gray-800 overflow-hidden">
        <div className={`hidden md:grid ${GRID} gap-2 px-4 py-2 text-xs text-gray-600 uppercase tracking-wider border-b border-gray-800`}>
          <span>Type</span>
          <span>Game</span>
          <span className="text-right">Opened</span>
          <span className="text-right">We project (move)</span>
          <span className="text-right">Take</span>
          <span className="text-right">Current</span>
          <span className="text-right">Moved</span>
          <span className="text-right">CLV</span>
          <span className="text-right">Dir</span>
          <span className="text-right">Result</span>
        </div>

        <div className="divide-y divide-gray-800/50">
          {visible.map(r => {
            const isTotal = r.bet_type === 'total'
            const pending = r.clv_points == null
            const clv = r.clv_points ?? 0
            const fmt = isTotal ? fmtNum : fmtLine
            const side = isTotal
              ? (r.predicted_side === 'over' ? 'Over' : 'Under')
              : (r.predicted_side === 'home' ? r.home_team : r.away_team)
            const pair = `${r.away_team}@${r.home_team}`
            const key = `${r.game_id}_${r.bet_type}`
            const bb = bestFor(r, best[pair])
            const isOpen = expanded === key
            return (
              <div key={key} className={
                r.qualifies
                  ? isTotal
                    ? 'bg-amber-950/20 border-l-2 border-l-amber-600'
                    : 'bg-green-950/20 border-l-2 border-l-green-600'
                  : ''
              }>
              <div role="button" tabIndex={0}
                   onClick={() => toggle(key, r)}
                   onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(key, r) } }}
                   className={`grid ${GRID} gap-2 px-4 py-2.5 text-sm items-center cursor-pointer hover:bg-gray-800/30`}>
                <div className="flex items-center gap-1">
                  {isOpen ? <ChevronDown size={12} className="text-gray-500 shrink-0" />
                          : <ChevronRight size={12} className="text-gray-600 shrink-0" />}
                  <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase tracking-wider ${
                    isTotal ? 'bg-amber-950 text-amber-500' : 'bg-gray-800 text-gray-400'
                  }`}>
                    {isTotal ? 'Tot' : 'Spr'}
                  </span>
                </div>
                <div className="min-w-0">
                  {r.qualifies && (
                    <Target size={11} className={`inline mr-1.5 ${isTotal ? 'text-amber-500' : 'text-green-500'}`} />
                  )}
                  <span className="text-gray-100">{r.away_team} @ {r.home_team}</span>
                  <span className="text-gray-600 text-xs ml-2">Wk {r.week}</span>
                </div>
                <div className="text-right text-gray-400 text-xs tabular-nums">{fmt(r.open_line)}</div>
                <div className="text-right text-xs tabular-nums">
                  <span className="text-gray-300">{fmt(r.projected_close)}</span>
                  <span className={`ml-1.5 ${sort === 'movement' ? 'text-gray-300' : 'text-gray-600'}`}>
                    ({fmtLine(r.predicted_movement)})
                  </span>
                </div>
                {/* Best number on the board right now, not DraftKings' —
                    shopping is worth ~+0.30 pts a bet and this is where it
                    gets captured. Falls back to the frozen line if no book
                    quotes have arrived yet. */}
                <div className="text-right text-xs leading-tight">
                  <div>
                    <span className={
                      r.qualifies
                        ? isTotal ? 'text-amber-300 font-semibold' : 'text-green-300 font-semibold'
                        : 'text-gray-200 font-medium'
                    }>{side}</span>
                    <span className="text-gray-300 ml-1 tabular-nums">
                      {bb ? fmt(bb.line) : fmt(r.taken_line)}
                    </span>
                    {bb?.price != null && (
                      <span className="text-gray-500 ml-1 tabular-nums">{fmtPrice(bb.price)}</span>
                    )}
                  </div>
                  <div className="text-[10px] text-gray-600 truncate">
                    {bb ? bookName(bb.book) : 'no book quotes yet'}
                  </div>
                </div>
                <div className="text-right text-gray-400 text-xs tabular-nums">{fmt(r.closing_line)}</div>
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
                {/* Graded at the frozen opener once final scores load (Tuesdays). */}
                <div className="text-right text-xs font-medium">
                  {r.result == null
                    ? <span className="text-gray-700">—</span>
                    : <span className={
                        r.result === 'win' ? 'text-green-400'
                        : r.result === 'loss' ? 'text-red-400' : 'text-gray-400'
                      }>{r.result === 'win' ? 'W' : r.result === 'loss' ? 'L' : 'Push'}</span>}
                </div>
              </div>
              {isOpen && (
                <RowDetail row={r} books={best[pair]} quotes={quotes[pair]}
                           loading={loadingQuotes && !quotes[pair]} />
              )}
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
