// Touchdown scorer board — anytime and first TD, priced across every book.
//
// NOTHING HERE IS A BET. This is phase 0 of the touchdown work: capture prices,
// grade from play-by-play, model nothing. The tab exists so the data can be
// looked at while it accumulates, not because an edge has been demonstrated.
//
// Three things worth understanding before reading the numbers:
//
//  1. SHOP is assumption-free. It is the best available price against the
//     median book, in payout terms. No de-vig, no model — just dispersion
//     between books, which was measured at a median +9% and a 90th percentile
//     of +24% across a full slate.
//
//  2. EV depends on a de-vig, and the de-vig is only exact for FIRST TD, where
//     the probabilities must sum to 1 because exactly one player scores first.
//     Anytime scales to the GAME's own expected scorer count, measured on 2,214
//     games as -0.101 + 0.0998 x game total -- near enough total/10, monotonic
//     across every total bucket, 3.7 scorers at a 38 total and 5.2 at 54. Still
//     an estimate (r=0.266: the total explains the mean, not the game), but no
//     longer the same 4.3 for a slog and a shootout.
//
//  3. Multiplicative de-vig overstates longshots, because books hold more on
//     them. Unfiltered, the top of an EV list is all +3000 bench players. The
//     "realistic" filter (default ON) hides anything longer than +1000 for
//     exactly that reason — turn it off to see the artifact for yourself.
import { useState, useEffect, useMemo } from 'react'
import { ChevronDown, ChevronRight, RefreshCw } from 'lucide-react'
import { supabase } from '../lib/supabase'

const fmtPrice = p => (p == null ? '—' : p > 0 ? `+${p}` : `${p}`)
const fmtPct = v => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)
const BOOKS = {
  draftkings: 'DK', fanduel: 'FD', betmgm: 'MGM', williamhill_us: 'Caesars',
  betrivers: 'BetRivers', espnbet: 'ESPN', betonlineag: 'BetOnline',
  lowvig: 'LowVig', bovada: 'Bovada', fanatics: 'Fanatics',
}
const book = k => BOOKS[k] ?? k

// Longer than this and the de-vig's longshot bias dominates the EV number.
const REALISTIC_MAX_PRICE = 1000

function fmtKick(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleString('en-US', {
    weekday: 'short', month: 'numeric', day: 'numeric',
    hour: 'numeric', minute: '2-digit',
  })
}

export default function TdPanel() {
  const [rows, setRows] = useState([])
  const [results, setResults] = useState([])
  const [market, setMarket] = useState('anytime')
  const [realistic, setRealistic] = useState(true)
  const [minBooks, setMinBooks] = useState(5)
  const [sort, setSort] = useState('ev')
  const [query, setQuery] = useState('')
  const [game, setGame] = useState('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => { load() }, [])

  async function load() {
    setLoading(true)
    const PAGE = 1000
    let all = []
    for (let from = 0; ; from += PAGE) {
      const { data, error } = await supabase.from('nfl_td_board').select('*')
        .order('commence_time').range(from, from + PAGE - 1)
      if (error || !data || !data.length) break
      all = all.concat(data)
      if (data.length < PAGE) break
    }
    setRows(all)
    const { data: res } = await supabase.from('nfl_td_results')
      .select('player, team, tds, scored_first, week, season')
      .order('week', { ascending: false }).limit(1000)
    setResults(res ?? [])
    setLoading(false)
  }

  const now = Date.now()

  const games = useMemo(() => {
    const m = new Map()
    rows.forEach(r => {
      if (!m.has(r.event_id)) {
        m.set(r.event_id, {
          id: r.event_id, label: `${r.away_team} @ ${r.home_team}`,
          kick: r.commence_time, total: r.game_total, scorers: r.assumed_scorers,
        })
      }
    })
    return [...m.values()].sort((a, b) => new Date(a.kick) - new Date(b.kick))
  }, [rows])

  const visible = useMemo(() => {
    let s = rows.filter(r => r.market === market)
    // Kicked-off games drop out: the price is gone and the row is history.
    s = s.filter(r => !r.commence_time || new Date(r.commence_time) > now)
    if (game !== 'all') s = s.filter(r => r.event_id === game)
    if (realistic) s = s.filter(r => r.best_price != null && r.best_price <= REALISTIC_MAX_PRICE)
    s = s.filter(r => (r.n_books ?? 0) >= minBooks)
    const q = query.trim().toLowerCase()
    if (q) s = s.filter(r => `${r.player} ${r.home_team} ${r.away_team}`.toLowerCase().includes(q))
    const by = {
      ev: (a, b) => (b.ev_pct ?? -9) - (a.ev_pct ?? -9),
      shop: (a, b) => (b.shop_gain ?? -9) - (a.shop_gain ?? -9),
      prob: (a, b) => (b.p_fair ?? 0) - (a.p_fair ?? 0),
      price: (a, b) => (a.best_price ?? 9e9) - (b.best_price ?? 9e9),
      game: (a, b) => new Date(a.commence_time) - new Date(b.commence_time),
    }
    return [...s].sort(by[sort] ?? by.ev)
  }, [rows, market, realistic, minBooks, sort, query, game, now])

  const scorers = useMemo(() => {
    const m = new Map()
    results.forEach(r => {
      const k = r.player
      const cur = m.get(k) ?? { tds: 0, firsts: 0, team: r.team }
      cur.tds += r.tds ?? 0
      if (r.scored_first) cur.firsts += 1
      m.set(k, cur)
    })
    return [...m.entries()].sort((a, b) => b[1].tds - a[1].tds).slice(0, 10)
  }, [results])

  const pill = (active, on, label, title) => (
    <button onClick={on} title={title}
      className={`px-2 py-1 rounded border text-xs ${active
        ? 'border-gray-500 text-gray-200 bg-gray-800'
        : 'border-gray-800 text-gray-500 hover:text-gray-300'}`}>
      {label}
    </button>
  )

  return (
    <div className="max-w-6xl mx-auto px-4 py-4 space-y-3">
      <div className="border border-amber-900/60 bg-amber-950/20 rounded p-3 text-xs text-amber-200/90 space-y-1">
        <div className="font-semibold uppercase tracking-wider text-amber-300">
          Nothing here is a bet — capture only
        </div>
        <p>
          This is the raw touchdown board while the price history accumulates. No
          model has been fitted and no edge has been demonstrated.{' '}
          <span className="text-amber-100">SHOP</span> is assumption-free: the
          best available price against the median book, in payout terms.{' '}
          <span className="text-amber-100">EV</span> needs a de-vig, which is
          exact only for first TD (probabilities must sum to 1); anytime scales
          to the game's own expected scorer count, which tracks the total at
          roughly total ÷ 10 (measured on 2,214 games, monotonic across every
          bucket: 3.7 scorers at a 38 total, 5.2 at 54).
        </p>
        <p className="text-amber-200/70">
          De-vigging multiplicatively overstates longshots because books hold
          more on them, so the <span className="text-amber-100">realistic</span>{' '}
          filter hides anything longer than +{REALISTIC_MAX_PRICE}. Turn it off
          and the EV list fills with +3000 bench players — that is the artifact,
          not an edge. Worth noticing: where the de-vig is exact (first TD)
          almost nothing prices as positive, which is the market telling you it
          is efficient on the obvious names.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1">
          {pill(market === 'anytime', () => setMarket('anytime'), 'Anytime',
            'Anytime touchdown scorer')}
          {pill(market === 'first', () => setMarket('first'), 'First TD',
            'First touchdown of the game — the de-vig is exact here')}
        </div>
        <div className="flex gap-1">
          {pill(sort === 'ev', () => setSort('ev'), 'EV')}
          {pill(sort === 'shop', () => setSort('shop'), 'Shop')}
          {pill(sort === 'prob', () => setSort('prob'), 'Likeliest')}
          {pill(sort === 'price', () => setSort('price'), 'Shortest price')}
          {pill(sort === 'game', () => setSort('game'), 'Kickoff')}
        </div>
        {pill(realistic, () => setRealistic(!realistic),
          `≤ +${REALISTIC_MAX_PRICE}`, 'Hide longshots, where the de-vig misleads')}
        <div className="flex gap-1">
          {[3, 5, 7].map(n => pill(minBooks === n, () => setMinBooks(n),
            `${n}+ books`, 'Minimum books quoting this player'))}
        </div>
        <select value={game} onChange={e => setGame(e.target.value)}
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-xs text-gray-300">
          <option value="all">All games</option>
          {games.map(g => (
            <option key={g.id} value={g.id}>
              {g.label}{g.total != null ? ` · ${g.total} → ${g.scorers} scorers` : ''}
            </option>
          ))}
        </select>
        <input value={query} onChange={e => setQuery(e.target.value)}
          placeholder="player or team"
          className="bg-gray-900 border border-gray-800 rounded px-2 py-1 text-xs text-gray-300 w-40" />
        <button onClick={load}
          className="ml-auto flex items-center gap-1 px-2 py-1 rounded border border-gray-800 text-xs text-gray-400 hover:text-gray-200">
          <RefreshCw size={11} className={loading ? 'animate-spin' : ''} /> Reload
        </button>
      </div>

      <div className="text-xs text-gray-500">
        {loading ? 'loading…'
          : `${visible.length} prices · ${games.length} games · best number across every book quoting`}
      </div>

      <div className="border border-gray-800 rounded overflow-hidden">
        <div className="hidden md:grid grid-cols-[1fr_150px_90px_76px_70px_70px_60px] gap-2 px-3 py-2 bg-gray-900/60 text-[10px] uppercase tracking-wider text-gray-500">
          <span>Player</span>
          <span>Game</span>
          <span className="text-right">Best</span>
          <span className="text-right">Fair %</span>
          <span className="text-right">EV</span>
          <span className="text-right">Shop</span>
          <span className="text-right">Books</span>
        </div>
        <div className="divide-y divide-gray-800/50">
          {visible.map(r => {
            const key = `${r.event_id}_${r.market}_${r.player}`
            const evGood = (r.ev_pct ?? 0) > 0
            return (
              <div key={key} className="hover:bg-gray-800/30">
                {/* phone: two lines */}
                <div className="md:hidden px-3 py-2">
                  <div className="flex items-center gap-2">
                    <span className="text-gray-100 text-sm font-medium truncate">{r.player}</span>
                    <span className="ml-auto text-sm tabular-nums text-gray-200 shrink-0">
                      {fmtPrice(r.best_price)}
                    </span>
                    <span className="text-[10px] text-gray-500 shrink-0">{book(r.best_book)}</span>
                  </div>
                  <div className="flex items-center gap-3 mt-0.5 text-xs">
                    <span className="text-gray-500 truncate">{r.away_team} @ {r.home_team}</span>
                    <span className="text-gray-400 tabular-nums">fair {fmtPct(r.p_fair)}</span>
                    <span className={`tabular-nums ${evGood ? 'text-green-400' : 'text-gray-600'}`}>
                      EV {fmtPct(r.ev_pct)}
                    </span>
                    <span className="ml-auto text-gray-600 tabular-nums">{r.n_books}b</span>
                  </div>
                </div>
                {/* desktop */}
                <div className="hidden md:grid grid-cols-[1fr_150px_90px_76px_70px_70px_60px] gap-2 px-3 py-2 text-sm items-center">
                  <span className="text-gray-100 truncate">{r.player}</span>
                  <span className="text-gray-500 text-xs truncate"
                        title={`${fmtKick(r.commence_time)}${r.game_total != null
                          ? ` · total ${r.game_total}, de-vig scaled to ${r.assumed_scorers} scorers`
                          : ' · no total, de-vig scaled to the 4.43 league mean'}`}>
                    {r.away_team} @ {r.home_team}
                    {r.game_total != null && (
                      <span className="text-gray-700 ml-1">{r.game_total}</span>
                    )}
                  </span>
                  <span className="text-right tabular-nums">
                    <span className="text-gray-200">{fmtPrice(r.best_price)}</span>
                    <span className="text-gray-600 text-[10px] ml-1">{book(r.best_book)}</span>
                  </span>
                  <span className="text-right tabular-nums text-gray-400 text-xs">{fmtPct(r.p_fair)}</span>
                  <span className={`text-right tabular-nums text-xs ${evGood ? 'text-green-400' : 'text-gray-600'}`}>
                    {fmtPct(r.ev_pct)}
                  </span>
                  <span className="text-right tabular-nums text-xs text-sky-300"
                        title="Best price vs the median book — no de-vig assumed">
                    {fmtPct(r.shop_gain)}
                  </span>
                  <span className="text-right tabular-nums text-xs text-gray-600">{r.n_books}</span>
                </div>
              </div>
            )
          })}
          {!loading && !visible.length && (
            <div className="px-3 py-6 text-center text-sm text-gray-500">
              No prices match this filter. Markets usually post a few days out.
            </div>
          )}
        </div>
      </div>

      {scorers.length > 0 && (
        <div className="border border-gray-800 rounded p-3">
          <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-2">
            Season so far — from play-by-play, not from any model
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-xs">
            {scorers.map(([name, s]) => (
              <div key={name} className="flex justify-between gap-2">
                <span className="text-gray-300 truncate">{name}</span>
                <span className="text-gray-500 tabular-nums shrink-0">
                  {s.tds} TD{s.firsts ? ` · ${s.firsts} 1st` : ''}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
