import { useState, useEffect } from 'react'
import { AlertTriangle, ChevronRight, ChevronDown } from 'lucide-react'
import { supabase } from '../lib/supabase'

// Preseason lines, for looking at only.
//
// Nothing here is modelled. Preseason results turn on which backups play and
// for how long — decided by a coach on the morning of the game — and no feature
// in this project observes that. There is no pick, no CLV and no edge claimed.
//
// These rows live in their own table for a reason: best_book_lines joins on the
// team pair alone, so a preseason DAL @ LAR in book_lines would collide with the
// regular-season DAL @ LAR and corrupt the CLV screen's "best number".

const BOOK_NAMES = {
  draftkings: 'DraftKings', fanduel: 'FanDuel', betmgm: 'BetMGM',
  williamhill_us: 'Caesars', betrivers: 'BetRivers', espnbet: 'ESPN Bet',
  betonlineag: 'BetOnline', lowvig: 'LowVig', bovada: 'Bovada',
}
const bookName = k => BOOK_NAMES[k] ?? k

const fmtLine = v => v == null ? '—' : v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1)
const fmtPrice = v => v == null ? '' : v > 0 ? `+${v}` : `${v}`

function fmtDate(s) {
  if (!s) return ''
  return new Date(s).toLocaleString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric',
    hour: 'numeric', minute: '2-digit',
  })
}

export default function PreseasonPanel() {
  const [games, setGames] = useState([])
  const [quotes, setQuotes] = useState([])
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      const [b, q] = await Promise.all([
        supabase.from('preseason_board').select('*').order('commence_time'),
        supabase.from('preseason_lines').select('*'),
      ])
      if (!cancelled) {
        setGames(b.data ?? [])
        setQuotes(q.data ?? [])
        setLoading(false)
      }
    }
    load()
    return () => { cancelled = true }
  }, [])

  if (loading) {
    return <div className="flex items-center justify-center h-48 text-gray-600 text-sm">Loading preseason lines…</div>
  }

  if (games.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2">
        <span>No preseason games on the board yet</span>
        <span className="text-xs text-gray-700">
          Run: <code className="bg-gray-900 px-1 rounded">python scripts/fetch_preseason_lines.py</code>
        </span>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-4 max-w-4xl mx-auto">

      <div className="flex items-start gap-2 text-xs bg-amber-950/30 border border-amber-900/60 rounded p-3">
        <AlertTriangle size={14} className="mt-0.5 shrink-0 text-amber-500" />
        <div className="space-y-1 text-amber-200/70">
          <div className="text-amber-300 font-semibold uppercase tracking-wider text-[11px]">
            Lines only — nothing here is modelled or bet
          </div>
          <div>
            Preseason turns on which backups play and for how long, decided the morning of the game.
            No feature in this project observes that, so there is no projection, no CLV and no pick —
            just what the books are hanging. The wide disagreement between them is itself the tell:
            they have no more idea than we do.
          </div>
        </div>
      </div>

      <div className="bg-gray-900 rounded border border-gray-800 overflow-hidden">
        <div className="hidden md:grid grid-cols-[24px_1.5fr_1fr_90px_90px_60px] gap-2 px-4 py-2 text-xs text-gray-600 uppercase tracking-wider border-b border-gray-800">
          <span />
          <span>Game</span>
          <span>Kickoff</span>
          <span className="text-right">Spread</span>
          <span className="text-right">Total</span>
          <span className="text-right">Books</span>
        </div>

        <div className="divide-y divide-gray-800/50">
          {games.map(g => {
            const isOpen = open === g.game_id
            const mine = quotes.filter(q => q.game_id === g.game_id)
              .sort((a, b) => (a.spread_home ?? 99) - (b.spread_home ?? 99))
            const spreadSpan = g.spread_high - g.spread_low
            return (
              <div key={g.game_id}>
                <div role="button" tabIndex={0}
                     onClick={() => setOpen(isOpen ? null : g.game_id)}
                     onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(isOpen ? null : g.game_id) } }}
                     className="grid grid-cols-[24px_1.5fr_1fr_90px_90px_60px] gap-2 px-4 py-2.5 text-sm items-center cursor-pointer hover:bg-gray-800/30">
                  <div>
                    {isOpen ? <ChevronDown size={12} className="text-gray-500" />
                            : <ChevronRight size={12} className="text-gray-600" />}
                  </div>
                  <div className="text-gray-100">{g.away_team} @ {g.home_team}</div>
                  <div className="text-gray-500 text-xs">{fmtDate(g.commence_time)}</div>
                  <div className="text-right text-gray-300 text-xs tabular-nums">
                    {fmtLine(g.consensus_spread)}
                  </div>
                  <div className="text-right text-gray-300 text-xs tabular-nums">
                    {g.consensus_total == null ? '—' : g.consensus_total.toFixed(1)}
                  </div>
                  <div className="text-right text-gray-500 text-xs tabular-nums">{g.n_books}</div>
                </div>

                {isOpen && (
                  <div className="px-4 py-3 bg-gray-950/60 border-t border-gray-800/70">
                    <div className="text-[10px] text-gray-500 uppercase tracking-wider mb-2">
                      Every book
                      {spreadSpan > 0 && (
                        <span className="ml-2 text-gray-600 normal-case tracking-normal">
                          spread spans {spreadSpan.toFixed(1)} pts ({fmtLine(g.spread_low)} to {fmtLine(g.spread_high)})
                          {spreadSpan >= 1.5 && ' — unusually wide, the market is guessing'}
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
                )}
              </div>
            )
          })}
        </div>
      </div>

      <p className="text-xs text-gray-600">
        Updated {games[0]?.updated_at ? fmtDate(games[0].updated_at) : '—'}. The board fills in through
        August as books post the rest of the slate.
      </p>
    </div>
  )
}
