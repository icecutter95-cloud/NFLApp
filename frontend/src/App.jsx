import { useState, useEffect, useMemo } from 'react'
import { supabase } from './lib/supabase'
import Header from './components/Header'
import FilterBar from './components/FilterBar'
import EdgeList from './components/EdgeList'
import PerformancePanel from './components/PerformancePanel'

const CUR_SEASON = 2025
const ALL_WEEKS = Array.from({ length: 18 }, (_, i) => i + 1)

export default function App() {
  const [season, setSeason] = useState(CUR_SEASON)
  const [week, setWeek] = useState(1)
  const [projections, setProjections] = useState([])
  const [bets, setBets] = useState([])
  const [loading, setLoading] = useState(true)
  const [showPerformance, setShowPerformance] = useState(false)
  const [filters, setFilters] = useState({
    betType: 'all',
    confidence: 'all',
    steamOnly: false,
    rlmOnly: false,
    minEdge: 0,
    sortBy: 'ev_pct',
  })

  useEffect(() => {
    fetchData()
  }, [season, week])

  async function fetchData() {
    setLoading(true)
    const [projRes, betRes] = await Promise.all([
      supabase
        .from('projections')
        .select('*')
        .eq('season', season)
        .eq('week', week)
        .order('ev_pct', { ascending: false }),
      supabase
        .from('bets')
        .select('*')
        .eq('season', season)
        .order('logged_at', { ascending: false }),
    ])
    setProjections(projRes.data ?? [])
    setBets(betRes.data ?? [])
    setLoading(false)
  }

  const filtered = useMemo(() => {
    let rows = projections
    if (filters.betType !== 'all') rows = rows.filter(p => p.bet_type === filters.betType)
    if (filters.confidence !== 'all') rows = rows.filter(p => p.confidence_tier === filters.confidence)
    if (filters.steamOnly) rows = rows.filter(p => p.steam_flag)
    if (filters.rlmOnly) rows = rows.filter(p => p.rlm_flag)
    if (filters.minEdge > 0) rows = rows.filter(p => (p.edge_points ?? 0) >= filters.minEdge)

    return [...rows].sort((a, b) => {
      if (filters.sortBy === 'ev_pct') return (b.ev_pct ?? 0) - (a.ev_pct ?? 0)
      if (filters.sortBy === 'edge') return (b.edge_points ?? 0) - (a.edge_points ?? 0)
      if (filters.sortBy === 'confidence') {
        const order = { A: 0, B: 1, C: 2, watch: 3 }
        return (order[a.confidence_tier] ?? 4) - (order[b.confidence_tier] ?? 4)
      }
      if (filters.sortBy === 'game_time') return new Date(a.game_time ?? 0) - new Date(b.game_time ?? 0)
      return 0
    })
  }, [projections, filters])

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-mono">
      <Header
        season={season}
        week={week}
        weeks={ALL_WEEKS}
        bets={bets}
        onSeasonChange={setSeason}
        onWeekChange={setWeek}
        showPerformance={showPerformance}
        onTogglePerformance={() => setShowPerformance(p => !p)}
      />

      {showPerformance ? (
        <PerformancePanel bets={bets} />
      ) : (
        <>
          <FilterBar filters={filters} onChange={setFilters} />
          <EdgeList projections={filtered} loading={loading} onBetLogged={fetchData} />
        </>
      )}
    </div>
  )
}
