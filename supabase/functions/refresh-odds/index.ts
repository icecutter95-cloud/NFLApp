// Supabase Edge Function: refresh-odds
// Triggered by pg_cron every 15 minutes during game week (Wed–Sun), every 2 hours off-peak.
// Fetches DraftKings spreads + totals from The Odds API and upserts to line_history.
// On first pull of the week (no existing rows for a game), sets is_opening = true.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const ODDS_API_BASE = "https://api.the-odds-api.com/v4";
const SPORT = "americanfootball_nfl";

interface OddsApiGame {
  id: string;
  home_team: string;
  away_team: string;
  commence_time: string;
  bookmakers: Array<{
    key: string;
    markets: Array<{
      key: string;
      outcomes: Array<{ name: string; point?: number; price: number }>;
    }>;
  }>;
}

function parseOddsResponse(data: OddsApiGame[]): Array<{
  game_id: string;
  spread_home: number | null;
  total: number | null;
  book: string;
  recorded_at: string;
}> {
  const rows = [];
  const now = new Date().toISOString();

  for (const game of data) {
    const dk = game.bookmakers.find((b) => b.key === "draftkings");
    if (!dk) continue;

    const spreadMarket = dk.markets.find((m) => m.key === "spreads");
    const totalMarket = dk.markets.find((m) => m.key === "totals");

    const homeSpreadOutcome = spreadMarket?.outcomes.find(
      (o) => o.name === game.home_team
    );
    const overOutcome = totalMarket?.outcomes.find((o) => o.name === "Over");

    rows.push({
      game_id: game.id,
      spread_home: homeSpreadOutcome?.point ?? null,
      total: overOutcome?.point ?? null,
      book: "draftkings",
      recorded_at: now,
    });
  }

  return rows;
}

serve(async () => {
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  const apiKey = Deno.env.get("ODDS_API_KEY")!;
  const url =
    `${ODDS_API_BASE}/sports/${SPORT}/odds` +
    `?regions=us&markets=spreads,totals&bookmakers=draftkings,fanduel&apiKey=${apiKey}`;

  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.text();
    return new Response(JSON.stringify({ error: body }), { status: 500 });
  }

  const data: OddsApiGame[] = await res.json();
  const rows = parseOddsResponse(data);

  // Check which game_ids already have rows this week (to set is_opening)
  const gameIds = rows.map((r) => r.game_id);
  const { data: existing } = await supabase
    .from("line_history")
    .select("game_id")
    .in("game_id", gameIds);

  const seenGameIds = new Set((existing ?? []).map((r: { game_id: string }) => r.game_id));

  const insertRows = rows.map((r) => ({
    ...r,
    is_opening: !seenGameIds.has(r.game_id),
  }));

  const { error } = await supabase.from("line_history").insert(insertRows);
  if (error) {
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }

  return new Response(
    JSON.stringify({ success: true, count: insertRows.length }),
    { headers: { "Content-Type": "application/json" } }
  );
});
