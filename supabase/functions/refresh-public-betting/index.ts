// Supabase Edge Function: refresh-public-betting
// Triggered by pg_cron every 2 hours.
// Fetches public bet % + money % from ActionNetwork and upserts to public_betting.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const ACTION_NETWORK_URL =
  "https://api.actionnetwork.com/web/v1/games?sport=nfl&include=all_market_percentages";

interface ANGame {
  id: number;
  home_team?: { abbr?: string };
  away_team?: { abbr?: string };
  markets?: Array<{
    market_type: string;
    bet_count?: number;
    bet_percentage?: number;
    money_percentage?: number;
  }>;
}

serve(async () => {
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  const res = await fetch(ACTION_NETWORK_URL, {
    headers: { "User-Agent": "nfl-betting-model/1.0" },
  });

  if (!res.ok) {
    return new Response(
      JSON.stringify({ error: `ActionNetwork returned ${res.status}` }),
      { status: 500 }
    );
  }

  const payload = await res.json();
  const games: ANGame[] = payload.games ?? [];

  const rows = [];
  const now = new Date().toISOString();

  for (const game of games) {
    const gameId = String(game.id);
    const spreadMarket = game.markets?.find(
      (m) => m.market_type === "game_spread"
    );
    const totalMarket = game.markets?.find(
      (m) => m.market_type === "game_total"
    );

    rows.push({
      game_id: gameId,
      recorded_at: now,
      bet_pct_home: spreadMarket?.bet_percentage ?? null,
      money_pct_home: spreadMarket?.money_percentage ?? null,
      bet_pct_over: totalMarket?.bet_percentage ?? null,
      money_pct_over: totalMarket?.money_percentage ?? null,
      source: "actionnetwork",
    });
  }

  if (rows.length === 0) {
    return new Response(JSON.stringify({ success: true, count: 0 }));
  }

  const { error } = await supabase.from("public_betting").insert(rows);
  if (error) {
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }

  return new Response(
    JSON.stringify({ success: true, count: rows.length }),
    { headers: { "Content-Type": "application/json" } }
  );
});
