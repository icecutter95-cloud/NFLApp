// Supabase Edge Function: refresh-injuries
// Triggered by pg_cron every 4 hours.
// Polls the ESPN injury API and upserts QB / key player status to injury_flags.
// The QB override flag is set manually from the UI; this function just keeps the
// roster status current so the UI has accurate data to show.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const ESPN_INJURIES_URL =
  "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries";

const KEY_POSITIONS = new Set(["QB", "WR1", "RB", "LT"]);

interface EspnInjury {
  team?: { abbreviation?: string };
  athlete?: { fullName?: string; position?: { abbreviation?: string } };
  status?: string;
  type?: { description?: string };
}

serve(async () => {
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );

  const res = await fetch(ESPN_INJURIES_URL, {
    headers: { "User-Agent": "nfl-betting-model/1.0" },
  });

  if (!res.ok) {
    return new Response(
      JSON.stringify({ error: `ESPN API returned ${res.status}` }),
      { status: 500 }
    );
  }

  const payload = await res.json();

  // ESPN injury response structure: payload.injuries (array) or similar
  const injuries: EspnInjury[] = payload.injuries ?? [];

  const rows = [];
  const now = new Date().toISOString();

  for (const injury of injuries) {
    const team = injury.team?.abbreviation;
    const playerName = injury.athlete?.fullName;
    const position = injury.athlete?.position?.abbreviation;
    const statusRaw = injury.status ?? "";

    if (!team || !playerName) continue;

    // Normalize status
    let status = "questionable";
    if (/out/i.test(statusRaw)) status = "out";
    else if (/doubtful/i.test(statusRaw)) status = "doubtful";
    else if (/questionable/i.test(statusRaw)) status = "questionable";
    else continue;  // skip "probable" and healthy

    const isQb = position === "QB";

    rows.push({
      team,
      player_name: playerName,
      position,
      status,
      is_qb_override: false,  // user sets this manually via UI toggle
      qb_downgrade_pts: 0,
      updated_at: now,
    });
  }

  if (rows.length > 0) {
    // Delete stale entries for this week before inserting fresh data
    // (simpler than upsert without a unique key on player+team+week)
    await supabase.from("injury_flags").delete().eq("is_qb_override", false);
    const { error } = await supabase.from("injury_flags").insert(rows);
    if (error) {
      return new Response(JSON.stringify({ error: error.message }), { status: 500 });
    }
  }

  return new Response(
    JSON.stringify({ success: true, count: rows.length }),
    { headers: { "Content-Type": "application/json" } }
  );
});
