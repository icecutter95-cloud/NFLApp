// Supabase Edge Function: refresh-injuries
// Triggered by pg_cron (job 4) daily.
// Polls the ESPN injury API and writes player status to injury_flags, which
// score_week.fetch_injury_aggregates() turns into the ten inj_* features the
// spread, residual and movement models were all trained on.
//
// FIXED 2026-10-08. This ran 486 times, reported "succeeded" every time, and
// wrote zero rows for the life of the app. ESPN returns injuries NESTED BY
// TEAM -- payload.injuries is 32 team objects, each with its own .injuries
// array -- and this function read it as a flat list of injuries looking for
// injury.team.abbreviation and injury.athlete. Every one of those lookups was
// undefined, the `if (!team || !playerName) continue` dropped all 32, and the
// function returned {success: true, count: 0}. pg_cron saw a 200 and agreed.
//
// The cost was not cosmetic: the models were trained on real injury counts and
// served zeros, a train/serve skew that fetch_injury_aggregates' own docstring
// warns about. inj_qb_out_home and inj_qb_out_away have been 0 for every game
// ever scored, which is why a starting quarterback ruled out moves our number
// not at all while it moves the market seven points.
//
// Two details that matter:
//   * "Injured Reserve" must count as out. The old status regex tested /out/i,
//     which does not match "Injured Reserve", so an IR player would have been
//     dropped even had the loop worked.
//   * ESPN says LAR and WSH; this pipeline says LA and WAS, matching
//     nfl_data_py. That exact mismatch once made every Rams game invisible
//     (see TEAM_NAME_TO_ABBR in scripts/fetch_historical_lines.py).

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const ESPN_INJURIES_URL =
  "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

// ESPN -> ours. Everything else matches on both sides.
const ABBR: Record<string, string> = { LAR: "LA", WSH: "WAS" };

function normStatus(raw: string): string | null {
  const s = (raw ?? "").toLowerCase();
  if (s.includes("injured reserve") || s === "ir") return "out";
  if (s.includes("out")) return "out";
  if (s.includes("doubtful")) return "doubtful";
  if (s.includes("questionable")) return "questionable";
  return null; // Active, probable, healthy -- not a flag
}

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });

  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    );

    const res = await fetch(ESPN_INJURIES_URL, {
      headers: { "User-Agent": "nfl-betting-model/1.0" },
    });
    if (!res.ok) {
      return new Response(
        JSON.stringify({ error: `ESPN API returned ${res.status}` }),
        { status: 500, headers: { ...CORS, "Content-Type": "application/json" } },
      );
    }

    const payload = await res.json();
    const teams: any[] = payload.injuries ?? [];

    const rows: Record<string, unknown>[] = [];
    const now = new Date().toISOString();
    const seen = new Set<string>();
    let skippedActive = 0;
    let skippedNoTeam = 0;

    for (const team of teams) {
      for (const inj of team.injuries ?? []) {
        const athlete = inj.athlete ?? {};
        const raw = athlete.team?.abbreviation ?? null;
        const abbr = raw ? (ABBR[raw] ?? raw) : null;
        const playerName = athlete.displayName ??
          [athlete.firstName, athlete.lastName].filter(Boolean).join(" ");
        const position = athlete.position?.abbreviation ?? null;
        const status = normStatus(inj.status ?? "");

        if (!abbr || !playerName) {
          skippedNoTeam++;
          continue;
        }
        if (!status) {
          skippedActive++;
          continue;
        }
        // ESPN can list the same player twice across a team's blocks.
        const key = `${abbr}|${playerName}`;
        if (seen.has(key)) continue;
        seen.add(key);

        rows.push({
          team: abbr,
          player_name: playerName,
          position,
          status,
          is_qb_override: false, // set by hand from the UI, never here
          qb_downgrade_pts: 0,
          updated_at: now,
        });
      }
    }

    // Refuse to wipe a populated table over an empty parse. Returning
    // "success" on a payload we understood as nothing is what hid this bug for
    // 486 runs, so an empty parse is now an explicit failure.
    if (rows.length === 0) {
      return new Response(
        JSON.stringify({
          error: "parsed 0 injuries from a 200 response -- payload shape " +
            "probably changed again",
          teams_seen: teams.length,
          skipped_active: skippedActive,
          skipped_no_team: skippedNoTeam,
        }),
        { status: 500, headers: { ...CORS, "Content-Type": "application/json" } },
      );
    }

    // Replace the machine-written rows; never touch a manual QB override.
    const del = await supabase.from("injury_flags").delete()
      .eq("is_qb_override", false);
    if (del.error) {
      return new Response(JSON.stringify({ error: del.error.message }), {
        status: 500,
        headers: { ...CORS, "Content-Type": "application/json" },
      });
    }
    const ins = await supabase.from("injury_flags").insert(rows);
    if (ins.error) {
      return new Response(JSON.stringify({ error: ins.error.message }), {
        status: 500,
        headers: { ...CORS, "Content-Type": "application/json" },
      });
    }

    const byStatus: Record<string, number> = {};
    for (const r of rows) {
      const s = String(r.status);
      byStatus[s] = (byStatus[s] ?? 0) + 1;
    }

    return new Response(
      JSON.stringify({
        success: true,
        count: rows.length,
        teams_seen: teams.length,
        by_status: byStatus,
        qbs_out: rows.filter((r) => r.position === "QB" && r.status === "out")
          .length,
      }),
      { headers: { ...CORS, "Content-Type": "application/json" } },
    );
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e).slice(0, 300) }), {
      status: 500,
      headers: { ...CORS, "Content-Type": "application/json" },
    });
  }
});
