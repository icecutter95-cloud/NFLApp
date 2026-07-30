// Supabase Edge Function: refresh-weather
// Fetches FORECAST conditions at kickoff for upcoming outdoor games and writes
// one row per game, joined to the rest of the pipeline on (home_team, away_team).
//
// Previous version had four bugs that made it a complete no-op downstream:
//   1. It wrote rows keyed `weather_<TEAM>` -- a placeholder that matched no
//      real game_id, so score_week.py's fetch_weather() never found a row.
//   2. It fetched CURRENT conditions, which say nothing about a game days away.
//   3. It looped over every stadium rather than the games actually scheduled.
//   4. Its DOME_TEAMS set used "LAR"; nfl_data_py (and our line data) use "LA",
//      so the Rams were treated as an outdoor team. It also omitted ARI/DET/DAL.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

// 5-day / 3-hour forecast — available on the free OpenWeatherMap tier.
const OWM_FORECAST = "https://api.openweathermap.org/data/2.5/forecast";

// How far ahead we can actually forecast. Games beyond this are skipped and
// picked up by a later run (the Wednesday cron catches that weekend's slate).
const FORECAST_HORIZON_DAYS = 5;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// Fixed roof or retractable — treated as weather-neutral.
// NOTE: the Rams are "LA", not "LAR", to match nfl_data_py and line_history.
const DOME_TEAMS = new Set([
  "NO", "ATL", "LV", "LA", "LAC", "MIN", "IND", "HOU", "ARI", "DET", "DAL",
]);

// Outdoor stadiums only — dome teams never need an API call.
const STADIUM_COORDS: Record<string, { name: string; lat: number; lon: number }> = {
  BUF: { name: "Highmark Stadium",          lat: 42.7738, lon: -78.7870 },
  MIA: { name: "Hard Rock Stadium",         lat: 25.9580, lon: -80.2389 },
  NE:  { name: "Gillette Stadium",          lat: 42.0909, lon: -71.2643 },
  NYJ: { name: "MetLife Stadium",           lat: 40.8135, lon: -74.0745 },
  NYG: { name: "MetLife Stadium",           lat: 40.8135, lon: -74.0745 },
  BAL: { name: "M&T Bank Stadium",          lat: 39.2779, lon: -76.6227 },
  CIN: { name: "Paycor Stadium",            lat: 39.0955, lon: -84.5160 },
  CLE: { name: "Cleveland Browns Stadium",  lat: 41.5061, lon: -81.6995 },
  PIT: { name: "Acrisure Stadium",          lat: 40.4468, lon: -80.0158 },
  JAX: { name: "EverBank Stadium",          lat: 30.3240, lon: -81.6373 },
  TEN: { name: "Nissan Stadium",            lat: 36.1665, lon: -86.7713 },
  DEN: { name: "Empower Field",             lat: 39.7439, lon: -105.0201 },
  KC:  { name: "Arrowhead Stadium",         lat: 39.0489, lon: -94.4839 },
  PHI: { name: "Lincoln Financial Field",   lat: 39.9008, lon: -75.1675 },
  WAS: { name: "Northwest Stadium",         lat: 38.9076, lon: -76.8645 },
  CHI: { name: "Soldier Field",             lat: 41.8623, lon: -87.6167 },
  GB:  { name: "Lambeau Field",             lat: 44.5013, lon: -88.0622 },
  CAR: { name: "Bank of America Stadium",   lat: 35.2258, lon: -80.8531 },
  TB:  { name: "Raymond James Stadium",     lat: 27.9759, lon: -82.5033 },
  SF:  { name: "Levi's Stadium",            lat: 37.4032, lon: -121.9700 },
  SEA: { name: "Lumen Field",               lat: 47.5952, lon: -122.3316 },
};

interface ForecastEntry {
  dt: number;
  main?: { temp?: number };
  wind?: { speed?: number; deg?: number };
  pop?: number; // probability of precipitation, 0-1
}

function windDirection(deg: number): string {
  const dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
  return dirs[Math.round(deg / 45) % 8];
}

/** Pick the forecast slot nearest the actual kickoff time. */
function nearestToKickoff(list: ForecastEntry[], kickoffEpoch: number): ForecastEntry | null {
  let best: ForecastEntry | null = null;
  let bestDelta = Infinity;
  for (const e of list) {
    const delta = Math.abs(e.dt - kickoffEpoch);
    if (delta < bestDelta) {
      bestDelta = delta;
      best = e;
    }
  }
  return best;
}

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });

  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
    );
    const owmKey = Deno.env.get("WEATHER_API_KEY") ?? Deno.env.get("OPENWEATHER_API_KEY")!;

    const now = new Date();
    const horizon = new Date(now.getTime() + FORECAST_HORIZON_DAYS * 86400_000);

    // Upcoming games, from the same source the rest of the pipeline uses.
    // line_open_close is one row per game, so this can't hit the 1000-row cap.
    const { data: games, error: gamesErr } = await supabase
      .from("line_open_close")
      .select("game_id, home_team, away_team, commence_time")
      .gte("commence_time", now.toISOString())
      .lte("commence_time", horizon.toISOString());

    if (gamesErr) {
      return new Response(JSON.stringify({ error: gamesErr.message }), {
        status: 500,
        headers: { ...CORS, "Content-Type": "application/json" },
      });
    }

    const upcoming = (games ?? []).filter((g) => g.home_team && g.commence_time);
    const rows = [];
    let apiCalls = 0;
    let skippedNoCoords = 0;
    const forecastCache = new Map<string, ForecastEntry[]>();

    for (const game of upcoming) {
      const team = game.home_team as string;
      const kickoff = new Date(game.commence_time as string);
      const kickoffEpoch = Math.floor(kickoff.getTime() / 1000);

      const base = {
        game_id: game.game_id,
        home_team: team,
        away_team: game.away_team,
        commence_time: game.commence_time,
        fetched_at: now.toISOString(),
      };

      if (DOME_TEAMS.has(team)) {
        rows.push({
          ...base,
          stadium: "dome",
          is_dome: true,
          wind_speed_mph: 0,
          wind_direction: null,
          temp_fahrenheit: 72,
          precipitation_prob: 0,
          forecast_for: game.commence_time,
        });
        continue;
      }

      const stadium = STADIUM_COORDS[team];
      if (!stadium) {
        skippedNoCoords++;
        continue;
      }

      // One forecast call covers all games at a venue within the window.
      let list = forecastCache.get(team);
      if (!list) {
        const url = `${OWM_FORECAST}?lat=${stadium.lat}&lon=${stadium.lon}` +
                    `&units=imperial&appid=${owmKey}`;
        try {
          const res = await fetch(url);
          apiCalls++;
          if (!res.ok) continue;
          const payload = await res.json();
          list = (payload.list ?? []) as ForecastEntry[];
          forecastCache.set(team, list);
        } catch {
          continue;
        }
      }
      if (!list.length) continue;

      const slot = nearestToKickoff(list, kickoffEpoch);
      if (!slot) continue;

      rows.push({
        ...base,
        stadium: stadium.name,
        is_dome: false,
        wind_speed_mph: Math.round((slot.wind?.speed ?? 0) * 10) / 10,
        wind_direction: windDirection(slot.wind?.deg ?? 0),
        temp_fahrenheit: Math.round(slot.main?.temp ?? 60),
        precipitation_prob: slot.pop ?? 0,
        forecast_for: new Date(slot.dt * 1000).toISOString(),
      });
    }

    if (rows.length === 0) {
      return new Response(
        JSON.stringify({
          success: true, count: 0, api_calls: apiCalls,
          note: `no games kicking off within ${FORECAST_HORIZON_DAYS} days`,
        }),
        { headers: { ...CORS, "Content-Type": "application/json" } }
      );
    }

    const { error } = await supabase
      .from("weather")
      .upsert(rows, { onConflict: "game_id" });

    if (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { ...CORS, "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({
        success: true,
        count: rows.length,
        dome: rows.filter((r) => r.is_dome).length,
        outdoor: rows.filter((r) => !r.is_dome).length,
        api_calls: apiCalls,
        skipped_no_coords: skippedNoCoords,
      }),
      { headers: { ...CORS, "Content-Type": "application/json" } }
    );
  } catch (err) {
    return new Response(JSON.stringify({ error: (err as Error).message }), {
      status: 500,
      headers: { ...CORS, "Content-Type": "application/json" },
    });
  }
});
