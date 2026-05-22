// Supabase Edge Function: refresh-weather
// Triggered by pg_cron on Wednesday at 9am UTC each game week.
// Fetches wind / temp / precip for each outdoor-stadium game this week.
// Skips dome stadiums. Upserts to weather table.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const OWM_BASE = "https://api.openweathermap.org/data/2.5/weather";

// Outdoor stadium coordinates — dome teams are omitted
const STADIUM_COORDS: Record<string, { name: string; lat: number; lon: number }> = {
  GB:  { name: "Lambeau Field",      lat: 44.5013,  lon: -88.0622 },
  CHI: { name: "Soldier Field",      lat: 41.8623,  lon: -87.6167 },
  BUF: { name: "Highmark Stadium",   lat: 42.7738,  lon: -78.7870 },
  PIT: { name: "Acrisure Stadium",   lat: 40.4468,  lon: -80.0158 },
  CLE: { name: "Cleveland Browns Stadium", lat: 41.5061, lon: -81.6995 },
  NE:  { name: "Gillette Stadium",   lat: 42.0909,  lon: -71.2643 },
  KC:  { name: "Arrowhead Stadium",  lat: 39.0489,  lon: -94.4839 },
  DEN: { name: "Empower Field",      lat: 39.7439,  lon: -105.0201 },
  NYG: { name: "MetLife Stadium",    lat: 40.8135,  lon: -74.0745 },
  NYJ: { name: "MetLife Stadium",    lat: 40.8135,  lon: -74.0745 },
  BAL: { name: "M&T Bank Stadium",   lat: 39.2780,  lon: -76.6227 },
  PHI: { name: "Lincoln Financial",  lat: 39.9008,  lon: -75.1675 },
  WAS: { name: "Northwest Stadium",  lat: 38.9079,  lon: -76.8644 },
  DAL: { name: "AT&T Stadium",       lat: 32.7473,  lon: -97.0945 },  // partially outdoor
  SF:  { name: "Levi's Stadium",     lat: 37.4033,  lon: -121.9694 },
  SEA: { name: "Lumen Field",        lat: 47.5952,  lon: -122.3316 },
  ARI: { name: "State Farm Stadium", lat: 33.5276,  lon: -112.2626 }, // retractable
  CAR: { name: "Bank of America",    lat: 35.2258,  lon: -80.8528 },
  TB:  { name: "Raymond James",      lat: 27.9759,  lon: -82.5033 },
  JAX: { name: "EverBank Stadium",   lat: 30.3240,  lon: -81.6373 },
  TEN: { name: "Nissan Stadium",     lat: 36.1665,  lon: -86.7713 },
  CIN: { name: "Paycor Stadium",     lat: 39.0954,  lon: -84.5160 },
  MIA: { name: "Hard Rock Stadium",  lat: 25.9579,  lon: -80.2388 },
};

const DOME_TEAMS = new Set(["NO", "ATL", "LV", "LAR", "LAC", "MIN", "IND", "HOU"]);

interface OWMResponse {
  wind?: { speed?: number; deg?: number };
  main?: { temp?: number };
  pop?: number;  // probability of precipitation (0-1)
}

serve(async () => {
  const supabase = createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );
  const owmKey = Deno.env.get("WEATHER_API_KEY") ?? Deno.env.get("OPENWEATHER_API_KEY")!;

  // Get games scheduled this week from line_history (games we're tracking)
  const { data: recentLines } = await supabase
    .from("line_history")
    .select("game_id")
    .eq("is_opening", true)
    .order("recorded_at", { ascending: false })
    .limit(50);

  // Extract home team abbreviations from game_id (format varies — use a lookup instead)
  // Here we just fetch weather for all outdoor stadiums and store by team key
  const rows = [];

  for (const [teamAbbr, stadium] of Object.entries(STADIUM_COORDS)) {
    if (DOME_TEAMS.has(teamAbbr)) continue;

    const url = `${OWM_BASE}?lat=${stadium.lat}&lon=${stadium.lon}&units=imperial&appid=${owmKey}`;
    let weatherData: OWMResponse;

    try {
      const res = await fetch(url);
      if (!res.ok) continue;
      weatherData = await res.json();
    } catch {
      continue;
    }

    const windDegrees = weatherData.wind?.deg ?? 0;
    const directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    const windDir = directions[Math.round(windDegrees / 45) % 8];

    // We store a row keyed by team (the refresh-odds function links game_id → home_team)
    // The Python scoring script reads weather by game_id; match via team on the frontend
    rows.push({
      game_id: `weather_${teamAbbr}`,  // placeholder; Python script joins by home_team
      stadium: stadium.name,
      is_dome: false,
      wind_speed_mph: Math.round((weatherData.wind?.speed ?? 0) * 10) / 10,
      wind_direction: windDir,
      temp_fahrenheit: Math.round(weatherData.main?.temp ?? 72),
      precipitation_prob: weatherData.pop ?? 0,
    });
  }

  const { error } = await supabase
    .from("weather")
    .upsert(rows, { onConflict: "game_id" });

  if (error) {
    return new Response(JSON.stringify({ error: error.message }), { status: 500 });
  }

  return new Response(
    JSON.stringify({ success: true, count: rows.length }),
    { headers: { "Content-Type": "application/json" } }
  );
});
