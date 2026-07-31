// Supabase Edge Function: refresh-odds
// Triggered by pg_cron every 15 minutes during game week (Wed–Sun), every 2 hours off-peak.
// Fetches spreads + totals from The Odds API.
//
// Two destinations, deliberately kept separate:
//   line_history — DraftKings ONLY, append-only. DK is the spine every validated
//                  number is built on (week_open_spread_home, closing lines, CLV).
//                  Widening it would make the entire track record non-comparable.
//   book_lines   — the full panel, upserted in place, used for line shopping.
//                  Measured worth ~+0.30 pts per bet across 2021-2025, which is
//                  roughly 15% on top of the qualifying filter's +1.93 CLV.
//
// Region stays "us" so the credit cost per call is unchanged. That covers 8 of
// the 9 liquid books; Pinnacle is EU-only and would double the burn, and it was
// only valuable as a model feature — which testing did not support shipping.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const ODDS_API_BASE = "https://api.the-odds-api.com/v4";
const SPORT = "americanfootball_nfl";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

// The Odds API returns full team names ("Los Angeles Rams"); the rest of the
// pipeline (nfl_data_py schedules, team_metrics, projections) keys everything
// by standard abbreviation. Without this translation, line_history rows were
// keyed only by the Odds API's own opaque event ID, which never matches the
// nfl_data_py game_id used everywhere else — so every downstream lookup
// silently fell back to a default line (0 spread / 45 total) for every game.
// NOTE: nfl_data_py labels the Rams "LA", not "LAR".
const TEAM_NAME_TO_ABBR: Record<string, string> = {
  "Arizona Cardinals": "ARI",
  "Atlanta Falcons": "ATL",
  "Baltimore Ravens": "BAL",
  "Buffalo Bills": "BUF",
  "Carolina Panthers": "CAR",
  "Chicago Bears": "CHI",
  "Cincinnati Bengals": "CIN",
  "Cleveland Browns": "CLE",
  "Dallas Cowboys": "DAL",
  "Denver Broncos": "DEN",
  "Detroit Lions": "DET",
  "Green Bay Packers": "GB",
  "Houston Texans": "HOU",
  "Indianapolis Colts": "IND",
  "Jacksonville Jaguars": "JAX",
  "Kansas City Chiefs": "KC",
  "Las Vegas Raiders": "LV",
  "Los Angeles Chargers": "LAC",
  "Los Angeles Rams": "LA",
  "Miami Dolphins": "MIA",
  "Minnesota Vikings": "MIN",
  "New England Patriots": "NE",
  "New Orleans Saints": "NO",
  "New York Giants": "NYG",
  "New York Jets": "NYJ",
  "Philadelphia Eagles": "PHI",
  "Pittsburgh Steelers": "PIT",
  "San Francisco 49ers": "SF",
  "Seattle Seahawks": "SEA",
  "Tampa Bay Buccaneers": "TB",
  "Tennessee Titans": "TEN",
  "Washington Commanders": "WAS",
};

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
  home_team: string | null;
  away_team: string | null;
  commence_time: string | null;
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
      home_team: TEAM_NAME_TO_ABBR[game.home_team] ?? null,
      away_team: TEAM_NAME_TO_ABBR[game.away_team] ?? null,
      // Kickoff time -- lets the line_open_close view derive the CLOSING line
      // as the last snapshot before kickoff, rather than guessing.
      commence_time: game.commence_time ?? null,
      spread_home: homeSpreadOutcome?.point ?? null,
      total: overOutcome?.point ?? null,
      book: "draftkings",
      recorded_at: now,
    });
  }

  return rows;
}

// Books that are liquid and consistently quoted. Offshore books with stale
// numbers inflated apparent line-shopping value to +1.28 pts in backtesting --
// pure max-of-N bias -- so the panel is an explicit allowlist, not "whatever
// the API returned".
const PANEL = [
  "draftkings", "fanduel", "betmgm", "williamhill_us",
  "betrivers", "espnbet", "betonlineag", "lowvig", "bovada",
];

function parseBookPanel(data: OddsApiGame[]) {
  const rows = [];
  const now = new Date().toISOString();

  for (const game of data) {
    const home = TEAM_NAME_TO_ABBR[game.home_team] ?? null;
    const away = TEAM_NAME_TO_ABBR[game.away_team] ?? null;

    for (const bk of game.bookmakers) {
      if (!PANEL.includes(bk.key)) continue;

      const spreads = bk.markets.find((m) => m.key === "spreads");
      const totals = bk.markets.find((m) => m.key === "totals");
      const sHome = spreads?.outcomes.find((o) => o.name === game.home_team);
      const sAway = spreads?.outcomes.find((o) => o.name === game.away_team);
      const over = totals?.outcomes.find((o) => o.name === "Over");
      const under = totals?.outcomes.find((o) => o.name === "Under");

      // A book with neither market priced tells us nothing; skip rather than
      // write an all-null row that would count toward n_books.
      if (sHome?.point == null && over?.point == null) continue;

      rows.push({
        game_id: game.id,
        book: bk.key,
        home_team: home,
        away_team: away,
        commence_time: game.commence_time ?? null,
        spread_home: sHome?.point ?? null,
        spread_home_price: sHome?.price ?? null,
        spread_away_price: sAway?.price ?? null,
        total: over?.point ?? null,
        over_price: over?.price ?? null,
        under_price: under?.price ?? null,
        updated_at: now,
      });
    }
  }
  return rows;
}

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });

  try {
    const supabase = createClient(
      Deno.env.get("SUPABASE_URL")!,
      Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
    );

    const apiKey = Deno.env.get("ODDS_API_KEY")!;
    const url =
      `${ODDS_API_BASE}/sports/${SPORT}/odds` +
      `?regions=us&markets=spreads,totals&bookmakers=${PANEL.join(",")}` +
      // The API defaults to DECIMAL odds (1.91). Everything in this project
      // speaks American (-110, and the 100/110 juice in every ROI figure), and
      // book_lines stores prices as integers.
      `&oddsFormat=american&apiKey=${apiKey}`;

    const res = await fetch(url);
    if (!res.ok) {
      const body = await res.text();
      return new Response(JSON.stringify({ error: body }), {
        status: 500,
        headers: { ...CORS, "Content-Type": "application/json" },
      });
    }

    const data: OddsApiGame[] = await res.json();
    const rows = parseOddsResponse(data);

    // Which games have we already recorded a line for? Query the
    // line_open_close VIEW rather than line_history directly: the view is
    // aggregated to ONE ROW PER GAME, so this can't be truncated by the
    // 1000-row response cap. Querying line_history directly (as this used to)
    // returns every historical snapshot, so once the table passed 1000 rows
    // the result would silently truncate, `seenGameIds` would come back
    // incomplete, and already-tracked games would be re-flagged as openers --
    // corrupting the opening line, which is the one number we most need to
    // preserve for closing-line-value tracking.
    const gameIds = rows.map((r) => r.game_id);
    const { data: existing } = await supabase
      .from("line_open_close")
      .select("game_id")
      .in("game_id", gameIds);

    const seenGameIds = new Set((existing ?? []).map((r: { game_id: string }) => r.game_id));

    const insertRows = rows.map((r) => ({
      ...r,
      is_opening: !seenGameIds.has(r.game_id),
    }));

    const { error } = await supabase.from("line_history").insert(insertRows);
    if (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { ...CORS, "Content-Type": "application/json" },
      });
    }

    // Panel refresh is best-effort: it powers line shopping, but a failure here
    // must not cost us the DK snapshot above, which is the one number CLV
    // tracking cannot reconstruct after the fact.
    const panelRows = parseBookPanel(data);
    let panelCount = 0;
    let panelError: string | null = null;
    if (panelRows.length > 0) {
      const { error: pErr } = await supabase
        .from("book_lines")
        .upsert(panelRows, { onConflict: "game_id,book" });
      if (pErr) panelError = pErr.message;
      else panelCount = panelRows.length;
    }

    return new Response(
      JSON.stringify({
        success: true,
        count: insertRows.length,
        books: panelCount,
        ...(panelError ? { panelError } : {}),
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
