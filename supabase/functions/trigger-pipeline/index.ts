/**
 * trigger-pipeline — triggers GitHub Actions workflows.
 *
 * Secrets required in Supabase (Dashboard → Edge Functions → Secrets):
 *   GITHUB_PAT   — Personal Access Token with "workflow" scope
 *   GITHUB_OWNER — your GitHub username
 *   GITHUB_REPO  — your repo name (e.g. "NFLApp")
 *
 * Request body:
 *   { action: 'score-week',     season: number, week: number }
 *   { action: 'update-metrics', season: number }
 *   { action: 'log-clv',        season?: number, week?: number }  // blank = auto-detect
 *   { action: 'fetch-results',  season?: number }                 // blank = current season
 */

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'

const CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
}

serve(async (req) => {
  if (req.method === 'OPTIONS') return new Response('ok', { headers: CORS })

  try {
    const { action, season, week } = await req.json()

    const owner = Deno.env.get('GITHUB_OWNER')
    const repo  = Deno.env.get('GITHUB_REPO')
    const pat   = Deno.env.get('GITHUB_PAT')

    if (!owner || !repo || !pat) {
      return new Response(
        JSON.stringify({ error: 'Missing GITHUB_OWNER, GITHUB_REPO, or GITHUB_PAT secrets' }),
        { status: 500, headers: { ...CORS, 'Content-Type': 'application/json' } },
      )
    }

    // Route the action to its workflow. log-clv takes optional inputs -- blank
    // means the script auto-detects the current season/week, which is what the
    // daily cron wants.
    const WORKFLOWS: Record<string, string> = {
      'update-metrics': 'update-metrics.yml',
      'log-clv':        'log-clv.yml',
      'fetch-results':  'fetch-results.yml',
      'score-week':     'score-week.yml',
    }
    const workflow = WORKFLOWS[action] ?? 'score-week.yml'

    let inputs: Record<string, string>
    if (action === 'update-metrics') {
      inputs = { season: String(season) }
    } else if (action === 'log-clv') {
      inputs = { season: season ? String(season) : '', week: week ? String(week) : '' }
    } else if (action === 'fetch-results') {
      inputs = { season: season ? String(season) : '' }
    } else {
      inputs = { season: String(season), week: String(week) }
    }

    const ghRes = await fetch(
      `https://api.github.com/repos/${owner}/${repo}/actions/workflows/${workflow}/dispatches`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${pat}`,
          Accept: 'application/vnd.github.v3+json',
          'Content-Type': 'application/json',
          'User-Agent': 'NFLApp/1.0',
        },
        body: JSON.stringify({ ref: 'main', inputs }),
      },
    )

    if (!ghRes.ok) {
      const detail = await ghRes.text()
      return new Response(
        JSON.stringify({ error: `GitHub API ${ghRes.status}`, detail }),
        { status: 502, headers: { ...CORS, 'Content-Type': 'application/json' } },
      )
    }

    return new Response(
      JSON.stringify({ success: true, message: `${workflow} triggered — season ${season}${week ? ` week ${week}` : ''}` }),
      { headers: { ...CORS, 'Content-Type': 'application/json' } },
    )
  } catch (err) {
    return new Response(
      JSON.stringify({ error: err.message }),
      { status: 500, headers: { ...CORS, 'Content-Type': 'application/json' } },
    )
  }
})
