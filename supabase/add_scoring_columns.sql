-- Migration: add points-per-game scoring columns to team_metrics
-- Run this once in the Supabase SQL Editor before re-uploading metrics.
ALTER TABLE team_metrics
    ADD COLUMN IF NOT EXISTS pts_scored_off_l4  REAL,
    ADD COLUMN IF NOT EXISTS pts_scored_off_l8  REAL,
    ADD COLUMN IF NOT EXISTS pts_allowed_def_l4 REAL,
    ADD COLUMN IF NOT EXISTS pts_allowed_def_l8 REAL;
