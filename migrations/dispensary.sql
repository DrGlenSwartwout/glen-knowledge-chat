-- In-house drop-ship dispensary (Phase 5).
-- Spec: docs/superpowers/specs/2026-06-01-practitioner-wholesale-portal-design.md §1.4
-- Additive + idempotent. Apply: psql "$SUPABASE_DB_URL" < migrations/dispensary.sql

-- Per-practitioner shareable dispensary code (for /dispensary/<code> attribution).
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS dispensary_code text;
CREATE UNIQUE INDEX IF NOT EXISTS practitioners_dispensary_code
  ON practitioners (dispensary_code) WHERE dispensary_code IS NOT NULL;

-- IMPORTANT: do NOT re-create v_practitioners_public; dispensary_code is harmless
-- to expose but we keep the stored SELECT * definition untouched for consistency.
