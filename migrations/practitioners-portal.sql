-- Practitioner wholesale portal: registration/account columns (Phase 3a).
-- Spec: docs/superpowers/specs/2026-06-01-practitioner-wholesale-portal-design.md
-- Additive + idempotent. Apply: psql "$SUPABASE_DB_URL" < migrations/practitioners-portal.sql

-- Which door the practitioner came in by, and the verification inputs.
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS portal_role text
  CHECK (portal_role IN ('licensed','coach'));
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS license_state text;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS license_number text;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS resale_license_number text;

-- Wholesale access gate: NULL until the practitioner is cleared to order.
-- Licensed = set on registration; coach = set once the first module is committed.
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS wholesale_unlocked_at timestamptz;

-- IMPORTANT: do NOT re-create v_practitioners_public. These columns (like the
-- wallet/cert columns) stay out of the public finder view by leaving the stored
-- SELECT * definition untouched.
