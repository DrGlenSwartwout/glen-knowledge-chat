-- Tier-2 wholesale: resale-license application + approval gate.
-- Spec: docs/superpowers/specs/2026-06-10-membership-consent-gate-design.md (Tier 2)
-- Additive + idempotent. Apply: psql "$SUPABASE_DB_URL" < migrations/practitioners-application.sql
--
-- Adds an explicit application -> approval path ALONGSIDE the existing auto-unlock
-- paths (licensed-on-register, coach-on-module). All three still converge on the
-- wholesale_unlocked_at gate that /api/practitioner/checkout already checks;
-- application_status is the descriptive state for the apply path.

ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS application_status text
  CHECK (application_status IN ('pending','approved','rejected'));
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS application_submitted_at timestamptz;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS approval_notes text;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS reviewed_at timestamptz;

-- Allow a non-licensed/coach applicant type for resellers who apply for approval.
ALTER TABLE practitioners DROP CONSTRAINT IF EXISTS practitioners_portal_role_check;
ALTER TABLE practitioners ADD CONSTRAINT practitioners_portal_role_check
  CHECK (portal_role IN ('licensed','coach','reseller'));

-- IMPORTANT: do NOT re-create v_practitioners_public — these columns stay out of
-- the public finder view by leaving the stored SELECT * definition untouched.
