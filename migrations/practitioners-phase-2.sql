-- Practitioner Finder Phase 2: additive columns for the inquiry bridge.
-- Idempotent (safe to re-run).

ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS accepts_inquiries boolean DEFAULT NULL;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS claim_token_hash text;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS claim_verified_at timestamptz;

CREATE UNIQUE INDEX IF NOT EXISTS practitioners_claim_token_hash_uniq
  ON practitioners (claim_token_hash)
  WHERE claim_token_hash IS NOT NULL;

-- Re-create the public view so a fresh rebuild exposes the columns added above.
-- A SELECT * view does NOT auto-expose columns added by a later ALTER, so the
-- view created in practitioners.sql would omit accepts_inquiries until refreshed
-- (this caused the /api/practitioner-finder/search 500 in prod, 2026-05-29).
-- This restates the FULL canonical definition (security_invoker + the
-- removal_requested/lat filters) so the refresh cannot drift into a bare
-- SELECT * that would expose removal-requested or un-geocoded rows.
CREATE OR REPLACE VIEW v_practitioners_public
WITH (security_invoker = on)
AS
SELECT * FROM practitioners
WHERE removal_requested = false
  AND lat IS NOT NULL;
