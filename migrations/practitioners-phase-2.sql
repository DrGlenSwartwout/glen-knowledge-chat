-- Practitioner Finder Phase 2: additive columns for the inquiry bridge.
-- Idempotent (safe to re-run).

ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS accepts_inquiries boolean DEFAULT NULL;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS claim_token_hash text;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS claim_verified_at timestamptz;

CREATE UNIQUE INDEX IF NOT EXISTS practitioners_claim_token_hash_uniq
  ON practitioners (claim_token_hash)
  WHERE claim_token_hash IS NOT NULL;
