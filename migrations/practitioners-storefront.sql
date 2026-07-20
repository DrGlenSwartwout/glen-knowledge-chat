-- Practitioner storefront editor: self-authored profile provenance + logo.
-- Spec: 00 System/deploy-chat-specs/2026-07-20-practitioner-storefront-editor.md
-- Additive + idempotent. Apply: psql "$SUPABASE_DB_URL" < migrations/practitioners-storefront.sql
--
-- profile_self_authored_at is null for every existing (scraped) row; the
-- storefront publishes bio/photo/services/location ONLY when it is set.
-- logo_url has no Postgres home today (it lived only in the sqlite branding blob).
--
-- Deliberately does NOT re-create v_practitioners_public: leaving the stored
-- view definition untouched keeps these columns OUT of the public finder view.
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS logo_url text;
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS profile_self_authored_at timestamptz;
