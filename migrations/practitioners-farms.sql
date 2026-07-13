-- Regenerative Farm Finder — integrate farms into the practitioner finder as a
-- new top-level category. APPLIED to prod (remedy-match/prd) 2026-07-13.
-- Spec: docs/superpowers/specs/2026-07-13-farm-finder-design.md
--
-- Farms are stored as `practitioners` rows so they reuse the entire finder
-- pipeline (search, map, radius, portal + public embed) unchanged:
--   - tier = 'farm'                     (new tier value; renderer shows farm card)
--   - specialties = ['regenerative_farms', <practice slugs...>]
--     -> the parent "Regenerative Farms" chip and practice sub-chips both resolve
--        via the existing `specialties && %s` filter, no query change.
--   - products / order_options          (new farm-only array columns; NULL for clinicians)

-- 1) Allow the new tier value. Rebuild the CHECK constraint to include 'farm'.
-- Must preserve every value already allowed by the live constraint (which
-- includes 'healing_oasis', added by a later migration than practitioners.sql)
-- or the ADD fails against existing rows.
ALTER TABLE practitioners DROP CONSTRAINT IF EXISTS practitioners_tier_check;
ALTER TABLE practitioners ADD CONSTRAINT practitioners_tier_check
  CHECK (tier IN (
    'org_member','eyehealing','panel_in_cert','panel_certified',
    'healing_oasis','farm'));

-- 2) Farm-only columns. Nullable; clinicians leave them NULL.
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS products      text[];
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS order_options text[];

-- practices are represented inside specialties[] (already GIN-indexed) so the
-- existing specialty filter works for farm sub-chips with no new index.

-- Opt-out reuses the existing practitioners.removal_requested flag (Glen's
-- decision) — v_practitioners_public already filters removal_requested = false.

-- 3) Expose products/order_options through the public view so the finder search
-- can return them for farm cards. The view's SELECT * was frozen at creation
-- (Postgres expands * once), so it does NOT auto-include later-added columns.
-- We must list columns EXPLICITLY and add ONLY the two new ones: a `SELECT *`
-- refresh would (a) fail on column-order changes and (b) newly expose sensitive
-- columns added since (wallet_balance_cents, license_number, portal_role, …)
-- through the PUBLIC search API. Keep this list = the frozen view + the 2 farm
-- columns.
CREATE OR REPLACE VIEW v_practitioners_public
WITH (security_invoker = on) AS
SELECT id, tier, source_org, source_url, fellowship_level, specialties, name,
       practice_name, credentials, phone, email, website, address1, city, state,
       postal, country, lat, lng, geocode_quality, photo_url, bio,
       accepting_new_patients, telehealth, ghl_contact_id, removal_requested,
       last_scraped_at, created_at, updated_at, accepts_inquiries,
       claim_token_hash, claim_verified_at, modules_completed, show_contact,
       products, order_options
FROM practitioners
WHERE removal_requested = false AND lat IS NOT NULL;
