-- Regenerative Farm Finder — integrate farms into the practitioner finder as a
-- new top-level category (PROPOSED — not yet applied to prod).
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
ALTER TABLE practitioners DROP CONSTRAINT IF EXISTS practitioners_tier_check;
ALTER TABLE practitioners ADD CONSTRAINT practitioners_tier_check
  CHECK (tier IN (
    'org_member','eyehealing','panel_in_cert','panel_certified','farm'));

-- 2) Farm-only columns. Nullable; clinicians leave them NULL.
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS products      text[];
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS order_options text[];

-- practices are represented inside specialties[] (already GIN-indexed) so the
-- existing specialty filter works for farm sub-chips with no new index.

-- Opt-out reuses the existing practitioners.removal_requested flag (Glen's
-- decision) — v_practitioners_public already filters removal_requested = false.
