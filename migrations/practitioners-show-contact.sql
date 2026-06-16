-- Per-record contact visibility for the public finder. Default hidden; a
-- practitioner (esp. certification-track) opts in to show email/phone.
-- Backward-compatible: adds one column; recreates the view to expose it.
ALTER TABLE practitioners ADD COLUMN IF NOT EXISTS show_contact boolean NOT NULL DEFAULT false;

CREATE OR REPLACE VIEW v_practitioners_public
WITH (security_invoker = on)
AS
 SELECT id, tier, source_org, source_url, fellowship_level, specialties, name,
        practice_name, credentials, phone, email, website, address1, city, state,
        postal, country, lat, lng, geocode_quality, photo_url, bio,
        accepting_new_patients, telehealth, ghl_contact_id, removal_requested,
        last_scraped_at, created_at, updated_at, accepts_inquiries,
        claim_token_hash, claim_verified_at, modules_completed, show_contact
   FROM practitioners
  WHERE removal_requested = false AND lat IS NOT NULL;
