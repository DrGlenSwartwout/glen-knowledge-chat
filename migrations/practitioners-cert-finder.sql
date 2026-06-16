-- Expose certification progress (modules_completed) to the public finder so the
-- "Certification" list can show each student's level (1-12). Backward-compatible:
-- adds one column to v_practitioners_public; all existing columns unchanged.
CREATE OR REPLACE VIEW v_practitioners_public
WITH (security_invoker = on)
AS
 SELECT id, tier, source_org, source_url, fellowship_level, specialties, name,
        practice_name, credentials, phone, email, website, address1, city, state,
        postal, country, lat, lng, geocode_quality, photo_url, bio,
        accepting_new_patients, telehealth, ghl_contact_id, removal_requested,
        last_scraped_at, created_at, updated_at, accepts_inquiries,
        claim_token_hash, claim_verified_at,
        modules_completed
   FROM practitioners
  WHERE removal_requested = false AND lat IS NOT NULL;
