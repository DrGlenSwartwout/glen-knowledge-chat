-- Practitioner Finder + Approved Portal shared table
-- Spec: docs/superpowers/specs/2026-05-26-practitioner-finder-design.md

CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'practitioner_geocode_quality') THEN
    CREATE TYPE practitioner_geocode_quality
      AS ENUM ('full', 'city', 'zip', 'state_only');
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS practitioners (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  tier            text NOT NULL CHECK (tier IN (
                    'org_member','eyehealing','panel_in_cert','panel_certified')),
  source_org      text,
  source_url      text,
  fellowship_level boolean DEFAULT false,
  specialties     text[] NOT NULL DEFAULT '{}',
  name            text NOT NULL,
  practice_name   text,
  credentials     text,
  phone           text,
  email           text,
  website         text,
  address1        text,
  city            text,
  state           text,
  postal          text,
  country         text DEFAULT 'US',
  lat             numeric(10,6),
  lng             numeric(10,6),
  geocode_quality practitioner_geocode_quality,
  photo_url       text,
  bio             text,
  accepting_new_patients boolean DEFAULT true,
  telehealth      boolean DEFAULT false,
  ghl_contact_id  text,
  removal_requested boolean DEFAULT false,
  last_scraped_at timestamptz,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS practitioners_specialties_gin
  ON practitioners USING GIN (specialties);
CREATE INDEX IF NOT EXISTS practitioners_lat_lng
  ON practitioners (lat, lng);
CREATE INDEX IF NOT EXISTS practitioners_state
  ON practitioners (state);
CREATE INDEX IF NOT EXISTS practitioners_tier
  ON practitioners (tier);
CREATE UNIQUE INDEX IF NOT EXISTS practitioners_source_url
  ON practitioners (source_url) WHERE source_url IS NOT NULL;

CREATE OR REPLACE VIEW v_practitioners_public AS
SELECT * FROM practitioners
WHERE removal_requested = false
  AND lat IS NOT NULL;
