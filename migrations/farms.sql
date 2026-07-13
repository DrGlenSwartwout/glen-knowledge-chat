-- Regenerative Farm Finder table (PROPOSED — not yet applied to prod).
-- Spec: docs/superpowers/specs/2026-07-13-farm-finder-design.md
-- Forks the practitioners table: same cube/earthdistance radius-search
-- machinery, but farm-shaped columns (practices/products/order_options).

CREATE EXTENSION IF NOT EXISTS cube;
CREATE EXTENSION IF NOT EXISTS earthdistance;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- for gen_random_uuid()

CREATE TABLE IF NOT EXISTS farms (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  source_org      text,                 -- e.g. 'Food for Humans'
  source_url      text,                 -- listing page; upsert key
  name            text NOT NULL,
  description     text,
  practices       text[] NOT NULL DEFAULT '{}',  -- regenerative markers
  products        text[] NOT NULL DEFAULT '{}',
  order_options   text[] NOT NULL DEFAULT '{}',
  phone           text,
  email           text,
  website         text,
  image_url       text,
  address1        text,
  city            text,
  state           text,
  postal          text,
  country         text DEFAULT 'US',
  lat             numeric(10,6),
  lng             numeric(10,6),
  geocode_quality text,                 -- 'source' when the directory supplied it
  removal_requested boolean DEFAULT false,
  last_scraped_at timestamptz,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS farms_practices_gin
  ON farms USING GIN (practices);
CREATE INDEX IF NOT EXISTS farms_products_gin
  ON farms USING GIN (products);
CREATE INDEX IF NOT EXISTS farms_lat_lng
  ON farms (lat, lng);
CREATE INDEX IF NOT EXISTS farms_state
  ON farms (state);
CREATE UNIQUE INDEX IF NOT EXISTS farms_source_url
  ON farms (source_url) WHERE source_url IS NOT NULL;

CREATE OR REPLACE VIEW v_farms_public
WITH (security_invoker = on)
AS
SELECT * FROM farms
WHERE removal_requested = false
  AND lat IS NOT NULL;
