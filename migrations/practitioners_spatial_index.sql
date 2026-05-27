-- Spatial GiST index for earth_distance() radius search on practitioners.
--
-- Without this, /api/practitioner-finder/search runs a full seq scan and
-- computes earth_distance() per row. The btree on (lat, lng) cannot help
-- because earth_distance() works on the ll_to_earth() expression, not the
-- raw columns. This index lets the planner use a GiST <-> operator scan.
--
-- Companion to migrations/practitioners.sql (which already enables the
-- cube + earthdistance extensions). NULL coordinates are naturally excluded
-- because ll_to_earth() is STRICT and returns NULL on NULL inputs, which
-- GiST does not index.

CREATE INDEX CONCURRENTLY IF NOT EXISTS practitioners_lat_lng_gist
  ON practitioners
  USING gist (ll_to_earth(lat, lng));
