"""practitioners table CRUD + search.

Two SQL-building functions are unit-tested as pure strings (build_search_sql,
upsert_sql_and_params). The actual execution helpers (run_upsert, run_search,
list_ungeocoded, update_geocode) are thin shims over psycopg2 — integration
tested in Task 13 against the real Supabase instance."""
from typing import Optional, Tuple

from db_supabase import supabase_cursor
from scrapers.practitioner_finder.normalize import normalize_country


MILES_TO_METERS = 1609.344


def build_search_sql(
    *,
    lat: float,
    lng: float,
    radius_miles: float,
    specialties: Optional[list[str]],
    tiers: Optional[list[str]],
    limit: int,
    fellowship_only: bool = False,
    countries: Optional[list[str]] = None,
) -> Tuple[str, list]:
    """Build the search SQL string and parameter tuple.

    psycopg2 substitutes %s in SQL-text order. SELECT appears before WHERE in
    the string, so its lat/lng come FIRST in params, then WHERE's lat/lng/radius,
    then optional filters. Param layout:
        [select_lat, select_lng,
         where_lat, where_lng, radius_meters,
         specialties? (if filtered),
         tiers? (if filtered)]

    fellowship_only=True narrows to rows where fellowship_level = true. Used
    by the UI's "Fellows Only" toggle to surface top-tier credentialed
    practitioners (FCOVD, MIAOMT, MIABDM, FOWNS, etc.).

    The WHERE pair `earth_box(...) @> ll_to_earth(lat, lng)` + the precise
    `earth_distance(...) < radius` is the canonical earthdistance pattern.
    earth_box yields a cube the GiST index on `ll_to_earth(lat, lng)` can
    accelerate via the @> operator (bounding-box pre-filter); the precise
    earth_distance clause then narrows to a true circle. Without the earth_box
    half, the planner does a Seq Scan even with the GiST index present."""
    radius_meters = radius_miles * MILES_TO_METERS
    where_clauses = [
        "earth_box(ll_to_earth(%s, %s), %s) @> ll_to_earth(lat, lng)",
        "earth_distance(ll_to_earth(%s, %s), ll_to_earth(lat, lng)) < %s",
    ]
    where_params: list = [
        lat, lng, radius_meters,   # earth_box bounding-box (GiST-accelerated)
        lat, lng, radius_meters,   # earth_distance precise circle
    ]

    if specialties:
        where_clauses.append("specialties && %s")
        where_params.append(specialties)

    if tiers:
        where_clauses.append("tier = ANY(%s)")
        where_params.append(tiers)

    if countries:
        # Defense-in-depth: a US-ZIP search constrains results to US rows so a
        # stray mis-geocoded coordinate (foreign row that slipped through) can
        # never surface in a domestic search. See geocode.mapbox_country_filter.
        where_clauses.append("country = ANY(%s)")
        where_params.append(countries)

    if fellowship_only:
        where_clauses.append("fellowship_level = true")

    sql = f"""
        SELECT id, tier, source_org, fellowship_level, specialties,
               name, practice_name, credentials,
               phone, email, website,
               address1, city, state, postal, country,
               lat, lng, geocode_quality,
               photo_url, bio, accepting_new_patients, telehealth,
               accepts_inquiries, modules_completed, show_contact,
               earth_distance(ll_to_earth(lat, lng), ll_to_earth(%s, %s)) / {MILES_TO_METERS:.4f}
                 AS distance_miles
        FROM v_practitioners_public
        WHERE {' AND '.join(where_clauses)}
        ORDER BY distance_miles ASC
        LIMIT {int(limit)}
    """
    # SELECT params (consumed first by psycopg2) come before WHERE params.
    params = [lat, lng] + where_params
    return sql, params


def upsert_sql_and_params(row_dict: dict) -> Tuple[str, list]:
    """Build INSERT ... ON CONFLICT (source_url) DO UPDATE SQL + params.

    Idempotent: re-running with the same source_url updates the existing row."""
    cols = list(row_dict.keys())
    params = [row_dict[c] for c in cols]
    col_sql = ", ".join(cols)
    placeholder_sql = ", ".join(["%s"] * len(cols))
    update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "source_url")

    # The unique index on source_url is partial (WHERE source_url IS NOT NULL),
    # so ON CONFLICT must repeat that predicate for Postgres to match it.
    # Otherwise: "no unique or exclusion constraint matching the ON CONFLICT specification".
    sql = f"""
        INSERT INTO practitioners ({col_sql}, last_scraped_at)
        VALUES ({placeholder_sql}, now())
        ON CONFLICT (source_url) WHERE source_url IS NOT NULL
        DO UPDATE SET {update_sql}, last_scraped_at = now(), updated_at = now()
    """
    return sql, params


def run_upsert(row_dict: dict) -> None:
    # Normalize the country to an ISO-2 code at the single write boundary so
    # every scraped row stores a clean value (adapters stay unchanged).
    if "country" in row_dict:
        row_dict["country"] = normalize_country(row_dict.get("country"))
    sql, params = upsert_sql_and_params(row_dict)
    with supabase_cursor() as cur:
        cur.execute(sql, params)


def run_search(
    *,
    lat: float,
    lng: float,
    radius_miles: float,
    specialties: Optional[list[str]],
    tiers: Optional[list[str]],
    limit: int = 200,
    fellowship_only: bool = False,
    countries: Optional[list[str]] = None,
) -> list[dict]:
    sql, params = build_search_sql(
        lat=lat, lng=lng, radius_miles=radius_miles,
        specialties=specialties, tiers=tiers, limit=limit,
        fellowship_only=fellowship_only, countries=countries,
    )
    with supabase_cursor() as cur:
        cur.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def list_ungeocoded() -> list[dict]:
    """Rows that have geocodable input but no lat/lng yet."""
    sql = """
        SELECT id, address1, city, state, postal, country
        FROM practitioners
        WHERE lat IS NULL
          AND removal_requested = false
          AND (
            (address1 IS NOT NULL AND city IS NOT NULL AND state IS NOT NULL)
            OR (city IS NOT NULL AND state IS NOT NULL)
            OR postal IS NOT NULL
            OR state IS NOT NULL
          )
    """
    with supabase_cursor() as cur:
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]


def update_geocode(
    practitioner_id: str,
    lat: Optional[float],
    lng: Optional[float],
    quality: Optional[str],
) -> None:
    sql = """
        UPDATE practitioners
        SET lat = %s, lng = %s, geocode_quality = %s, updated_at = now()
        WHERE id = %s
    """
    with supabase_cursor() as cur:
        cur.execute(sql, (lat, lng, quality, practitioner_id))
