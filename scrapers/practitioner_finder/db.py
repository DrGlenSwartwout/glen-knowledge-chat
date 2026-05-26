"""practitioners table CRUD + search.

Two SQL-building functions are unit-tested as pure strings (build_search_sql,
upsert_sql_and_params). The actual execution helpers (run_upsert, run_search,
list_ungeocoded, update_geocode) are thin shims over psycopg2 — integration
tested in Task 13 against the real Supabase instance."""
from typing import Optional, Tuple

from db_supabase import supabase_cursor


MILES_TO_METERS = 1609.344


def build_search_sql(
    *,
    lat: float,
    lng: float,
    radius_miles: float,
    specialties: Optional[list[str]],
    tiers: Optional[list[str]],
    limit: int,
) -> Tuple[str, list]:
    """Build the search SQL string and parameter tuple.

    Parameter order: lat, lng, radius_meters [, specialties] [, tiers], lat, lng
    The trailing lat/lng feed the SELECT distance expression; all WHERE params
    come first so params[0..2] are always lat, lng, radius_meters."""
    where_clauses = [
        "earth_distance(ll_to_earth(%s, %s), ll_to_earth(lat, lng)) < %s",
    ]
    params: list = [lat, lng, radius_miles * MILES_TO_METERS]

    if specialties:
        where_clauses.append("specialties && %s")
        params.append(specialties)

    if tiers:
        where_clauses.append("tier = ANY(%s)")
        params.append(tiers)

    # Append lat/lng for the SELECT distance expression (after all WHERE params)
    params.extend([lat, lng])

    sql = f"""
        SELECT id, tier, source_org, fellowship_level, specialties,
               name, practice_name, credentials,
               phone, email, website,
               address1, city, state, postal, country,
               lat, lng, geocode_quality,
               photo_url, bio, accepting_new_patients, telehealth,
               earth_distance(ll_to_earth(lat, lng), ll_to_earth(%s, %s)) / {MILES_TO_METERS:.4f}
                 AS distance_miles
        FROM v_practitioners_public
        WHERE {' AND '.join(where_clauses)}
        ORDER BY distance_miles ASC
        LIMIT {int(limit)}
    """
    return sql, params


def upsert_sql_and_params(row_dict: dict) -> Tuple[str, list]:
    """Build INSERT ... ON CONFLICT (source_url) DO UPDATE SQL + params.

    Idempotent: re-running with the same source_url updates the existing row."""
    cols = list(row_dict.keys())
    params = [row_dict[c] for c in cols]
    col_sql = ", ".join(cols)
    placeholder_sql = ", ".join(["%s"] * len(cols))
    update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "source_url")

    sql = f"""
        INSERT INTO practitioners ({col_sql}, last_scraped_at)
        VALUES ({placeholder_sql}, now())
        ON CONFLICT (source_url)
        DO UPDATE SET {update_sql}, last_scraped_at = now(), updated_at = now()
    """
    return sql, params


def run_upsert(row_dict: dict) -> None:
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
) -> list[dict]:
    sql, params = build_search_sql(
        lat=lat, lng=lng, radius_miles=radius_miles,
        specialties=specialties, tiers=tiers, limit=limit,
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
