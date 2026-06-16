import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.practitioner_finder.db import (
    build_search_sql,
    upsert_sql_and_params,
)


def test_search_sql_with_specialties_and_radius():
    sql, params = build_search_sql(
        lat=21.3099, lng=-157.8581, radius_miles=25,
        specialties=["eye_care", "syntonic"], tiers=None, limit=200,
    )
    # earth_box pre-filter (GiST-indexable) + precise earth_distance circle
    assert "earth_box(ll_to_earth(%s, %s), %s) @> ll_to_earth(lat, lng)" in sql
    assert "earth_distance" in sql
    assert "%s" in sql  # parameterized
    # Specialty filter using && (array overlap)
    assert "specialties && %s" in sql
    # Tier filter NOT applied when tiers is None
    assert "tier = ANY" not in sql
    # Limit applied
    assert "LIMIT 200" in sql
    # Params in SQL-text order: SELECT lat, lng (for distance display),
    # then earth_box (lat, lng, radius), then earth_distance (lat, lng, radius),
    # then filter values.
    radius_meters = 25 * 1609.344
    assert params[0] == 21.3099       # SELECT lat
    assert params[1] == -157.8581     # SELECT lng
    assert params[2] == 21.3099       # earth_box lat
    assert params[3] == -157.8581     # earth_box lng
    assert abs(params[4] - radius_meters) < 0.01  # earth_box radius
    assert params[5] == 21.3099       # earth_distance lat
    assert params[6] == -157.8581     # earth_distance lng
    assert abs(params[7] - radius_meters) < 0.01  # earth_distance radius
    assert params[8] == ["eye_care", "syntonic"]  # specialty filter


def test_search_sql_no_specialties_no_filter():
    sql, params = build_search_sql(
        lat=21.0, lng=-157.0, radius_miles=50,
        specialties=None, tiers=None, limit=200,
    )
    assert "specialties &&" not in sql
    assert ["eye_care"] not in params


def test_search_sql_with_tier_filter():
    sql, params = build_search_sql(
        lat=21.0, lng=-157.0, radius_miles=50,
        specialties=None, tiers=["eyehealing"], limit=200,
    )
    assert "tier = ANY(%s)" in sql
    assert ["eyehealing"] in params


def test_search_sql_selects_modules_completed():
    """The Certification list shows each student's level, so modules_completed
    must be returned for every search result (null for non-cert rows)."""
    sql, _ = build_search_sql(
        lat=21.0, lng=-157.0, radius_miles=50,
        specialties=None, tiers=None, limit=200,
    )
    assert "modules_completed" in sql


def test_search_sql_cert_tier_filter():
    """Selecting the Certification category filters by tier = ANY(...) with the
    cert tiers, exactly as the frontend's tier[] params drive it."""
    sql, params = build_search_sql(
        lat=21.0, lng=-157.0, radius_miles=50,
        specialties=None, tiers=["panel_in_cert", "panel_certified"], limit=200,
    )
    assert "tier = ANY(%s)" in sql
    assert "modules_completed" in sql
    assert ["panel_in_cert", "panel_certified"] in params


def test_search_sql_fellowship_only_adds_clause():
    sql, params = build_search_sql(
        lat=21.0, lng=-157.0, radius_miles=50,
        specialties=None, tiers=None, limit=200,
        fellowship_only=True,
    )
    assert "fellowship_level = true" in sql


def test_search_sql_fellowship_only_default_false_omits_clause():
    sql, _ = build_search_sql(
        lat=21.0, lng=-157.0, radius_miles=50,
        specialties=None, tiers=None, limit=200,
    )
    # `fellowship_level` appears in the SELECT clause (always returned),
    # but the WHERE-clause filter `= true` should only appear when fellowship_only=True
    assert "fellowship_level = true" not in sql


def test_search_sql_countries_guard_adds_clause():
    """Defense-in-depth: a US-ZIP search constrains results to US rows so a
    stray mis-geocoded foreign coordinate can never surface."""
    sql, params = build_search_sql(
        lat=42.75, lng=-73.76, radius_miles=25,
        specialties=None, tiers=None, limit=200,
        countries=["US"],
    )
    assert "country = ANY(%s)" in sql
    assert ["US"] in params


def test_search_sql_countries_default_omits_clause():
    sql, _ = build_search_sql(
        lat=42.75, lng=-73.76, radius_miles=25,
        specialties=None, tiers=None, limit=200,
    )
    assert "country = ANY(%s)" not in sql


def test_upsert_sql_params_match_dict():
    row_dict = {
        "tier": "eyehealing",
        "name": "Dr. Jane Doe",
        "specialties": ["eye_care"],
        "source_url": "https://eyehealingcenter.com/some-id",
        "city": "Honolulu",
        "state": "HI",
    }
    sql, params = upsert_sql_and_params(row_dict)
    # ON CONFLICT clause for idempotency
    assert "ON CONFLICT (source_url)" in sql
    assert "DO UPDATE SET" in sql
    # All columns in row_dict appear in params
    for key in row_dict.keys():
        assert key in sql, f"missing column {key}"
    # updated_at is appended automatically
    assert "updated_at = now()" in sql
