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
    # Earthdistance query
    assert "earth_distance" in sql
    assert "ll_to_earth" in sql
    assert "%s" in sql  # parameterized
    # Specialty filter using && (array overlap)
    assert "specialties && %s" in sql
    # Tier filter NOT applied when tiers is None
    assert "tier = ANY" not in sql
    # Limit applied
    assert "LIMIT 200" in sql
    # Params include radius in METERS (25 mi * 1609.344)
    assert params[0] == 21.3099
    assert params[1] == -157.8581
    assert abs(params[2] - 25 * 1609.344) < 0.01
    assert params[3] == ["eye_care", "syntonic"]


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
