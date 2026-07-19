"""Fixture-driven test for the Food for Humans farm-finder adapter.

Mirrors the tests/test_pf_*.py convention: parse a saved listing-page HTML
fixture and assert the NormalizedFarmRow fields. No network."""
import os

from scrapers.farm_finder.foodforhumans import parse_listing
from scrapers.farm_finder.mapping import (
    PARENT_SPECIALTY,
    TIER_FARM,
    practice_slug,
    to_practitioner_row,
)

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "farm_finder", "meadowdale.html"
)
URL = "https://findfoodforhumans.com/listing/meadowdale-farm-and-sawmill/"


def _row():
    with open(FIXTURE, encoding="utf-8") as fh:
        return parse_listing(fh.read(), URL)


def test_core_identity_and_provenance():
    row = _row()
    assert row is not None
    assert row.name == "Meadowdale Farm and Sawmill"
    assert row.source_org == "Food for Humans"
    assert row.source_url == URL


def test_contact_fields():
    row = _row()
    assert row.email == "Meadowdalefarmtn@gmail.com"
    assert row.phone == "(802) 380-1014"
    assert row.website == "https://www.meadowdalefarm.com/"


def test_location_is_pre_geocoded():
    row = _row()
    assert row.address1 == "1400 Buttermilk Road"   # street line only
    assert row.city == "Lenoir City"
    assert row.state == "TN"          # normalized from "Tennessee"
    assert row.postal == "37771"
    assert row.country == "US"
    assert round(row.lat, 4) == 35.8760
    assert round(row.lng, 4) == -84.3202
    assert row.geocode_quality == "source"


def test_products_from_makes_offer():
    row = _row()
    assert row.products == ["Chicken", "Turkey", "Eggs"]


def test_practices_are_segmented_not_mixed_with_order_options():
    row = _row()
    # Regenerative practice markers land in practices ...
    assert "Rotational Grazing" in row.practices
    assert "Pasture-Raised" in row.practices
    assert "Non-GMO" in row.practices
    assert "No Till" in row.practices
    # ... and ordering options do NOT bleed into practices.
    assert "Farm Pickup" not in row.practices
    assert "Farm Pickup" in row.order_options
    assert "Bulk Orders" in row.order_options


def test_no_localbusiness_returns_none():
    assert parse_listing("<html><body>not a farm</body></html>", URL) is None


# --- integration mapping (farm -> practitioners row) ---

def test_practice_slug():
    assert practice_slug("Pasture-Raised") == "pasture_raised"
    assert practice_slug("Non-GMO") == "non_gmo"
    assert practice_slug("No Till") == "no_till"


def test_mapped_row_is_a_farm_tier_with_parent_specialty():
    pr = to_practitioner_row(_row())
    assert pr["tier"] == TIER_FARM
    # parent tag first so the "Regenerative Farms" chip matches every farm ...
    assert pr["specialties"][0] == PARENT_SPECIALTY
    # ... and each practice is a filterable sub-tag slug.
    assert "pasture_raised" in pr["specialties"]
    assert "rotational_grazing" in pr["specialties"]
    # no duplicate specialties
    assert len(pr["specialties"]) == len(set(pr["specialties"]))


def test_mapped_row_carries_farm_columns_and_maps_geocode_quality():
    pr = to_practitioner_row(_row())
    assert pr["products"] == ["Chicken", "Turkey", "Eggs"]
    assert "Farm Pickup" in pr["order_options"]
    assert pr["website"] == "https://www.meadowdalefarm.com/"
    # 'source' precision maps onto the existing enum's 'full'
    assert pr["geocode_quality"] == "full"
    assert pr["name"] == "Meadowdale Farm and Sawmill"


# --- ingest runner (dry run writes nothing, no DB import) ---

def test_ingest_dry_run_maps_but_writes_nothing():
    from scrapers.farm_finder import ingest as ingest_mod

    # Inject a single fake source rather than monkeypatching a module import —
    # the multi-source ingest takes a `sources` override for exactly this.
    fake = [("foodforhumans", lambda limit=None, sleep=0: [_row()])]
    summary = ingest_mod.ingest(sources=fake, apply=False, log=lambda *_: None)
    assert summary == {
        "per_source": {"foodforhumans": 1},
        "scraped": 1, "deduped": 1, "mapped": 1, "written": 0, "applied": False,
        "with_geo": 1, "with_website": 1,
    }
