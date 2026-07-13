"""Fixture-driven test for the Food for Humans farm-finder adapter.

Mirrors the tests/test_pf_*.py convention: parse a saved listing-page HTML
fixture and assert the NormalizedFarmRow fields. No network."""
import os

from scrapers.farm_finder.foodforhumans import parse_listing

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
    assert row.address1.startswith("1400 Buttermilk Road")
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
