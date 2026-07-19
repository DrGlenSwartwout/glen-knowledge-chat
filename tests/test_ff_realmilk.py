"""Fixture-driven test for the Real Milk (realmilk.com) wpbdp adapter. No network."""
import os

from scrapers.farm_finder.realmilk import parse_listing, BASE_PRACTICE
from scrapers.farm_finder.mapping import PARENT_SPECIALTY, TIER_FARM, to_practitioner_row

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "farm_finder", "realmilk_listing.html"
)
URL = "https://www.realmilk.com/farm-directory/warrior-food-nj/"


def _row():
    with open(FIXTURE, encoding="utf-8") as fh:
        return parse_listing(fh.read(), URL)


def test_identity_and_provenance():
    row = _row()
    assert row is not None
    assert row.name == "Warrior Food – New Jersey"   # ld+json, entity-unescaped
    assert row.source_org == "A Campaign for Real Milk"
    assert row.source_url == URL


def test_location_from_wpbdp_fields():
    row = _row()
    assert row.city == "Long Valley"
    assert row.state == "CT"          # realmilk's own field value -> code
    assert row.postal == "07853"
    assert row.country == "US"


def test_contact_and_no_coords():
    row = _row()
    assert row.email == "LandoMilkandHoney@yahoo.com"   # entity-decoded
    assert row.website is None
    assert row.lat is None and row.lng is None          # geocoded by sweep
    assert row.geocode_quality is None


def test_practice_and_order_options():
    row = _row()
    assert row.practices == [BASE_PRACTICE]              # raw_milk
    assert "Home Delivery" in row.order_options          # type_of_location


def test_maps_to_farm_practitioner_row():
    pr = to_practitioner_row(_row())
    assert pr["tier"] == TIER_FARM
    assert PARENT_SPECIALTY in pr["specialties"]
    assert "raw_milk" in pr["specialties"]
    # No coords -> geocode_quality stays unset so the global sweep picks it up.
    assert "geocode_quality" not in pr
