"""Fixture-driven test for the USDA Local Food Directories adapter.

Real records captured from the keyless bulk endpoint (farmersmarket directory).
No network: parse_row is pure."""
import json
import os

from scrapers.farm_finder.usda import parse_row, _parse_address, DIRECTORIES
from scrapers.farm_finder.mapping import PARENT_SPECIALTY, TIER_FARM, to_practitioner_row

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "farm_finder", "usda_farmersmarket.json"
)


def _records():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _rows():
    return [parse_row(r, "farmersmarket") for r in _records()]


def test_core_identity_and_provenance():
    row = _rows()[0]
    assert row is not None
    assert row.name == "Colorado Farm and Art Market"
    assert row.source_org == "USDA Local Food Directories"
    assert row.source_url.endswith("/fe/farmersmarket/300002")


def test_full_address_split():
    row = _rows()[0]
    assert row.address1 == "7350 Pine Creek Road"
    assert row.city == "Colorado Springs"
    assert row.state == "CO"          # 'Colorado' -> 'CO'
    assert row.postal == "80919"


def test_streetless_address_split():
    # "Rochester, Indiana 46975" — no street line.
    row = _rows()[1]
    assert row.address1 is None
    assert row.city == "Rochester"
    assert row.state == "IN"
    assert row.postal == "46975"


def test_coords_from_x_y_and_quality():
    row = _rows()[0]
    assert round(row.lat, 4) == 38.9377   # y = latitude
    assert round(row.lng, 4) == -104.8147  # x = longitude
    assert row.geocode_quality == "source"


def test_practices_include_directory_and_organic():
    row = _rows()[0]
    assert "farmers_market" in row.practices
    assert "usda_organic" in row.practices    # 'Organic (USDA Certified);' -> existing chip


def test_order_options_snap_vs_wic():
    snap_row, wic_row = _rows()
    assert "SNAP/EBT" in snap_row.order_options       # FNAP has SNAP + EBT
    assert "SNAP/EBT" not in wic_row.order_options     # WIC-only, no SNAP/EBT


def test_no_website_in_bulk_export():
    assert all(r.website is None for r in _rows())


def test_maps_to_farm_practitioner_row():
    pr = to_practitioner_row(_rows()[0])
    assert pr["tier"] == TIER_FARM
    assert PARENT_SPECIALTY in pr["specialties"]       # parent 'Regenerative Farms'
    assert "farmers_market" in pr["specialties"]        # sub-chip
    assert pr["geocode_quality"] == "full"              # 'source' -> enum 'full'


def test_parse_address_edge_cases():
    assert _parse_address(None) == (None, None, None, None)
    assert _parse_address("") == (None, None, None, None)
    # City + 2-letter state + zip, no street
    s, c, st, z = _parse_address("Detroit, MI 48260")
    assert (c, st, z) == ("Detroit", "MI", "48260")


def test_parse_address_new_england_leading_zero_zip():
    # USDA drops the leading 0 -> "6484"; we pad it back and still resolve state.
    s, c, st, z = _parse_address("Some Rd, Deep River, Connecticut 6484")
    assert st == "CT"
    assert z == "06484"


def test_parse_address_trailing_state_code():
    # "Tennessee TN" — full name plus a redundant trailing code.
    _s, _c, st, _z = _parse_address("123 Farm Ln, Franklin, Tennessee TN 37064")
    assert st == "TN"


def test_parse_address_strips_country_suffix():
    s, c, st, z = _parse_address("1400 Buttermilk Rd, Lenoir City, Tennessee 37771, United States")
    assert st == "TN"
    assert z == "37771"
    assert c == "Lenoir City"


def test_directories_exclude_agritourism_by_default():
    assert "agritourism" not in DIRECTORIES
    assert "farmersmarket" in DIRECTORIES
