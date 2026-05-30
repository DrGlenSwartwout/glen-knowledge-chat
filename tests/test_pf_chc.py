"""Unit tests for the Council for Homeopathic Certification (CHC) adapter.

CHC's "Find a Homeopath" directory paginates randomly in its HTML view, so the
adapter drives the deterministic WordPress REST API for the complete listing
index (name + slug + permalink) and parses each listing's detail page for the
labeled wpbdp fields (Phone / Website / Address / ZIP Code). The detail HTML
samples below mirror the real markup captured 2026-05-29.

CHC publishes no practitioner email (email=None), issues the CCH credential, and
has no fellow tier (fellowship_level=False).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scrapers.practitioner_finder.chc import (  # noqa: E402
    parse_index_json,
    parse_detail_html,
    _parse_location,
    _clean_website,
    BASE,
    CREDENTIAL,
)


def _detail(*field_blocks: str) -> str:
    """Wrap labeled field blocks in the real wpbdp detail-field container."""
    inner = "\n".join(
        f'<div class="wpbdp-field-display wpbdp-field wpbdp-field-value field">{b}</div>'
        for b in field_blocks
    )
    return f'<div class="listing-details">{inner}</div>'


CATEGORY = '<div class="value"><a href="/find-a-homeopath/wpbdp_category/homeopath/" rel="tag">Homeopath</a></div>'
PHONE = '<span class="field-label">Phone:</span> <div class="value"><a href="tel:2674465646">2674465646</a></div>'
WEBSITE = '<span class="field-label">Website:</span> <div class="value"><a href="https://www.heirloomministries.org/" rel="" target="_self" title="x">https://www.heirloomministries.org/</a></div>'
ADDRESS = '<span class="field-label">Address:</span> <div class="value">310 Broad St, Suite E, Harleysville, PA</div>'
ZIP = '<span class="field-label">ZIP Code:</span> <div class="value">19438</div>'
CERT = '<span class="field-label">Certified Since:</span> <div class="value">Nov 21, 2023</div>'

DETAIL_FULL = _detail(CATEGORY, PHONE, WEBSITE, ADDRESS, ZIP, CERT)


# ---------------------------------------------------------------------------
# REST index parser
# ---------------------------------------------------------------------------

def test_parse_index_json():
    items = [
        {"slug": "lauren-messina", "link": f"{BASE}/find-a-homeopath/lauren-messina/",
         "title": {"rendered": "Lauren Messina"}},
        {"slug": "jos-amp-eacute", "link": f"{BASE}/find-a-homeopath/jose/",
         "title": {"rendered": "Jos&#233;"}},   # entity-decoded
        {"slug": "x", "link": None, "title": {"rendered": "No URL"}},   # dropped
        {"slug": "y", "link": f"{BASE}/find-a-homeopath/y/", "title": {"rendered": ""}},  # dropped
    ]
    rows = parse_index_json(items)
    assert len(rows) == 2
    assert rows[0] == {"name": "Lauren Messina", "slug": "lauren-messina",
                       "url": f"{BASE}/find-a-homeopath/lauren-messina/"}
    assert rows[1]["name"] == "José"


def test_parse_index_json_handles_bad_input():
    assert parse_index_json(None) == []
    assert parse_index_json([]) == []
    assert parse_index_json(["notadict", 3]) == []


# ---------------------------------------------------------------------------
# Detail-page parser
# ---------------------------------------------------------------------------

def _row(html, name="Lauren Messina", url=None):
    return parse_detail_html(html, name, url or f"{BASE}/find-a-homeopath/lauren-messina/")


def test_full_detail_record():
    r = _row(DETAIL_FULL)
    assert r.name == "Lauren Messina"
    assert r.source_org == "CHC"
    assert r.specialties == ["homeopathy", "holistic_health"]
    assert r.credentials == CREDENTIAL == "CCH"
    assert r.fellowship_level is False
    assert r.email is None
    assert r.phone == "2674465646"
    assert r.website == "https://www.heirloomministries.org/"
    assert r.address1 == "310 Broad St, Suite E"
    assert r.city == "Harleysville"
    assert r.state == "PA"
    assert r.postal == "19438"
    assert r.country == "US"
    assert r.source_url == f"{BASE}/find-a-homeopath/lauren-messina/"


def test_city_only_address_no_street():
    html = _detail(
        '<span class="field-label">Phone:</span> <div class="value"><a href="tel:9186880316">918 688-0316</a></div>',
        '<span class="field-label">Website:</span> <div class="value"><a href="https://marianasullivan.com">site</a></div>',
        '<span class="field-label">Address:</span> <div class="value">Arlington, TX</div>',
    )
    r = _row(html, name="Mariana Sullivan")
    assert r.address1 is None
    assert r.city == "Arlington" and r.state == "TX"
    assert r.postal is None
    assert r.website == "https://marianasullivan.com"
    assert r.phone == "918 688-0316"


def test_canadian_province_sets_country_ca():
    html = _detail('<span class="field-label">Address:</span> <div class="value">6051 Gleneagles Drive, West Vancouver, BC</div>')
    r = _row(html, name="Coleen Davis-Stanton")
    assert r.state == "BC" and r.country == "CA" and r.city == "West Vancouver"


def test_address_with_inline_zip_is_stripped():
    html = _detail('<span class="field-label">Address:</span> <div class="value">310 Broad St, Suite E, Harleysville, PA 19438</div>')
    r = _row(html)
    assert r.state == "PA" and r.city == "Harleysville"
    assert r.address1 == "310 Broad St, Suite E"


def test_profile_not_public_keeps_name_only():
    """A non-public listing detail has no contact fields -> name + url only."""
    html = _detail(CATEGORY)  # category only, no labeled fields
    r = _row(html, name="Autumn Louise Schaefer",
             url=f"{BASE}/find-a-homeopath/autumn-louise-schaefer/")
    assert r.name == "Autumn Louise Schaefer"
    assert r.phone is None and r.website is None
    assert r.address1 is None and r.city is None and r.state is None and r.postal is None
    assert r.country == "US"


def test_polluted_website_trimmed():
    html = _detail('<span class="field-label">Website:</span> <div class="value"><a href="http://www.tahoe-homeopathy.comwww.marinhomeopathy.com">x</a></div>')
    r = _row(html)
    assert r.website == "http://www.tahoe-homeopathy.com"


def test_parse_detail_requires_name_and_url():
    assert parse_detail_html(DETAIL_FULL, "", "url") is None
    assert parse_detail_html(DETAIL_FULL, "Name", "") is None


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_parse_location_variants():
    assert _parse_location("Arlington, TX") == {
        "address1": None, "city": "Arlington", "state": "TX", "country": "US"}
    assert _parse_location("310 Broad St, Suite E, Harleysville, PA") == {
        "address1": "310 Broad St, Suite E", "city": "Harleysville",
        "state": "PA", "country": "US"}
    assert _parse_location("3009 Pelham Rd,Madison, WI") == {
        "address1": "3009 Pelham Rd", "city": "Madison",
        "state": "WI", "country": "US"}
    assert _parse_location("6051 Gleneagles Drive, West Vancouver, BC")["country"] == "CA"
    assert _parse_location("123 Main St, Denver, CO 80202")["state"] == "CO"
    assert _parse_location("Somewhere overseas") == {
        "address1": None, "city": None, "state": None, "country": None}
    assert _parse_location("") == {
        "address1": None, "city": None, "state": None, "country": None}


def test_clean_website():
    assert _clean_website("https://marianasullivan.com") == "https://marianasullivan.com"
    assert _clean_website("https://www.heirloomministries.org/") == "https://www.heirloomministries.org/"
    assert _clean_website(
        "http://www.tahoe-homeopathy.comwww.marinhomeopathy.com"
    ) == "http://www.tahoe-homeopathy.com"
    assert _clean_website("") is None
