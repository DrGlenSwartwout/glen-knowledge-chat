"""Unit tests for the Integrative Psychiatry Institute (IPI) adapter.

IPI publishes its provider directory through a GeoDirectory v2 REST endpoint
that returns every listing as flat JSON. The fixture here is a real response
captured 2026-05-29:

- ipi_listings_page_1.json — the full directory, per_page=100 (66 listings).
  Covers the credential matrix (CIPP / Psychedelic-only / no-cert), degrees
  and licenses combinations, and listings with missing zip / phone / website.

At capture time the certification split was:
  14  Certified Integrative Psychiatric Provider   (CIPP = IPI Fellowship)
  40  Certified Psychedelic Assisted Therapy Provider
  15  (no IPI certification listed)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.ipi import (  # noqa: E402
    parse_directory_json,
    _country_iso2,
    _field_list,
    _format_credentials,
    _is_fellowship,
    _normalize_website,
    _title,
)


def _load(name: str):
    return json.loads((FIXTURE_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_returns_all_listings():
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    # Every fixture record has a title -> a row.
    assert len(rows) == 66
    assert len(rows) > 0


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are constant per spec."""
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "IPI"
        assert r.specialties == ["integrative_psychiatry", "holistic_health"]
        assert r.source_url
        assert r.source_url.startswith("https://directory.psychiatryinstitute.com/")
        # Portal-managed fields stay None per spec.
        assert r.lat is None
        assert r.lng is None
        assert r.photo_url is None
        assert r.bio is None


def test_fields_populate_on_full_record():
    """CIPP holder Elizabeth L. Ward — full contact + address extraction."""
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    ward = next(r for r in rows if "Elizabeth L. Ward" in r.name)
    assert ward.practice_name == "Integrative Psychiatrist"  # job_title
    assert ward.credentials == "MD"
    assert ward.phone == "415-498-0481"
    assert ward.email == "elizabethwardmd@gmail.com"
    assert ward.website == "https://www.elizabethwardmd.com/"
    assert ward.address1 == "447 Sutter Street Suite 405-1339"
    assert ward.city == "San Francisco"
    assert ward.state == "California"
    assert ward.country == "US"
    # CIPP -> fellowship
    assert ward.fellowship_level is True


def test_credentials_combine_degrees_and_licenses():
    """Camilla Coakley carries degree MSW + license LCSW."""
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    cam = next(r for r in rows if "Camilla Coakley" in r.name)
    assert cam.credentials == "MSW, LCSW"


def test_fellowship_set_for_cipp_holders():
    """Exactly the 14 CIPP listings get fellowship_level=True."""
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    fellows = [r for r in rows if r.fellowship_level]
    assert len(fellows) == 14


def test_fellowship_not_set_for_psychedelic_only():
    """A listing carrying only the Psychedelic provider cert is NOT a fellow."""
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    cam = next(r for r in rows if "Camilla Coakley" in r.name)
    assert cam.fellowship_level is False


def test_fellowship_not_set_for_no_cert_listing():
    """A listing with no IPI certification is NOT a fellow."""
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    potocko = next(r for r in rows if "Joshua Potocko" in r.name)
    assert potocko.fellowship_level is False
    # degrees still combine into credentials
    assert potocko.credentials == "MD, MPH"


def test_country_normalized_to_iso2():
    """Free-text country names map to ISO2 — 65 US listings + 1 Canadian."""
    rows = parse_directory_json(_load("ipi_listings_page_1.json"))
    assert sum(1 for r in rows if r.country == "US") == 65
    giberson = next(r for r in rows if "Camber Isaac Giberson" in r.name)
    assert giberson.country == "CA"
    assert giberson.state == "British Columbia"
    # No raw country names leak through.
    assert all(r.country in ("US", "CA") for r in rows)


def test_source_url_stable_and_unique():
    """source_urls are the dedup keys: stable across re-runs and all distinct."""
    payload = _load("ipi_listings_page_1.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_parser_accepts_wrapped_dict_and_json_string():
    payload = _load("ipi_listings_page_1.json")
    via_list = parse_directory_json(payload)
    via_dict = parse_directory_json({"data": payload})
    via_str = parse_directory_json((FIXTURE_DIR / "ipi_listings_page_1.json").read_text())
    assert len(via_list) == len(via_dict) == len(via_str) == 66


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_field_list_reads_rendered():
    rec = {"degrees": {"raw": "MD,MPH", "rendered": ["MD", "MPH"]}}
    assert _field_list(rec, "degrees") == ["MD", "MPH"]


def test_field_list_handles_blank_and_missing():
    assert _field_list({"licenses": {"raw": "", "rendered": []}}, "licenses") == []
    assert _field_list({}, "degrees") == []
    assert _field_list({"degrees": None}, "degrees") == []


def test_field_list_falls_back_to_raw_comma_split():
    rec = {"x": {"raw": "A, B ,C"}}
    assert _field_list(rec, "x") == ["A", "B", "C"]


def test_title_reads_raw_or_plain():
    assert _title({"title": {"raw": "Jane Doe", "rendered": "Jane Doe"}}) == "Jane Doe"
    assert _title({"title": "John"}) == "John"
    assert _title({}) is None


def test_format_credentials_dedupes_and_orders():
    assert _format_credentials(["MD"], ["LCSW"]) == "MD, LCSW"
    assert _format_credentials(["MD"], ["MD"]) == "MD"
    assert _format_credentials([], []) is None


def test_normalize_website_adds_scheme():
    assert _normalize_website("example.com") == "https://example.com"
    assert _normalize_website("https://x.com/") == "https://x.com/"
    assert _normalize_website(None) is None
    assert _normalize_website("") is None


def test_country_iso2():
    assert _country_iso2("United States") == "US"
    assert _country_iso2("canada") == "CA"
    assert _country_iso2("Atlantis") is None
    assert _country_iso2(None) is None


def test_is_fellowship_matches_cipp_only():
    assert _is_fellowship(["Certified Integrative Psychiatric Provider"]) is True
    assert _is_fellowship(["Certified Psychedelic Assisted Therapy Provider"]) is False
    assert _is_fellowship([]) is False
    # case-insensitive substring match
    assert _is_fellowship(["certified integrative psychiatric provider"]) is True
