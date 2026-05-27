"""Unit tests for the OEPF (Optometric Extension Program Foundation) adapter.

OEPF's directory is per-listing (not a single index page like
eyehealingcenter). One listing page parses to 1 or 2 rows (primary doctor
+ optional second doctor / therapist). Each fixture covers one shape:

- oepf_listing_single_doctor.html: one fellowship-level (F.C.O.V.D.)
  practitioner with US address.
- oepf_listing_two_doctors.html: two practitioners under one practice with
  an international (Canada) address.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.oepf import (  # noqa: E402
    parse_directory_listing_html,
    _split_address,
    _extract_credentials,
    _is_fellowship,
)


def test_parse_single_doctor_listing_returns_one_row():
    html = (FIXTURE_DIR / "oepf_listing_single_doctor.html").read_text()
    rows = parse_directory_listing_html(html)
    assert len(rows) == 1
    r = rows[0]

    # Locked invariants from spec
    assert r.tier == "org_member"
    assert r.source_org == "OEPF"
    assert "functional" in r.specialties
    assert "eye_care" in r.specialties
    assert r.source_url and r.source_url.startswith("https://www.oepf.org/")
    assert "#doctor-1" in r.source_url

    # Spot-check extracted fields
    assert r.name
    assert "Valenti" in r.name
    assert r.credentials and "O.D." in r.credentials
    assert r.fellowship_level is True  # F.C.O.V.D. present
    assert r.email == "ovdcenter@yahoo.com"
    assert r.phone and "858" in r.phone
    assert r.website and "optometric-vision-development-center" in r.website
    assert r.practice_name == "Optometric Vision Development Center"
    assert r.city == "La Jolla"
    assert r.state == "CA"
    assert r.postal == "92037"
    assert r.country == "US"


def test_parse_two_doctor_listing_returns_two_rows():
    html = (FIXTURE_DIR / "oepf_listing_two_doctors.html").read_text()
    rows = parse_directory_listing_html(html)
    assert len(rows) == 2
    # Both rows share the same practice
    practice_names = {r.practice_name for r in rows}
    assert practice_names == {"Cowichan Eyecare"}

    # source_urls are distinct (so upsert ON CONFLICT works)
    urls = {r.source_url for r in rows}
    assert len(urls) == 2
    assert any(u.endswith("#doctor-1") for u in urls)
    assert any(u.endswith("#doctor-2") for u in urls)

    # Both carry the locked tier/source/specialties
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "OEPF"
        assert r.specialties == ["functional", "eye_care"]

    # Different doctor emails
    emails = {r.email for r in rows if r.email}
    assert "drrebecca@myeyecare.ca" in emails
    assert "drangela@myeyecare.ca" in emails


def test_parse_two_doctor_listing_preserves_phone_website():
    """Practice-level phone/website apply to both doctors."""
    html = (FIXTURE_DIR / "oepf_listing_two_doctors.html").read_text()
    rows = parse_directory_listing_html(html)
    for r in rows:
        assert r.phone == "250-743-8899"
        assert r.website and "myeyecare.ca" in r.website


def test_specialties_are_locked_taxonomy():
    """Per Phase 2 spec: every OEPF row must carry ['functional','eye_care']."""
    html_a = (FIXTURE_DIR / "oepf_listing_single_doctor.html").read_text()
    html_b = (FIXTURE_DIR / "oepf_listing_two_doctors.html").read_text()
    all_rows = parse_directory_listing_html(html_a) + parse_directory_listing_html(html_b)
    assert all_rows
    for r in all_rows:
        assert set(r.specialties) == {"functional", "eye_care"}


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same HTML must produce identical source_urls (dedup key)."""
    html = (FIXTURE_DIR / "oepf_listing_two_doctors.html").read_text()
    a = parse_directory_listing_html(html)
    b = parse_directory_listing_html(html)
    assert [r.source_url for r in a] == [r.source_url for r in b]


# ---------- pure helpers ----------

def test_split_address_full_us():
    a1, city, state, postal, country = _split_address(
        "8950 Villa La Jolla Drive, Ste B128, La Jolla, CA, USA 92037"
    )
    assert a1 == "8950 Villa La Jolla Drive, Ste B128"
    assert city == "La Jolla"
    assert state == "CA"
    assert postal == "92037"
    assert country == "US"


def test_split_address_canada():
    a1, city, state, postal, country = _split_address(
        "56-1400 Cowichan Bay Road, Cobble Hill, BC, Canada"
    )
    assert country == "CA"
    # State / postal should NOT be populated for non-US even though BC looks
    # like a 2-letter abbrev — it is not a US state code.
    assert state is None
    assert postal is None
    # Full string preserved in address1 so the geocoder still has signal
    assert a1 and "Cowichan Bay" in a1


def test_split_address_street_only_keeps_raw():
    """Partial 'street only' addresses keep the raw string in address1."""
    a1, city, state, postal, country = _split_address("335 Park Avenue")
    assert a1 == "335 Park Avenue"
    assert city is None
    assert state is None
    assert country is None  # Can't infer


def test_extract_credentials_with_dotted_designations():
    name, creds = _extract_credentials("Claude Valenti. O.D., F.C.O.V.D.")
    assert name == "Claude Valenti"
    assert creds and "O.D." in creds
    assert creds and "F.C.O.V.D." in creds


def test_extract_credentials_plain_name():
    name, creds = _extract_credentials("Brian Thamel")
    assert name == "Brian Thamel"
    assert creds is None


def test_extract_credentials_dr_prefix_preserved():
    name, creds = _extract_credentials("Dr. Angela Dobson")
    assert name == "Dr. Angela Dobson"
    assert creds is None


def test_fellowship_detection_variants():
    assert _is_fellowship("O.D., F.C.O.V.D.") is True
    assert _is_fellowship("FCOVD") is True
    assert _is_fellowship("OD") is False
    assert _is_fellowship(None) is False
    # via the original name field
    assert _is_fellowship(None, "Jane Doe, F.C.O.V.D.") is True
