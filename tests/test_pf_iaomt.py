"""Unit tests for the IAOMT (International Academy of Oral Medicine and
Toxicology) adapter.

IAOMT publishes its directory through a single AJAX endpoint that returns
JSON pages of practitioner records. Fixtures here are real responses
captured 2026-05-27:

- iaomt_page_1.json   — page 1 @ per_page=100 (top-tier Master/Fellow
                        practitioners; mostly US, with Canada / UK / Turkey
                        / Singapore / Brazil / Chile examples).
- iaomt_page_15.json  — page 15 @ per_page=50 (lower-tier records:
                        General / Biological_Dental_Hygiene_Accredited;
                        international skew with Turkey, UAE, Egypt,
                        Costa Rica, France, Denmark, Spain).

These two cover the credential matrix (Master / Fellow / Accredited /
Hygiene-Accredited / General / SMART-only) and the geographic spread
(US with state+zip, Canadian postal codes, international with no state).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.iaomt import (  # noqa: E402
    parse_directory_json,
    _build_source_url,
    _country_iso2,
    _format_credentials,
    _is_fellowship,
    _normalize_website,
    _parse_degrees,
)


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


def test_parse_page_1_returns_full_batch():
    payload = _load("iaomt_page_1.json")
    rows = parse_directory_json(payload)
    # Page 1 fixture is per_page=100; every record has Account_Name.
    assert len(rows) == 100


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are constant per spec."""
    rows = parse_directory_json(_load("iaomt_page_1.json"))
    rows += parse_directory_json(_load("iaomt_page_15.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "IAOMT"
        assert r.specialties == ["biological", "dental"]
        # source_url is always populated (used as upsert dedup key)
        assert r.source_url
        assert r.source_url.startswith("https://iaomt.org/for-patients/members/")


def test_spot_check_first_practitioner_full_fields():
    """First record on page 1 — Crystal Robyn Abramczyk, Master+SMART,
    full US address. Validates name + address + contact extraction."""
    rows = parse_directory_json(_load("iaomt_page_1.json"))
    abramczyk = next(r for r in rows if "Abramczyk" in r.name)

    assert abramczyk.name == "Crystal Robyn Abramczyk"
    assert abramczyk.practice_name == "Smile Ranch Dentistry"
    assert abramczyk.city == "Heath"
    assert abramczyk.state == "Texas"
    assert abramczyk.postal == "75032"
    assert abramczyk.country == "US"
    assert abramczyk.address1 == "6700 Horizon Road"
    assert abramczyk.phone == "9727727645"
    assert abramczyk.email == "smileranchdentistry@yahoo.com"
    assert abramczyk.website == "https://smileranchdentistry.com/"
    # Master = fellowship-tier
    assert abramczyk.fellowship_level is True
    # Credentials should contain the dental degrees AND the IAOMT
    # designation from Other_Degrees.
    assert abramczyk.credentials
    assert "DDS" in abramczyk.credentials
    assert "MIAOMT" in abramczyk.credentials


def test_fellowship_detection_accredited_only():
    """An Accredited Member (no Master/Fellow) is still fellowship-tier
    per the spec ('Accredited Member or Master or Fellow')."""
    rows = parse_directory_json(_load("iaomt_page_1.json"))
    butcher = next(r for r in rows if "Jessica Butcher" in r.name)
    assert butcher.fellowship_level is True
    assert butcher.state == "North Carolina"


def test_fellowship_detection_hygiene_accredited():
    """Hygiene Accredited counts as fellowship-tier per spec interpretation."""
    rows = parse_directory_json(_load("iaomt_page_15.json"))
    tritz = next(r for r in rows if "Barbara Tritz" in r.name)
    assert tritz.fellowship_level is True


def test_fellowship_not_set_for_general_member():
    """A General member without Master/Fellow/Accredited/Hygiene-tier
    flags is NOT fellowship-tier (SMART certification alone also
    doesn't qualify per spec)."""
    rows = parse_directory_json(_load("iaomt_page_15.json"))
    nalan = next(r for r in rows if "Nalan" in r.name)
    assert nalan.fellowship_level is False


def test_international_record_maps_country_to_iso2():
    """Canadian record on page 1 — country mapped to 'CA',
    Canadian postal code preserved as-is in postal."""
    rows = parse_directory_json(_load("iaomt_page_1.json"))
    shapka = next(r for r in rows if "Nestor Shapka" in r.name)
    assert shapka.country == "CA"
    assert shapka.state == "Alberta"
    assert shapka.postal == "T9N 2G4"


def test_international_record_with_no_state_kept_intact():
    """Turkish record from page 15 has no State_Province — must not
    crash, must still produce a valid row."""
    rows = parse_directory_json(_load("iaomt_page_15.json"))
    nalan = next(r for r in rows if "Nalan" in r.name)
    assert nalan.country == "TR"
    assert nalan.state is None
    assert nalan.city == "Istanbul"


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the dedup keys for ON CONFLICT upsert."""
    payload = _load("iaomt_page_1.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    # And all distinct (no two practitioners share a source_url).
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_format_matches_iaomt_member_link():
    """source_url should mirror the on-site `/for-patients/members/<slug>/`
    pattern so the URL is meaningful when clicked, and have a stable
    module_id fragment for tiebreak."""
    rows = parse_directory_json(_load("iaomt_page_1.json"))
    abramczyk = next(r for r in rows if "Abramczyk" in r.name)
    assert "crystal-robyn-abramczyk" in abramczyk.source_url
    # Degrees are appended (BS, DDS, NMD, MS) — order matters for stability.
    assert "bs-dds-nmd-ms" in abramczyk.source_url
    assert "?ppage=dashboard" in abramczyk.source_url
    # Module-id fragment for uniqueness
    assert "#4861911000069880430" in abramczyk.source_url


def test_parser_accepts_raw_records_list():
    """parse_directory_json should accept either the wrapped dict or just
    the inner records list — the migrate CLI calls it both ways."""
    payload = _load("iaomt_page_1.json")
    via_dict = parse_directory_json(payload)
    via_list = parse_directory_json(payload["data"])
    assert len(via_dict) == len(via_list)
    assert [r.source_url for r in via_dict] == [r.source_url for r in via_list]


def test_parser_accepts_json_string():
    """A JSON string of the response is also valid input."""
    raw = (FIXTURE_DIR / "iaomt_page_1.json").read_text()
    rows = parse_directory_json(raw)
    assert len(rows) == 100


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_parse_degrees_handles_json_list():
    assert _parse_degrees('["BS","DDS","NMD","MS"]') == ["BS", "DDS", "NMD", "MS"]


def test_parse_degrees_handles_missing_and_blank():
    assert _parse_degrees(None) == []
    assert _parse_degrees("") == []
    assert _parse_degrees("not-json") == []


def test_format_credentials_dedupes_and_appends_other_degrees():
    assert _format_credentials(["DDS", "NMD"], "MIAOMT") == "DDS, NMD, MIAOMT"
    # Other_Degrees that duplicates a degree is not re-added.
    assert _format_credentials(["DDS"], "DDS, FIAOMT") == "DDS, FIAOMT"
    # No degrees + no extras -> None
    assert _format_credentials([], None) is None


def test_normalize_website_adds_scheme_when_missing():
    assert _normalize_website("smileranchdentistry.com") == "https://smileranchdentistry.com"
    assert _normalize_website("https://x.com/") == "https://x.com/"
    assert _normalize_website("http://x.com") == "http://x.com"
    assert _normalize_website(None) is None
    assert _normalize_website("") is None


def test_country_iso2_canonicalizes_common_names():
    assert _country_iso2("United States") == "US"
    assert _country_iso2("united kingdom") == "GB"
    assert _country_iso2("Canada") == "CA"
    assert _country_iso2("United Arab Emirates") == "AE"
    assert _country_iso2("Atlantis") is None  # unknown -> None (caller decides)
    assert _country_iso2(None) is None


def test_is_fellowship_flag_matrix():
    """Any of Master/Fellow/Accredited/Hygiene-* qualifies; SMART alone
    or General alone does not."""
    assert _is_fellowship({"Master": "1"}) is True
    assert _is_fellowship({"Fellow": "1"}) is True
    assert _is_fellowship({"Accredited": "1"}) is True
    assert _is_fellowship({"Hygiene_Master": "1"}) is True
    assert _is_fellowship({"Hygiene_Fellow": "1"}) is True
    assert _is_fellowship({"Biological_Dental_Hygiene_Accredited": "1"}) is True
    # SMART-only / General-only / empty
    assert _is_fellowship({"Smart": "1"}) is False
    assert _is_fellowship({"General": "1"}) is False
    assert _is_fellowship({}) is False
    # Empty string is the API's "false" representation
    assert _is_fellowship({"Master": ""}) is False


def test_build_source_url_matches_js_slug_rules():
    """The JS slug rule: lowercase, hyphenated name + concatenated lowercase
    degrees without commas/parens. Our builder must match so the URL
    actually resolves at iaomt.org/for-patients/members/."""
    url = _build_source_url("Crystal Robyn Abramczyk", ["BS", "DDS", "NMD", "MS"], "4861911000069880430")
    assert url == (
        "https://iaomt.org/for-patients/members/"
        "crystal-robyn-abramczyk-bs-dds-nmd-ms/?ppage=dashboard"
        "#4861911000069880430"
    )


def test_build_source_url_handles_no_degrees():
    url = _build_source_url("Jane Doe", [], "abc123")
    assert url == "https://iaomt.org/for-patients/members/jane-doe/?ppage=dashboard#abc123"


def test_build_source_url_handles_no_module_id():
    """Missing module_id still produces a slug-only stable URL."""
    url = _build_source_url("Jane Doe", ["DDS"], None)
    assert url == "https://iaomt.org/for-patients/members/jane-doe-dds/?ppage=dashboard"
