"""Unit tests for the ACAM (American College for Advancement in Medicine)
adapter.

ACAM publishes its members map at ``https://www.acam.org/page/MembersState``
as an iframe pointing at ZeeMaps. The primary acam.org site is fronted
by a Cloudflare managed-challenge that blocks every plain-curl request
(both the WordPress / YourMembership REST endpoints and the static
``custom.asp`` search forms), but the ZeeMaps ``emarkers`` JSON endpoint
is reachable directly outside Cloudflare. Fixtures here are real
responses captured 2026-05-27:

- acam_markers_full.json     -> full emarkers payload (121 marker
                                records: US-heavy mix of MDs / DOs /
                                NDs, plus international entries from
                                Mexico, Canada, UAE, Japan, Slovak
                                Republic, China, Slovenia, Switzerland,
                                Spain, India, Italy, US-Klimenko with
                                inline credentials).
- acam_markers_sample.json   -> first 5 records (fast spot-check
                                fixture for parser smoke tests).
- acam_markers_intl.json     -> the 19 non-US records (exercises the
                                FIPS-vs-ISO country-code disambiguation
                                — JA=Japan, SZ=Switzerland, SP=Spain,
                                CH=China, SI=Slovenia, "Slovak Republic"
                                full-name lookup, etc.).
- acam_zeemaps_page.html     -> the wrapping zeemaps.com/pub page so
                                a future maintainer can re-derive the
                                marker endpoint shape if ZeeMaps
                                changes its URL convention.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.acam import (  # noqa: E402
    parse_directory_json,
    _build_source_url,
    _country_iso2,
    _fips_ch_disambiguate,
    _is_fellowship_name,
    _normalize_postal,
    _normalize_state,
    _split_name_credentials,
)


def _load(name: str) -> list:
    return json.loads((FIXTURE_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_full_payload_returns_all_rows():
    """The full ACAM payload has 121 marker records, all with a usable
    name in either ``nm`` or ``ov`` — the adapter must produce 121 rows."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    assert len(rows) == 121


def test_sample_payload_returns_five_rows():
    """The 5-record sample fixture must parse cleanly to 5 rows."""
    rows = parse_directory_json(_load("acam_markers_sample.json"))
    assert len(rows) == 5


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are constant per spec."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "ACAM"
        assert r.specialties == ["functional_medicine", "holistic_health"]
        # source_url is always populated (dedup key)
        assert r.source_url
        assert r.source_url.startswith("https://www.acam.org/page/MembersState#zm-")


def test_spot_check_kenneth_bock_us_record():
    """Canonical US record: Kenneth Bock, Red Hook NY. Validates name +
    address + state-abbr preservation + ISO country resolution for the
    most common code path (cty='US' bare string)."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    bock = next(r for r in rows if "Kenneth Bock" in r.name)
    assert bock.name == "Kenneth Bock"
    assert bock.address1 == "50 Old Farm Rd"
    assert bock.city == "Red Hook"
    assert bock.state == "NY"
    assert bock.postal == "12571"
    assert bock.country == "US"
    # ZeeMaps marker payload carries no credential/practice/contact info;
    # those stay None until ACAM exposes them outside Cloudflare.
    assert bock.credentials is None
    assert bock.practice_name is None
    assert bock.phone is None
    assert bock.email is None
    assert bock.website is None


def test_spot_check_klimenko_inline_credentials():
    """Elena Klimenko's nm field carries 'Elena Klimenko, MD IFMCP -
    Functional Medicine' — the comma split must isolate the name from
    the credential suffix. Country is the bizarre 'United States' full
    string (not ISO code), so this also exercises the full-name lookup."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    klim = next(r for r in rows if "Klimenko" in r.name)
    assert klim.name == "Elena Klimenko"
    assert klim.credentials == "MD IFMCP - Functional Medicine"
    assert klim.country == "US"
    assert klim.state == "NY"
    assert klim.city == "New York"


def test_spot_check_canadian_postal_preserved():
    """Chris Gordillo (Alberta, CA) — the Canadian postal code 'T7X 4P9'
    with its embedded space must survive normalisation, and state must
    be dropped (ZeeMaps stores '01' as a region-id placeholder for the
    province, which is not a usable state name)."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    gord = next(r for r in rows if "Gordillo" in r.name)
    assert gord.country == "CA"
    assert gord.postal == "T7X 4P9"
    assert gord.city == "Spruce Grove"
    # '01' is a ZeeMaps internal region id, not a province name -> drop.
    assert gord.state is None


def test_fips_japan_code_resolves_to_iso_jp():
    """ZeeMaps writes 'JA' (FIPS Japan) where ISO would write 'JP'. The
    adapter must translate to JP. Tadashi Mitsuo in Shibuya-Ku, Tokyo
    is the canonical record."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    mit = next(r for r in rows if "Tadashi Mitsuo" in r.name)
    assert mit.country == "JP"
    assert mit.city == "Shibuya-Ku"


def test_fips_switzerland_code_resolves_to_iso_ch():
    """ZeeMaps writes 'SZ' (FIPS Switzerland) where ISO would write
    'CH'. Karsten Ostermann in Winterthur is the canonical record."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    ost = next(r for r in rows if "Ostermann" in r.name)
    assert ost.country == "CH"
    assert ost.city == "Winterthur"


def test_fips_ch_disambiguates_to_china_for_shanghai_address():
    """The FIPS code 'CH' means China, but it COLLIDES with ISO 3166's
    CH which means Switzerland. ZeeMaps stamps 'CH' on yueyue Guan's
    record. The Shanghai-anchored address must steer the resolver to
    CN (China), not CH (Switzerland)."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    guan = next(r for r in rows if "Guan" in r.name)
    assert guan.country == "CN"
    assert guan.city == "Shanghai"


def test_full_country_name_string_resolves():
    """Some records carry the full English country name as a string
    instead of any code — 'Slovak Republic' for Judita Durovska, 'Italy'
    for Pietro Mammana. Both must resolve to ISO2."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    dur = next(r for r in rows if "Durovska" in r.name)
    assert dur.country == "SK"
    mam = next(r for r in rows if "Mammana" in r.name)
    assert mam.country == "IT"
    assert mam.state == "Sicilia"


def test_intl_numeric_state_placeholder_dropped():
    """For international records, ZeeMaps sometimes stuffs a numeric
    region id ('02', '15', '23', '40', '03') into the state field
    instead of a real subdivision name. These are not usable — they
    must be dropped, not echoed into the row's state."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    mit = next(r for r in rows if "Tadashi Mitsuo" in r.name)
    # Raw state='40' in the payload, dropped because numeric placeholder.
    assert mit.state is None
    fati = next(r for r in rows if "Fatima Ibragimova" in r.name)
    # Raw state='' but country='AE' — must not be 'AE' echoed as state.
    assert fati.country == "AE"


def test_us_country_distribution_majority():
    """ACAM is US-headquartered; the directory is US-dominant. ~104 of
    121 records must resolve to country='US' after FIPS/ISO/full-name
    normalisation."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    us = [r for r in rows if r.country == "US"]
    # Allow a wider band so a future minor data shift doesn't flake the
    # test; the structural assertion is "US is the majority."
    assert len(us) >= 90
    assert len(us) > len(rows) / 2


def test_fellowship_default_false_when_no_credentials():
    """Per the module docstring fellowship rule: ZeeMaps does not expose
    a Diplomate flag, so fellowship_level defaults to False unless the
    nm field carries an explicit DABCMT / ABCMT / Diplomate marker.

    In the live data, ZERO records meet that bar — so the entire
    fellowship_level column must be False after parsing. (This pins
    the conservative default so a future relaxation of the rule is a
    deliberate change, not a silent regression.)"""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    assert rows
    assert sum(1 for r in rows if r.fellowship_level) == 0


def test_fellowship_set_when_diplomate_credential_present():
    """When a practitioner's nm string DOES include a DABCMT / ABCMT /
    Diplomate marker, fellowship_level flips to True. Synthetic record
    because the production data doesn't carry any such markers right
    now (would still parse correctly when ACAM updates display)."""
    rec = {
        "id": 999000001,
        "gid": 3473180,
        "nm": "Jane Doe, MD, DABCMT",
        "s": "1 Main St",
        "city": "Boston",
        "state": "MA",
        "zip": "02108",
        "cty": "US",
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].name == "Jane Doe"
    assert rows[0].credentials == "MD, DABCMT"
    assert rows[0].fellowship_level is True


def test_fellowship_set_for_literal_diplomate_word():
    """Explicit 'Diplomate' word in the credential suffix also qualifies
    (case-insensitive)."""
    rec = {
        "id": 999000002,
        "gid": 3473180,
        "nm": "John Roe, MD, Diplomate ABCMT",
        "s": "2 Main St",
        "city": "Boston",
        "state": "MA",
        "zip": "02108",
        "cty": "US",
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].fellowship_level is True


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the dedup keys for ON CONFLICT upsert."""
    payload = _load("acam_markers_full.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    # All distinct (no two practitioners share a source_url).
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_format():
    """Anchored at ACAM MembersState page with #zm-<marker_id> fragment.
    Kenneth Bock's marker id is 494151006."""
    rows = parse_directory_json(_load("acam_markers_full.json"))
    bock = next(r for r in rows if "Kenneth Bock" in r.name)
    assert bock.source_url == "https://www.acam.org/page/MembersState#zm-494151006"


def test_parser_accepts_json_string():
    """A raw JSON string of the response list is also valid input."""
    raw = (FIXTURE_DIR / "acam_markers_sample.json").read_text()
    rows = parse_directory_json(raw)
    assert len(rows) == 5


def test_parser_accepts_dict_payload():
    """Defensive: a {id: marker} dict shape (the CSO convention) is
    accepted too. ZeeMaps' emarkers returns a list, but if a future
    version changes that, the parser must keep working."""
    sample = _load("acam_markers_sample.json")
    as_dict = {str(r["id"]): r for r in sample}
    rows = parse_directory_json(as_dict)
    assert len(rows) == 5


def test_parser_skips_non_dict_records():
    """Defensive: a payload with junk entries mixed in must skip them
    without crashing."""
    rows = parse_directory_json([None, 42, "string", {}])
    assert rows == []


def test_intl_fixture_excludes_us_records():
    """The intl-only fixture must be entirely non-US after parsing,
    and every row must have a resolved (non-None) country."""
    rows = parse_directory_json(_load("acam_markers_intl.json"))
    assert rows
    assert all(r.country != "US" for r in rows)
    assert all(r.country for r in rows)


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_country_iso2_handles_iso_codes():
    """Bare ISO 3166 codes pass through unchanged."""
    assert _country_iso2("US") == "US"
    assert _country_iso2("us") == "US"
    assert _country_iso2("CA") == "CA"
    assert _country_iso2("MX") == "MX"
    assert _country_iso2("AE") == "AE"


def test_country_iso2_translates_fips_codes():
    """FIPS-only 2-letter codes map to ISO 3166."""
    assert _country_iso2("JA") == "JP"   # FIPS Japan
    assert _country_iso2("SZ") == "CH"   # FIPS Switzerland
    assert _country_iso2("SP") == "ES"   # FIPS Spain
    assert _country_iso2("TU") == "TR"   # FIPS Turkey
    assert _country_iso2("KS") == "KR"   # FIPS South Korea


def test_country_iso2_translates_full_names():
    """Full English country names (case-insensitive)."""
    assert _country_iso2("United States") == "US"
    assert _country_iso2("united states") == "US"
    assert _country_iso2("Slovak Republic") == "SK"
    assert _country_iso2("Italy") == "IT"
    assert _country_iso2("United Kingdom") == "GB"
    assert _country_iso2("Canada") == "CA"


def test_country_iso2_unknown_returns_none():
    """Unrecognised input -> None (caller defaults to 'US')."""
    assert _country_iso2("Atlantis") is None
    assert _country_iso2(None) is None
    assert _country_iso2("") is None
    assert _country_iso2("   ") is None


def test_country_iso2_ch_disambiguates_via_address():
    """The 'CH' collision: FIPS = China, ISO = Switzerland. The resolver
    must pick based on the address hint passed alongside the code."""
    # No address -> default to Switzerland (ISO convention).
    assert _country_iso2("CH") == "CH"
    # China-anchored address -> CN.
    assert _country_iso2("CH", "No 1 Main St, Shanghai 200000") == "CN"
    assert _country_iso2("CH", "Beijing CN") == "CN"
    # Switzerland-anchored address -> CH.
    assert _country_iso2("CH", "Bahnhofstrasse 1, Zurich") == "CH"
    assert _country_iso2("CH", "Geneva, Switzerland") == "CH"


def test_fips_ch_disambiguate_pure():
    """The disambiguator itself: explicit China hints win; otherwise
    default to Switzerland."""
    assert _fips_ch_disambiguate("Shanghai") == "CN"
    assert _fips_ch_disambiguate("china") == "CN"
    assert _fips_ch_disambiguate("Beijing 100000") == "CN"
    assert _fips_ch_disambiguate("Zurich") == "CH"
    assert _fips_ch_disambiguate("Switzerland") == "CH"
    assert _fips_ch_disambiguate("") == "CH"
    assert _fips_ch_disambiguate(None) == "CH"


def test_split_name_credentials_basic():
    """Comma is the structural separator. Everything before -> name,
    everything after -> credentials."""
    name, creds = _split_name_credentials("Jane Doe, MD")
    assert name == "Jane Doe"
    assert creds == "MD"

    name, creds = _split_name_credentials("Elena Klimenko, MD IFMCP - Functional Medicine")
    assert name == "Elena Klimenko"
    assert creds == "MD IFMCP - Functional Medicine"


def test_split_name_credentials_no_comma():
    """No comma -> the entire value is the name; no credentials."""
    name, creds = _split_name_credentials("Kenneth Bock")
    assert name == "Kenneth Bock"
    assert creds is None


def test_split_name_credentials_handles_empty():
    """Empty / None input."""
    assert _split_name_credentials(None) == ("", None)
    assert _split_name_credentials("") == ("", None)
    assert _split_name_credentials("   ") == ("", None)


def test_split_name_credentials_dash_not_separator():
    """A bare dash is NOT a separator — only the comma is structural.
    'Jose Diaz-Barboza' must stay whole; the dash inside the name
    must not be misinterpreted as a credential boundary."""
    name, creds = _split_name_credentials("Jose Diaz-Barboza")
    assert name == "Jose Diaz-Barboza"
    assert creds is None


def test_is_fellowship_name_matrix():
    """Fellowship triggers: DABCMT, ABCMT, Diplomate, Board Certified
    (case-insensitive, word-bounded). Anything else is False."""
    # Trigger cases.
    assert _is_fellowship_name("Jane Doe, MD, DABCMT") is True
    assert _is_fellowship_name("Jane Doe, dabcmt") is True
    assert _is_fellowship_name("Jane Doe, ABCMT Member") is True
    assert _is_fellowship_name("Jane Doe, MD, Diplomate ABCMT") is True
    assert _is_fellowship_name("Dr Roe, Board Certified Functional Med") is True
    # Non-trigger cases.
    assert _is_fellowship_name("Kenneth Bock") is False
    assert _is_fellowship_name("Elena Klimenko, MD IFMCP") is False
    assert _is_fellowship_name("") is False
    assert _is_fellowship_name(None) is False
    # Word boundary: 'ABCMTX' must NOT trigger ABCMT match.
    assert _is_fellowship_name("Jane Doe, ABCMTX") is False


def test_normalize_state_us_passthrough():
    """For US records, state is passed through verbatim (no
    abbreviation-vs-name canonicalisation in this adapter — the geocoder
    handles full names downstream)."""
    assert _normalize_state("NY", "US") == "NY"
    assert _normalize_state("Texas", "US") == "Texas"
    assert _normalize_state("", "US") is None
    assert _normalize_state(None, "US") is None


def test_normalize_state_intl_drops_numeric_placeholder():
    """International records: ZeeMaps stuffs a numeric region id
    placeholder ('02', '15', '40') into state. Must drop."""
    assert _normalize_state("02", "MX") is None
    assert _normalize_state("15", "MX") is None
    assert _normalize_state("40", "JP") is None
    assert _normalize_state("01", "CA") is None
    # But real names survive.
    assert _normalize_state("Sicilia", "IT") == "Sicilia"
    assert _normalize_state("British Columbia", "CA") == "British Columbia"


def test_normalize_postal_basic():
    """Postal: pass through; drop obvious placeholders."""
    assert _normalize_postal("12345") == "12345"
    assert _normalize_postal("T7X 4P9") == "T7X 4P9"
    assert _normalize_postal("82104") == "82104"
    assert _normalize_postal("") is None
    assert _normalize_postal(None) is None
    assert _normalize_postal("0") is None
    assert _normalize_postal("None") is None


def test_build_source_url():
    """Format: https://www.acam.org/page/MembersState#zm-<id>"""
    url = _build_source_url({"id": 494151006})
    assert url == "https://www.acam.org/page/MembersState#zm-494151006"
    # String id also works.
    url = _build_source_url({"id": "494151006"})
    assert url == "https://www.acam.org/page/MembersState#zm-494151006"
    # Missing id falls back to 'unknown' (still produces a stable URL).
    url = _build_source_url({})
    assert url == "https://www.acam.org/page/MembersState#zm-unknown"
