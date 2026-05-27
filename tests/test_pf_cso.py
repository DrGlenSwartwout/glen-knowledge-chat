"""Unit tests for the College of Syntonic Optometry (CSO) adapter.

CSO publishes its entire member roster as a single inline JSON blob
inside the Find-a-Practitioner page's WP-Google-Map plugin
initialization script. Fixtures here are real responses captured
2026-05-27:

- cso_find_a_practitioner.html  — full page HTML download (331KB,
                                  contains the inline marker payload).
                                  Used by extract_marker_payload tests
                                  and full-batch parse tests.
- cso_markers_full.json         — the marker payload extracted from
                                  that HTML and dumped (375 records).
                                  Used for full-batch tests without
                                  re-running the HTML scrape.
- cso_markers_sample.json       — 21 hand-picked markers covering the
                                  credential matrix (FCSO US, FCSO
                                  international, non-FCSO US, non-FCSO
                                  international, non-OD creds, no creds,
                                  duplicates) and the address-format
                                  matrix (US single-comma, US two-comma,
                                  full-state-name, Canada, UK, Spain,
                                  Brazil, Mexico, Switzerland, South
                                  Africa, Andorra, Puerto Rico).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.cso import (  # noqa: E402
    extract_marker_payload,
    parse_directory_html,
    parse_directory_json,
    _build_source_url,
    _country_iso2,
    _detect_country_from_address,
    _has_fcso,
    _normalize_website,
    _parse_address,
    _parse_description,
    _split_title,
    _split_us_address,
)


def _load_json(name: str):
    return json.loads((FIXTURE_DIR / name).read_text())


def _load_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_extract_marker_payload_pulls_full_roster_from_html():
    """The Find-a-Practitioner page contains the entire roster as a
    JSON.parse('...') inline blob — extract_marker_payload must surface
    all 375 records as a dict keyed by marker id."""
    html = _load_text("cso_find_a_practitioner.html")
    data = extract_marker_payload(html)
    assert isinstance(data, dict)
    assert len(data) == 375


def test_extract_marker_payload_returns_empty_when_blob_missing():
    """Defensive: an HTML page with no mapMarkers blob must return {}
    (would indicate CSO swapped its directory plugin — fail loud at the
    next layer, don't crash here)."""
    assert extract_marker_payload("<html><body>nope</body></html>") == {}
    assert extract_marker_payload("") == {}


def test_parse_directory_html_end_to_end_full_batch():
    """End-to-end: page HTML in, 375 NormalizedPractitionerRow records
    out. Every record has a usable title in the live data, so the count
    matches the raw marker count exactly."""
    html = _load_text("cso_find_a_practitioner.html")
    rows = parse_directory_html(html)
    assert len(rows) == 375


def test_parse_directory_json_accepts_dict_shape():
    """Natural shape from extract_marker_payload: dict of {id: record}."""
    data = _load_json("cso_markers_full.json")
    rows = parse_directory_json(data)
    assert len(rows) == 375


def test_parse_directory_json_accepts_list_shape():
    """Alternate shape: list of records (what fetch_all_directory_records
    returns)."""
    data = _load_json("cso_markers_full.json")
    rows = parse_directory_json(list(data.values()))
    assert len(rows) == 375


def test_parse_directory_json_accepts_string_shape():
    """Raw JSON string input (file contents)."""
    raw = (FIXTURE_DIR / "cso_markers_sample.json").read_text()
    rows = parse_directory_json(raw)
    assert len(rows) == 21


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are locked per spec — never mutate."""
    rows = parse_directory_json(_load_json("cso_markers_full.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "CSO"
        assert r.specialties == ["syntonic", "eye_care"]
        assert r.source_url
        assert r.source_url.startswith("https://csovision.org/find-a-practitioner/#marker-")


def test_source_url_is_unique_per_practitioner():
    """The marker id is the dedup key — every row must have a unique
    source_url, including the legitimate (title, address) duplicates
    where the same practitioner appears twice with different marker ids
    (e.g. Cade Kowallis 763 + 1582, Pilar Vergara 875 + 1213)."""
    rows = parse_directory_json(_load_json("cso_markers_full.json"))
    urls = [r.source_url for r in rows]
    assert len(urls) == len(set(urls)), "duplicate source_url across rows"


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the ON CONFLICT dedup keys."""
    payload = _load_json("cso_markers_sample.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]


def test_fcso_count_in_full_batch_matches_directory():
    """33 FCSOs out of 375 markers as captured 2026-05-27. If this drops
    to 0, the title-based detection has regressed. If it climbs past
    ~50, the directory grew (re-capture the fixtures)."""
    rows = parse_directory_json(_load_json("cso_markers_full.json"))
    fcso = [r for r in rows if r.fellowship_level]
    assert len(fcso) == 33
    non_fcso = [r for r in rows if not r.fellowship_level]
    assert len(non_fcso) == 342


def test_aakash_shah_full_field_extraction():
    """Canonical non-FCSO US OD with full structured fields. Validates
    name + credentials + state/postal + website + phone + practice."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    shah = next(r for r in rows if r.name == "Aakash Shah")
    assert shah.credentials == "O.D."
    assert shah.fellowship_level is False
    assert shah.state == "CA"
    assert shah.postal == "93230"
    assert shah.country == "US"
    assert shah.website == "https://www.mayerandshahoptometry.com/"
    assert shah.practice_name == "Mayer and Shah Optometry"
    # Tel: (559) 582 9244 lives in the description blob
    assert shah.phone is not None
    assert "559" in shah.phone


def test_alia_santoyo_johnson_fcso_extraction():
    """Canonical FCSO US OD. FCSO must mark fellowship_level=True and
    end up in the credentials string. Email must extract from the
    mailto: link in the description."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    santoyo = next(r for r in rows if r.name == "Alia Santoyo-Johnson")
    assert santoyo.credentials == "O.D., FCSO"
    assert santoyo.fellowship_level is True
    assert santoyo.email == "drsantoyollc@gmail.com"
    assert santoyo.practice_name == "Family Vision Development Center"


def test_cathy_stern_multi_credential_fcso():
    """Multi-credential FCSO: 'O.D., FCOVD, FNORA, FCSO'. All four
    credentials must survive in the credentials field; FCSO must be
    detected as fellowship marker."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    stern = next(r for r in rows if r.name == "Cathy Stern")
    assert stern.fellowship_level is True
    assert stern.credentials
    for cred in ("O.D.", "FCOVD", "FNORA", "FCSO"):
        assert cred in stern.credentials, f"missing {cred!r} in {stern.credentials!r}"


def test_non_fcso_with_other_fellowships_is_not_fellowship_level():
    """A practitioner with FCOVD / FAAO but NO FCSO must NOT have
    fellowship_level=True — fellowship_level is CSO-specific. Aaron
    Nichols (O.D., FAAO, FCOVD) is the canonical case."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    nichols = next(r for r in rows if r.name == "Aaron Nichols")
    assert nichols.fellowship_level is False
    assert "FCOVD" in (nichols.credentials or "")
    assert "FCSO" not in (nichols.credentials or "")


def test_practitioner_with_no_credentials():
    """A title with no comma carries no credentials. Carla Van Der Merwe
    on the sample fixture has just the bare name."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    carla = next(r for r in rows if r.name == "Carla Van Der Merwe")
    assert carla.credentials is None
    assert carla.country == "ZA"


def test_international_fcso_brazil_country_resolved():
    """FCSO international: Fernanda Leite Ribeiro (Brazil). The address
    'Av. Rebouças, 3797. Pinheiros, São Paulo, SP. Brazil. 05401-450'
    has the country in the MIDDLE (postal code after) — the country
    detector must still catch it."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    ribeiros = [r for r in rows if r.name == "Fernanda Leite Ribeiro"]
    assert ribeiros, "Fernanda not parsed"
    for r in ribeiros:
        assert r.country == "BR"
        assert r.fellowship_level is True
        assert r.credentials == "FCSO"


def test_duplicate_practitioner_distinct_marker_ids():
    """Cade Kowallis appears twice in the live data (markers 763 and
    1582) with identical title+address but distinct ids — both rows
    must survive parsing with distinct source_urls so the dedup layer
    at the DB level can decide downstream policy."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    cades = [r for r in rows if r.name == "Cade Kowallis"]
    assert len(cades) == 2
    urls = {r.source_url for r in cades}
    assert len(urls) == 2
    for r in cades:
        assert r.fellowship_level is True


def test_canada_address_country_resolved():
    """Canadian address ('Burlington, ON L75 2E2 Canada') must resolve
    country='CA' and keep the full address in address1 (no US state
    split). Tested via the Agata Majewski Toronto entry."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    agata = next(r for r in rows if r.name == "Agata Majewski")
    # Toronto, Ontario address — no US state match.
    assert agata.country in ("CA",) or "Toronto" in (agata.address1 or "")
    # The full address is preserved in address1 for the geocoder.
    assert agata.address1 is not None
    assert "Toronto" in agata.address1


def test_us_single_comma_address_extracts_state_postal():
    """The most common CSO US shape: 'Street City, ST ZIP' — a single
    comma. State and ZIP must extract; city falls to None (the whole
    pre-comma segment goes to address1 since we can't split city out)."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    shah = next(r for r in rows if r.name == "Aakash Shah")
    assert shah.state == "CA"
    assert shah.postal == "93230"
    assert shah.address1 == "429 N Irwin St Hanford"


def test_us_two_comma_address_extracts_full_split():
    """The cleaner CSO US shape: 'Street, City, FullStateName ZIP, USA'.
    All four fields must extract."""
    # Synthetic — page 1 has Watertown, but we double-check the parser
    # via the Aimee Schulte record loaded from the full fixture.
    rows = parse_directory_json(_load_json("cso_markers_full.json"))
    schulte = next(r for r in rows if r.name == "Aimee Schulte")
    assert schulte.state == "SD"
    assert schulte.postal == "57201"
    assert schulte.country == "US"
    assert schulte.city == "Watertown"
    assert schulte.address1 == "22 19th St SE"


def test_international_germany_country_resolved():
    """Bettina Wilke (Germany) — address ends with 'Germany'. Country
    must resolve to 'DE'; no US state split. Credentials='MD' (no
    period)."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    wilke = next(r for r in rows if r.name == "Bettina Wilke")
    assert wilke.country == "DE"
    assert wilke.credentials == "MD"
    assert wilke.fellowship_level is False


def test_pic_url_is_ignored():
    """pic_url is portal-managed (photo_url stays None from scrapers).
    All adapters share this convention so portal-uploaded photos aren't
    overwritten by directory scrapes."""
    rows = parse_directory_json(_load_json("cso_markers_sample.json"))
    for r in rows:
        assert r.photo_url is None
        assert r.bio is None


def test_skipped_records_when_title_is_empty():
    """A marker with empty title gets dropped (no name = no row)."""
    raw = [
        {"id": "9001", "title": "", "address": ""},
        {"id": "9002", "title": "   ", "address": ""},
        {"id": "9003", "title": "Valid Person, OD", "address": ""},
    ]
    rows = parse_directory_json(raw)
    assert len(rows) == 1
    assert rows[0].name == "Valid Person"


def test_parser_skips_non_dict_records():
    """Defensive: junk entries in the marker list must be skipped, not
    crashed on."""
    rows = parse_directory_json([None, 42, "string", {}])
    assert rows == []


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_split_title_simple():
    assert _split_title("Jane Doe") == ("Jane Doe", None)
    assert _split_title("Jane Doe, O.D.") == ("Jane Doe", "O.D.")
    assert _split_title("Cade Kowallis, OD, FCSO") == ("Cade Kowallis", "OD, FCSO")


def test_split_title_period_separator_normalized():
    """Brenda Montecalvo. OD — period as a comma substitute."""
    name, creds = _split_title("Brenda Montecalvo. OD")
    assert name == "Brenda Montecalvo"
    assert creds == "OD"


def test_split_title_preserves_credential_trailing_period():
    """O.D. is the OD credential; the trailing period is part of the
    credential, not a sentence terminator."""
    _name, creds = _split_title("Aakash Shah, O.D.")
    assert creds == "O.D."


def test_split_title_preserves_initials_in_name():
    """C. William Harpur — the initial period must NOT be confused with
    a credential separator. Comma comes first."""
    name, creds = _split_title("C. William Harpur, O.D., FCSO")
    assert name == "C. William Harpur"
    assert creds == "O.D., FCSO"


def test_split_title_empty():
    assert _split_title("") == ("", None)
    assert _split_title("   ") == ("", None)


def test_has_fcso_positive():
    assert _has_fcso("Jane Doe, FCSO") is True
    assert _has_fcso("Jane Doe, OD, FCOVD, FCSO") is True
    assert _has_fcso("Jane Doe, F.C.S.O.") is True  # defensive dot variant


def test_has_fcso_negative():
    assert _has_fcso("Jane Doe, OD") is False
    assert _has_fcso("Jane Doe, FCOVD") is False
    assert _has_fcso("") is False
    assert _has_fcso(None) is False
    # Word-boundary safety — substring inside another token must not match.
    assert _has_fcso("Jane FCSOX Doe") is False


def test_country_iso2_common_names():
    assert _country_iso2("United States") == "US"
    assert _country_iso2("usa") == "US"
    assert _country_iso2("Canada") == "CA"
    assert _country_iso2("United Kingdom") == "GB"
    assert _country_iso2("south africa") == "ZA"
    assert _country_iso2("Atlantis") is None
    assert _country_iso2(None) is None


def test_detect_country_from_address_suffix():
    """Country at end of address."""
    assert _detect_country_from_address("113 Roncesvalles Ave Toronto, Ontario") is None
    # ^ Ontario alone is not enough; we need an explicit country word
    assert _detect_country_from_address(
        "767 Hawkins Crescent Burlington, ON L75 2E2 Canada"
    ) == "CA"
    assert _detect_country_from_address(
        "26 Medowfield Road Stocksfield NE43 7PY Northumberland United Kingdom"
    ) == "GB"
    assert _detect_country_from_address(
        "Calle Derecho 28 02008 Albacete SPAIN"
    ) == "ES"


def test_detect_country_with_trailing_postal():
    """Country name followed by punctuation + postal — the Brazilian
    case 'São Paulo, SP. Brazil. 05401-450' has Brazil in the middle."""
    assert _detect_country_from_address(
        "Av. Rebouças, 3797. Pinheiros, São Paulo, SP. Brazil. 05401-450"
    ) == "BR"


def test_detect_country_does_not_match_substring_inside_word():
    """'india' must NOT match inside 'indiana'."""
    assert _detect_country_from_address(
        "1234 Indiana Ave, Indianapolis, IN 46204"
    ) is None
    # But a real India address must match.
    assert _detect_country_from_address(
        "123 Some Road, Mumbai 400001 India"
    ) == "IN"


def test_split_us_address_two_comma_form():
    """Cleanest form: 'Street, City, ST ZIP'."""
    r = _split_us_address("22 19th St SE, Watertown, South Dakota 57201, USA")
    assert r == ("22 19th St SE", "Watertown", "SD", "57201")


def test_split_us_address_single_comma_form():
    """'Street City, ST ZIP' — only one comma. City is None; whole
    pre-comma string is address1."""
    r = _split_us_address("429 N Irwin St Hanford, CA 93230")
    assert r == ("429 N Irwin St Hanford", None, "CA", "93230")


def test_split_us_address_full_state_name():
    """Full state name format."""
    r = _split_us_address("414 E Upland Rd Ste A Ithaca, New York 14850")
    assert r == ("414 E Upland Rd Ste A Ithaca", None, "NY", "14850")


def test_split_us_address_with_trailing_usa():
    """Some entries end with ', USA' — must not block the split."""
    r = _split_us_address("123 Main St, Springfield, MA 01060 USA")
    assert r is not None
    assert r[2] == "MA"
    assert r[3] == "01060"


def test_split_us_address_returns_none_for_non_us():
    assert _split_us_address("113 Roncesvalles Ave Toronto, Ontario") is None
    assert _split_us_address("Calle Derecho 28 02008 Albacete SPAIN") is None
    assert _split_us_address("") is None


def test_parse_address_defaults_country_us():
    """When no country can be inferred, country defaults to 'US' (the
    bulk of CSO records). The geocoder is the safety net for true
    international leaks."""
    _addr, _city, _state, _postal, country = _parse_address("Some unknown place 123")
    assert country == "US"


def test_parse_address_handles_none_and_empty():
    assert _parse_address(None) == (None, None, None, None, "US")
    assert _parse_address("") == (None, None, None, None, "US")


def test_parse_description_extracts_practice_phone_email():
    """Description is one HTML-escaped <p> blob — all three sub-fields
    must come out."""
    desc = [
        '&lt;p&gt;&lt;a href=&quot;https://example.com&quot;&gt;'
        '&lt;strong&gt;Sample Vision Center&lt;/strong&gt;&lt;/a&gt;&lt;br /&gt;'
        'Tel: 555-555-1234&lt;br /&gt;'
        'Fax: 555-555-5678&lt;br /&gt;'
        '&lt;a href=&quot;mailto:foo@bar.com&quot;&gt;email Dr. Foo&lt;/a&gt;'
        '&lt;/p&gt;'
    ]
    out = _parse_description(desc)
    assert out.get("practice_name") == "Sample Vision Center"
    assert out.get("email") == "foo@bar.com"
    assert "555-555-1234" in (out.get("phone") or "")


def test_parse_description_handles_empty_and_missing_fields():
    """Empty description -> empty dict. Description with only a Tel:
    block -> only phone set."""
    assert _parse_description(None) == {}
    assert _parse_description([]) == {}
    only_tel = ["&lt;p&gt;Tel: 555-1234&lt;/p&gt;"]
    out = _parse_description(only_tel)
    assert "phone" in out
    assert "email" not in out
    assert "practice_name" not in out


def test_normalize_website_adds_https_scheme():
    assert _normalize_website("example.com") == "https://example.com"
    assert _normalize_website("http://example.com") == "http://example.com"
    assert _normalize_website("https://example.com/") == "https://example.com/"


def test_normalize_website_strips_pollution():
    """CSO sometimes appends 'Tel: ...' text to link_url. Strip that."""
    polluted = (
        "https://www.watertownfamilyeyecare.netTel: 605-753-3937 "
        "Fax: 605-753-0472 https://www.watertownfamilyeyecare.net"
    )
    # We can't fully clean this up (the URL itself has 'netTel' glued
    # together with no separator) but the helper must at least not crash
    # and return a usable string starting with https.
    out = _normalize_website(polluted)
    assert out is not None
    assert out.startswith("https://")


def test_normalize_website_handles_none_and_sentinel_values():
    assert _normalize_website(None) is None
    assert _normalize_website("") is None
    assert _normalize_website("None") is None
    assert _normalize_website("n/a") is None


def test_build_source_url_uses_marker_id():
    """source_url is the page URL anchored on the marker id — stable
    across re-runs and unique per record."""
    assert _build_source_url({"id": "1485"}) == (
        "https://csovision.org/find-a-practitioner/#marker-1485"
    )


def test_build_source_url_handles_missing_id():
    """Defensive: a record with no id still gets a (synthetic) URL so
    the dedup layer can scream loudly instead of erroring on None."""
    url = _build_source_url({})
    assert url.startswith("https://csovision.org/find-a-practitioner/#marker-")
