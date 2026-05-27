"""Unit tests for the ABFM (American Board of Functional Medicine) adapter.

ABFM's certified-practitioner directory lives at FMU (Functional Medicine
University) — the redirect destination of functionalmedicinedoctors.com.
Its public marker endpoint returns a single JSON document of ~1,005
``locations``. Fixtures here are real responses captured 2026-05-27,
trimmed (template HTML stripped) for fixture leanness:

- abfm_markers_us_sample.json        — 60 US-based records, mix of
                                       CFMP-certified and non-CFMP
                                       training-program graduates;
                                       includes the canonical Kaminsky
                                       (DC + CFMP, with hide-address=1)
                                       and Keith Lewis (CFMP in MGCF
                                       cert field but NOT in LAST_NAME).
- abfm_markers_intl_sample.json      — 50 international records: CA,
                                       UK, MX, FR, LB, ... — covers
                                       free-text country names, the
                                       "STATE=UNITED KINGDOM" dup
                                       pattern, and the FR-record
                                       placeholder STATE='1'.
- abfm_markers_non_cfmp_sample.json  — 30 non-CFMP-name records, all
                                       used to confirm fellowship=False
                                       except where MGCF cert
                                       independently lists CFMP.

These three cover the credential matrix (CFMP / non-CFMP / CFMP-only-
in-MGCF-cert), the country matrix (US / CA / UK / MX / FR), the
hide-address opt-out, and the comma vs no-comma vs mixed-form
LAST_NAME parsing.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.abfm import (  # noqa: E402
    DIRECTORY_PAGE,
    parse_directory_json,
    _build_source_url,
    _country_iso2,
    _has_cfmp_credential,
    _is_placeholder,
    _normalize_state,
    _normalize_website,
    _split_last_name_and_credentials,
)


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_us_sample_returns_full_batch():
    """US fixture has 60 raw records, all with usable names — adapter
    must produce 60 rows."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    assert len(rows) == 60


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are constant per spec."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    rows += parse_directory_json(_load("abfm_markers_intl_sample.json"))
    rows += parse_directory_json(_load("abfm_markers_non_cfmp_sample.json"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "ABFM"
        # Locked-per-spec specialty pair, NOT the IABDM ['biological','dental']
        assert r.specialties == ["functional_medicine", "holistic_health"]
        # source_url is always populated (dedup key)
        assert r.source_url
        assert r.source_url.startswith(DIRECTORY_PAGE)


def test_spot_check_kaminsky_full_fields():
    """First US record — Jarrett Kaminsky, DC + CFMP. Validates the
    canonical CFMP-tier extraction: name split, credentials, contact,
    location, and hide-address suppression of address1."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    k = next(r for r in rows if "Kaminsky" in r.name)

    assert k.name == "Jarrett Kaminsky"
    assert k.practice_name == "Professional Chiropractic Associates"
    assert k.city == "Scranton"
    # STATE='Pa' in raw -> upper-cased
    assert k.state == "PA"
    assert k.postal == "18508"
    assert k.country == "US"
    assert k.phone == "570-348-1158"
    assert k.email == "dr.kaminsky@live.com"
    # CFMP credential present -> fellowship-tier
    assert k.fellowship_level is True
    assert k.credentials
    assert "DC" in k.credentials
    assert "CFMP" in k.credentials
    # hide-address=1 -> address1 must be suppressed
    assert k.address1 is None


def test_us_state_two_letter_codes_are_uppercased():
    """STATE comes in mixed case ('Pa', 'OR', 'TX'). 2-letter ASCII
    codes get normalized to upper. Longer codes keep original casing."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    weintraub = next(r for r in rows if "Weintraub" in r.name)
    # raw STATE 'OR' -> 'OR'
    assert weintraub.state == "OR"
    brittain = next(r for r in rows if "Brittain" in r.name)
    assert brittain.state == "TX"


def test_fellowship_set_for_cfmp_in_last_name():
    """The canonical fellowship case: CFMP appears in LAST_NAME."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    k = next(r for r in rows if "Kaminsky" in r.name)
    assert k.fellowship_level is True


def test_fellowship_set_for_cfmp_in_mgcf_only():
    """Keith Lewis has 'DC, DABAAHP, ...' in LAST_NAME (no CFMP) but
    CFMP IS in MGCF_PROFESSIONAL_DEGREECERTIFICATION_. Adapter must
    still set fellowship_level=True — we check both fields."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    lewis = next(r for r in rows if "Keith Lewis" in r.name)
    assert lewis.fellowship_level is True


def test_fellowship_false_for_non_cfmp_practitioner():
    """Non-CFMP practitioners — those who went through FMU training
    but haven't earned the board designation — must NOT be marked
    fellowship-tier. Skye Weintraub is the canonical case: 'ND' only."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    w = next(r for r in rows if "Weintraub" in r.name)
    assert w.fellowship_level is False


def test_non_cfmp_fixture_is_mostly_non_fellowship():
    """The non-CFMP fixture was filtered on LAST_NAME absence of CFMP,
    so the bulk of rows must be fellowship=False. A small number may
    still be True because their MGCF cert field independently lists
    CFMP — that's expected and tested separately."""
    rows = parse_directory_json(_load("abfm_markers_non_cfmp_sample.json"))
    assert rows
    non_fellow = sum(1 for r in rows if not r.fellowship_level)
    # Overwhelming majority must be non-fellowship (>= 80%).
    assert non_fellow >= int(len(rows) * 0.8)


def test_credentials_deduped_when_short_and_last_name_overlap():
    """The trailing creds in LAST_NAME and the MGCF cert field commonly
    overlap. The adapter must NOT emit the same token twice. Ralph
    Irani is the canonical case: LAST_NAME ends with '...M.Sc, CFMP'
    and MGCF cert is 'PhD, MD.AM, PGDip.HM, M.Sc' — final credentials
    must list each token once."""
    rows = parse_directory_json(_load("abfm_markers_intl_sample.json"))
    irani = next(r for r in rows if "Irani" in r.name)
    assert irani.credentials
    # 'PhD' must appear exactly once.
    assert irani.credentials.count("PhD") == 1
    assert irani.credentials.count("CFMP") == 1


def test_credentials_dedup_long_form_against_short_form():
    """When LAST_NAME already carries 'DC' and MGCF_DEGREES_HELD lists
    'Doctor of Chiropractic', the long form is suppressed — same fact,
    one slot. Kaminsky's credentials must be 'DC, CFMP' (NOT 'DC, CFMP,
    Doctor of Chiropractic')."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    k = next(r for r in rows if "Kaminsky" in r.name)
    assert k.credentials
    # Three-slot triplicate must NOT appear.
    assert "Doctor of Chiropractic" not in k.credentials


def test_last_name_trailing_credential_is_peeled_off_bare_surname():
    """The bare surname must NOT carry the trailing credential token.
    'Weintraub ND' -> surname 'Weintraub' + creds 'ND'. Full name
    output therefore is 'Skye Weintraub', NOT 'Skye Weintraub ND'."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    w = next(r for r in rows if "Weintraub" in r.name)
    assert w.name == "Skye Weintraub"


def test_mixed_form_last_name_peeled_correctly():
    """'Dempster ND, FAARFM' has BOTH a trailing-credential token in
    the pre-comma section AND post-comma credentials. The bare surname
    must drop 'ND' and the credentials tail must include both 'ND'
    AND 'FAARFM'. John Dempster on the intl fixture is the canonical
    case."""
    rows = parse_directory_json(_load("abfm_markers_intl_sample.json"))
    d = next(r for r in rows if "Dempster" in r.name)
    assert d.name == "John Dempster"
    assert d.credentials
    assert "ND" in d.credentials
    assert "FAARFM" in d.credentials


def test_international_record_maps_to_iso2():
    """Canadian record on intl fixture — country -> 'CA', province ON
    preserved, Canadian postal code intact."""
    rows = parse_directory_json(_load("abfm_markers_intl_sample.json"))
    d = next(r for r in rows if "Dempster" in r.name)
    assert d.country == "CA"
    assert d.state == "ON"
    assert d.postal == "M4W 3Y6"
    assert d.city == "Toronto"


def test_uk_record_country_iso_and_dup_state_suppression():
    """UK record where raw STATE='UNITED KINGDOM' (country dup). The
    adapter must map COUNTRY 'United Kingdom' -> 'GB' AND drop the
    country-dup state. Belinda Asonganyi is the canonical case."""
    rows = parse_directory_json(_load("abfm_markers_intl_sample.json"))
    a = next(r for r in rows if "Asonganyi" in r.name)
    assert a.country == "GB"
    assert a.state is None  # 'UNITED KINGDOM' suppressed (= country)
    assert a.postal == "DA10 1AP"
    assert a.city == "Kent"


def test_placeholder_state_value_dropped():
    """When STATE is a placeholder like '1' or 'x' (FMU's 'no data'
    fallback) we drop it. Nathalie Skogland on the intl fixture has
    STATE='1' — adapter must emit state=None."""
    rows = parse_directory_json(_load("abfm_markers_intl_sample.json"))
    s = next(r for r in rows if "Skogland" in r.name)
    assert s.state is None
    assert s.country == "FR"
    assert s.city == "Nice"


def test_hide_address_suppresses_address1_but_keeps_city_state():
    """MGCF_MAPPING_HIDE_ADDRESS=1 means the practitioner asked to hide
    their street address. The adapter must drop address1 but keep
    city/state/postal (these are still publicly shown in the directory
    UI). Validates the privacy promise documented in the docstring."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    k = next(r for r in rows if "Kaminsky" in r.name)
    assert k.address1 is None
    assert k.city == "Scranton"
    assert k.state == "PA"
    assert k.postal == "18508"


def test_practice_name_suppressed_when_matches_practitioner_name():
    """When COMPANY field is literally the practitioner's name (solo
    practice where they typed their own name as the office name), the
    adapter suppresses the duplicate."""
    # Synthetic record because the production fixtures don't have this
    # exact pattern (FMU mostly has real company names).
    rec = {
        "USERID": "test123",
        "FIRST_NAME": "Solo",
        "LAST_NAME": "Practitioner, DC",
        "COMPANY": "Solo Practitioner",  # exact duplicate of name
        "COUNTRY": "United States",
    }
    rows = parse_directory_json([rec])
    assert len(rows) == 1
    assert rows[0].practice_name is None


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the ON CONFLICT dedup keys."""
    payload = _load("abfm_markers_us_sample.json")
    a = parse_directory_json(payload)
    b = parse_directory_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    # All distinct (no two practitioners share a source_url).
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_uses_userid():
    """When USERID is present (the normal case), source_url derives
    from it. Kaminsky has USERID='jdk15'."""
    rows = parse_directory_json(_load("abfm_markers_us_sample.json"))
    k = next(r for r in rows if "Kaminsky" in r.name)
    assert k.source_url.endswith("#user=jdk15")


def test_parser_accepts_locations_dict_or_raw_list():
    """The marker endpoint returns {'locations': [...]} but the parser
    must also accept a bare list (defensive in case the upstream API
    shape changes), and accept a JSON string of either."""
    # Bare list
    rows1 = parse_directory_json([
        {"USERID": "x1", "FIRST_NAME": "A", "LAST_NAME": "B", "COUNTRY": "United States"},
    ])
    assert len(rows1) == 1

    # Raw JSON string
    rows2 = parse_directory_json(
        (FIXTURE_DIR / "abfm_markers_non_cfmp_sample.json").read_text()
    )
    assert len(rows2) == 30


def test_parser_skips_non_dict_records():
    """Defensive: a payload with junk entries mixed in must skip them
    without crashing."""
    rows = parse_directory_json([None, 42, "string", {}])
    assert rows == []


def test_parser_skips_record_without_name():
    """A record with neither FIRST_NAME nor LAST_NAME nor title must
    produce no row (rather than emitting a blank-name entry)."""
    rows = parse_directory_json([{"USERID": "noname", "COUNTRY": "United States"}])
    assert rows == []


def test_full_directory_record_count_matches_locations():
    """All three fixtures' records must round-trip 1:1 — no silent drops
    on the production data shape."""
    us_data = _load("abfm_markers_us_sample.json")
    intl_data = _load("abfm_markers_intl_sample.json")
    nc_data = _load("abfm_markers_non_cfmp_sample.json")
    assert len(parse_directory_json(us_data)) == len(us_data["locations"])
    assert len(parse_directory_json(intl_data)) == len(intl_data["locations"])
    assert len(parse_directory_json(nc_data)) == len(nc_data["locations"])


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_has_cfmp_credential_matches_anywhere_in_name():
    """CFMP detection is case-insensitive and works whether the token
    is trailing, embedded mid-list, or even space-separated."""
    assert _has_cfmp_credential("Kaminsky, DC, CFMP") is True
    assert _has_cfmp_credential("Martin, DC, FASA, BCIM, CGP, CFMP, CCIP") is True
    # Comma-less form with trailing token
    assert _has_cfmp_credential("Walton  MSc. BSc. CFMP") is True
    # Case-insensitive
    assert _has_cfmp_credential("cfmp") is True
    # Negative: nothing remotely resembling CFMP
    assert _has_cfmp_credential("Weintraub ND") is False
    assert _has_cfmp_credential("") is False
    assert _has_cfmp_credential(None) is False
    # Must not false-positive on substrings like 'CFMPRO' — token boundary required
    assert _has_cfmp_credential("Lastname, CFMPRO") is False


def test_has_cfmp_credential_checks_multiple_fields():
    """Pass multiple credential strings — True if ANY contains CFMP."""
    assert _has_cfmp_credential("DC", "MD, CFMP") is True
    assert _has_cfmp_credential("DC", "MD") is False
    assert _has_cfmp_credential(None, "CFMP") is True


def test_is_placeholder_recognizes_fmu_no_data_sentinels():
    """FMU users type 'x' / '1' / 'X' / 'N/A' when they don't fill the
    field. The adapter treats those as missing data."""
    for s in ("x", "X", "1", "0", "n/a", "N/A", "NA", "-", "."):
        assert _is_placeholder(s) is True, f"expected placeholder for {s!r}"
    # Real values are NOT placeholders
    for s in ("DC", "MD", "Doctor of Chiropractic", "CFMP"):
        assert _is_placeholder(s) is False, f"expected NOT placeholder for {s!r}"


def test_split_last_name_handles_comma_form():
    """Standard 'Lastname, CRED1, CRED2' form."""
    name, creds = _split_last_name_and_credentials("Kaminsky, DC, CFMP")
    assert name == "Kaminsky"
    assert creds == "DC, CFMP"


def test_split_last_name_handles_no_comma_form():
    """'Weintraub ND' — trailing whitespace-delimited credential."""
    name, creds = _split_last_name_and_credentials("Weintraub ND")
    assert name == "Weintraub"
    assert creds == "ND"


def test_split_last_name_handles_dotted_credential():
    """'Garcia Monroy D.O.' — dotted credential gets recognized as a
    credential token (compact form is uppercase letters)."""
    name, creds = _split_last_name_and_credentials("Garcia Monroy D.O.")
    assert name == "Garcia Monroy"
    # Trailing period preserved as written, the helper just strips the
    # final dot when needed.
    assert creds is not None
    assert "D.O" in creds


def test_split_last_name_handles_mixed_form():
    """'Dempster ND, FAARFM' — pre-comma still has a trailing credential.
    Both ND and FAARFM must end up in the credentials tail; surname
    is just 'Dempster'."""
    name, creds = _split_last_name_and_credentials("Dempster ND, FAARFM")
    assert name == "Dempster"
    assert creds is not None
    assert "ND" in creds
    assert "FAARFM" in creds


def test_split_last_name_no_credentials():
    """Lone surname with no credentials -> (surname, None)."""
    name, creds = _split_last_name_and_credentials("Mercier")
    assert name == "Mercier"
    assert creds is None


def test_split_last_name_blank_input():
    """Empty / None input -> (None, None)."""
    assert _split_last_name_and_credentials("") == (None, None)
    assert _split_last_name_and_credentials(None) == (None, None)


def test_country_iso2_canonicalizes_common_names():
    assert _country_iso2("United States") == "US"
    assert _country_iso2("USA") == "US"
    assert _country_iso2("United Kingdom") == "GB"
    assert _country_iso2("Canada") == "CA"
    assert _country_iso2("Mexico") == "MX"
    # FMU has a 'Columbia' (sic) typo for Colombia
    assert _country_iso2("Columbia") == "CO"
    assert _country_iso2("Colombia") == "CO"
    # Case-insensitive
    assert _country_iso2("united arab emirates") == "AE"
    # Unrecognized -> None
    assert _country_iso2("Atlantis") is None
    assert _country_iso2(None) is None


def test_normalize_state_uppercases_two_letter_codes():
    """US/CA 2-letter state/province codes upper-case to ISO style."""
    assert _normalize_state("Pa", "US") == "PA"
    assert _normalize_state("OR", "US") == "OR"
    assert _normalize_state("on", "CA") == "ON"


def test_normalize_state_drops_country_duplicate():
    """When STATE accidentally holds a country name that matches the
    record's country, drop it (it's a UI dup, not a real state value)."""
    assert _normalize_state("UNITED KINGDOM", "GB") is None
    assert _normalize_state("Ireland", "IE") is None


def test_normalize_state_drops_placeholders():
    """'1' / 'x' / 'X' state values are FMU placeholders, NOT real data."""
    assert _normalize_state("1", "FR") is None
    assert _normalize_state("x", "US") is None
    assert _normalize_state(None, "US") is None
    assert _normalize_state("", "US") is None


def test_normalize_state_preserves_long_real_state_names():
    """Real long-form state/province names ('British Columbia',
    'Antioquia') must NOT be lower-cased or modified."""
    assert _normalize_state("Antioquia", "CO") == "Antioquia"
    assert _normalize_state("British Columbia", "CA") == "British Columbia"


def test_normalize_website_adds_scheme():
    """Bare domains get an https:// scheme; full URLs pass through."""
    assert _normalize_website("example.com") == "https://example.com"
    assert _normalize_website("www.example.com") == "https://www.example.com"
    assert _normalize_website("https://example.com/") == "https://example.com/"
    assert _normalize_website("http://example.com") == "http://example.com"


def test_normalize_website_rejects_placeholders_and_garbage():
    assert _normalize_website("x") is None
    assert _normalize_website("1") is None
    assert _normalize_website("") is None
    assert _normalize_website(None) is None
    # No dot, no scheme -> not a domain
    assert _normalize_website("just a sentence") is None
    assert _normalize_website("nodot") is None


def test_build_source_url_prefers_userid():
    """USERID is the most stable per-account identifier on the
    MemberGate platform — prefer it over MEMBER_NUMBER when both
    present."""
    url = _build_source_url({"USERID": "jdk15", "MEMBER_NUMBER": 53})
    assert url.endswith("#user=jdk15")
    assert url.startswith(DIRECTORY_PAGE)


def test_build_source_url_falls_back_to_member_number():
    """No USERID -> fall back to MEMBER_NUMBER."""
    url = _build_source_url({"MEMBER_NUMBER": 99})
    assert url.endswith("#member=99")


def test_build_source_url_last_resort_uses_name_and_city():
    """Defensive: when neither USERID nor MEMBER_NUMBER is present,
    synthesize a slug from name + city so dedup still has a key."""
    url = _build_source_url({
        "FIRST_NAME": "Jane",
        "LAST_NAME": "Doe",
        "CITY": "Anywhere",
    })
    assert "unknown-jane-doe-anywhere" in url
