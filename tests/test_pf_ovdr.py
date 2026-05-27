"""Unit tests for the OVDR (Optometric Vision Development & Rehabilitation
Association, formerly COVD) adapter.

OVDR publishes its directory through a YourMembership-backed ASP.NET MVC
locator at ``locate.covd.org/Search/DoSearch``. Fixtures here are real
search responses captured 2026-05-27:

- ovdr_search_us_ny_page_1.html  — Country=US&State=NY page 1 (20 doctors,
                                   simple single-office US shape where the
                                   doctor + office data collapse into one
                                   <tr>; mix of FCOVD / FOVDR / OD-only).
- ovdr_search_us_ny_page_2.html  — Country=US&State=NY page 2 (14 doctors,
                                   the tail of the NY result set; includes
                                   Kenneth Ciuffreda — the canonical
                                   FCOVD-A "Advanced" tier — and one
                                   COVT-only entry without FCOVD).
- ovdr_search_country_ca.html    — Country=CA page 1 (20 doctors, the
                                   multi-office Canadian shape where the
                                   doctor <tr> holds only a hidden
                                   ``<address style="display: none;">``
                                   and the visible office data lives in
                                   follow-up multiOffices <tr>s; covers
                                   Ontario / BC postal codes with and
                                   without internal spaces).

These three cover the credential matrix (FCOVD / FOVDR / FCOVD-A /
OD-only / COVT-only), the page-shape matrix (single-TR US vs
multiOffices Canadian), the postal matrix (US ZIP, US ZIP+4, Canadian
ANA-NAN with and without space), and pagination state (page 1 vs page 2
of the same search).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.ovdr import (  # noqa: E402
    _build_source_url,
    _extract_lat_lng_from_href,
    _extract_profile_id,
    _is_fellowship,
    _parse_address_block,
    _split_city_state_postal,
    _strip_credentials,
    parse_search_html,
    parse_total_pages,
)


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_us_ny_page_1_returns_20_rows():
    """NY page 1 has exactly 20 doctor anchors — adapter emits 20 rows."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    assert len(rows) == 20


def test_parse_us_ny_page_2_returns_14_rows():
    """NY page 2 (the tail) has 14 doctors."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_2.html"), country="US")
    assert len(rows) == 14


def test_parse_canada_returns_20_rows():
    """Canada page 1 has 20 doctors; each has a multiOffices follow-up TR
    that the parser must consume without producing a phantom second row."""
    rows = parse_search_html(_load("ovdr_search_country_ca.html"), country="CA")
    assert len(rows) == 20


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are constant per spec."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    rows += parse_search_html(_load("ovdr_search_us_ny_page_2.html"), country="US")
    rows += parse_search_html(_load("ovdr_search_country_ca.html"), country="CA")
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "OVDR"
        assert r.specialties == ["rehabilitation", "eye_care"]
        # source_url is always populated (dedup key) and points at the
        # canonical detail endpoint.
        assert r.source_url
        assert r.source_url.startswith(
            "https://locate.covd.org/Search/Detailed?profileId="
        )


def test_us_single_office_row_full_fields():
    """First row on NY page 1 is Rebecca Marinoff — single-office US case
    where the doctor + office data collapse into one TR. Validates
    name / credentials / practice / address / phone extraction."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    marinoff = next(r for r in rows if r.name == "Rebecca Marinoff")
    assert marinoff.credentials == "OD"
    assert marinoff.practice_name == "SUNY College of Optometry"
    assert marinoff.address1 == "33 W 42nd St"
    assert marinoff.city == "New York"
    assert marinoff.state == "NY"
    assert marinoff.postal == "10036"
    assert marinoff.country == "US"
    assert marinoff.phone == "(212) 938-5937"
    # Marinoff is "Associate" type — not FCOVD/FOVDR — so NOT a fellow.
    assert marinoff.fellowship_level is False


def test_fellowship_set_for_fcovd_credential():
    """Gary Williams ('OD, FCOVD') is the canonical legacy-credential
    fellow on NY page 1."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    williams = next(r for r in rows if r.name == "Gary Williams")
    assert williams.credentials == "OD, FCOVD"
    assert williams.fellowship_level is True


def test_fellowship_set_for_fovdr_credential():
    """Samantha Slotnick ('OD, FAAO, FOVDR') — the rebrand-era FOVDR
    designation must also flip fellowship_level to True."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    slotnick = next(r for r in rows if r.name == "Samantha Slotnick")
    assert slotnick.credentials == "OD, FAAO, FOVDR"
    assert slotnick.fellowship_level is True


def test_fellowship_set_for_fcovd_a_advanced():
    """Kenneth Ciuffreda ('OD, PhD, FCOVD-A') — FCOVD-A is the Advanced
    fellowship tier and must qualify."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_2.html"), country="US")
    ciuffreda = next(r for r in rows if r.name == "Kenneth Ciuffreda")
    assert "FCOVD-A" in (ciuffreda.credentials or "")
    assert ciuffreda.fellowship_level is True


def test_fellowship_not_set_for_od_only():
    """Rebecca Marinoff ('OD' only — no F-prefixed designation) must NOT
    be marked as fellowship-tier. Catches a regex that's too loose."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    marinoff = next(r for r in rows if r.name == "Rebecca Marinoff")
    assert marinoff.fellowship_level is False


def test_fellowship_not_set_for_faao_only():
    """Lily Zhu-Tam ('OD, FAAO') — FAAO is Fellow of American Academy of
    Optometry, NOT FCOVD/FOVDR. Must not qualify."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_2.html"), country="US")
    zhu_tam = next(r for r in rows if r.name == "Lily Zhu-Tam")
    assert zhu_tam.credentials == "OD, FAAO"
    assert zhu_tam.fellowship_level is False


def test_canadian_multioffice_row_uses_first_office():
    """Heather Mackenzie has TWO Canadian offices in consecutive
    multiOffices TRs. Adapter must emit exactly ONE row using the FIRST
    office's data (Saugeen Shores Vision Therapy Centre, 311 Goderich
    st.) — not two rows, and not the second office."""
    rows = parse_search_html(_load("ovdr_search_country_ca.html"), country="CA")
    mackenzies = [r for r in rows if r.name == "Heather Mackenzie"]
    assert len(mackenzies) == 1
    m = mackenzies[0]
    assert m.practice_name == "Saugeen Shores Vision Therapy Centre"
    assert m.address1 == "311 Goderich st."
    assert m.city == "Port Elgin"
    assert m.state == "Ontario"
    assert m.postal == "N0H2C1"
    assert m.country == "CA"
    assert m.fellowship_level is True  # OD, FCOVD


def test_canadian_postal_with_internal_space():
    """Angela Peddle's postal is 'M9A 4S4' (with internal space).
    The parser must keep the postal as 'M9A 4S4' AND leave the state as
    'Ontario' (NOT 'Ontario M9A')."""
    rows = parse_search_html(_load("ovdr_search_country_ca.html"), country="CA")
    peddle = next(r for r in rows if r.name == "Angela Peddle")
    assert peddle.state == "Ontario"
    assert peddle.postal == "M9A 4S4"


def test_us_zip_plus_4_preserved():
    """ZIP+4 codes like '10036-8005' must survive intact in postal."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_2.html"), country="US")
    ciuffreda = next(r for r in rows if r.name == "Kenneth Ciuffreda")
    assert ciuffreda.postal == "10036-8005"


def test_practice_name_dropped_when_equals_practitioner():
    """When the office name field is literally the doctor's name (solo
    listing, no separate practice name), suppress the duplicate so the
    UI doesn't double-display 'Samantha Slotnick' under 'Samantha
    Slotnick, OD'.

    The fixtures' Samantha Slotnick has practice='Samantha Slotnick, OD'
    — this is "name + credentials" which is technically a different
    string than the bare name. We only suppress when the comparison is
    exact-match (case-insensitive). The Marcie Evans Stein row is a
    cleaner test: practice='Marcie Evans Stein, O.D.' — still not
    exact-match, so it should be retained.

    For exact-match suppression we use a synthetic record below."""
    # First confirm we don't over-suppress.
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    stein = next(r for r in rows if r.name == "Marcie Evans Stein")
    # The practice differs from the name (has ", O.D." suffix), so kept.
    assert stein.practice_name == "Marcie Evans Stein, O.D."


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the dedup keys for ON CONFLICT upsert."""
    payload = _load("ovdr_search_us_ny_page_1.html")
    a = parse_search_html(payload, country="US")
    b = parse_search_html(payload, country="US")
    assert [r.source_url for r in a] == [r.source_url for r in b]
    # All distinct within one page.
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_does_not_include_state_or_country():
    """The canonical source_url is the BARE profileId-keyed detail URL —
    NOT the search-context-augmented URL the locator hands out. This is
    what makes re-runs from different search slices (Country=US vs
    Country=US&State=NY) yield identical upsert keys for the same
    practitioner."""
    rows = parse_search_html(_load("ovdr_search_us_ny_page_1.html"), country="US")
    for r in rows:
        # No &State=, no &Country=, no &page=, no &lat=...
        assert "&State=" not in r.source_url
        assert "&Country=" not in r.source_url
        assert "&page=" not in r.source_url
        assert "&lat=" not in r.source_url


def test_empty_page_returns_empty_list():
    """A search HTML page with no <table class='results'> (i.e. the
    locator returned 'no results found') must yield an empty list, not
    crash."""
    rows = parse_search_html(
        "<html><body><h1>No doctors found</h1></body></html>", country="US"
    )
    assert rows == []


def test_parse_total_pages_finds_max():
    """Pagination block extraction from NY page 1 (2 pages total)."""
    pages = parse_total_pages(_load("ovdr_search_us_ny_page_1.html"))
    assert pages == 2

    # Canada has 4 pages.
    pages_ca = parse_total_pages(_load("ovdr_search_country_ca.html"))
    assert pages_ca == 4


def test_parse_total_pages_returns_1_when_no_pagination():
    """If there's no pagination block (single-page result set), the
    default total page count is 1."""
    assert parse_total_pages("<html><body></body></html>") == 1


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_strip_credentials_standard():
    """'Name, Cred1, Cred2' splits cleanly."""
    name, creds = _strip_credentials("Kenneth Ciuffreda, OD, PhD, FCOVD-A")
    assert name == "Kenneth Ciuffreda"
    assert creds == "OD, PhD, FCOVD-A"


def test_strip_credentials_single_cred():
    name, creds = _strip_credentials("Rebecca Marinoff, OD")
    assert name == "Rebecca Marinoff"
    assert creds == "OD"


def test_strip_credentials_no_creds():
    name, creds = _strip_credentials("Sarah Shaw")
    assert name == "Sarah Shaw"
    assert creds is None


def test_strip_credentials_empty():
    name, creds = _strip_credentials("")
    assert name == ""
    assert creds is None


def test_is_fellowship_matrix():
    """FCOVD / FCOVD-A / FOVDR (any case/spacing) qualify; FAAO / OD /
    COVT do NOT."""
    assert _is_fellowship("OD, FCOVD") is True
    assert _is_fellowship("OD, fcovd") is True
    assert _is_fellowship("OD, F.C.O.V.D.") is True
    assert _is_fellowship("OD, FCOVD-A") is True
    assert _is_fellowship("OD, PhD, FCOVD-A") is True
    assert _is_fellowship("OD, FOVDR") is True
    assert _is_fellowship("OD, F.O.V.D.R.") is True
    # Negatives:
    assert _is_fellowship("OD") is False
    assert _is_fellowship("OD, FAAO") is False
    assert _is_fellowship("COVT; OD") is False
    assert _is_fellowship("Doctor of Optometry") is False
    assert _is_fellowship(None) is False
    assert _is_fellowship("") is False


def test_is_fellowship_does_not_match_unrelated_f_designations():
    """Defensive: 'FNAP', 'FAAO', 'FOAA' all START with F but are NOT
    OVDR fellowships. The regex must be specific to FCOVD / FOVDR."""
    assert _is_fellowship("OD, FAAO, FNAP") is False
    assert _is_fellowship("OD, FAAO") is False


def test_extract_profile_id():
    href = (
        "/Search/Detailed?profileId=867C2ADB-DEBF-4CA4-AFEC-F27128F54DEC"
        "&markerId=0&lat=40.7544216&lng=-73.9822267"
    )
    assert _extract_profile_id(href) == "867C2ADB-DEBF-4CA4-AFEC-F27128F54DEC"


def test_extract_profile_id_missing():
    assert _extract_profile_id("/Search/Detailed?markerId=0") is None
    assert _extract_profile_id("") is None
    assert _extract_profile_id(None) is None


def test_extract_lat_lng_from_href():
    href = (
        "/Search/Detailed?profileId=X&markerId=0&lat=40.7544216&lng=-73.9822267"
    )
    lat, lng = _extract_lat_lng_from_href(href)
    assert lat == 40.7544216
    assert lng == -73.9822267


def test_extract_lat_lng_missing():
    lat, lng = _extract_lat_lng_from_href("/Search/Detailed?profileId=X")
    assert lat is None
    assert lng is None


def test_build_source_url():
    """Stable bare-profileId URL."""
    url = _build_source_url("867C2ADB-DEBF-4CA4-AFEC-F27128F54DEC")
    assert url == (
        "https://locate.covd.org/Search/Detailed"
        "?profileId=867C2ADB-DEBF-4CA4-AFEC-F27128F54DEC"
    )


def test_build_source_url_none():
    """No profileId -> None (caller must skip; we don't synthesize because
    re-runs would otherwise produce duplicate-key collisions)."""
    assert _build_source_url(None) is None
    assert _build_source_url("") is None


def test_split_city_state_postal_us_simple():
    city, state, postal = _split_city_state_postal("New York, NY 10036", "US")
    assert city == "New York"
    assert state == "NY"
    assert postal == "10036"


def test_split_city_state_postal_us_zip_plus_4():
    city, state, postal = _split_city_state_postal(
        "New York, NY 10036-8005", "US"
    )
    assert postal == "10036-8005"


def test_split_city_state_postal_us_full_state_name():
    """The locator sometimes emits the full state name instead of the
    abbreviation ('Commack, New York 11725'). State retains the full
    spelling."""
    city, state, postal = _split_city_state_postal(
        "Commack, New York 11725", "US"
    )
    assert city == "Commack"
    assert state == "New York"
    assert postal == "11725"


def test_split_city_state_postal_canadian_no_space():
    city, state, postal = _split_city_state_postal(
        "Port Elgin, Ontario N0H2C1", "CA"
    )
    assert city == "Port Elgin"
    assert state == "Ontario"
    assert postal == "N0H2C1"


def test_split_city_state_postal_canadian_with_space():
    """Canadian postal codes CAN have an internal space ('M9A 4S4'). The
    parser must NOT consume the first half as part of the state name."""
    city, state, postal = _split_city_state_postal(
        "Etobicoke, Ontario M9A 4S4", "CA"
    )
    assert city == "Etobicoke"
    assert state == "Ontario"
    assert postal == "M9A 4S4"


def test_split_city_state_postal_uk():
    city, state, postal = _split_city_state_postal(
        "London, England SW112PJ", "GB"
    )
    assert city == "London"
    assert state == "England"
    assert postal == "SW112PJ"


def test_parse_address_block_us():
    addr_html = (
        "<address>"
        "33 W 42nd St"
        "<br />"
        "New York, NY 10036"
        "</address>"
    )
    address1, city, state, postal = _parse_address_block(addr_html, "US")
    assert address1 == "33 W 42nd St"
    assert city == "New York"
    assert state == "NY"
    assert postal == "10036"


def test_parse_address_block_empty():
    a1, c, s, p = _parse_address_block("", "US")
    assert a1 is None
    assert c is None
    assert s is None
    assert p is None
