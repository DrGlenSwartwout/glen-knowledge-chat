"""Unit tests for the AANP (American Association of Naturopathic Physicians)
adapter.

AANP publishes its public "Find an ND" directory through a YourMembership
/ AssociationVoice CMS. The search form at /search/custom.asp?id=5613 hits
/search/search.asp which returns a shell with an iframe pointing at
/searchserver/people.aspx?id=<one-shot-session-uuid>. The iframe response
carries the paginated list-grid; each row holds the practitioner name,
detail-page URL (``/members/?id=<id>``), and the address columns. Detail
pages carry the credentials, phone, website, email, and practice name.

The site is Cloudflare-protected (HTTP 403 for static-UA clients), so
fixtures here were captured 2026-05-27 via the Internet Archive Wayback
Machine — the wayback URL-rewrites were stripped during capture so the
fixtures look like real production HTML.

Fixtures:

- aanp_search_page_1.html  — 281-result search, page 1 of 12 (25 rows;
                             US-centric mix incl. 2 privacy-suppressed
                             entries that lack any address divs).
- aanp_search_page_2.html  — 117-result search, page 1 of 5 (25 rows;
                             includes Sarah Abel with full address).
- aanp_search_page_3.html  — 255-result search, page 1 of 11 (25 rows;
                             every row has full address; no
                             privacy-suppressed entries).

- aanp_profile_60515148.html — Dr. Joshua Levitt (ND, Hamden CT, phone-
                               only, solo practice = employer link
                               equals the practitioner name; tests
                               solo-name suppression).
- aanp_profile_60515163.html — Dr. Kiera Lane (LAc, NMD, Other; Chandler
                               AZ, website-only, no phone; tests multi-
                               credential extraction + the Visit-Website
                               anchor).
- aanp_profile_60515573.html — Dr. Sharon Hunter (ND, West Hartford CT,
                               "Bloom Natural Health" practice name +
                               2-line street; tests genuine practice
                               name + multi-line address pickup).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.aanp import (  # noqa: E402
    BASE,
    _build_source_url,
    _country_iso2_from_name,
    _detect_fellowship_creds,
    _infer_country_from_state,
    _normalize_website,
    _strip_credentials,
    _strip_html_tags,
    parse_page_info,
    parse_profile_html,
    parse_record_count,
    parse_search_results_html,
)


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Search-grid (list page) tests
# ---------------------------------------------------------------------------

def test_parse_page_1_returns_25_rows():
    """Each iframe response page holds up to 25 lineitem rows. Page 1
    fixtures all carry exactly 25 (the YourMembership default page size)."""
    rows = parse_search_results_html(_load("aanp_search_page_1.html"))
    assert len(rows) == 25


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties / source_url are constant per spec
    across every parsed row from every fixture page."""
    rows = []
    for fixture in (
        "aanp_search_page_1.html",
        "aanp_search_page_2.html",
        "aanp_search_page_3.html",
    ):
        rows += parse_search_results_html(_load(fixture))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "AANP"
        assert r.specialties == ["naturopathy", "holistic_health"]
        assert r.source_url
        assert r.source_url.startswith("https://naturopathic.org/members/?id=")


def test_full_row_with_address_lee_aberle():
    """Page 1's first row is Dr. Lee Aberle — full US address with
    suite line. Validates name + 2-line address + city/state/postal
    extraction."""
    rows = parse_search_results_html(_load("aanp_search_page_1.html"))
    aberle = next(r for r in rows if "Lee Aberle" in r.name)
    assert aberle.name == "Dr. Lee Aberle"
    assert aberle.address1 == "22 Wilson Ave NE, Ste 205"
    assert aberle.city == "Saint Cloud"
    assert aberle.state == "Minnesota"
    assert aberle.postal == "56304-0440"
    assert aberle.country == "US"
    assert aberle.source_url == "https://naturopathic.org/members/?id=60516495"


def test_row_with_only_city_no_street():
    """Some directory entries list a city + state but no street address
    (Dr. Ezenwanyi Ahaghotu — Katy, Texas only). The parser must still
    populate city/state/postal and leave address1=None rather than
    pushing the city into address1."""
    rows = parse_search_results_html(_load("aanp_search_page_2.html"))
    ah = next(r for r in rows if "Ahaghotu" in r.name)
    assert ah.address1 is None
    assert ah.city == "Katy"
    assert ah.state == "Texas"
    assert ah.postal == "77494-7823"


def test_row_with_privacy_suppressed_address_keeps_name_only():
    """Privacy-suppressed entries have all 7 address divs empty. The
    parser must still emit the row (it has a name + member_id, which
    is enough for the org-member tier) with address1/city/state all
    None. Melissa Barber on page 1 is the canonical no-address case."""
    rows = parse_search_results_html(_load("aanp_search_page_1.html"))
    barber = next(r for r in rows if "Melissa Barber" in r.name)
    assert barber.name == "Melissa Barber"
    assert barber.source_url == "https://naturopathic.org/members/?id=68162190"
    assert barber.address1 is None
    assert barber.city is None
    assert barber.state is None
    assert barber.postal is None
    # Country still defaults to US even without state.
    assert barber.country == "US"


def test_record_count_extracted():
    """The DocCount span at the top of the result grid carries the total
    matching practitioners across all pages."""
    assert parse_record_count(_load("aanp_search_page_1.html")) == 281
    assert parse_record_count(_load("aanp_search_page_2.html")) == 117
    assert parse_record_count(_load("aanp_search_page_3.html")) == 255


def test_page_info_extracted():
    """The 'Page X of Y' text gives us pagination bounds."""
    assert parse_page_info(_load("aanp_search_page_1.html")) == (1, 12)
    assert parse_page_info(_load("aanp_search_page_2.html")) == (1, 5)
    assert parse_page_info(_load("aanp_search_page_3.html")) == (1, 11)


def test_source_url_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — source_url is the upsert dedup key."""
    payload = _load("aanp_search_page_1.html")
    a = parse_search_results_html(payload)
    b = parse_search_results_html(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]


def test_source_url_uses_member_id():
    """The canonical detail URL is built from the numeric member id, which
    is the YourMembership account id and is stable across re-runs."""
    rows = parse_search_results_html(_load("aanp_search_page_1.html"))
    for r in rows:
        # Every URL ends with the numeric id pattern.
        assert r.source_url.startswith(f"{BASE}/members/?id=")
        # No trailing fragment / query slug.
        tail = r.source_url.split("?id=", 1)[1]
        assert tail.isdigit()


def test_search_parser_skips_non_string_input():
    """Defensive: a non-str payload must return [] not crash."""
    assert parse_search_results_html(None) == []
    assert parse_search_results_html(b"<html>") == []
    assert parse_search_results_html(12345) == []


def test_search_parser_skips_missing_grid():
    """An HTML doc without ``id="SearchResultsGrid"`` (e.g. a login wall
    or a Cloudflare challenge) must return [] not raise."""
    assert parse_search_results_html("<html><body>blocked</body></html>") == []
    assert parse_search_results_html("") == []


def test_search_parser_ignores_featured_member_sidebar():
    """The search shell page (custom.asp / search.asp) carries a
    ``Featured Members`` sidebar with 2 member-id links that are NOT
    search results. The parser must ignore them because they aren't
    inside ``id="SearchResultsGrid"``."""
    # Synthetic page with the sidebar but no grid — adapter must return [].
    html = (
        "<html><body>"
        '<div id="featured">'
        '  <a href="/members/?id=99999999">Fake Featured Member</a>'
        "</div>"
        "</body></html>"
    )
    assert parse_search_results_html(html) == []


# ---------------------------------------------------------------------------
# Profile (detail page) tests
# ---------------------------------------------------------------------------

def test_profile_solo_practice_suppresses_practice_name():
    """When the employer link equals the practitioner's bare name
    ('Joshua Levitt' for 'Dr. Joshua Levitt'), the practice_name is
    suppressed — it's a duplicate, not a real practice brand. Per the
    same pattern used by the IABDM adapter."""
    row = parse_profile_html(_load("aanp_profile_60515148.html"))
    assert row is not None
    assert row.name == "Dr. Joshua Levitt"
    assert row.practice_name is None
    assert row.credentials == "ND"
    assert row.phone == "203 288-8283"
    assert row.email == "admin@wholehealthct.com"
    assert row.address1 == "2838 Old Dixwell Ave"
    assert row.city == "Hamden"
    assert row.state == "Connecticut"
    assert row.postal == "06518-3137"
    assert row.country == "US"
    assert row.source_url == "https://naturopathic.org/members/?id=60515148"
    assert row.fellowship_level is False
    # tier/source_org/specialties invariants
    assert row.tier == "org_member"
    assert row.source_org == "AANP"
    assert row.specialties == ["naturopathy", "holistic_health"]


def test_profile_multi_credential_and_website_only():
    """Dr. Kiera Lane has multi-credentials (LAc, NMD, Other), no phone,
    a Visit Website link, and no separate practice-name link. Validates
    multi-credential pickup + website extraction + practice=None when
    the employer block has no employer anchor."""
    row = parse_profile_html(_load("aanp_profile_60515163.html"))
    assert row is not None
    assert row.name == "Dr. Kiera Lane"
    assert row.practice_name is None
    assert row.credentials == "LAc, NMD, Other"
    assert row.phone is None
    assert row.website == "https://www.aznaturalmedicine.com/"
    assert row.email == "drklane@aznaturalmedicine.com"
    assert row.city == "Chandler"
    assert row.state == "Arizona"
    assert row.postal == "85224-3558"


def test_profile_with_practice_and_multiline_street():
    """Dr. Sharon Hunter has a real practice name ('Bloom Natural
    Health') plus a 2-line street ('95 S Main St', 'Fl 2'). The
    parser must collect both street lines into address1 and keep
    the practice name distinct."""
    row = parse_profile_html(_load("aanp_profile_60515573.html"))
    assert row is not None
    assert row.name == "Dr. Sharon Hunter"
    assert row.practice_name == "Bloom Natural Health"
    assert row.address1 == "95 S Main St, Fl 2"
    assert row.city == "West Hartford"
    assert row.state == "Connecticut"
    assert row.postal == "06107-2506"
    assert row.phone == "860 310-5559"


def test_profile_parser_handles_non_string():
    """Defensive: non-string input returns None."""
    assert parse_profile_html(None) is None
    assert parse_profile_html(b"<html>") is None


def test_profile_parser_handles_empty_input():
    """An empty/blocked profile page returns None (no title -> no name)."""
    assert parse_profile_html("") is None
    assert parse_profile_html("<html><body>blocked</body></html>") is None


# ---------------------------------------------------------------------------
# Fellowship-detection rule (the key per-adapter contract decision)
# ---------------------------------------------------------------------------

def test_fellowship_defaults_to_false_for_plain_nd():
    """The standard credential 'ND' does NOT qualify for fellowship_level
    — naturopathy has no public 'Fellow' tier analogous to OEPF/IAOMT."""
    assert _detect_fellowship_creds("ND") is False
    assert _detect_fellowship_creds("Dr.") is False
    assert _detect_fellowship_creds(None) is False
    assert _detect_fellowship_creds("") is False


def test_fellowship_true_for_specialty_board_fellows():
    """FAANP / FABNO / FACO / FABNE are vetted specialty-board fellow
    credentials and qualify per the AANP-specific rule."""
    assert _detect_fellowship_creds("ND, FAANP") is True
    assert _detect_fellowship_creds("ND FABNO") is True
    assert _detect_fellowship_creds("ND, FACO") is True
    assert _detect_fellowship_creds("ND, FABNE") is True
    # Case-insensitive.
    assert _detect_fellowship_creds("nd, faanp") is True


def test_fellowship_true_for_diplomate_status():
    """Long-form 'Diplomate' (e.g. 'Diplomate of the American Board of
    Naturopathic Oncology') counts."""
    assert _detect_fellowship_creds("ND, Diplomate") is True
    assert _detect_fellowship_creds("ND Diplomate ABNO") is True


def test_fellowship_ignores_other_unrelated_credentials():
    """Credentials like LAc, NMD, MS, DC, PhD do NOT qualify."""
    assert _detect_fellowship_creds("LAc, NMD, Other") is False
    assert _detect_fellowship_creds("ND, MS, PhD") is False
    assert _detect_fellowship_creds("DC, ND") is False


def test_profile_row_fellowship_when_creds_qualify():
    """When the profile credential field carries a qualifying token, the
    parsed row's fellowship_level is True. Synthetic profile because none
    of the production fixtures carry FAANP / FABNO."""
    html = (
        "<html><head><title>Dr. Sample Fellow - "
        "American Association of Naturopathic Physicians</title></head>"
        '<body><a href="/members/?id=12345">edit</a>'
        '<td id="tdEmployerName">'
        '<a href="/search/search.asp?txt_state=Oregon">Oregon</a>'
        "</td>"
        '<td id="tdWorkPhone"></td>'
        '<tr><td><label class="CstmFldLbl">Credentials:</label></td>'
        '<td class="CstmFldVal">ND, FAANP</td></tr>'
        "</body></html>"
    )
    row = parse_profile_html(html)
    assert row is not None
    assert row.credentials == "ND, FAANP"
    assert row.fellowship_level is True


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_strip_credentials_paren_form():
    """'(ND)' style suffix lifts to credentials."""
    name, creds = _strip_credentials("Jane Doe (ND)")
    assert name == "Jane Doe"
    assert creds == "ND"


def test_strip_credentials_comma_form():
    """'Name, Cred1, Cred2' splits cleanly."""
    name, creds = _strip_credentials("Dr. Jane Doe, ND, FABNO")
    assert name == "Dr. Jane Doe"
    assert "ND" in creds
    assert "FABNO" in creds


def test_strip_credentials_preserves_dr_honorific():
    """'Dr.' is preserved on the name (it's part of the title, not a
    trailing credential)."""
    name, _ = _strip_credentials("Dr. Joshua Levitt")
    assert name == "Dr. Joshua Levitt"


def test_strip_credentials_no_creds():
    name, creds = _strip_credentials("Sarah Abel")
    assert name == "Sarah Abel"
    assert creds is None


def test_strip_html_tags_collapses_blocks_to_space():
    """Block-level tags become spaces; inline tags vanish so anchor-
    delimited comma lists don't get whitespace inserted around their
    commas."""
    assert _strip_html_tags("<a>LAc</a>, <a>NMD</a>") == "LAc, NMD"
    assert _strip_html_tags("<div>Foo</div><div>Bar</div>") == "Foo Bar"
    assert _strip_html_tags("95 S Main St<br/>Fl 2") == "95 S Main St Fl 2"
    assert _strip_html_tags(None) == ""
    assert _strip_html_tags("") == ""


def test_normalize_website_adds_scheme():
    assert _normalize_website("aznaturalmedicine.com") == "https://aznaturalmedicine.com"
    assert _normalize_website("https://x.com/") == "https://x.com/"
    assert _normalize_website("http://x.com") == "http://x.com"


def test_normalize_website_rejects_garbage():
    """Mailto / javascript / fragment-only links are not real websites."""
    assert _normalize_website("mailto:x@y.com") is None
    assert _normalize_website("javascript:void(0)") is None
    assert _normalize_website("#") is None
    assert _normalize_website(None) is None
    assert _normalize_website("") is None


def test_country_iso2_from_name():
    assert _country_iso2_from_name("United States") == "US"
    assert _country_iso2_from_name("Canada") == "CA"
    assert _country_iso2_from_name("united kingdom") == "GB"
    assert _country_iso2_from_name("Atlantis") is None
    assert _country_iso2_from_name(None) is None


def test_infer_country_from_state():
    """Inference fallback for the (common) row case where the country
    column is blank — US state names => 'US', Canadian provinces => 'CA',
    unrecognized => 'US' (safe default for naturopathic.org)."""
    assert _infer_country_from_state("California") == "US"
    assert _infer_country_from_state("Texas") == "US"
    assert _infer_country_from_state("Ontario") == "CA"
    assert _infer_country_from_state("British Columbia") == "CA"
    assert _infer_country_from_state("Unknown") == "US"
    assert _infer_country_from_state(None) == "US"


def test_build_source_url_uses_canonical_pattern():
    """URL pattern is the canonical YourMembership detail page."""
    assert _build_source_url("12345") == "https://naturopathic.org/members/?id=12345"
    assert _build_source_url("60515148") == "https://naturopathic.org/members/?id=60515148"
