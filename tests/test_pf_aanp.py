"""Unit tests for the AANP (American Association of Naturopathic Physicians)
adapter.

AANP publishes its public "Find an ND" directory through a YourMembership
/ AssociationVoice CMS. The search form at /search/custom.asp?id=5613 hits
/search/search.asp which returns a shell with an iframe pointing at
/searchserver/people2.aspx?id=<one-shot-session-uuid>. The iframe response
carries a paginated card list (``<ul id="search-results">`` of
``<li><div class="memb-result-item">`` cards). Each card carries only
the name, member_id, and a city/state/postal snippet — street, phone,
email, website, credentials, practice_name all live on the per-member
profile page at ``/members/?id=<id>`` and are merged in by the migrate
runner.

The site is Cloudflare-protected (HTTP 403 for static-UA clients), so
the live migrate runner uses Playwright. Fixtures here are real captures
taken 2026-05-27 against the live site through Playwright.

Live fixtures (the new canonical set, post-2024 site migration):

- aanp_search_iframe_live.html — real /searchserver/people2.aspx page 1
                                  (24 cards, "1000+" record count,
                                   "Page 1 of 46").
- aanp_profile_60515743_live.html — Dr. Nancy Aagenes (mostly empty
                                     profile — name only, no phone /
                                     address / credentials).
- aanp_profile_60515403_live.html — Dr. Lise Alschuler (FABNO, ND;
                                     Tempe AZ; phone + website + email
                                     + practice name; fellowship=True).
- aanp_profile_60520396_live.html — Dr. Dawn Alden (ND; Sacramento CA;
                                     website + email, no phone, no
                                     practice anchor).

Legacy fixtures (kept for the profile-parser regression suite — the
profile-page structure tdEmployerName / tdWorkPhone / CstmFldLbl is
unchanged across the migration so these still apply):

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

The legacy aanp_search_page_*.html table-grid fixtures captured the
pre-migration layout; they no longer match the live HTML and the list-
page tests against them were removed when the parser was rewritten.
The fixture files are retained as historical reference but are not
exercised by this suite.
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
# Search-results (list page) tests — live ``ul#search-results`` card layout
# ---------------------------------------------------------------------------

def test_parse_iframe_live_returns_24_cards():
    """The live ``/searchserver/people2.aspx`` iframe page packs 24
    ``memb-result-item`` cards per page (down from the 25-row table-grid
    page that the original layout used)."""
    rows = parse_search_results_html(_load("aanp_search_iframe_live.html"))
    assert len(rows) == 24


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties / source_url are constant per spec
    across every parsed row."""
    rows = parse_search_results_html(_load("aanp_search_iframe_live.html"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "AANP"
        assert r.specialties == ["naturopathy", "holistic_health"]
        assert r.source_url
        assert r.source_url.startswith("https://naturopathic.org/members/?id=")


def test_card_with_full_address_hilliary_abbott():
    """Card 2 on the iframe page is Dr. Hilliary Abbott — full US city /
    state / postal triple. Validates city + state + postal extraction
    from the ``<p class="address">`` br-delimited token list."""
    rows = parse_search_results_html(_load("aanp_search_iframe_live.html"))
    abbott = next(r for r in rows if "Hilliary Abbott" in r.name)
    assert abbott.name == "Dr. Hilliary Abbott"
    assert abbott.city == "Lynnwood"
    assert abbott.state == "Washington"
    assert abbott.postal == "98036-6921"
    assert abbott.country == "US"
    # Street is profile-only on the new layout — list-page card never
    # carries it.
    assert abbott.address1 is None
    assert abbott.source_url == "https://naturopathic.org/members/?id=74147396"


def test_card_with_empty_address_keeps_name_only():
    """Some live cards have ``<p class="address"></p>`` (privacy-suppressed
    or empty member profiles). The parser must still emit the row (name +
    member_id are enough for the org-member tier) with city/state/postal
    all None. Dr. Nancy Aagenes is the canonical no-address case in the
    live iframe page (the first card on page 1 of an A-Z sorted query)."""
    rows = parse_search_results_html(_load("aanp_search_iframe_live.html"))
    aagenes = next(r for r in rows if "Nancy Aagenes" in r.name)
    assert aagenes.name == "Dr. Nancy Aagenes"
    assert aagenes.source_url == "https://naturopathic.org/members/?id=60515743"
    assert aagenes.address1 is None
    assert aagenes.city is None
    assert aagenes.state is None
    assert aagenes.postal is None
    # Country still defaults to US even without state.
    assert aagenes.country == "US"


def test_record_count_extracted_from_live_iframe():
    """The DocCount span on the live iframe page renders ``1000+`` for
    unbounded queries — the parser strips the ``+`` and returns 1000.
    The true total is established by walking pages until exhausted."""
    assert parse_record_count(_load("aanp_search_iframe_live.html")) == 1000


def test_page_info_extracted_from_live_iframe():
    """The 'Page X of Y' text gives us pagination bounds (page 1 of 46
    on the live capture)."""
    assert parse_page_info(_load("aanp_search_iframe_live.html")) == (1, 46)


def test_source_url_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — source_url is the upsert dedup key."""
    payload = _load("aanp_search_iframe_live.html")
    a = parse_search_results_html(payload)
    b = parse_search_results_html(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]


def test_source_url_uses_member_id():
    """The canonical detail URL is built from the numeric member id, which
    is the YourMembership account id and is stable across re-runs."""
    rows = parse_search_results_html(_load("aanp_search_iframe_live.html"))
    for r in rows:
        assert r.source_url.startswith(f"{BASE}/members/?id=")
        # No trailing fragment / query slug.
        tail = r.source_url.split("?id=", 1)[1]
        assert tail.isdigit()


def test_search_parser_skips_non_string_input():
    """Defensive: a non-str payload must return [] not crash."""
    assert parse_search_results_html(None) == []
    assert parse_search_results_html(b"<html>") == []
    assert parse_search_results_html(12345) == []


def test_search_parser_skips_missing_list():
    """An HTML doc without ``id="search-results"`` (e.g. a login wall,
    a Cloudflare challenge, or the pre-migration table-grid page) must
    return [] not raise."""
    assert parse_search_results_html("<html><body>blocked</body></html>") == []
    assert parse_search_results_html("") == []


def test_search_parser_ignores_member_links_outside_search_results():
    """Member-id anchors that live OUTSIDE the ``<ul id="search-results">``
    list (sidebar widgets, featured member tiles, footer breadcrumbs) must
    NOT be parsed as search results."""
    # Synthetic page with a stray member link but no search-results ul —
    # adapter must return [].
    html = (
        "<html><body>"
        '<div id="featured">'
        '  <a href="/members/?id=99999999" class="normalName">Fake</a>'
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
# Live profile-page tests (post-migration captures, with caller-supplied
# member_id since the live page body no longer carries the id in an
# in-page anchor)
# ---------------------------------------------------------------------------

def test_profile_live_minimal_aagenes():
    """Dr. Nancy Aagenes — a near-empty live profile (name only, no phone,
    no address, no credentials, no email). Validates that the parser still
    emits a row carrying the locked invariants when contact fields are
    blank, and that the caller-supplied member_id is reflected in the
    source_url."""
    row = parse_profile_html(
        _load("aanp_profile_60515743_live.html"), member_id="60515743"
    )
    assert row is not None
    assert row.name == "Dr. Nancy Aagenes"
    assert row.practice_name is None
    assert row.credentials is None
    assert row.phone is None
    assert row.email is None
    assert row.website is None
    assert row.address1 is None
    assert row.city is None
    assert row.state is None
    assert row.postal is None
    assert row.country == "US"
    assert row.source_url == "https://naturopathic.org/members/?id=60515743"
    # Locked invariants
    assert row.tier == "org_member"
    assert row.source_org == "AANP"
    assert row.specialties == ["naturopathy", "holistic_health"]
    assert row.fellowship_level is False


def test_profile_live_full_alschuler_fellowship_true():
    """Dr. Lise Alschuler — full live profile with FABNO credential
    (fellowship-qualifying), real practice name, phone + website + email,
    and a single-line street address."""
    row = parse_profile_html(
        _load("aanp_profile_60515403_live.html"), member_id="60515403"
    )
    assert row is not None
    assert row.name == "Dr. Lise Alschuler"
    assert row.practice_name == "Naturopathic Specialists"
    assert row.credentials == "FABNO, ND"
    assert row.phone == "480 990-1111"
    assert row.email == "lnalschuler@comcast.net"
    assert row.website == "http://www.listenandcare.com"
    assert row.address1 == "2140 E Broadway Rd"
    assert row.city == "Tempe"
    assert row.state == "Arizona"
    assert row.postal == "85282"
    assert row.country == "US"
    assert row.source_url == "https://naturopathic.org/members/?id=60515403"
    # FABNO is a vetted specialty-board fellow credential.
    assert row.fellowship_level is True


def test_profile_live_website_only_alden():
    """Dr. Dawn Alden — live profile with website + email but no phone,
    no practice-name anchor (employer block opens with a bare street),
    and an extended postal (ZIP+4)."""
    row = parse_profile_html(
        _load("aanp_profile_60520396_live.html"), member_id="60520396"
    )
    assert row is not None
    assert row.name == "Dr. Dawn Alden"
    assert row.practice_name is None
    assert row.credentials == "ND"
    assert row.phone is None
    assert row.email == "info@eastsacramentoconcierge.com"
    assert row.website == "https://www.aldennd.com/"
    assert row.address1 == "3800 J St"
    assert row.city == "Sacramento"
    assert row.state == "California"
    assert row.postal == "95816-5551"
    assert row.country == "US"
    assert row.source_url == "https://naturopathic.org/members/?id=60520396"
    assert row.fellowship_level is False


def test_profile_member_id_fallback_to_in_page_anchor():
    """When ``member_id`` is omitted, the parser falls back to scraping
    a ``/members/?id=NNN`` anchor from the page body. The legacy fixture
    aanp_profile_60515148.html carries such an anchor in the edit-profile
    link, so source_url still resolves without a caller-supplied id."""
    row = parse_profile_html(_load("aanp_profile_60515148.html"))
    assert row is not None
    assert row.source_url == "https://naturopathic.org/members/?id=60515148"


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
