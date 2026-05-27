"""Unit tests for the NANP (National Association of Nutrition Professionals)
adapter.

NANP and AANP run on the same YourMembership / AssociationVoice CMS
vendor — sister deployments. The search form on ``mynanp.nanp.org`` GETs
a shell at ``/search/newsearch.asp`` which embeds an iframe pointing at
``/searchserver/people2.aspx?id=<session-uuid>``. The iframe response
carries a paginated card list (``<ul id="search-results">`` of
``<li><div class="memb-result-item">`` cards). Each card carries only
the name, member_id, and a city/state snippet — street, phone, email,
website, credentials, BCHN-flag, practice_name all live on the per-member
profile page at ``/members/?id=<id>`` and are merged in by the migrate
runner.

The site is Cloudflare-protected, so the live migrate runner uses
Playwright. Fixtures here are real captures taken 2026-05-27 against the
live site through Playwright.

Live fixtures:

- nanp_search_iframe_live.html — real ``/searchserver/people2.aspx``
                                  page 1 (24 cards, DocCount=673,
                                  Page 1 of 29).
- nanp_profile_80983337_live.html — Laura Andrews (BCHN=No,
                                     fellowship_level=False;
                                     website-only, no phone, Kentucky).
- nanp_profile_81477610_live.html — Charity Allen (BCHN=Yes,
                                     fellowship_level=True;
                                     phone in tdHomePhone, mailto-anchor
                                     email, West Virginia).
- nanp_profile_80176432_live.html — Millie H Abplanalp (BCHN=Yes,
                                     fellowship_level=True; website-only,
                                     no phone, Utah, no street).

The synthesized fixtures from the initial spec
(``nanp_search_form.html``, ``nanp_search_results_page_1.html``,
``nanp_member_profile_*.html``) were built against the wrong URL pattern
(``/profile/?ID=<n>``) and an invented results-table shape (``<table
class="search-results">``); the live deployment uses the iframe-card
pattern instead. The synthesized files are retained on disk for
historical reference but are no longer exercised by this suite.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.nanp import (  # noqa: E402
    BASE,
    _build_source_url,
    _country_iso2_from_name,
    _detect_fellowship_creds,
    _infer_country_from_state,
    _normalize_credential_chunk,
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
    ``memb-result-item`` cards per page on NANP — same per-page density
    as AANP (the two sites are sister YourMembership deployments)."""
    rows = parse_search_results_html(_load("nanp_search_iframe_live.html"))
    assert len(rows) == 24


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties / source_url are constant per spec
    across every parsed row. source_org and specialties are NANP-
    specific (the only differences from the AANP card invariants)."""
    rows = parse_search_results_html(_load("nanp_search_iframe_live.html"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "NANP"
        assert r.specialties == ["nutrition", "holistic_health"]
        assert r.source_url
        assert r.source_url.startswith("https://mynanp.nanp.org/members/?id=")


def test_card_with_full_city_state_carla_abate():
    """Card 1 on the iframe page is Carla Abate — full US city/state
    pair. Validates city + state extraction from the
    ``<p class="address">`` br-delimited token list. Note: the live
    NANP card markup has NO postal token (postal is profile-only)."""
    rows = parse_search_results_html(_load("nanp_search_iframe_live.html"))
    abate = next(r for r in rows if "Carla Abate" in r.name)
    assert abate.name == "Carla Abate"
    assert abate.city == "Golden"
    assert abate.state == "Colorado"
    assert abate.country == "US"
    # Postal + street are profile-only on the live layout — list-page
    # card never carries them.
    assert abate.postal is None
    assert abate.address1 is None


def test_card_with_canadian_province_card_3():
    """Card 3 is Afnan M. Abdelwahed in London, Ontario — exercises the
    Canadian-province path (state lookup hits ``_CA_PROVINCES``, country
    inference yields ``CA``)."""
    rows = parse_search_results_html(_load("nanp_search_iframe_live.html"))
    afnan = next(r for r in rows if "Afnan" in r.name)
    assert afnan.city == "London"
    assert afnan.state == "Ontario"
    assert afnan.country == "CA"


def test_record_count_extracted_from_live_iframe():
    """The DocCount span on the live NANP iframe page renders the exact
    record count (673) without the ``+`` suffix that AANP shows on
    unbounded queries."""
    assert parse_record_count(_load("nanp_search_iframe_live.html")) == 673


def test_page_info_extracted_from_live_iframe():
    """The 'Page X of Y' text gives us pagination bounds — page 1 of 29
    on the live NANP capture (673 records / 24 per page = 28.04, rounded
    up to 29)."""
    assert parse_page_info(_load("nanp_search_iframe_live.html")) == (1, 29)


def test_source_url_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — source_url is the upsert dedup key."""
    payload = _load("nanp_search_iframe_live.html")
    a = parse_search_results_html(payload)
    b = parse_search_results_html(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]


def test_source_url_uses_member_id():
    """The canonical detail URL is built from the numeric member id, which
    is the YourMembership account id and is stable across re-runs."""
    rows = parse_search_results_html(_load("nanp_search_iframe_live.html"))
    for r in rows:
        assert r.source_url.startswith(f"{BASE}/members/?id=")
        tail = r.source_url.split("?id=", 1)[1]
        assert tail.isdigit()


def test_search_parser_skips_non_string_input():
    """Defensive: a non-str payload must return [] not crash."""
    assert parse_search_results_html(None) == []
    assert parse_search_results_html(b"<html>") == []
    assert parse_search_results_html(12345) == []


def test_search_parser_skips_missing_list():
    """An HTML doc without ``id="search-results"`` (login wall,
    Cloudflare challenge, etc.) must return [] not raise."""
    assert parse_search_results_html("<html><body>blocked</body></html>") == []
    assert parse_search_results_html("") == []


def test_search_parser_ignores_member_links_outside_search_results():
    """Member-id anchors that live OUTSIDE the search-results UL (sidebar
    widgets, featured-member tiles, footer breadcrumbs) must NOT be
    parsed as search results."""
    html = (
        "<html><body>"
        '<div id="featured">'
        '  <a href="/members/?id=99999999" class="normalName">Fake</a>'
        "</div>"
        "</body></html>"
    )
    assert parse_search_results_html(html) == []


# ---------------------------------------------------------------------------
# Live profile-page tests
# ---------------------------------------------------------------------------

def test_profile_live_bchn_yes_full_row_abplanalp():
    """Millie H Abplanalp — BCHN=Yes via the custom-field block, no
    practice name, no street, no phone (only a Visit-Website anchor in
    ``tdWorkPhone``). Validates that ``fellowship_level=True`` fires off
    the canonical custom-field signal even when the credential token is
    NOT in the title."""
    row = parse_profile_html(
        _load("nanp_profile_80176432_live.html"), member_id="80176432"
    )
    assert row is not None
    assert row.name == "Millie H Abplanalp"
    # No street + no practice — the employer block opens with the city
    # anchor.
    assert row.practice_name is None
    assert row.address1 is None
    assert row.city == "Spanish Fork"
    assert row.state == "Utah"
    assert row.postal == "84660"
    assert row.country == "US"
    # No phone — tdWorkPhone only holds the Visit Website anchor.
    assert row.phone is None
    assert row.website == "http://www.nutritionforwellnesscenter.com"
    # No email on this profile.
    assert row.email is None
    assert row.source_url == "https://mynanp.nanp.org/members/?id=80176432"
    # BCHN=Yes via the custom-field block.
    assert row.fellowship_level is True
    # Locked invariants
    assert row.tier == "org_member"
    assert row.source_org == "NANP"
    assert row.specialties == ["nutrition", "holistic_health"]


def test_profile_live_bchn_no_andrews():
    """Laura Andrews — BCHN=No via the custom-field block,
    fellowship_level=False. Practice name = "Certified Holistic Nutrition
    Consultant" (the first plain-text fragment in the employer block,
    above the city anchor). Website-only, no phone, no email."""
    row = parse_profile_html(
        _load("nanp_profile_80983337_live.html"), member_id="80983337"
    )
    assert row is not None
    assert row.name == "Laura Andrews"
    assert row.practice_name == "Certified Holistic Nutrition Consultant"
    assert row.address1 is None
    assert row.city == "Georgetown"
    assert row.state == "Kentucky"
    assert row.postal == "40324"
    assert row.country == "US"
    assert row.phone is None
    assert row.website == "http://lovejoynutrition.com"
    assert row.email is None
    assert row.source_url == "https://mynanp.nanp.org/members/?id=80983337"
    # BCHN=No — fellowship_level must be False.
    assert row.fellowship_level is False
    # Locked invariants
    assert row.tier == "org_member"
    assert row.source_org == "NANP"
    assert row.specialties == ["nutrition", "holistic_health"]


def test_profile_live_bchn_yes_phone_email_allen():
    """Charity Allen — BCHN=Yes, fellowship_level=True. Has a street
    line (``424 Breckenridge Way``), phone in ``tdHomePhone`` (NOT
    ``tdWorkPhone``), and an email rendered via a JS-decrypted
    ``mailto:`` anchor in the right-column header. No practice name."""
    row = parse_profile_html(
        _load("nanp_profile_81477610_live.html"), member_id="81477610"
    )
    assert row is not None
    assert row.name == "Charity Allen"
    assert row.practice_name is None
    assert row.address1 == "424 Breckenridge Way"
    assert row.city == "Shenandoah Junction"
    assert row.state == "West Virginia"
    assert row.postal == "25442"
    assert row.country == "US"
    # Phone lives in tdHomePhone — the parser must accept either phone cell.
    assert row.phone == "931 2205391"
    # Email picked from the right-column JS-decrypted mailto anchor.
    assert row.email == "callen8492@gmail.com"
    # No Visit Website anchor on this profile.
    assert row.website is None
    assert row.source_url == "https://mynanp.nanp.org/members/?id=81477610"
    assert row.fellowship_level is True


def test_profile_parser_handles_non_string():
    """Defensive: non-string input returns None."""
    assert parse_profile_html(None) is None
    assert parse_profile_html(b"<html>") is None


def test_profile_parser_handles_empty_input():
    """An empty/blocked profile page returns None (no title -> no name)."""
    assert parse_profile_html("") is None
    assert parse_profile_html("<html><body>blocked</body></html>") is None


# ---------------------------------------------------------------------------
# Fellowship-detection rule (the BCHN-specific contract)
# ---------------------------------------------------------------------------

def test_fellowship_defaults_to_false_when_no_bchn():
    """No BCHN credential, no signal — fellowship_level stays False.
    Professional Member without the exam-vetted board cert is the
    canonical False case."""
    assert _detect_fellowship_creds(None) is False
    assert _detect_fellowship_creds("") is False
    assert _detect_fellowship_creds("MS") is False
    assert _detect_fellowship_creds("CNP") is False


def test_fellowship_false_for_cdsp_alone():
    """CDSP is a separate certificate (dietary-supplement specialty),
    NOT a higher tier than BCHN. Per the NANP-specific rule, CDSP alone
    does not trigger fellowship_level."""
    assert _detect_fellowship_creds("CDSP") is False
    assert _detect_fellowship_creds("MS, CDSP") is False
    assert _detect_fellowship_creds("Linda Martinez, CDSP") is False


def test_fellowship_true_for_bchn_in_any_form():
    """BCHN token in any spelling/decoration flips fellowship to True.
    The trademark glyph (R) / HTML entity &reg; and interspersed dots
    all collapse to the bare letter run before testing."""
    assert _detect_fellowship_creds("BCHN") is True
    assert _detect_fellowship_creds("MS, BCHN") is True
    assert _detect_fellowship_creds("BCHN®") is True
    assert _detect_fellowship_creds("BCHN&reg;") is True
    assert _detect_fellowship_creds("B.C.H.N.") is True
    # Case insensitive.
    assert _detect_fellowship_creds("bchn") is True
    # Co-occurring with CDSP (BCHN wins).
    assert _detect_fellowship_creds("Sarah Henderson, MS, BCHN®, CDSP™") is True


def test_fellowship_no_false_match_on_embedded_letters():
    """BCHN as a substring of another token must NOT false-match. The
    word-boundary anchors prevent ``ABCHNYZ`` from triggering."""
    assert _detect_fellowship_creds("ABCHNYZ") is False


def test_profile_row_fellowship_when_bchn_token_in_title():
    """When the custom-field block is absent but the H1 / title carries
    a BCHN credential token, fellowship_level still fires off the
    title-creds fallback. Synthetic profile mirrors NANP's structure."""
    html = (
        "<html><head><title>Dr. Sample Practitioner, BCHN&reg; - "
        "National Association of Nutrition Professionals (NANP)</title></head>"
        "<body>"
        '<td id="tdEmployerName"></td>'
        '<td id="tdWorkPhone"></td>'
        "</body></html>"
    )
    row = parse_profile_html(html, member_id="999")
    assert row is not None
    assert row.fellowship_level is True
    assert row.credentials is not None
    assert "BCHN" in row.credentials


def test_profile_row_fellowship_false_when_bchn_field_says_no():
    """When the custom-field block explicitly says ``BCHN: No`` AND no
    BCHN token is in the title, fellowship_level is False."""
    html = (
        "<html><head><title>Linda Martinez, MS - "
        "National Association of Nutrition Professionals (NANP)</title></head>"
        "<body>"
        '<td id="tdEmployerName"></td>'
        '<td id="tdWorkPhone"></td>'
        '<tr><td><label class="CstmFldLbl">BCHN&reg;:</label></td>'
        '<td class="CstmFldVal">No</td></tr>'
        "</body></html>"
    )
    row = parse_profile_html(html, member_id="998")
    assert row is not None
    assert row.fellowship_level is False
    assert row.credentials == "MS"


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_strip_credentials_paren_form():
    """'(BCHN)' style suffix lifts to credentials."""
    name, creds = _strip_credentials("Jane Doe (BCHN)")
    assert name == "Jane Doe"
    assert creds == "BCHN"


def test_strip_credentials_comma_form():
    """'Name, Cred1, Cred2' splits cleanly."""
    name, creds = _strip_credentials("Sarah Henderson, MS, BCHN")
    assert name == "Sarah Henderson"
    assert "MS" in creds
    assert "BCHN" in creds


def test_strip_credentials_preserves_dr_honorific():
    """'Dr.' is preserved on the name (part of the title, not a trailing
    credential)."""
    name, _ = _strip_credentials("Dr. Jennifer Walsh, BCHN")
    assert name == "Dr. Jennifer Walsh"


def test_strip_credentials_strips_trademark_glyphs():
    """The (R) / (TM) trademark decorations are stripped from the
    credential tokens so downstream comparison ignores them."""
    name, creds = _strip_credentials("Sarah Henderson, BCHN®")
    assert name == "Sarah Henderson"
    assert creds == "BCHN"
    name, creds = _strip_credentials("Maya Gupta, MS, BCHN®, CDSP™")
    assert name == "Maya Gupta"
    assert "BCHN" in creds
    assert "CDSP" in creds


def test_strip_credentials_no_creds():
    name, creds = _strip_credentials("Linda Martinez")
    assert name == "Linda Martinez"
    assert creds is None


def test_normalize_credential_chunk_strips_trademark_glyphs():
    """(R) / ® and (TM) / ™ all collapse to empty so credential
    comparison ignores trademark decorations."""
    assert _normalize_credential_chunk("BCHN®") == "BCHN"
    assert _normalize_credential_chunk("BCHN&reg;") == "BCHN"
    assert _normalize_credential_chunk("CDSP™") == "CDSP"
    assert _normalize_credential_chunk("CDSP&trade;") == "CDSP"
    assert _normalize_credential_chunk("") == ""


def test_strip_html_tags_collapses_blocks_to_space():
    """Block-level tags become spaces; inline tags vanish so anchor-
    delimited comma lists don't get whitespace around their commas."""
    assert _strip_html_tags("<a>BCHN</a>, <a>MS</a>") == "BCHN, MS"
    assert _strip_html_tags("<div>Foo</div><div>Bar</div>") == "Foo Bar"
    assert _strip_html_tags("424 Breckenridge Way<br/>") == "424 Breckenridge Way"
    assert _strip_html_tags(None) == ""
    assert _strip_html_tags("") == ""


def test_normalize_website_adds_scheme():
    assert _normalize_website("lovejoynutrition.com") == "https://lovejoynutrition.com"
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
    """Inference fallback for the common case where country is blank —
    US state names => 'US', Canadian provinces => 'CA', unrecognized =>
    'US' (safe default for the NANP directory)."""
    assert _infer_country_from_state("Utah") == "US"
    assert _infer_country_from_state("West Virginia") == "US"
    assert _infer_country_from_state("Ontario") == "CA"
    assert _infer_country_from_state("British Columbia") == "CA"
    assert _infer_country_from_state("Unknown") == "US"
    assert _infer_country_from_state(None) == "US"


def test_build_source_url_uses_canonical_pattern():
    """URL pattern is the canonical YourMembership detail page on the
    mynanp.nanp.org subdomain."""
    assert _build_source_url("12345") == "https://mynanp.nanp.org/members/?id=12345"
    assert _build_source_url("80176432") == "https://mynanp.nanp.org/members/?id=80176432"
