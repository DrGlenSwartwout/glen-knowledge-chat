"""Unit tests for the NANP (National Association of Nutrition Professionals)
adapter.

NANP runs the public website on WordPress at nanp.org but the member
directory itself lives on a separate YourMembership Classic AMS subdomain
at ``mynanp.nanp.org``. The "Find a Practitioner" link on the WP site
hands off to ``mynanp.nanp.org/search/custom.asp?id=7551`` (the search
form), which POSTs to ``/search/newsearch.asp`` for results, which link
to ``/profile/?ID=<numeric>`` for per-member detail pages.

Fixtures captured / synthesized 2026-05-27:

- nanp_search_form.html
    REAL fixture — the live ``/search/custom.asp?id=7551`` page (captured
    via Wayback 2025-11-04; the live URL was Cloudflare-gated at scrape
    time). Carries the filter contract (cdlMemberTypeID=1705148, BCHN /
    CDSP custom-field flags) but is NOT a results page — used only to
    validate discovery-layer expectations.

- nanp_search_results_page_1.html
    Synthesized YourMembership Classic results page reflecting NANP's
    actual published roster shape: 22 members spanning US states, Canada,
    UK, Germany, Italy, Poland, Brazil; mix of BCHN-credentialed and
    non-BCHN entries; one BCHN+CDSP combo; one blank-name opt-out row
    that must be skipped. The YM Classic results template
    (``<table class="search-results">`` with name-anchor / city / state /
    country columns and a pagination block) is platform-stable and
    documented across the dozens of association sites using the AMS.

- nanp_member_profile_bchn.html, nanp_member_profile_professional.html,
  nanp_member_profile_intl_canada.html
    Synthesized YourMembership profile pages. The BCHN fixture has the
    BCHN custom-field set 'Yes' AND the credential in the H1; the
    professional fixture has both signals negative; the Canadian fixture
    has BCHN=Yes via custom field, full Canadian-postal address, AND
    international phone.

The BCHN fellowship rule is THE key spec deliverable for this adapter —
the tests below exhaust the matrix:

  1. Custom-field row 'BCHN' = 'Yes'         -> fellowship_level=True
  2. H1 credential token 'BCHN' (with ®/&reg;/dotted variants)
                                              -> fellowship_level=True
  3. Stub-name (list page) carries BCHN even if profile body is sparse
                                              -> fellowship_level=True
  4. Custom-field 'BCHN' = 'No' AND no token   -> fellowship_level=False
  5. CDSP™ alone (no BCHN)                    -> fellowship_level=False
  6. Professional Member with no creds         -> fellowship_level=False
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.nanp import (  # noqa: E402
    LOCKED_SPECIALTIES,
    PROFILE_URL,
    _build_source_url,
    _country_iso2,
    _extract_profile_id,
    _has_bchn,
    _normalize_credential_chunk,
    _normalize_website,
    _parse_address_block,
    _strip_credentials,
    parse_member_profile_html,
    parse_search_results_html,
    parse_total_pages,
)


def _read(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Stage 1 — search-results page parser
# ---------------------------------------------------------------------------

def test_search_results_emits_one_stub_per_valid_row():
    """Page 1 of the synthesized results fixture has 22 <tr> rows under
    the results table, but one is a blank-name opt-out (must be skipped),
    so the parser emits 21 stubs."""
    stubs = parse_search_results_html(_read("nanp_search_results_page_1.html"))
    assert len(stubs) == 21


def test_search_results_extract_profile_id_and_source_url():
    """Each stub carries the YM numeric profile id and a canonical
    source_url derived from it. The first stub is Sarah Henderson
    (ID=12345678)."""
    stubs = parse_search_results_html(_read("nanp_search_results_page_1.html"))
    s = stubs[0]
    assert s["profile_id"] == "12345678"
    assert s["source_url"] == "https://mynanp.nanp.org/profile/?ID=12345678"
    assert s["name"].startswith("Sarah Henderson")


def test_search_results_extract_geo_columns():
    """City / State / Country come from the three <td>s after the name
    cell. The 5th US row in the synthesized fixture is Thomas Becker in
    Minneapolis MN."""
    stubs = parse_search_results_html(_read("nanp_search_results_page_1.html"))
    becker = next(s for s in stubs if "Thomas Becker" in s["name"])
    assert becker["city"] == "Minneapolis"
    assert becker["state"] == "MN"
    assert becker["country_raw"] == "United States"


def test_search_results_international_state_blank():
    """International rows have a blank state cell (the directory only
    asks for state on US entries). Anna Rossi in Rome has state=None."""
    stubs = parse_search_results_html(_read("nanp_search_results_page_1.html"))
    rossi = next(s for s in stubs if "Anna Rossi" in s["name"])
    assert rossi["city"] == "Rome"
    assert rossi["state"] is None
    assert rossi["country_raw"] == "Italy"


def test_search_results_skips_blank_name_row():
    """A profile-photo-only / opt-out anchor with empty link text must
    not produce a stub — that row's ID=12345687 must NOT appear in the
    stub list."""
    stubs = parse_search_results_html(_read("nanp_search_results_page_1.html"))
    assert all(s["profile_id"] != "12345687" for s in stubs)


def test_search_results_total_pages():
    """Pagination block: three numbered pages -> total_pages=3."""
    html = _read("nanp_search_results_page_1.html")
    assert parse_total_pages(html) == 3


def test_search_results_empty_html_returns_empty():
    """No table at all -> empty stub list. Defensive against an unusual
    Cloudflare interstitial."""
    assert parse_search_results_html("<html><body>blocked</body></html>") == []


def test_search_results_dedups_within_page():
    """A page with two anchors pointing at the same profile_id only
    emits one stub (defensive — shouldn't happen on real YM Classic
    pages but the parser handles it cleanly)."""
    html = """
    <table class="search-results">
      <tr><td><a href="/profile/?ID=999">Alice Smith</a></td><td>X</td></tr>
      <tr><td><a href="/profile/?ID=999">Alice Smith again</a></td><td>X</td></tr>
    </table>"""
    stubs = parse_search_results_html(html)
    assert len(stubs) == 1
    assert stubs[0]["profile_id"] == "999"


# ---------------------------------------------------------------------------
# Stage 2 — profile-page parser (full row)
# ---------------------------------------------------------------------------

def test_profile_bchn_full_row():
    """The canonical BCHN-credentialed profile yields a fully-populated
    row with fellowship_level=True. Locked invariants checked too."""
    row = parse_member_profile_html(
        _read("nanp_member_profile_bchn.html"),
        profile_id="12345678",
    )
    assert row is not None
    assert row.tier == "org_member"
    assert row.source_org == "NANP"
    assert row.specialties == ["nutrition", "holistic_health"]
    assert row.source_url == "https://mynanp.nanp.org/profile/?ID=12345678"
    assert row.name == "Sarah Henderson"
    assert row.practice_name == "Boulder Holistic Nutrition LLC"
    assert row.address1 == "1234 Pearl Street, Suite 200"
    assert row.city == "Boulder"
    assert row.state == "CO"
    assert row.postal == "80302"
    assert row.country == "US"
    assert row.phone == "303-555-0100"
    assert row.email == "sarah@boulderholisticnutrition.com"
    assert row.website == "http://www.boulderholisticnutrition.com"
    # Fellowship-level via the canonical 'BCHN: Yes' custom field.
    assert row.fellowship_level is True
    # Credentials carry BCHN, with the ® glyph stripped.
    assert row.credentials is not None
    assert "BCHN" in row.credentials


def test_profile_professional_member_not_fellowship():
    """A Professional Member with no BCHN credential at all — the
    canonical fellowship_level=False case. Custom-field row 'BCHN: No'
    is the negative signal."""
    row = parse_member_profile_html(
        _read("nanp_member_profile_professional.html"),
        profile_id="12345680",
    )
    assert row is not None
    assert row.name == "Linda Martinez"
    assert row.fellowship_level is False
    assert row.city == "Austin"
    assert row.state == "TX"
    assert row.postal == "78704"
    assert row.country == "US"
    # The H1 still has 'MS' so credentials are present but BCHN is not.
    assert row.credentials == "MS"


def test_profile_canada_intl_address_with_bchn():
    """Canadian profile — BCHN credential present (Yes), postal code
    in Canadian format ('M5V 3A8'), state as full province name."""
    row = parse_member_profile_html(
        _read("nanp_member_profile_intl_canada.html"),
        profile_id="12345683",
    )
    assert row is not None
    assert row.name == "Priya Sharma"
    assert row.city == "Toronto"
    assert row.state == "Ontario"
    assert row.postal == "M5V 3A8"
    assert row.country == "CA"
    assert row.fellowship_level is True
    # BCHN is in the H1 — credentials reflect that. The CDSP custom-field
    # is also set on this profile but lives in the secondary table, not
    # the H1, so it doesn't surface in the credentials string. The
    # fellowship_level signal would be True even if CDSP were also in
    # the H1 (BCHN takes precedence per spec).
    assert "BCHN" in (row.credentials or "")


def test_profile_pulls_profile_id_from_canonical_when_not_passed():
    """When profile_id arg is omitted, the parser falls back to og:url /
    canonical link on the page."""
    row = parse_member_profile_html(_read("nanp_member_profile_bchn.html"))
    assert row is not None
    assert row.source_url == "https://mynanp.nanp.org/profile/?ID=12345678"


def test_profile_uses_stub_country_when_block_omits_it():
    """A 2-line address block lacks the country line; the list-page
    stub's country_raw must backfill. Synthetic: bare profile body with
    a country-less address."""
    html = """
    <html><head>
    <meta property="og:url" content="https://mynanp.nanp.org/profile/?ID=42">
    </head><body><div id="SpContent_Container">
    <h1>Jane Roe, BCHN&reg;</h1>
    <table><tr><th>Address:</th><td>10 Main St<br>Vancouver, British Columbia V6B 1A1</td></tr></table>
    </div></body></html>"""
    stub = {"profile_id": "42", "name": "Jane Roe, BCHN®",
            "city": "Vancouver", "state": "British Columbia",
            "country_raw": "Canada"}
    row = parse_member_profile_html(html, profile_id="42", stub=stub)
    assert row is not None
    assert row.country == "CA"
    assert row.postal == "V6B 1A1"
    assert row.fellowship_level is True


def test_profile_falls_back_to_stub_name_when_h1_missing():
    """Defensive: a profile page that didn't render an H1 (rare YM
    Classic edge case) still produces a row when the stub has the name."""
    html = """
    <html><body><div id="SpContent_Container">
    <table><tr><th>Email:</th><td>x@y.com</td></tr></table>
    </div></body></html>"""
    stub = {"profile_id": "777", "name": "Pat Smith, BCHN®",
            "city": "X", "state": "CA", "country_raw": "United States"}
    row = parse_member_profile_html(html, profile_id="777", stub=stub)
    assert row is not None
    assert row.name == "Pat Smith"
    assert row.fellowship_level is True


def test_profile_returns_none_when_no_name_or_id():
    """A page with no H1, no canonical, no stub, no profile_id arg
    cannot produce a usable row — must return None rather than emit
    garbage."""
    html = "<html><body><p>nothing</p></body></html>"
    assert parse_member_profile_html(html) is None


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_extract_profile_id_handles_param_variants():
    """Both ``?ID=N`` and ``&ID=N`` are accepted; case-insensitive."""
    assert _extract_profile_id("/profile/?ID=12345678") == "12345678"
    assert _extract_profile_id("/profile/?lang=en&ID=42") == "42"
    assert _extract_profile_id("/profile/?id=99") == "99"
    assert _extract_profile_id("/profile/") is None
    assert _extract_profile_id("") is None
    assert _extract_profile_id(None) is None


def test_build_source_url_is_canonical():
    """source_url is always the bare /profile/?ID=<n> form — no extra
    filter / breadcrumb params that would drift the dedup key."""
    assert _build_source_url("999") == "https://mynanp.nanp.org/profile/?ID=999"
    assert _build_source_url("12345678").startswith(PROFILE_URL)


def test_has_bchn_matches_glyph_and_entity_variants():
    """The fellowship trigger must match BCHN written every way the
    directory might surface it — bare, with ®, with &reg;, dotted."""
    assert _has_bchn("BCHN") is True
    assert _has_bchn("BCHN®") is True
    assert _has_bchn("BCHN&reg;") is True
    assert _has_bchn("B.C.H.N.") is True
    assert _has_bchn("Sarah Henderson, MS, BCHN®, CDSP™") is True
    # Case insensitive (rare but possible).
    assert _has_bchn("bchn") is True
    # CDSP alone must NOT count — spec: only BCHN flips fellowship.
    assert _has_bchn("Linda Martinez, CDSP") is False
    assert _has_bchn("Linda Martinez, MS") is False
    assert _has_bchn("") is False
    assert _has_bchn(None) is False
    # Substring of another token must not false-match (e.g. embedded
    # alphabetic context). \b boundary anchors prevent this.
    assert _has_bchn("ABCHNYZ") is False


def test_normalize_credential_chunk_strips_trademark_glyphs():
    """&reg; / ® and &trade; / ™ all collapse to empty so credential
    comparison ignores trademark decorations."""
    assert _normalize_credential_chunk("BCHN®") == "BCHN"
    assert _normalize_credential_chunk("BCHN&reg;") == "BCHN"
    assert _normalize_credential_chunk("CDSP™") == "CDSP"
    assert _normalize_credential_chunk("CDSP&trade;") == "CDSP"
    assert _normalize_credential_chunk("") == ""


def test_strip_credentials_handles_h1_form():
    """The YM Classic H1 is 'Name, Cred1, Cred2, ...'. The ®/™ glyphs
    are stripped before the split."""
    name, creds = _strip_credentials("Sarah Henderson, BCHN®")
    assert name == "Sarah Henderson"
    assert creds == "BCHN"

    name, creds = _strip_credentials("Maya Gupta, MS, BCHN®, CDSP™")
    assert name == "Maya Gupta"
    assert "MS" in creds
    assert "BCHN" in creds
    assert "CDSP" in creds


def test_strip_credentials_preserves_honorific():
    name, _ = _strip_credentials("Dr. Jennifer Walsh, BCHN®")
    assert name == "Dr. Jennifer Walsh"


def test_strip_credentials_no_creds():
    name, creds = _strip_credentials("Linda Martinez")
    assert name == "Linda Martinez"
    assert creds is None


def test_country_iso2_canonicalizes_directory_spread():
    """Cover the country values NANP's directory actually surfaces:
    US dominant, plus Canada / UK / EU / Asia entries."""
    assert _country_iso2("United States") == "US"
    assert _country_iso2("Canada") == "CA"
    assert _country_iso2("United Kingdom") == "GB"
    assert _country_iso2("England") == "GB"
    assert _country_iso2("Germany") == "DE"
    assert _country_iso2("Brazil") == "BR"
    assert _country_iso2("japan") == "JP"
    assert _country_iso2("Australia") == "AU"
    assert _country_iso2("Atlantis") is None
    assert _country_iso2(None) is None


def test_normalize_website_adds_scheme():
    """Bare-domain anchor text gets an https:// prefix; already-scheme'd
    URLs pass through untouched."""
    assert _normalize_website("torontowellness.ca") == "https://torontowellness.ca"
    assert _normalize_website("lmnutrition.com") == "https://lmnutrition.com"
    assert _normalize_website("https://x.com/") == "https://x.com/"
    assert _normalize_website("http://x.com") == "http://x.com"
    assert _normalize_website(None) is None
    assert _normalize_website("") is None


def test_parse_address_block_three_line_us():
    """Canonical three-line US address block from the profile page."""
    address1, city, state, postal, country = _parse_address_block(
        "1234 Pearl Street, Suite 200\nBoulder, CO 80302\nUnited States"
    )
    assert address1 == "1234 Pearl Street, Suite 200"
    assert city == "Boulder"
    assert state == "CO"
    assert postal == "80302"
    assert country == "US"


def test_parse_address_block_three_line_canada():
    """Canadian address block — postal is ANA NAN with a space; state is
    the full province name."""
    address1, city, state, postal, country = _parse_address_block(
        "350 King Street West, Suite 410\nToronto, Ontario M5V 3A8\nCanada"
    )
    assert address1 == "350 King Street West, Suite 410"
    assert city == "Toronto"
    assert state == "Ontario"
    assert postal == "M5V 3A8"
    assert country == "CA"


def test_parse_address_block_two_line_omits_country():
    """A 2-line block (no country line) yields country=None — the
    profile parser then backfills from the list-page stub."""
    address1, city, state, postal, country = _parse_address_block(
        "500 South Lamar Boulevard\nAustin, TX 78704"
    )
    assert address1 == "500 South Lamar Boulevard"
    assert city == "Austin"
    assert state == "TX"
    assert postal == "78704"
    assert country is None


def test_parse_address_block_empty():
    """Empty input -> all-None tuple, no crash."""
    assert _parse_address_block("") == (None, None, None, None, None)
    assert _parse_address_block(None) == (None, None, None, None, None)


def test_locked_specialties_value():
    """Defensive: the LOCKED_SPECIALTIES constant must never drift —
    downstream cross-adapter filters key off this exact list."""
    assert LOCKED_SPECIALTIES == ["nutrition", "holistic_health"]


def test_source_url_stable_across_reruns():
    """Re-parsing the same fixture twice yields identical source_urls in
    identical order — these are the dedup keys for ON CONFLICT upsert."""
    html = _read("nanp_search_results_page_1.html")
    a = parse_search_results_html(html)
    b = parse_search_results_html(html)
    assert [s["source_url"] for s in a] == [s["source_url"] for s in b]
    # All distinct (no two practitioners share a profile_id).
    urls = [s["source_url"] for s in a]
    assert len(urls) == len(set(urls))
