"""Unit tests for the NCCAOM (National Certification Commission for
Acupuncture and Oriental Medicine) adapter.

NCCAOM publishes its Find-a-Practitioner directory through an ASP.NET
MVC application at ``directory.nccaom.org``. The public form POSTs to
``/FAP/SearchPractitioners`` but the same query parameters work as a
GET against ``/FAP/SearchResultWithoutMap`` (which is what we use).

Fixtures here are real responses captured 2026-05-27 via the Internet
Archive Wayback Machine (the production endpoint is Cloudflare-gated;
the parser is fully decoupled from fetch so the same parse logic runs
against fixtures, archived snapshots, or live HTML alike):

  - nccaom_search_wa.html      — Country=USA & State=WA results page 1
                                 (20 practitioners, 789 total / 40 pages).
                                 Validates: locked invariants, card
                                 parsing, address extraction, cert-code
                                 -> credentials mapping, pagination
                                 metadata, fellowship default True.
  - nccaom_search_name.html    — Name search (FirstName=eileen
                                 LastName=li, 1 result). Validates the
                                 alternative ``citySearchList__content``
                                 card layout and the
                                 ``/FAPPractitionerProfile/<id>=`` link
                                 form (vs the list-page's
                                 ``/FAP/PractitionerDetail?AgencyClientId=<id>=``).
  - nccaom_search_empty.html   — Synthetic "0 Practitioners found" page.
                                 Validates that an empty result page
                                 yields zero rows and total_pages=0.
  - nccaom_profile.html        — Detail page (Willow E. Hammer). Only
                                 used for spot-checking the profile-URL
                                 helpers; the live adapter does not fetch
                                 detail pages (the list view carries
                                 every field we need).

Fellowship rule: every NCCAOM-listed practitioner is board-certified by
definition (NCCAOM IS a credentialing body — the directory only lists
Dipl. Ac. / Dipl. C.H. / Dipl. O.M. / Dipl. ABT certificate holders).
The adapter defaults ``fellowship_level=True`` and only downgrades to
False when the per-card status text carries an inactive marker
(Expired / Inactive / Retired / Recertification Pending / Suspended /
Revoked). The production fixtures only contain "Certified Diplomate"
status, so the downgrade branch is exercised via synthetic fixtures
in the tests below.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.nccaom import (  # noqa: E402
    parse_search_html,
    parse_total_pages,
    parse_total_count,
    _build_source_url,
    _card_to_row,
    _country_iso2,
    _detect_inactive_status,
    _extract_agency_client_id,
    _normalize_phone,
    _normalize_website,
    _parse_address_line,
    _parse_card,
    _strip_credentials,
)


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_wa_returns_full_page_batch():
    """The WA fixture is page 1 of 40 — 20 cards per page. Adapter must
    produce all 20 rows."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    assert len(rows) == 20


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are constant per spec."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    rows += parse_search_html(_load("nccaom_search_name.html"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "NCCAOM"
        assert r.specialties == ["acupuncture_tcm", "holistic_health"]
        # source_url is always populated (it's the dedup key).
        assert r.source_url
        assert r.source_url.startswith(
            "https://directory.nccaom.org/FAP/PractitionerDetail?AgencyClientId="
        )


def test_fellowship_default_true_for_all_listed():
    """NCCAOM is a credentialing body — every practitioner in the public
    directory holds at least one Dipl. designation, so fellowship_level
    defaults True. The WA fixture is all Certified Diplomates."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    assert rows
    for r in rows:
        assert r.fellowship_level is True


def test_spot_check_zhenbo_li_full_fields():
    """First card on the WA page — Zhenbo Li, Dipl. Ac. + Dipl. C.H.,
    full US address. Validates name + address + phone + website +
    credentials extraction."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    li = next(r for r in rows if r.name == "Zhenbo Li")
    assert li.fellowship_level is True
    assert li.phone == "360-984-6489"
    assert li.website == "http://www.lotusacupuncturefertility.com"
    assert li.address1 == "513 N Morrison Rd"
    assert li.city == "Vancouver"
    assert li.state == "WA"
    assert li.country == "US"
    # Both AC and CH certs are listed for this practitioner.
    assert li.credentials
    assert "Dipl. Ac. (NCCAOM)" in li.credentials
    assert "Dipl. C.H. (NCCAOM)" in li.credentials


def test_credentials_combine_multiple_cert_codes():
    """Practitioners with multiple cert badges (AC + CH + OM) get all of
    them in the credentials string. Yufang Xue on the WA page has both
    CH and AC."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    xue = next(r for r in rows if r.name == "Yufang Xue")
    assert xue.credentials
    assert "Dipl. C.H. (NCCAOM)" in xue.credentials
    assert "Dipl. Ac. (NCCAOM)" in xue.credentials


def test_credentials_extract_om_cert():
    """OM Certification (Dipl. O.M.) is a distinct cert-code from AC and
    CH — must be expanded to its full form. Yun Xiao on the WA page has
    the OM cert."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    xiao = next(r for r in rows if r.name == "Yun Xiao")
    assert xiao.credentials
    assert "Dipl. O.M. (NCCAOM)" in xiao.credentials


def test_name_with_trailing_degree_stripped_into_credentials():
    """A practitioner whose displayed name ends in 'L.Ac.' or 'DAOM' has
    that degree pulled into credentials so the name field stays clean.
    Youl Park on the WA page is the canonical case
    (displayed as 'Youl Park L.Ac.')."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    park = next(r for r in rows if r.name.startswith("Youl Park"))
    assert park.name == "Youl Park"
    assert park.credentials
    assert "L.Ac." in park.credentials
    # The cert-code Dipl. Ac. is still preserved.
    assert "Dipl. Ac. (NCCAOM)" in park.credentials


def test_website_not_available_is_none():
    """Cards whose globe block reads 'Not Available' must yield
    website=None (not the literal string). Zhaoyang Chen on the WA page
    has 'Not Available' for both website and other optional fields."""
    rows = parse_search_html(_load("nccaom_search_wa.html"))
    chen = next(r for r in rows if r.name == "Zhaoyang Chen")
    assert chen.website is None
    assert chen.phone == "212-974-2880"  # phone IS available


def test_name_search_layout_parsed():
    """The name-search page uses a DIFFERENT card layout
    (citySearchList__content) and a different link form
    (/FAPPractitionerProfile/<id>=). Same parser must handle it."""
    rows = parse_search_html(_load("nccaom_search_name.html"))
    assert len(rows) == 1
    eileen = rows[0]
    assert eileen.name == "Eileen Li"
    assert eileen.source_url == (
        "https://directory.nccaom.org/FAP/PractitionerDetail?"
        "AgencyClientId=tYzwDnfj8Pg="
    )
    assert eileen.country == "US"
    assert eileen.state == "CT"
    assert eileen.city == "Old Greenwich"
    # Name-search page omits phone — 'Not Available' shows in both phone
    # and website blocks for this entry.
    assert eileen.phone is None
    assert eileen.website is None


def test_empty_result_page_yields_zero_rows():
    """An '0 Practitioners found' page produces no rows and pagination
    reports zero pages — caller's empty-page break stops the walk."""
    html = _load("nccaom_search_empty.html")
    rows = parse_search_html(html)
    assert rows == []
    assert parse_total_pages(html) == 0
    assert parse_total_count(html) == 0


def test_total_pages_and_count_from_real_page():
    """The pagination + result-banner helpers pull WA's 40 pages / 789
    total directly off the fixture."""
    html = _load("nccaom_search_wa.html")
    assert parse_total_pages(html) == 40
    assert parse_total_count(html) == 789


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in the same order — these are the dedup keys for ON CONFLICT
    upsert."""
    html = _load("nccaom_search_wa.html")
    a = parse_search_html(html)
    b = parse_search_html(html)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    # All distinct.
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_source_url_format_uses_canonical_form():
    """Both list-page anchors (?AgencyClientId=<id>) and name-search
    anchors (/FAPPractitionerProfile/<id>) normalize to the same
    canonical ``/FAP/PractitionerDetail?AgencyClientId=<id>`` source_url."""
    rows_list = parse_search_html(_load("nccaom_search_wa.html"))
    rows_name = parse_search_html(_load("nccaom_search_name.html"))
    # Both layouts produce source_urls in the canonical detail form.
    for r in rows_list + rows_name:
        assert "/FAP/PractitionerDetail?AgencyClientId=" in r.source_url


# ---------------------------------------------------------------------------
# Fellowship downgrade tests — synthetic fixtures (production fixtures
# only contain "Certified Diplomate" status)
# ---------------------------------------------------------------------------

_EXPIRED_CARD = """
<div class="result-card__item">
  <div class="info-box">
    <p class="name"><a href="/FAP/PractitionerDetail?AgencyClientId=EXPIREDtest1=" title="View the profile of Jane Expired"> Jane Expired</a></p>
    <p class="gendar">Expired | Female</p>
    <div class="iconic-callout">
      <div class="iconic-callout__item">
        <i class="icon-call-end"></i>
        <p class="copy"><em>555-555-5555</em></p>
      </div>
      <div class="iconic-callout__item">
        <i class="icon-globe"></i>
        <p class="copy"><em>Not Available</em></p>
      </div>
      <div class="iconic-callout__item">
        <i class="icon-location-pin"></i>
        <p class="copy"><em id="addressdata_1">100 Test St, Townville, NY, USA</em></p>
      </div>
    </div>
  </div>
  <div class="cert-box">
    <div class="cert-box__item"><div class="content"><p class="copy">AC Certification</p></div></div>
  </div>
</div>
"""


def test_fellowship_false_for_expired_status():
    """An 'Expired' status downgrades fellowship_level to False even
    though the practitioner still has a cert-code badge."""
    rows = parse_search_html(_EXPIRED_CARD)
    assert len(rows) == 1
    assert rows[0].name == "Jane Expired"
    assert rows[0].fellowship_level is False


def test_fellowship_false_for_inactive_status():
    """'Inactive' status downgrades fellowship_level."""
    html = _EXPIRED_CARD.replace(">Expired |", ">Inactive |").replace(
        "EXPIREDtest1=", "INACTIVEtest1="
    )
    rows = parse_search_html(html)
    assert rows[0].fellowship_level is False


def test_fellowship_false_for_retired_status():
    """'Retired' status downgrades fellowship_level."""
    html = _EXPIRED_CARD.replace(">Expired |", ">Retired |").replace(
        "EXPIREDtest1=", "RETIREDtest1="
    )
    rows = parse_search_html(html)
    assert rows[0].fellowship_level is False


def test_fellowship_false_for_recertification_pending_status():
    """'Recertification Pending' downgrades fellowship_level (the
    practitioner is in the middle of renewal and not currently
    actively certified)."""
    html = _EXPIRED_CARD.replace(
        ">Expired |", ">Recertification Pending |"
    ).replace("EXPIREDtest1=", "RECERTtest1=")
    rows = parse_search_html(html)
    assert rows[0].fellowship_level is False


def test_fellowship_true_for_certified_diplomate_status():
    """The canonical production status — 'Certified Diplomate' — keeps
    fellowship_level=True."""
    html = _EXPIRED_CARD.replace(
        ">Expired |", ">Certified Diplomate |"
    ).replace("EXPIREDtest1=", "CERTIFIEDtest1=")
    rows = parse_search_html(html)
    assert rows[0].fellowship_level is True


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_extract_agency_client_id_from_list_page_link():
    """The list page uses ?AgencyClientId=<id>= query-param form."""
    assert (
        _extract_agency_client_id(
            "/FAP/PractitionerDetail?AgencyClientId=rTjSagQVHkY="
        )
        == "rTjSagQVHkY="
    )


def test_extract_agency_client_id_from_name_search_link():
    """The name-search page uses /FAPPractitionerProfile/<id>= path
    form. The opaque ID is identical."""
    assert (
        _extract_agency_client_id("/FAPPractitionerProfile/tYzwDnfj8Pg=")
        == "tYzwDnfj8Pg="
    )


def test_extract_agency_client_id_handles_absent_id():
    """No href / no recognizable ID -> None."""
    assert _extract_agency_client_id("") is None
    assert _extract_agency_client_id("/some/other/path") is None
    assert _extract_agency_client_id(None) is None  # type: ignore[arg-type]


def test_build_source_url_canonical_form():
    """source_url is always the bare ?AgencyClientId=<id> form."""
    assert _build_source_url("abc123=") == (
        "https://directory.nccaom.org/FAP/PractitionerDetail?AgencyClientId=abc123="
    )
    assert _build_source_url(None) is None
    assert _build_source_url("") is None


def test_detect_inactive_status_matrix():
    """Status header detection — case-insensitive, only the segment
    BEFORE the ``|`` (so 'Expired | Female' matches Expired)."""
    assert _detect_inactive_status("Certified Diplomate | Female") is False
    assert _detect_inactive_status("Expired | Female") is True
    assert _detect_inactive_status("Inactive | Male") is True
    assert _detect_inactive_status("Retired | Female") is True
    assert _detect_inactive_status("Recertification Pending | Male") is True
    assert _detect_inactive_status("Suspended | Female") is True
    assert _detect_inactive_status("Revoked | Male") is True
    # Substring match also OK (defensive against future status variants).
    assert _detect_inactive_status("Expired Certification | Female") is True
    # Empty / None safe.
    assert _detect_inactive_status(None) is False
    assert _detect_inactive_status("") is False


def test_country_iso2_canonicalizes_common_names():
    assert _country_iso2("USA") == "US"
    assert _country_iso2("United States") == "US"
    assert _country_iso2("Canada") == "CA"
    assert _country_iso2("United Kingdom") == "GB"
    # 2-letter codes pass through.
    assert _country_iso2("US") == "US"
    assert _country_iso2("CA") == "CA"
    assert _country_iso2(None) is None
    assert _country_iso2("Atlantis") is None


def test_normalize_website_handles_sentinels():
    """'Not Available' / 'N/A' / empty -> None; bare domain gets https."""
    assert _normalize_website(None) is None
    assert _normalize_website("") is None
    assert _normalize_website("Not Available") is None
    assert _normalize_website("N/A") is None
    assert _normalize_website("http://example.com") == "http://example.com"
    assert _normalize_website("https://example.com") == "https://example.com"
    assert _normalize_website("example.com") == "https://example.com"


def test_normalize_phone_handles_sentinels():
    """'Not Available' phone => None; otherwise trimmed pass-through."""
    assert _normalize_phone(None) is None
    assert _normalize_phone("") is None
    assert _normalize_phone("Not Available") is None
    assert _normalize_phone("  555-123-4567  ") == "555-123-4567"


def test_strip_credentials_handles_trailing_degree_no_comma():
    """'Youl Park L.Ac.' -> name='Youl Park', creds='L.Ac.'. The NCCAOM
    directory frequently lists the degree this way (no comma)."""
    name, creds = _strip_credentials("Youl Park L.Ac.")
    assert name == "Youl Park"
    assert creds == "L.Ac."


def test_strip_credentials_handles_trailing_degree_with_comma():
    """Comma form is also supported."""
    name, creds = _strip_credentials("Jane Doe, DAOM")
    assert name == "Jane Doe"
    assert creds == "DAOM"


def test_strip_credentials_no_creds():
    name, creds = _strip_credentials("Zhenbo Li")
    assert name == "Zhenbo Li"
    assert creds is None


def test_parse_address_line_us_full():
    """Full US address: 'street, city, state postal, country'."""
    out = _parse_address_line("143 Piccadilly Downs , Lynbrook, NY 11563-3117, USA")
    assert out["address1"] == "143 Piccadilly Downs"
    assert out["city"] == "Lynbrook"
    assert out["state"] == "NY"
    assert out["postal"] == "11563-3117"
    assert out["country"] == "USA"


def test_parse_address_line_us_no_postal():
    """The WA list page omits postal for many rows."""
    out = _parse_address_line("513 N Morrison Rd , Vancouver, WA, USA")
    assert out["address1"] == "513 N Morrison Rd"
    assert out["city"] == "Vancouver"
    assert out["state"] == "WA"
    assert out.get("postal") is None
    assert out["country"] == "USA"


def test_parse_address_line_no_street():
    """When the practitioner only listed city/state/country (no street),
    everything still parses without crashing."""
    out = _parse_address_line("Seattle, WA, USA")
    assert out.get("address1") is None
    assert out["city"] == "Seattle"
    assert out["state"] == "WA"
    assert out["country"] == "USA"


def test_parse_address_line_empty_or_garbage():
    assert _parse_address_line("") == {}
    # Single-token addresses without a comma are left as address1.
    out = _parse_address_line("Unknown")
    assert out.get("address1") == "Unknown"


def test_parse_card_extracts_all_fields():
    """End-to-end: a single result-card chunk parses to the expected
    fields dict."""
    card = _EXPIRED_CARD
    parsed = _parse_card(card)
    assert parsed is not None
    assert parsed["name"] == "Jane Expired"
    assert "EXPIREDtest1=" in parsed["href"]
    assert parsed["status"].startswith("Expired")
    assert parsed["phone"] == "555-555-5555"
    assert parsed["address"] == "100 Test St, Townville, NY, USA"
    assert parsed["cert_codes"] == ["AC"]


def test_parse_card_returns_none_when_no_anchor():
    """Defensive: a chunk with no practitioner anchor at all -> None."""
    assert _parse_card("<div>no anchor here</div>") is None
    assert _parse_card("") is None


def test_card_to_row_returns_none_without_agency_id():
    """A card with a name but no AgencyClientId in the href cannot
    produce a stable source_url; we drop it rather than synthesize one
    (re-runs would otherwise create duplicates)."""
    card = {
        "name": "Jane Test",
        "href": "/somewhere/else",
        "cert_codes": ["AC"],
        "address": "1 Test St, Test, NY, USA",
    }
    assert _card_to_row(card) is None


def test_parser_handles_non_string_input():
    """Defensive: a non-string payload (None, bytes, dict) yields an
    empty list rather than crashing."""
    assert parse_search_html(None) == []  # type: ignore[arg-type]
    assert parse_search_html(b"<html></html>") == []  # type: ignore[arg-type]
    assert parse_search_html({}) == []  # type: ignore[arg-type]
