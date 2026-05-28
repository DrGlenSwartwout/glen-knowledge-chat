"""Unit tests for the NCBAHM (formerly NCCAOM) "Find a Practitioner"
adapter.

Background
----------
The organization rebranded in 2026 from NCCAOM (National Certification
Commission for Acupuncture and Oriental Medicine) to NCBAHM (National
Board for Acupuncture and Herbal Medicine). The directory now lives at
``https://directory.ncbahm.org/``. The module/file is still named
``nccaom.py`` and emits ``source_org="NCCAOM"`` to keep the database +
orchestrator + UI key stable across the rebrand; only the credential
strings updated to ``Dipl. <X> (NCBAHM)``.

Fixtures
--------
Five real captured-live fixtures (2026-05-27):

  - ncbahm_search_wa_live.html      — WA page 1 (20 cards, 800 total, 40 pages)
  - ncbahm_search_ny_live.html      — NY page 1 (20 cards, 1198 total, 60 pages)
  - ncbahm_search_ca_live.html      — CA page 1 (20 cards, 2007 total, 101 pages)
  - ncbahm_search_hi_live.html      — HI page 1 (20 cards, 151 total, 8 pages)
  - ncbahm_search_hi_p2_live.html   — HI page 2 (pagination round-trip check)

Each fixture exercises:
  - 20 cards per page (back-end-enforced PageSize=20)
  - Locked invariants (tier, source_org, specialties, fellowship_level)
  - hdnlastpage parser
  - "<N> Practitioners found" total-count parser
  - Phone / address / credentials extraction per row
  - Multi-cert practitioners (NY: Adam J. French has AC + CH + OM)
  - "Not Available" sentinel handling (HI: Aaron Bullington has no
    phone or address)

Fellowship rule: every NCBAHM-listed practitioner is board-certified by
definition (NCBAHM IS a credentialing body — the directory only lists
Dipl. <X> certificate holders). Default ``fellowship_level=True``; only
downgraded to False when the per-card status carries an inactive marker
(Expired / Inactive / Retired / Recertification Pending / Suspended /
Revoked). Production fixtures only contain "Certified Diplomate" so
the downgrade branch is exercised via synthetic fixtures below.
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


# Per-fixture expected pagination metadata (verified at capture time).
_EXPECTED = {
    "ncbahm_search_wa_live.html": {"rows": 20, "total": 800,  "pages": 40,  "state": "WA"},
    "ncbahm_search_ny_live.html": {"rows": 20, "total": 1198, "pages": 60,  "state": "NY"},
    "ncbahm_search_ca_live.html": {"rows": 20, "total": 2007, "pages": 101, "state": "CA"},
    "ncbahm_search_hi_live.html": {"rows": 20, "total": 151,  "pages": 8,   "state": "HI"},
    "ncbahm_search_hi_p2_live.html": {"rows": 20, "total": 151,  "pages": 8, "state": "HI"},
}


# ---------------------------------------------------------------------------
# Fixture sweep — each live fixture must parse to 20 rows with locked
# invariants
# ---------------------------------------------------------------------------

def test_each_live_fixture_parses_to_twenty_rows():
    """Every captured live fixture is a single result page (PageSize=20)
    — the parser must return all 20 cards from each."""
    for fname, meta in _EXPECTED.items():
        rows = parse_search_html(_load(fname))
        assert len(rows) == meta["rows"], (
            f"{fname}: expected {meta['rows']} rows, got {len(rows)}"
        )


def test_each_live_fixture_pagination_metadata():
    """hdnlastpage + 'N Practitioners found' parse from every live fixture."""
    for fname, meta in _EXPECTED.items():
        html = _load(fname)
        assert parse_total_pages(html) == meta["pages"], fname
        assert parse_total_count(html) == meta["total"], fname


def test_all_live_rows_carry_locked_invariants():
    """tier / source_org / specialties / source_url are constant per spec
    across every live fixture row."""
    for fname in _EXPECTED:
        rows = parse_search_html(_load(fname))
        assert rows, fname
        for r in rows:
            assert r.tier == "org_member", fname
            # NCBAHM is the new brand; source_org keeps the historical
            # NCCAOM string (see module docstring).
            assert r.source_org == "NCCAOM", fname
            assert r.specialties == ["acupuncture_tcm", "holistic_health"], fname
            assert r.source_url, fname
            assert r.source_url.startswith(
                "https://directory.ncbahm.org/FAP/PractitionerDetail?AgencyClientId="
            ), fname


def test_all_live_rows_default_fellowship_true():
    """Every NCBAHM listing is board-certified by definition. All live
    fixtures contain only 'Certified Diplomate' status, so all rows must
    default fellowship_level=True."""
    for fname in _EXPECTED:
        rows = parse_search_html(_load(fname))
        assert rows, fname
        for r in rows:
            assert r.fellowship_level is True, f"{fname}: {r.name}"


def test_no_duplicate_source_urls_within_a_page():
    """A single page must not emit duplicate source_urls."""
    for fname in _EXPECTED:
        rows = parse_search_html(_load(fname))
        urls = [r.source_url for r in rows]
        assert len(urls) == len(set(urls)), fname


# ---------------------------------------------------------------------------
# Per-fixture spot-check rows — at least 2 sample rows per fixture
# verified end-to-end
# ---------------------------------------------------------------------------

def test_wa_spot_check_abigail_coble_hoehne():
    """First row on WA page 1, descending-sort: Abigail Coble Hoehne."""
    rows = parse_search_html(_load("ncbahm_search_wa_live.html"))
    abigail = next(r for r in rows if r.name == "Abigail Coble Hoehne")
    assert abigail.phone == "206-920-7979"
    assert abigail.address1 == "524 N 67th St"
    assert abigail.city == "Seattle"
    assert abigail.state == "WA"
    assert abigail.country == "US"
    assert abigail.credentials and "Dipl. Ac. (NCBAHM)" in abigail.credentials
    assert abigail.fellowship_level is True


def test_wa_spot_check_multi_cert_allen_sayigh():
    """WA card #14 — Allen Adnan Sayigh holds AC + CH; credentials must
    combine both into the canonical Dipl. <X> (NCBAHM) strings."""
    rows = parse_search_html(_load("ncbahm_search_wa_live.html"))
    allen = next(r for r in rows if r.name == "Allen Adnan Sayigh")
    assert allen.credentials
    assert "Dipl. Ac. (NCBAHM)" in allen.credentials
    assert "Dipl. CH. (NCBAHM)" in allen.credentials


def test_ny_spot_check_a_li_song():
    """First row on NY page 1 (descending-sort): A Li Song."""
    rows = parse_search_html(_load("ncbahm_search_ny_live.html"))
    song = next(r for r in rows if r.name == "A Li Song")
    assert song.phone == "917-807-0898"
    assert song.address1 == "6801 Jericho Tpke"
    assert song.city == "Syosset"
    assert song.state == "NY"
    assert song.country == "US"
    assert song.credentials and "Dipl." in song.credentials


def test_ny_spot_check_adam_french_three_certs_plus_trailing_lac():
    """NY card #6 — Adam J. French L.Ac. holds AC + CH + OM AND has the
    L.Ac. degree trailing on his displayed name. Credentials must
    include all three Dipl. strings AND the trailing L.Ac. degree;
    name field must be cleaned to 'Adam J. French'."""
    rows = parse_search_html(_load("ncbahm_search_ny_live.html"))
    adam = next(r for r in rows if r.name == "Adam J. French")
    assert adam.credentials
    assert "Dipl. Ac. (NCBAHM)" in adam.credentials
    assert "Dipl. CH. (NCBAHM)" in adam.credentials
    assert "Dipl. OM. (NCBAHM)" in adam.credentials
    assert "L.Ac." in adam.credentials
    assert adam.website == "http://adamfrenchlac.com"


def test_ca_spot_check_a_young_kim():
    """First row on CA page 1: A Young Kim."""
    rows = parse_search_html(_load("ncbahm_search_ca_live.html"))
    kim = next(r for r in rows if r.name == "A Young Kim")
    assert kim.phone == "661-251-5930"
    assert kim.address1 == "18261 Soledad Canyon Rd"
    assert kim.city == "Canyon Country"
    assert kim.state == "CA"
    assert kim.country == "US"
    assert kim.credentials and "Dipl." in kim.credentials


def test_ca_spot_check_multi_cert_abbey_tucker_seiden():
    """CA card #4 — Abbey Tucker Seiden holds AC + OM."""
    rows = parse_search_html(_load("ncbahm_search_ca_live.html"))
    abbey = next(r for r in rows if r.name == "Abbey Tucker Seiden")
    assert abbey.credentials
    assert "Dipl. Ac. (NCBAHM)" in abbey.credentials
    assert "Dipl. OM. (NCBAHM)" in abbey.credentials


def test_hi_spot_check_aaron_bullington_no_phone_no_address():
    """HI card #1 — Aaron Bullington has 'Not Available' for both phone
    AND address. Must emit phone=None, address fields=None (NOT the
    literal 'Not Available' string), but website IS available."""
    rows = parse_search_html(_load("ncbahm_search_hi_live.html"))
    aaron = next(r for r in rows if r.name == "Aaron Bullington")
    assert aaron.phone is None
    assert aaron.address1 is None
    assert aaron.city is None
    assert aaron.state is None
    assert aaron.website == "http://www.bodyrealms.com"
    # Cert is OM for this practitioner.
    assert aaron.credentials and "Dipl. OM. (NCBAHM)" in aaron.credentials


def test_hi_spot_check_aaron_tsutomo_ishigo():
    """HI card #2 — Aaron Tsutomo Ishigo with full address + phone +
    AC cert, NO website (the 'Not Available' globe sentinel)."""
    rows = parse_search_html(_load("ncbahm_search_hi_live.html"))
    aaron = next(r for r in rows if r.name == "Aaron Tsutomo Ishigo")
    assert aaron.phone == "808-934-9858"
    assert aaron.address1 == "82 Keaa St"
    assert aaron.city == "Hilo"
    assert aaron.state == "HI"
    assert aaron.country == "US"
    assert aaron.website is None  # globe block is "Not Available"
    assert aaron.credentials and "Dipl. Ac. (NCBAHM)" in aaron.credentials


def test_hi_pagination_round_trip_p1_and_p2_both_parse():
    """Both HI page 1 and page 2 fixtures must parse to 20 rows with the
    same total + last-page metadata (the back-end emits the same
    pagination header on every page within a search)."""
    p1_html = _load("ncbahm_search_hi_live.html")
    p2_html = _load("ncbahm_search_hi_p2_live.html")
    p1 = parse_search_html(p1_html)
    p2 = parse_search_html(p2_html)
    assert len(p1) == 20
    assert len(p2) == 20
    assert parse_total_pages(p1_html) == parse_total_pages(p2_html) == 8
    assert parse_total_count(p1_html) == parse_total_count(p2_html) == 151


# ---------------------------------------------------------------------------
# Source URL stability — dedup key contract
# ---------------------------------------------------------------------------

def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice yields identical source_urls
    in the same order — these are the dedup keys for ON CONFLICT
    upsert."""
    html = _load("ncbahm_search_wa_live.html")
    a = parse_search_html(html)
    b = parse_search_html(html)
    assert [r.source_url for r in a] == [r.source_url for r in b]


def test_source_url_canonical_form_across_fixtures():
    """Every emitted source_url uses the canonical
    /FAP/PractitionerDetail?AgencyClientId=<id> form."""
    for fname in _EXPECTED:
        for r in parse_search_html(_load(fname)):
            assert "/FAP/PractitionerDetail?AgencyClientId=" in r.source_url, (
                fname, r.name, r.source_url
            )


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
    """'Recertification Pending' downgrades fellowship_level."""
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
            "/FAP/PractitionerDetail?AgencyClientId=FdSPJbqR6f4="
        )
        == "FdSPJbqR6f4="
    )


def test_extract_agency_client_id_from_legacy_path_form():
    """The legacy name-search layout used /FAPPractitionerProfile/<id>=
    path form. The opaque ID is identical."""
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
    """source_url is always the bare ?AgencyClientId=<id> form on the
    new ncbahm.org host."""
    assert _build_source_url("abc123=") == (
        "https://directory.ncbahm.org/FAP/PractitionerDetail?AgencyClientId=abc123="
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
    """'Youl Park L.Ac.' -> name='Youl Park', creds='L.Ac.'. The NCBAHM
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


def test_parse_address_line_not_available_sentinel():
    """The 'Not Available' literal (HI Aaron Bullington pattern) yields
    an empty dict — caller's address fields stay None."""
    assert _parse_address_line("Not Available") == {}
    assert _parse_address_line("not available") == {}


def test_parse_address_line_empty_or_garbage():
    assert _parse_address_line("") == {}
    # Single-token addresses without a comma are left as address1.
    out = _parse_address_line("Unknown")
    assert out.get("address1") == "Unknown"


def test_parse_card_extracts_all_fields_from_synthetic_card():
    """End-to-end: a single result-card chunk parses to the expected
    fields dict."""
    parsed = _parse_card(_EXPIRED_CARD)
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


def test_empty_page_yields_zero_rows_and_zero_pagination():
    """A page with no result cards and no banner -> zero rows, zero
    last-page; callers' empty-page break stops the walk."""
    html = "<html><body>no results here</body></html>"
    assert parse_search_html(html) == []
    assert parse_total_pages(html) == 0
    assert parse_total_count(html) is None
