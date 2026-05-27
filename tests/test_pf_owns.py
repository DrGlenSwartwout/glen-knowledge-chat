"""Unit tests for the OWNS (Ocular Wellness & Nutrition Society) adapter.

OWNS publishes its directory via a WordPress ``portfolio`` custom post
type. Fixtures here are real responses captured 2026-05-26:

- owns_portfolio_page_1.json
    Page 1 @ per_page=100 — the entire 37-row directory. Mix of US
    optometrists (32), Canadian optometrists (5 with Province field),
    one UK optometrist (using County field), and three FOWNS Fellows
    (Kaleb Abbott, Daniel Walker, Mila Ioussifova).

- owns_portfolio_categories.json
    The portfolio_category taxonomy term list (52 terms: 50 US states +
    Canada + United Kingdom). Used to build the {term_id: ISO2} map
    that drives country resolution.

These two cover the credential matrix (FOWNS Fellow vs non-Fellow),
the country matrix (US / CA / GB), the region-line label matrix
(State / Province / County), the zip-label matrix (Zip Code / Postal
Code), and the cross-parent specialty tagging convention (all four
tags on every row).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.owns import (  # noqa: E402
    LOCKED_SPECIALTIES,
    build_category_country_map,
    parse_directory_json,
    _build_source_url,
    _coerce_str,
    _credentials_from_title,
    _extract_fields,
    _href_from_field,
    _is_fellowship_title,
    _name_from_title,
    _normalize_website,
    _resolve_country,
    _strip_credentials,
)


def _load(name: str):
    return json.loads((FIXTURE_DIR / name).read_text())


def _cat_map():
    """Build the {term_id: ISO2} map from the real categories fixture."""
    return build_category_country_map(_load("owns_portfolio_categories.json"))


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_returns_full_directory():
    """The portfolio fixture holds the entire 37-row directory — adapter
    must produce 37 rows (every record has a usable title)."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    assert len(rows) == 37


def test_all_rows_carry_locked_invariants():
    """tier / source_org are constant. specialties carries all FOUR cross-
    parent tags so the Finder filter chips surface OWNS practitioners
    under both Eye Care AND Holistic Health."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "OWNS"
        # Source URL is always populated (dedup key).
        assert r.source_url
        assert r.source_url.startswith("https://ocularnutritionsociety.org/")
        # All four cross-parent specialty tags must be present on every
        # row — the Finder's filter chip logic depends on this exact set.
        assert "nutritional_eye_care" in r.specialties
        assert "eye_care" in r.specialties
        assert "nutrition" in r.specialties
        assert "holistic_health" in r.specialties
        assert len(r.specialties) == 4


def test_locked_specialties_constant_exposes_all_four_tags():
    """The exported LOCKED_SPECIALTIES list itself must carry the canonical
    four-tag set, in case downstream callers introspect it."""
    assert set(LOCKED_SPECIALTIES) == {
        "nutritional_eye_care",
        "eye_care",
        "nutrition",
        "holistic_health",
    }


def test_spot_check_walker_shaffer_full_us_fields():
    """First/newest record on the page — Walker Shaffer, full US address
    in the canonical State / Zip Code labels."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    shaffer = next(r for r in rows if "Walker Shaffer" in r.name)

    assert shaffer.name == "Walker Shaffer"
    assert shaffer.credentials == "O.D."
    assert shaffer.practice_name == "Eyecare of Lehi"
    assert shaffer.address1 == "75 W State St"
    assert shaffer.city == "Lehi"
    assert shaffer.state == "Utah"
    assert shaffer.postal == "84043"
    assert shaffer.country == "US"
    assert shaffer.phone == "(801) 768-4100"
    assert shaffer.email == "drshaffer@eyecareoflehi.com"
    # Website href comes from the <a href="..."> inside the Website field
    assert shaffer.website == "https://eyecareoflehi.com/"
    # Non-Fellow
    assert shaffer.fellowship_level is False


def test_canadian_record_uses_province_label_and_iso2_country():
    """Canadian records use ``<strong>Province:</strong>`` instead of State
    and ``<strong>Postal Code:</strong>`` instead of Zip Code. Country
    resolves to 'CA' via the portfolio_category 'canada' term."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    irene = next(r for r in rows if "Irene Kokoshko" in r.name)
    assert irene.city == "Toronto"
    assert irene.state == "Ontario"  # Province label collapses into state slot
    assert irene.postal == "M5J 2X1"
    assert irene.country == "CA"
    assert irene.practice_name == "KOKO Vision"


def test_uk_record_uses_county_label_but_drops_country_value():
    """UK record (David Burghardt) uses ``<strong>County:</strong>`` with
    the literal value 'United Kingdom' — that's a country name, not a
    region, so the parser must DROP it from the state slot rather than
    surfacing 'United Kingdom' as a state. Country resolves to 'GB' via
    the portfolio_category 'united-kingdom' term."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    burghardt = next(r for r in rows if "David Burghardt" in r.name)
    assert burghardt.country == "GB"
    assert burghardt.state is None
    assert burghardt.city == "Nettleham"
    assert burghardt.postal == "LN2 2PD"
    assert burghardt.practice_name == "David Burghardt Vision Care"


def test_fellowship_detection_fowns_credential():
    """FOWNS = Fellow OWNS = top tier, must set fellowship_level=True.
    Three of 37 practitioners carry FOWNS as of discovery."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    fellows = [r for r in rows if r.fellowship_level]
    fellow_names = {r.name for r in fellows}
    # Three FOWNS Fellows visible on the page.
    assert len(fellows) == 3
    assert "Kaleb Abbott" in fellow_names
    assert "Daniel Walker" in fellow_names
    assert "Mila Ioussifova" in fellow_names


def test_fellowship_not_set_for_non_fowns_practitioners():
    """A practitioner with FAAO / Diplomate ABO but NO FOWNS in the title
    is not fellowship-tier in the OWNS adapter — FAAO is the American
    Academy of Optometry fellowship, a separate organization. Pamela
    Lowe (OD, FAAO, Dipl ABO) is the canonical case."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    lowe = next(r for r in rows if "Pamela Lowe" in r.name)
    assert "FAAO" in (lowe.credentials or "")
    assert lowe.fellowship_level is False


def test_credentials_extracted_from_title():
    """Trailing comma-separated credentials must end up in the credentials
    field, with the practitioner name stripped clean."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    abbott = next(r for r in rows if "Kaleb Abbott" in r.name)
    assert abbott.name == "Kaleb Abbott"
    assert abbott.credentials
    assert "OD" in abbott.credentials
    assert "FOWNS" in abbott.credentials


def test_name_without_credentials_keeps_full_string():
    """Records with no comma in the title (just a bare name) yield the
    full name in `name` and None in `credentials`. Lynda Jones is the
    canonical case — three of 37 records carry just a name."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    jones = next(r for r in rows if r.name == "Lynda Jones")
    assert jones.credentials is None


def test_country_defaults_to_us_when_category_map_empty():
    """Caller may omit the category map (e.g. for a quick parse during
    testing). Every record then defaults to country='US' — the
    overwhelming majority of OWNS practitioners are US-based."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"))
    assert rows
    for r in rows:
        assert r.country == "US"


def test_country_defaults_to_us_for_uncategorized_record():
    """A synthetic record with no portfolio_category list (or an empty
    one) falls back to 'US' — the safest default for a brand-new
    uncategorized record on a US-focused society site."""
    rec = {
        "id": 99999,
        "slug": "jane-uncategorized",
        "link": "https://ocularnutritionsociety.org/practitioners/jane-uncategorized/",
        "title": {"rendered": "Jane Uncategorized, OD"},
        "content": {"rendered": "<p><strong>Clinic Name:</strong> Test Clinic</p>"},
        "portfolio_category": [],
    }
    rows = parse_directory_json([rec], _cat_map())
    assert len(rows) == 1
    assert rows[0].country == "US"


def test_source_url_uses_canonical_link_when_present():
    """The WP REST canonical ``link`` becomes source_url, with ``#<id>``
    appended as a stable tiebreak fragment."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    shaffer = next(r for r in rows if "Walker Shaffer" in r.name)
    assert shaffer.source_url == (
        "https://ocularnutritionsociety.org/practitioners/walker-shaffer-o-d/#8250"
    )


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order — these are the dedup keys for ON CONFLICT
    upsert. All distinct so no two practitioners collide."""
    payload = _load("owns_portfolio_page_1.json")
    cm = _cat_map()
    a = parse_directory_json(payload, cm)
    b = parse_directory_json(payload, cm)
    assert [r.source_url for r in a] == [r.source_url for r in b]
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_address_composition_includes_street_city_state_postal():
    """Every parsed row must populate the address ladder fields the
    geocoder feeds on. Spot-check a couple of US records — both should
    have all of address1 / city / state / postal populated when the
    source record has them."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    daniels = next(r for r in rows if "Kenneth Daniels" in r.name)
    assert daniels.address1 == "84 East Broad Street"
    assert daniels.city == "Hopewell"
    assert daniels.state == "New Jersey"
    assert daniels.postal == "08525"
    assert daniels.country == "US"


def test_solo_practitioner_practice_name_dedup_suppression():
    """When the clinic name matches the practitioner name exactly (self-
    employed listings), the practice_name is suppressed to avoid showing
    'Jane Smith' as both name and practice. Synthetic record — none of
    the real OWNS data triggers this, but the dedup is locked behavior
    shared with the IABDM adapter."""
    rec = {
        "id": 88888,
        "slug": "jane-solo",
        "link": "https://ocularnutritionsociety.org/practitioners/jane-solo/",
        "title": {"rendered": "Jane Solo, OD"},
        "content": {
            "rendered": (
                "<p><strong>Clinic Name:</strong> Jane Solo</p>"
                "<p><strong>City:</strong> Portland<br />"
                "<strong>State:</strong> Oregon</p>"
            )
        },
        "portfolio_category": [43],  # Oregon
    }
    rows = parse_directory_json([rec], _cat_map())
    assert len(rows) == 1
    assert rows[0].name == "Jane Solo"
    assert rows[0].practice_name is None


def test_email_href_preferred_over_visible_text():
    """The Email value HTML wraps the canonical address in
    <a href="mailto:..."> — the href is authoritative because the
    visible text sometimes truncates. Spot-check Walker Shaffer where
    visible text and href match (sanity check) and a synthetic where
    they differ (correctness check)."""
    rec = {
        "id": 77777,
        "slug": "diff-email",
        "link": "https://ocularnutritionsociety.org/practitioners/diff-email/",
        "title": {"rendered": "Diff Email, OD"},
        "content": {
            "rendered": (
                "<p><strong>Email:</strong> "
                '<a href="mailto:canonical@example.com">contact us</a></p>'
            )
        },
        "portfolio_category": [50],
    }
    rows = parse_directory_json([rec], _cat_map())
    assert len(rows) == 1
    assert rows[0].email == "canonical@example.com"


def test_website_href_preferred_over_visible_text():
    """Same as email — the <a href="..."> is authoritative because
    the OWNS layout often shows 'www.example.com' as visible text while
    the href is 'https://example.com/' (or vice versa)."""
    rows = parse_directory_json(_load("owns_portfolio_page_1.json"), _cat_map())
    shaffer = next(r for r in rows if "Walker Shaffer" in r.name)
    # Visible text on the page is 'www.eyecareoflehi.com'; the href is the
    # real URL. The parser must emit the href, not the visible text.
    assert shaffer.website == "https://eyecareoflehi.com/"


def test_parser_accepts_json_string():
    """A raw JSON string of the response list is also valid input."""
    raw = (FIXTURE_DIR / "owns_portfolio_page_1.json").read_text()
    rows = parse_directory_json(raw, _cat_map())
    assert len(rows) == 37


def test_parser_skips_non_dict_records():
    """Defensive: a payload with junk entries mixed in must skip them
    without crashing."""
    rows = parse_directory_json([None, 42, "string", {}], _cat_map())
    assert rows == []


def test_parser_skips_record_with_no_title():
    """A record with an empty title yields no row — name is mandatory."""
    rec = {
        "id": 55555,
        "slug": "no-title",
        "link": "https://ocularnutritionsociety.org/practitioners/no-title/",
        "title": {"rendered": ""},
        "content": {"rendered": "<p>some body</p>"},
        "portfolio_category": [50],
    }
    rows = parse_directory_json([rec], _cat_map())
    assert rows == []


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_build_category_country_map_canonicalizes_terms():
    """The map keys are integer term ids; values are ISO2 country codes."""
    cm = _cat_map()
    # 'utah' = id 50 -> US
    assert cm.get(50) == "US"
    # 'california' = id 11 -> US
    assert cm.get(11) == "US"
    # 'canada' = id 57 -> CA
    assert cm.get(57) == "CA"
    # 'united-kingdom' = id 58 -> GB
    assert cm.get(58) == "GB"


def test_build_category_country_map_skips_invalid_terms():
    """Defensive: terms missing id/slug or that are not dicts are skipped."""
    cm = build_category_country_map(
        [
            None,
            42,
            "string",
            {"id": "not-an-int", "slug": "utah"},
            {"id": 99, "slug": None},
            {"id": 50, "slug": "utah"},  # valid -> US
        ]
    )
    assert cm == {50: "US"}


def test_build_category_country_map_skips_unknown_slugs():
    """A category slug we don't know (e.g. a future 'mexico' entry) is
    intentionally omitted so the parser falls back to its own default
    rather than mis-flagging."""
    cm = build_category_country_map([{"id": 999, "slug": "mexico"}])
    assert cm == {}


def test_resolve_country_uses_category_map():
    """A record with a known category id resolves to that country."""
    cm = {50: "US", 57: "CA", 58: "GB"}
    assert _resolve_country({"portfolio_category": [57]}, cm) == "CA"
    assert _resolve_country({"portfolio_category": [58]}, cm) == "GB"
    assert _resolve_country({"portfolio_category": [50, 99]}, cm) == "US"


def test_resolve_country_falls_back_to_us():
    """No portfolio_category, an empty list, or an all-unknown list all
    fall back to 'US' (the safe default for a US-focused society)."""
    assert _resolve_country({}, {50: "US"}) == "US"
    assert _resolve_country({"portfolio_category": []}, {50: "US"}) == "US"
    assert _resolve_country({"portfolio_category": [999]}, {50: "US"}) == "US"


def test_is_fellowship_title_only_fires_on_fowns():
    """FOWNS marks fellowship; FAAO / FCOVD / FOVDR do NOT (those are
    other organizations' fellowships)."""
    assert _is_fellowship_title("OD, MS, FAAO, FOWNS") is True
    assert _is_fellowship_title("OD, FOWNS, FCOVD") is True
    assert _is_fellowship_title("fowns") is True  # case-insensitive
    assert _is_fellowship_title("OD, FAAO, Dipl ABO") is False
    assert _is_fellowship_title("OD") is False
    assert _is_fellowship_title("") is False
    assert _is_fellowship_title(None) is False
    # Word-boundary guard: a stray 'FOWNSY' shouldn't match. (Not a real
    # credential — defensive.)
    assert _is_fellowship_title("OD, FOWNSY") is False


def test_normalize_website_adds_scheme():
    assert _normalize_website("example.com") == "https://example.com"
    assert _normalize_website("https://x.com/") == "https://x.com/"
    assert _normalize_website("http://x.com") == "http://x.com"
    assert _normalize_website(None) is None
    assert _normalize_website("") is None


def test_strip_credentials_handles_simple_comma_form():
    """The OWNS title format is consistently 'Name, Cred1, Cred2, ...'"""
    name, creds = _strip_credentials("Walker Shaffer, O.D.")
    assert name == "Walker Shaffer"
    assert creds == "O.D."

    name, creds = _strip_credentials("Kaleb Abbott, OD, MS, FAAO, FOWNS")
    assert name == "Kaleb Abbott"
    assert creds == "OD, MS, FAAO, FOWNS"


def test_strip_credentials_preserves_honorific():
    """Honorifics ('Dr.') stay on the name side of the split."""
    name, _creds = _strip_credentials("Dr. Ingryd Lorenzana, FAAO")
    assert name == "Dr. Ingryd Lorenzana"


def test_strip_credentials_no_creds():
    """A bare name (no comma) yields the name and None for credentials."""
    name, creds = _strip_credentials("Lynda Jones")
    assert name == "Lynda Jones"
    assert creds is None


def test_coerce_str_strips_zero_width_space():
    """Every OWNS title ends with U+200B (zero-width space). The coercer
    must strip it so equality checks ("Walker Shaffer") work."""
    assert _coerce_str("Walker Shaffer​") == "Walker Shaffer"
    assert _coerce_str({"rendered": "Walker Shaffer​"}) == "Walker Shaffer"
    assert _coerce_str("  spaced  ") == "spaced"
    assert _coerce_str("") is None
    assert _coerce_str(None) is None


def test_name_from_title_strips_creds_and_zwsp():
    assert _name_from_title("Walker Shaffer, O.D.​") == "Walker Shaffer"
    assert _name_from_title("Lynda Jones​") == "Lynda Jones"
    assert _name_from_title("") == ""


def test_credentials_from_title():
    assert _credentials_from_title("Walker Shaffer, O.D.​") == "O.D."
    assert _credentials_from_title("Lynda Jones") is None
    assert _credentials_from_title("") is None


def test_extract_fields_pulls_labelled_lines():
    """The labelled-line extractor must split <strong>Label:</strong>
    Value blocks even when separated by <br /> or </p><p>. The OWNS
    layout uses a mix of both."""
    body = (
        "<p><strong>Clinic Name:</strong> Test Clinic</p>"
        "<p><strong>Address:</strong> 123 Main<br />"
        "<strong>City:</strong> Townsville<br />"
        "<strong>State:</strong> Texas<br />"
        "<strong>Zip Code:</strong> 75001</p>"
        "<p><strong>Phone: </strong>(555) 123-4567</p>"
    )
    f = _extract_fields(body)
    assert f["clinic name"] == "Test Clinic"
    assert f["address"] == "123 Main"
    assert f["city"] == "Townsville"
    assert f["state"] == "Texas"
    assert f["zip code"] == "75001"
    assert f["phone"] == "(555) 123-4567"


def test_extract_fields_empty_body():
    assert _extract_fields("") == {}
    assert _extract_fields(None) == {}  # type: ignore[arg-type]


def test_href_from_field_picks_mailto():
    """Email href extraction strips the mailto: scheme prefix."""
    html = '<a href="mailto:jane@example.com">contact</a>'
    assert _href_from_field(html, scheme="mailto:") == "jane@example.com"
    # Wrong scheme returns None.
    assert _href_from_field(html, scheme="http:") is None
    # No href returns None.
    assert _href_from_field("plain text", scheme="mailto:") is None
    assert _href_from_field(None) is None


def test_href_from_field_picks_website():
    """Website href extraction returns the URL as-is (no scheme strip)."""
    html = '<a href="https://example.com/" rel="noopener">www.example.com</a>'
    assert _href_from_field(html) == "https://example.com/"


def test_build_source_url_prefers_link():
    """When the API gives us a canonical link, use it (with #id fragment)."""
    url = _build_source_url(
        {
            "link": "https://ocularnutritionsociety.org/practitioners/jane-doe/",
            "slug": "jane-doe",
            "id": 9999,
        }
    )
    assert url == "https://ocularnutritionsociety.org/practitioners/jane-doe/#9999"


def test_build_source_url_falls_back_to_slug():
    """No link -> synthesize /practitioners/<slug>/."""
    url = _build_source_url({"slug": "jane-doe", "id": 9999})
    assert url == "https://ocularnutritionsociety.org/practitioners/jane-doe/#9999"


def test_build_source_url_falls_back_to_id_only():
    """No link, no slug -> /practitioners/practitioner-<id>/ keeps URL stable."""
    url = _build_source_url({"id": 9999})
    assert url == "https://ocularnutritionsociety.org/practitioners/practitioner-9999/#9999"
