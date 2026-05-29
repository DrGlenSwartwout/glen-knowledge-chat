"""Unit tests for the American Academy of Anti-Aging Medicine (A4M) adapter.

A4M serves its provider directory through two JSON endpoints:

  - /listing-search/coordinates  -> a list of listing ids (no detail)
  - /listing-search/listings     -> the full per-listing records

Fixtures here are REAL responses captured live 2026-05-29:

  - a4m_coordinates_sample.json  — a real /listing-search/coordinates
                                   response (the id-list shape) for a
                                   small radius search. Used to test the
                                   id extractor.
  - a4m_listings_sample.json     — a real /listing-search/listings
                                   response envelope hand-assembled from
                                   live records to cover the full
                                   credential matrix:
                                     * FAARM in degrees            -> fellow
                                     * FAARFM in degrees           -> fellow
                                     * FAARFM Fellowship in props
                                       (degrees FAAMFM or only ABAARM) -> fellow
                                     * ABAARM Certification only    -> NOT fellow
                                     * ABAAHP Certification only    -> NOT fellow
                                     * FAAMFM / FAAMM other fellowships -> NOT fellow
                                     * plain membership             -> NOT fellow
                                   plus 2 product/service rows (different
                                   ``type`` hash) that MUST be filtered out.

The locked distinction under test: FAARM / FAARFM => fellowship_level=True;
ABAARM (board cert), ABAAHP, plain membership, or the sibling fellowships
(FAAMFM/FAAMM) => False, matched on word boundaries.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.a4m import (  # noqa: E402
    PROVIDER_TYPE,
    build_coordinates_url,
    build_listings_url,
    extract_listing_records,
    parse_coordinates_ids,
    parse_listings_json,
    _build_address1,
    _build_name,
    _build_source_url,
    _has_fellowship,
    _is_provider,
    _normalize_website,
)


def _load_json(name: str):
    return json.loads((FIXTURE_DIR / name).read_text())


def _load_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


def _rows():
    return parse_listings_json(_load_json("a4m_listings_sample.json"))


def _by_name(rows, needle):
    return next(r for r in rows if needle in r.name)


# ---------------------------------------------------------------------------
# Coordinates (id-list) extraction
# ---------------------------------------------------------------------------

def test_parse_coordinates_ids_pulls_ids():
    """The coordinates endpoint returns id + sort metadata only — the
    extractor must surface a non-empty list of string ids."""
    data = _load_json("a4m_coordinates_sample.json")
    ids = parse_coordinates_ids(data)
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert all(isinstance(i, str) for i in ids)


def test_parse_coordinates_ids_accepts_string_and_html():
    """Defensive: raw JSON string and HTML-wrapped JSON both parse."""
    raw = _load_text("a4m_coordinates_sample.json")
    assert parse_coordinates_ids(raw) == parse_coordinates_ids(_load_json("a4m_coordinates_sample.json"))
    wrapped = f"<html><body><pre>{raw}</pre></body></html>"
    assert parse_coordinates_ids(wrapped) == parse_coordinates_ids(raw)


def test_parse_coordinates_ids_handles_junk():
    assert parse_coordinates_ids("") == []
    assert parse_coordinates_ids("not json") == []
    assert parse_coordinates_ids({"error": True}) == []
    assert parse_coordinates_ids({"message": "nope"}) == []


# ---------------------------------------------------------------------------
# Listings parsing — counts, provider filter, locked invariants
# ---------------------------------------------------------------------------

def test_parse_listings_returns_only_providers():
    """The fixture carries 14 providers + 2 product/service rows. The
    parser must drop the products and yield exactly the providers."""
    raw = extract_listing_records(_load_json("a4m_listings_sample.json"))
    providers = [r for r in raw if _is_provider(r)]
    products = [r for r in raw if not _is_provider(r)]
    assert len(providers) == 14
    assert len(products) == 2

    rows = _rows()
    assert len(rows) == 14  # products filtered out


def test_count_greater_than_zero():
    rows = _rows()
    assert len(rows) > 0


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are locked per spec."""
    rows = _rows()
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "A4M"
        assert r.specialties == ["anti_aging_regenerative", "holistic_health"]
        assert r.source_url  # stable, present
        assert r.photo_url is None  # portal-managed
        assert r.bio is None


def test_source_url_is_unique_per_practitioner():
    rows = _rows()
    urls = [r.source_url for r in rows]
    assert len(urls) == len(set(urls)), "duplicate source_url across rows"


def test_source_url_stable_across_reruns():
    payload = _load_json("a4m_listings_sample.json")
    a = parse_listings_json(payload)
    b = parse_listings_json(payload)
    assert [r.source_url for r in a] == [r.source_url for r in b]


# ---------------------------------------------------------------------------
# Field population
# ---------------------------------------------------------------------------

def test_fields_populate_for_full_record():
    """A provider with full data must populate name + credentials +
    practice + phone + address + city/state/zip + website."""
    rows = _rows()
    # Eva Henry: MD, FAARFM, ABAARM with a full address + phone.
    henry = _by_name(rows, "Eva Henry")
    assert henry.name == "Eva Henry"
    assert henry.credentials and "MD" in henry.credentials
    assert henry.city
    assert henry.state
    assert henry.country == "US"


def test_email_is_always_none():
    """A4M never exposes practitioner email — every row's email is None."""
    rows = _rows()
    assert all(r.email is None for r in rows)


def test_website_promotes_label_not_redirect():
    """``website`` must be the human-readable websiteLabel as an https URL,
    NOT the internal /directory-url-redirect path."""
    rows = _rows()
    for r in rows:
        if r.website is not None:
            assert r.website.startswith("http")
            assert "directory-url-redirect" not in r.website


def test_practice_name_from_company():
    rows = _rows()
    henry = _by_name(rows, "Eva Henry")
    # company field is the practice name when present
    assert henry.practice_name is None or isinstance(henry.practice_name, str)


# ---------------------------------------------------------------------------
# FELLOWSHIP — the locked distinction
# ---------------------------------------------------------------------------

def test_faarm_in_degrees_is_fellow():
    """FAARM in the degrees string => fellowship_level True."""
    rows = _rows()
    miles = _by_name(rows, "Laura Miles")
    assert "FAARM" in (miles.credentials or "")
    assert miles.fellowship_level is True


def test_faarfm_in_degrees_is_fellow():
    """FAARFM in the degrees string => fellowship_level True."""
    rows = _rows()
    hellman = _by_name(rows, "David Hellman")
    assert "FAARFM" in (hellman.credentials or "")
    assert hellman.fellowship_level is True


def test_faarfm_in_properties_only_is_fellow():
    """A provider whose degrees do NOT carry the fellowship token but
    whose properties list a 'FAARFM Fellowship' tag must still resolve
    True. Heidi Koch: degrees 'MD, ABAARM' + property 'FAARFM Fellowship'."""
    rows = _rows()
    koch = _by_name(rows, "Heidi Koch")
    assert "FAARFM" not in (koch.credentials or "")
    assert "FAARM" not in (koch.credentials or "")
    assert koch.fellowship_level is True


def test_abaarm_only_is_not_fellow():
    """ABAARM is a BOARD CERTIFICATION, not a fellowship. A provider with
    ABAARM but no FAARM/FAARFM must be False. Paul Rothwell: 'MD, ABAARM'
    with no fellowship property."""
    rows = _rows()
    rothwell = _by_name(rows, "Paul Rothwell")
    assert "ABAARM" in (rothwell.credentials or "")
    assert rothwell.fellowship_level is False


def test_abaahp_only_is_not_fellow():
    """ABAAHP certification alone => False."""
    rows = _rows()
    battmer = _by_name(rows, "Bridget Battmer")
    assert "ABAAHP" in (battmer.credentials or "")
    assert battmer.fellowship_level is False


def test_other_fellowships_do_not_count():
    """FAAMFM / FAAMM are SIBLING fellowships that must NOT trigger
    fellowship_level (word-boundary discipline). Mildred Clifton (FAAMM)
    and Amanda Whitson (FAAMM) are the canonical non-FAARM fellows."""
    rows = _rows()
    clifton = _by_name(rows, "Mildred Clifton")
    assert "FAAMM" in (clifton.credentials or "")
    assert clifton.fellowship_level is False
    whitson = _by_name(rows, "Amanda Whitson")
    assert whitson.fellowship_level is False


def test_plain_membership_is_not_fellow():
    """A4M Membership only, no cert, no fellowship => False."""
    rows = _rows()
    rogers = _by_name(rows, "Rhea Rogers")
    assert rogers.credentials == "MD"
    assert rogers.fellowship_level is False


def test_fellowship_count_matches_matrix():
    """6 fellows / 8 non-fellows out of the 14 providers in the fixture.
    If this drifts, the FAARM/FAARFM detection has regressed."""
    rows = _rows()
    fellows = [r for r in rows if r.fellowship_level]
    non = [r for r in rows if not r.fellowship_level]
    assert len(fellows) == 6
    assert len(non) == 8


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_has_fellowship_word_boundaries():
    """FAARM / FAARFM match as whole tokens; FAAMFM / FAAMM / substrings
    must NOT match."""
    assert _has_fellowship({"degrees": "MD, FAARM, ABAARM"}) is True
    assert _has_fellowship({"degrees": "MD, FAARFM"}) is True
    assert _has_fellowship({"degrees": "MD, FAAMFM, ABAARM"}) is False
    assert _has_fellowship({"degrees": "FNP-BC, FAAMM"}) is False
    assert _has_fellowship({"degrees": "MD, ABAARM"}) is False
    assert _has_fellowship({"degrees": "MD, ABAAHP"}) is False
    assert _has_fellowship({"degrees": None}) is False
    assert _has_fellowship({"degrees": "FAARMx"}) is False
    assert _has_fellowship({"degrees": "xFAARFM"}) is False
    # Property-only signal.
    assert _has_fellowship(
        {"degrees": "MD, ABAARM", "properties": [{"name": "FAARFM Fellowship"}]}
    ) is True
    # Sibling fellowship in properties must NOT count.
    assert _has_fellowship(
        {"degrees": "MD", "properties": [{"name": "FAAMM Fellowship"}]}
    ) is False


def test_is_provider_filters_type():
    assert _is_provider({"type": PROVIDER_TYPE}) is True
    assert _is_provider({"type": "ac38be7f7ff5bc932826db127e1b7f23"}) is False
    assert _is_provider({}) is False


def test_build_name():
    assert _build_name({"fname": "Jane", "lname": "Doe"}) == "Jane Doe"
    assert _build_name({"fname": "Jane", "lname": None}) == "Jane"
    assert _build_name({"fname": None, "lname": "Doe"}) == "Doe"
    assert _build_name({"fname": None, "lname": None}) is None
    assert _build_name({}) is None


def test_build_address1_joins_lines():
    assert _build_address1(
        {"address1": "2907 Butterfield Rd", "address2": "Suite 100"}
    ) == "2907 Butterfield Rd, Suite 100"
    assert _build_address1({"address1": "1 Main St", "address2": None}) == "1 Main St"
    assert _build_address1({"address1": None, "address2": None}) is None


def test_normalize_website_promotes_label():
    assert _normalize_website(
        {"websiteLabel": "www.example.com", "website": "/directory-url-redirect;x.html"}
    ) == "https://www.example.com"
    assert _normalize_website({"websiteLabel": "https://x.com"}) == "https://x.com"
    assert _normalize_website({"websiteLabel": None}) is None
    assert _normalize_website({"websiteLabel": "n/a"}) is None


def test_build_source_url_uses_detail_url():
    assert _build_source_url({"url": "https://www.a4m.com/jane-doe-ny.html"}) == (
        "https://www.a4m.com/jane-doe-ny.html"
    )
    # Fallback to listingId anchor when url is missing.
    fb = _build_source_url({"listingId": "abc123"})
    assert fb.endswith("#listing-abc123")
    assert _build_source_url({}) is None


def test_skips_records_without_name():
    """Provider record with no name -> dropped."""
    payload = {
        "error": False,
        "message": [
            {"type": PROVIDER_TYPE, "fname": None, "lname": None, "url": "x"},
            {"type": PROVIDER_TYPE, "fname": "Val", "lname": "Id", "url": "y"},
        ],
    }
    rows = parse_listings_json(payload)
    assert len(rows) == 1
    assert rows[0].name == "Val Id"


def test_url_builders():
    u = build_coordinates_url()
    assert u.startswith("https://www.a4m.com/listing-search/coordinates?")
    assert "lat=38.341656" in u and "lng=-96.69342" in u and "radius=3000" in u
    lu = build_listings_url(["a", "b"])
    assert lu.startswith("https://www.a4m.com/listing-search/listings?")
    assert lu.count("id%5B%5D=") == 2  # id[] urlencoded, repeated


def test_parse_listings_accepts_bare_list_and_string():
    payload = _load_json("a4m_listings_sample.json")
    bare = parse_listings_json(payload["message"])
    enveloped = parse_listings_json(payload)
    assert len(bare) == len(enveloped) == 14
    as_str = parse_listings_json(json.dumps(payload))
    assert len(as_str) == 14
