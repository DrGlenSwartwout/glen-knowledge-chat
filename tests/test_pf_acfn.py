"""Unit tests for the American College of Functional Neurology (ACFN) adapter.

ACFN publishes its Fellows directory at https://acfn.org/directory/ as a
single static page whose entire roster ships inline as one HTML
``<table class="posts-data-table">``. Fixtures captured 2026-05-29 via a
plain requests GET (no login, no JS, no Cloudflare):

- acfn_directory.html  — full directory page download (119KB, 151 fellow
                         rows in one table).
- acfn_sample.html     — a 7-row trimmed table covering the credential
                         matrix: single FACFN, single FABVR, multi
                         (FABVR+FACFN), the three non-fellow placeholders
                         (Uncategorized, CABCDD-candidate, Retired), and
                         international rows (South Korea, Canada).
- acfn_profile_ron_mcmorris.html — one fellow profile page (used by the
                         opt-in profile-enrichment parser tests).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.acfn import (  # noqa: E402
    parse_directory_html,
    parse_profile_html,
    _build_source_url,
    _country_iso2,
    _fellowships_from_cell,
    _has_fellowship,
    _normalize_state,
    _normalize_website,
)


def _load_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_full_directory_yields_rows():
    """The full directory page parses to one row per fellow. 151 fellows
    as captured 2026-05-29 — assert > 0 and pin the captured count so a
    regression in the table locator screams."""
    rows = parse_directory_html(_load_text("acfn_directory.html"))
    assert len(rows) > 0
    assert len(rows) == 151


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are locked per spec — never mutate."""
    rows = parse_directory_html(_load_text("acfn_directory.html"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "ACFN"
        assert r.specialties == ["functional_neurology", "holistic_health"]
        assert r.source_url
        # lat/lng/photo/bio are never set by the scraper.
        assert r.lat is None
        assert r.lng is None
        assert r.photo_url is None
        assert r.bio is None


def test_source_url_is_unique_per_practitioner():
    """Every fellow has a distinct profile URL — the dedup / ON CONFLICT
    key. No collisions across the full roster."""
    rows = parse_directory_html(_load_text("acfn_directory.html"))
    urls = [r.source_url for r in rows]
    assert len(urls) == len(set(urls)), "duplicate source_url across rows"
    for r in rows:
        assert r.source_url.startswith("https://acfn.org/")


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice yields identical source_urls in
    identical order (the upsert keys must be deterministic)."""
    html = _load_text("acfn_sample.html")
    a = parse_directory_html(html)
    b = parse_directory_html(html)
    assert [r.source_url for r in a] == [r.source_url for r in b]


def test_required_fields_populate_across_roster():
    """Field-fill sanity on the full roster: name, credentials, practice,
    state, and country should each be populated for the large majority of
    rows (the listing table carries all of these)."""
    rows = parse_directory_html(_load_text("acfn_directory.html"))
    n = len(rows)
    assert all(r.name for r in rows)                       # name is mandatory
    assert sum(1 for r in rows if r.credentials) >= n * 0.9
    assert sum(1 for r in rows if r.practice_name) >= n * 0.6
    assert sum(1 for r in rows if r.state) >= n * 0.6
    assert all(r.country for r in rows)                    # defaults to US


def test_listing_table_does_not_supply_contact_fields():
    """phone/email/website/address1/postal are NOT in the listing table;
    they stay None from the listing parse (profile enrichment is opt-in)."""
    rows = parse_directory_html(_load_text("acfn_directory.html"))
    for r in rows:
        assert r.phone is None
        assert r.email is None
        assert r.website is None
        assert r.address1 is None
        assert r.postal is None


def test_fellowship_flag_true_for_facfn_and_subfellowships():
    """FACFN and each sub-fellowship token flips fellowship_level=True."""
    rows = parse_directory_html(_load_text("acfn_sample.html"))
    by_name = {r.name: r for r in rows}

    facfn = by_name["Ron McMorris"]      # single FACFN
    assert facfn.fellowship_level is True
    assert facfn.credentials == "FACFN"

    fabvr = by_name["Chad Billiris"]     # single sub-fellowship FABVR
    assert fabvr.fellowship_level is True
    assert fabvr.credentials == "FABVR"

    multi = by_name["Seung il Youn"]     # FABVR + FACFN
    assert multi.fellowship_level is True
    assert "FABVR" in multi.credentials
    assert "FACFN" in multi.credentials


def test_fellowship_flag_false_for_non_fellow_placeholders():
    """Uncategorized / Retired / CABCDD(candidate) are NOT fellowships."""
    rows = parse_directory_html(_load_text("acfn_sample.html"))
    by_name = {r.name: r for r in rows}

    assert by_name["Christopher Caffery"].fellowship_level is False  # Uncategorized
    assert by_name["Michael Swank"].fellowship_level is False        # Retired
    # CABCDD is a Candidate token — must NOT trip the FABCDD sub-fellowship.
    rosa = by_name["Raijose Rosa"]
    assert rosa.credentials == "CABCDD"
    assert rosa.fellowship_level is False


def test_sample_has_both_fellows_and_non_fellows():
    """The sample fixture deliberately mixes fellows and non-fellows so the
    flag is exercised in both directions."""
    rows = parse_directory_html(_load_text("acfn_sample.html"))
    fellows = [r for r in rows if r.fellowship_level]
    non = [r for r in rows if not r.fellowship_level]
    assert len(fellows) >= 1
    assert len(non) >= 1


def test_full_roster_fellowship_split():
    """143 fellows / 8 non-fellows as captured 2026-05-29. If fellows drop
    to 0 the token detection regressed; if the split shifts a lot the
    directory grew (re-capture)."""
    rows = parse_directory_html(_load_text("acfn_directory.html"))
    fellows = [r for r in rows if r.fellowship_level]
    non = [r for r in rows if not r.fellowship_level]
    assert len(fellows) == 143
    assert len(non) == 8


def test_us_row_full_field_extraction():
    """Canonical US fellow: name, credentials, practice, city, state(abbr),
    country, and a unique profile source_url."""
    rows = parse_directory_html(_load_text("acfn_sample.html"))
    ron = next(r for r in rows if r.name == "Ron McMorris")
    assert ron.credentials == "FACFN"
    assert ron.practice_name == "My Elite Chiro"
    assert ron.city == "Livingston"
    assert ron.state == "LA"            # 'Louisiana' canonicalized to abbr
    assert ron.country == "US"
    assert ron.source_url == "https://acfn.org/fellows/ron-mcmorris/"
    assert ron.fellowship_level is True


def test_international_rows_resolve_country():
    """Non-US fellows resolve their country to ISO2; state stays raw text
    (no US-abbr canonicalization applied off-US)."""
    rows = parse_directory_html(_load_text("acfn_sample.html"))
    by_name = {r.name: r for r in rows}
    assert by_name["Seung il Youn"].country == "KR"   # South Korea
    assert by_name["William Farrell"].country == "CA"  # Canada


def test_misspelled_state_falls_through_as_raw_text():
    """'New Jeresey' (sic, as the directory writes it) is not in the abbr
    map, so it survives as cleaned raw text rather than being dropped."""
    rows = parse_directory_html(_load_text("acfn_sample.html"))
    chad = next(r for r in rows if r.name == "Chad Billiris")
    assert chad.state == "New Jeresey"
    assert chad.country == "US"


def test_parse_empty_or_tableless_html_returns_empty():
    """Defensive: no roster table -> empty list (do not crash)."""
    assert parse_directory_html("") == []
    assert parse_directory_html("<html><body>no table here</body></html>") == []


# ---------------------------------------------------------------------------
# Opt-in profile-enrichment parser
# ---------------------------------------------------------------------------

def test_parse_profile_html_extracts_contact_block():
    """The fellow profile page's 'Clinic Information:' block yields phone
    and website; the ACFN org email (secretary@acfn.org) is excluded."""
    fields = parse_profile_html(_load_text("acfn_profile_ron_mcmorris.html"))
    assert fields.get("phone") == "225-271-4083"
    assert fields.get("website") is not None
    assert "myelitechiro" in fields["website"].lower()
    # The only mailto on the page is the org footer — must be filtered out.
    assert fields.get("email") in (None,)


def test_parse_profile_html_handles_empty():
    assert parse_profile_html("") == {}
    assert parse_profile_html("<html><body>nothing</body></html>") == {}


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_has_fellowship_positive():
    assert _has_fellowship("FACFN") is True
    assert _has_fellowship("FABVR, FACFN") is True
    assert _has_fellowship("FABBIR") is True
    assert _has_fellowship("FABHP") is True          # spec token, not yet live
    assert _has_fellowship("od, facfn") is True       # case-insensitive


def test_has_fellowship_negative():
    assert _has_fellowship("CABCDD") is False         # candidate, not fellow
    assert _has_fellowship("Uncategorized") is False
    assert _has_fellowship("Retired") is False
    assert _has_fellowship("") is False
    assert _has_fellowship(None) is False
    # Word-boundary safety: CABCDD must not match the FABCDD sub-token.
    assert _has_fellowship("CABCDD") is False
    assert _has_fellowship("XFABVR") is False


def test_country_iso2():
    assert _country_iso2("United States") == "US"
    assert _country_iso2("canada") == "CA"
    assert _country_iso2("South Korea") == "KR"
    assert _country_iso2("Germany") == "DE"
    assert _country_iso2("Atlantis") is None
    assert _country_iso2("") is None
    assert _country_iso2(None) is None


def test_normalize_state_us_abbr_and_passthrough():
    assert _normalize_state("Louisiana", "US") == "LA"
    assert _normalize_state("california", "US") == "CA"
    # Unknown US state text passes through unchanged.
    assert _normalize_state("New Jeresey", "US") == "New Jeresey"
    # Non-US: never canonicalize against the US abbr map.
    assert _normalize_state("Ontario", "CA") == "Ontario"
    assert _normalize_state(None, "US") is None


def test_build_source_url_prefers_profile_href():
    assert _build_source_url("https://acfn.org/fellows/jane-doe/", "Jane Doe") == (
        "https://acfn.org/fellows/jane-doe/"
    )
    # Relative href is absolutized.
    assert _build_source_url("/fellows/jane/", "Jane") == "https://acfn.org/fellows/jane/"


def test_build_source_url_falls_back_to_slug():
    """No href -> a deterministic slugged directory anchor (so the dedup
    key never collapses to None)."""
    url = _build_source_url(None, "Jane Q. Doe")
    assert url == "https://acfn.org/directory/#fellow-jane-q-doe"


def test_normalize_website_scheme():
    assert _normalize_website("www.example.com") == "https://www.example.com"
    assert _normalize_website("http://example.com") == "http://example.com"
    assert _normalize_website(None) is None
    assert _normalize_website("n/a") is None


def test_fellowships_from_cell_span_and_text_paths():
    """Span-based extraction joins tokens cleanly; text fallback splits on
    commas. Both normalize whitespace."""
    from bs4 import BeautifulSoup
    span_cell = BeautifulSoup(
        '<td><span data-slug="fabvr">FABVR</span>, '
        '<span data-slug="facfn">FACFN</span></td>',
        "html.parser",
    ).td
    assert _fellowships_from_cell(span_cell) == "FABVR, FACFN"

    text_cell = BeautifulSoup("<td>FABBIR , FACFN</td>", "html.parser").td
    assert _fellowships_from_cell(text_cell) == "FABBIR, FACFN"

    empty_cell = BeautifulSoup("<td>   </td>", "html.parser").td
    assert _fellowships_from_cell(empty_cell) is None
