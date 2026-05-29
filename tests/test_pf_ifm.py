"""Unit tests for the Institute for Functional Medicine (IFM) adapter.

IFM's public Find-a-Practitioner directory is a Drupal "view" that
server-renders result cards (no public JSON endpoint). The fixture here is a
real, trimmed capture (2026-05-29) of the ``/practitioner-listings/...``
results page for a Los Angeles 40km search, de-duplicated across the first
few pager pages into a representative card set:

- ifm_listings_la.html — 13 unique ``<li class="practitioner-card">`` cards.
  Covers BOTH IFM-certified (badge present) and non-certified members,
  varied credential post-nominals, a card with no credential at all, a card
  with no profile slug (id-anchored source_url fallback), a card with no
  email, and a card whose website href carries free-text pollution.

At capture time the certification split in the fixture was:
   7  IFM-certified (ifm-certified-label badge)  -> fellowship_level=True
   6  non-certified                              -> fellowship_level=False

The live public directory is NOT IFMCP-only: across metros it mixes
certified and non-certified members (NYC was all-certified; LA ~1/3 and
Chicago ~1/2 non-certified). The fellowship flag is driven by the badge, a
structural signal independent of credential-string spelling.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.ifm import (  # noqa: E402
    parse_listings_html,
    _split_name_credentials,
    _is_certified,
    _extract_email,
    _extract_website,
    _build_source_url,
)


def _load(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


def _rows():
    return parse_listings_html(_load("ifm_listings_la.html"))


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_parse_returns_rows():
    rows = _rows()
    assert len(rows) == 13
    assert len(rows) > 0


def test_locked_invariants():
    rows = _rows()
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "IFM"
        assert r.specialties == ["functional_medicine", "holistic_health"]
        assert r.source_url
        assert r.source_url.startswith("https://www.ifm.org/")
        # Portal-managed / geocoder-owned fields stay None per spec.
        assert r.lat is None
        assert r.lng is None
        assert r.photo_url is None
        assert r.bio is None
        # List cards carry no street/phone — profile-only per spec.
        assert r.phone is None
        assert r.address1 is None
        assert r.city is None
        assert r.state is None
        assert r.postal is None


def test_fields_populate_on_certified_record():
    rows = _rows()
    marino = next(r for r in rows if "Christina Marino" in r.name)
    assert marino.name == "Christina Marino"
    assert marino.credentials == "L.AC, DAOM, DIPL.OM, DIPL.Ac. MS, IFMCP"
    assert marino.email == "ahcconcepts@aol.com"
    assert marino.website == "http://www.alternativehealthcareconcepts.com"
    assert marino.fellowship_level is True
    assert marino.source_url == "https://www.ifm.org/practitioners/christina-marino"


def test_email_and_website_fill_rates():
    rows = _rows()
    assert sum(1 for r in rows if r.email) == 12
    assert sum(1 for r in rows if r.website) == 12
    assert sum(1 for r in rows if r.credentials) == 12


# ---------------------------------------------------------------------------
# Fellowship flag behaviour
# ---------------------------------------------------------------------------

def test_fellowship_count_matches_badged_cards():
    rows = _rows()
    fellows = [r for r in rows if r.fellowship_level]
    assert len(fellows) == 7
    non = [r for r in rows if not r.fellowship_level]
    assert len(non) == 6


def test_fellowship_true_for_badged():
    rows = _rows()
    vojdani = next(r for r in rows if "Elroy Vojdani" in r.name)
    assert vojdani.fellowship_level is True


def test_fellowship_false_for_non_badged():
    rows = _rows()
    shine = next(r for r in rows if "Kim Shine" in r.name)
    assert shine.fellowship_level is False
    waters = next(r for r in rows if "Amy Waters" in r.name)
    assert waters.fellowship_level is False


# ---------------------------------------------------------------------------
# Edge cases captured in the fixture
# ---------------------------------------------------------------------------

def test_card_without_credentials():
    """Christy T. Evans has no comma in the heading -> credentials None."""
    rows = _rows()
    evans = next(r for r in rows if "Christy T. Evans" in r.name)
    assert evans.credentials is None
    assert evans.name == "Christy T. Evans"


def test_card_without_slug_uses_id_anchor():
    """Edmund H. Lew's card has no /practitioners/<slug> link; the
    source_url falls back to the stable id-anchored find-a-practitioner URL."""
    rows = _rows()
    lew = next(r for r in rows if "Edmund H. Lew" in r.name)
    assert lew.source_url == "https://www.ifm.org/find-a-practitioner/#practitioner-4333"
    assert lew.fellowship_level is True


def test_card_without_email():
    """Paris Whitney has no mailto; email stays None but website still fills."""
    rows = _rows()
    whitney = next(r for r in rows if "Paris Whitney" in r.name)
    assert whitney.email is None
    assert whitney.website == "https://www.ParisTheDoctor.com"


def test_polluted_website_is_trimmed():
    """Whitney's href was 'https://...ParisTheDoctor.com and Lifespanmedicine.com'
    — the free-text suffix is cut so a clean single URL is stored."""
    rows = _rows()
    whitney = next(r for r in rows if "Paris Whitney" in r.name)
    assert " " not in (whitney.website or "")
    assert "and" not in (whitney.website or "").split("//")[-1].split("/")[0]


def test_source_urls_stable_and_unique():
    a = _rows()
    b = _rows()
    assert [r.source_url for r in a] == [r.source_url for r in b]
    urls = [r.source_url for r in a]
    assert len(urls) == len(set(urls))


def test_parser_handles_non_string_and_empty():
    assert parse_listings_html(None) == []
    assert parse_listings_html("") == []
    assert parse_listings_html("<html><body>no cards</body></html>") == []


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_split_name_credentials():
    assert _split_name_credentials("Jane Doe, MD, IFMCP") == ("Jane Doe", "MD, IFMCP")
    assert _split_name_credentials("John Smith") == ("John Smith", None)
    assert _split_name_credentials("<h3>  Amy  Lee , ND </h3>") == ("Amy Lee", "ND")
    assert _split_name_credentials("") == ("", None)


def test_is_certified_badge_variants():
    assert _is_certified('<img class="ibfmc_image" src="/themes/ifm/img/IFM-FMCP-M-Button.png">') is True
    assert _is_certified('<img class="ibfmc_image" src="/themes/ifm/img/IFM-FMCP-Button.png">') is True
    assert _is_certified('<div>no badge here</div>') is False


def test_extract_email_shape_validated():
    assert _extract_email('<a href="mailto:x@y.com">x</a>') == "x@y.com"
    assert _extract_email('<a href="mailto:not-an-email">n</a>') is None
    assert _extract_email("<div>nothing</div>") is None


def test_extract_website_skips_ifm_and_trims():
    card = ('<a href="https://www.ifm.org/practitioners/x">internal</a>'
            '<a href="https://practice.com and other.com">site</a>')
    assert _extract_website(card) == "https://practice.com"
    assert _extract_website("<div>nothing</div>") is None


def test_build_source_url_prefers_slug():
    assert _build_source_url("/practitioners/jane-doe", "999") == "https://www.ifm.org/practitioners/jane-doe"
    assert _build_source_url(None, "999") == "https://www.ifm.org/find-a-practitioner/#practitioner-999"
    assert _build_source_url(None, None) is None
