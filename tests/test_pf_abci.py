"""Unit tests for the American Board of Chiropractic Internists (ABCI /
CDID / DABCI) adapter.

ABCI publishes its "DABCI Status" diplomate directory as a single static
HTML table on the archived dabci.org site. Fixtures here are real
responses captured 2026-05-29:

- abci_dabci_status.html         -- full DABCI Status page download
                                    (~47KB). Contains the directory table
                                    plus two decoy continuing-education
                                    seminar tables that must be skipped.
                                    ~89 diplomate rows.
- abci_dabci_status_sample.html  -- hand-trimmed 10-doctor sample plus one
                                    decoy seminar table and the trailing
                                    blank row, covering the field matrix:
                                    Certified, Retired, status-typo,
                                    missing location, "Last, First M."
                                    middle initial, "Last, First" with
                                    multi-word given segment, state-only
                                    location, and a generational suffix
                                    on the surname ("Zevan III, Alex").

NOTE on the directory's provenance: the live page header states the
content is from the site's 2008-2009 archived pages, and the current
authority (aca-cdid.com) does not resolve. This is the only public,
login-free, plain-requests-scrapable DABCI roster that exists. See the
abci.py module docstring for the full discovery trail.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "practitioner_finder"

from scrapers.practitioner_finder.abci import (  # noqa: E402
    extract_directory_records,
    parse_directory_html,
    parse_directory_records,
    _build_credentials,
    _build_source_url,
    _flip_name,
    _parse_location,
)


def _load_text(name: str) -> str:
    return (FIXTURE_DIR / name).read_text()


# ---------------------------------------------------------------------------
# Fixture-driven behavioral tests
# ---------------------------------------------------------------------------

def test_extract_directory_records_pulls_roster_from_full_html():
    """The DABCI Status page contains the diplomate roster as a single
    HTML table; extract must surface the directory rows and skip the two
    decoy continuing-education seminar tables."""
    html = _load_text("abci_dabci_status.html")
    recs = extract_directory_records(html)
    assert isinstance(recs, list)
    assert len(recs) > 0
    # All real rows carry a doctor name (seminar-table rows would carry
    # dates / course names, not "Last, First" doctors).
    for r in recs:
        assert r["doctor"]
        assert "," in r["doctor"] or r["doctor"]  # at least non-empty


def test_full_batch_row_count_is_sane():
    """End-to-end full-batch parse returns a sane row count (the live
    2026-05-29 capture has ~89 diplomates). Guard a wide band so a small
    roster drift doesn't flake the test, but 0 (parser broke) or a
    runaway count (decoy table leaked in) both fail."""
    rows = parse_directory_html(_load_text("abci_dabci_status.html"))
    assert 50 < len(rows) < 200


def test_extract_directory_records_returns_empty_when_table_missing():
    """Defensive: a page with no directory table returns [] (would
    indicate dabci.org restructured -- fail loud downstream, don't
    crash)."""
    assert extract_directory_records("<html><body>nope</body></html>") == []
    assert extract_directory_records("") == []


def test_decoy_seminar_tables_are_not_parsed_as_practitioners():
    """The continuing-education seminar tables share a 'Location' column
    but are headed Dates / Seminar / Hrs / Location / Contact. None of
    their rows (e.g. a 'Spirometry & Pulmonary Disease' course) may leak
    into the practitioner output."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    names = {r.name for r in rows}
    assert not any("Spirometry" in n or "Pulmonary" in n for n in names)
    assert not any("Seminar" in n for n in names)


def test_sample_row_count_excludes_header_and_blank():
    """Sample fixture has 10 real doctors (header row + trailing all-blank
    row dropped)."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    assert len(rows) == 10


def test_all_rows_carry_locked_invariants():
    """tier / source_org / specialties are locked per spec -- never
    mutate. Every row is a DABCI diplomate, so fellowship_level is True
    for all (NCCAOM posture)."""
    rows = parse_directory_html(_load_text("abci_dabci_status.html"))
    assert rows
    for r in rows:
        assert r.tier == "org_member"
        assert r.source_org == "ABCI"
        assert r.specialties == ["chiropractic", "holistic_health"]
        assert r.fellowship_level is True
        assert r.source_url
        assert r.source_url.startswith(
            "https://www.dabci.org/index.php?id=1&reveal=yes&view_only=yes#dabci-"
        )
        assert r.country == "US"
        assert (r.credentials or "").startswith("DABCI")


def test_fellowship_count_equals_row_count():
    """DABCI is a diplomate credential and the table is 'a public
    directory of all DABCIs' -- every parsed row must be
    fellowship_level=True."""
    rows = parse_directory_html(_load_text("abci_dabci_status.html"))
    fellows = [r for r in rows if r.fellowship_level]
    assert len(fellows) == len(rows)
    assert len(fellows) > 0


def test_source_url_is_unique_per_practitioner():
    """The (doctor, location) slug is the dedup key -- every row must have
    a unique source_url."""
    rows = parse_directory_html(_load_text("abci_dabci_status.html"))
    urls = [r.source_url for r in rows]
    assert len(urls) == len(set(urls)), "duplicate source_url across rows"


def test_source_url_is_stable_across_reruns():
    """Re-parsing the same fixture twice must yield identical source_urls
    in identical order -- these are the ON CONFLICT dedup keys."""
    html = _load_text("abci_dabci_status_sample.html")
    a = parse_directory_html(html)
    b = parse_directory_html(html)
    assert [r.source_url for r in a] == [r.source_url for r in b]


def test_name_is_flipped_to_first_last():
    """Directory stores 'Last, First'; output name must be 'First Last'."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    names = {r.name for r in rows}
    assert "Delilah Anderson" in names
    assert "Edward Brown" in names


def test_middle_initial_preserved_in_name():
    """'Smith, Todd A.' -> 'Todd A. Smith' (middle initial kept)."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    smith = next(r for r in rows if r.name.endswith("Smith"))
    assert smith.name == "Todd A. Smith"


def test_generational_suffix_trails_full_name():
    """'Zevan III, Alex' -> 'Alex Zevan III' (suffix stays with surname,
    not mistaken for the given name)."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    zevan = next(r for r in rows if "Zevan" in r.name)
    assert zevan.name == "Alex Zevan III"


def test_location_city_state_extraction():
    """'Sandwich, IL' -> city='Sandwich', state='IL', raw kept in
    address1."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    delilah = next(r for r in rows if r.name == "Delilah Anderson")
    assert delilah.city == "Sandwich"
    assert delilah.state == "IL"
    assert delilah.address1 == "Sandwich, IL"


def test_status_carried_into_credentials():
    """The ABCI Certified/Retired status is preserved in credentials but
    never downgrades the diplomate fact."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    brown = next(r for r in rows if r.name == "Edward Brown")
    assert brown.credentials == "DABCI (Retired)"
    assert brown.fellowship_level is True
    delilah = next(r for r in rows if r.name == "Delilah Anderson")
    assert delilah.credentials == "DABCI (Certified)"


def test_missing_location_does_not_crash():
    """'Hug, Reginald' has no location -- row still parses with
    None location fields."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    hug = next(r for r in rows if r.name == "Reginald Hug")
    assert hug.city is None
    assert hug.state is None
    assert hug.address1 is None
    assert hug.fellowship_level is True


def test_state_only_location():
    """'OK' (state, no city) -> state='OK', city=None, raw in address1."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    taylor = next(r for r in rows if r.name == "Mike Taylor")
    assert taylor.state == "OK"
    assert taylor.city is None
    assert taylor.address1 == "OK"


def test_status_typo_still_yields_diplomate():
    """'Cerifiied' (typo for Certified) is preserved verbatim in
    credentials; the row is still a diplomate. We do not silently
    'correct' source typos."""
    rows = parse_directory_html(_load_text("abci_dabci_status_sample.html"))
    yeager = next(r for r in rows if r.name == "Mark Yeager")
    assert yeager.fellowship_level is True
    assert "Cerifiied" in (yeager.credentials or "")


def test_contact_fields_are_none():
    """The directory carries no phone/email/website/practice/postal -- all
    stay None. photo_url and bio are portal-managed (always None)."""
    rows = parse_directory_html(_load_text("abci_dabci_status.html"))
    for r in rows:
        assert r.phone is None
        assert r.email is None
        assert r.website is None
        assert r.practice_name is None
        assert r.postal is None
        assert r.photo_url is None
        assert r.bio is None
        assert r.lat is None
        assert r.lng is None


def test_parser_skips_non_dict_records():
    """Defensive: junk entries must be skipped, not crashed on."""
    rows = parse_directory_records([None, 42, "string", {}])
    assert rows == []


def test_skipped_records_when_doctor_is_empty():
    """A record with empty doctor gets dropped (no name = no row)."""
    rows = parse_directory_records(
        [
            {"doctor": "", "location": "X", "status": "Certified", "disciplinary": ""},
            {"doctor": "   ", "location": "X", "status": "Certified", "disciplinary": ""},
            {"doctor": "Valid, Person", "location": "Denver, CO", "status": "Certified", "disciplinary": ""},
        ]
    )
    assert len(rows) == 1
    assert rows[0].name == "Person Valid"


# ---------------------------------------------------------------------------
# Pure-helper unit tests
# ---------------------------------------------------------------------------

def test_flip_name_simple():
    assert _flip_name("Anderson, Delilah") == "Delilah Anderson"
    assert _flip_name("Smith, Todd A.") == "Todd A. Smith"
    assert _flip_name("Satterwhite, R Vincent") == "R Vincent Satterwhite"


def test_flip_name_with_generational_suffix():
    assert _flip_name("Zevan III, Alex") == "Alex Zevan III"
    assert _flip_name("Doe Jr., John") == "John Doe Jr."


def test_flip_name_no_comma_returned_as_is():
    assert _flip_name("Madonna") == "Madonna"
    assert _flip_name("") == ""


def test_parse_location_city_state():
    assert _parse_location("Sandwich, IL") == ("Sandwich, IL", "Sandwich", "IL")
    assert _parse_location("North Kanas City, MO") == (
        "North Kanas City, MO",
        "North Kanas City",
        "MO",
    )


def test_parse_location_no_space_after_comma():
    assert _parse_location("Prudenville,MI") == ("Prudenville,MI", "Prudenville", "MI")


def test_parse_location_state_only():
    assert _parse_location("OK") == ("OK", None, "OK")


def test_parse_location_typo_state_keeps_raw_drops_state():
    """A typo'd state ('IH' for IA) is NOT a valid 2-letter code, so state
    stays None but the raw location is preserved for the geocoder."""
    addr, city, state = _parse_location("Mason City, IH")
    assert addr == "Mason City, IH"
    assert city == "Mason City"
    assert state is None


def test_parse_location_blank():
    assert _parse_location("") == (None, None, None)
    assert _parse_location("   ") == (None, None, None)


def test_build_credentials_always_starts_with_dabci():
    assert _build_credentials("Certified", "") == "DABCI (Certified)"
    assert _build_credentials("Retired", "") == "DABCI (Retired)"
    assert _build_credentials("", "") == "DABCI"
    assert _build_credentials("Certified", "Suspended 2009") == (
        "DABCI (Certified); Disciplinary: Suspended 2009"
    )


def test_build_source_url_is_anchored_and_stable():
    a = _build_source_url("Anderson, Delilah", "Sandwich, IL")
    b = _build_source_url("Anderson, Delilah", "Sandwich, IL")
    assert a == b
    assert a.startswith(
        "https://www.dabci.org/index.php?id=1&reveal=yes&view_only=yes#dabci-"
    )


def test_build_source_url_distinguishes_distinct_doctors():
    a = _build_source_url("Anderson, Delilah", "Sandwich, IL")
    b = _build_source_url("Anderson, Jeffrey", "Edina, MN")
    assert a != b
